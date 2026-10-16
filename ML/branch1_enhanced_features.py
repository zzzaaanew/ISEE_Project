"""Telemetry-only Branch 1 feature extension (Falcon-Augmented).

This module keeps the base telemetry extractor and appends:
1. Cross-Metric Physical Interaction features (7 features)
2. Intra-node Spatial Topology features (8 features)
3. Falcon Peer-Relative Normalization features (5 features: z-scores & intra-node ranks)
4. Falcon Temporal Dynamics & Persistence features (3 features: slopes & thermal persistence)

XID history features are strictly excluded: ``xid_count_30d`` and
``days_since_xid`` belong exclusively to the History-only Branch 2 contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import run_bidirectional_adst_fusion as base


ENHANCED_CROSS_METRIC_FEATURES = (
    "thermal_efficiency",
    "power_temp_ratio",
    "util_fb_coupling",
    "power_per_util",
    "temp_fb_divergence",
    "thermal_headroom",
    "power_headroom",
)

ENHANCED_TOPOLOGY_FEATURES = (
    "topo_neighbor_temp_max",
    "topo_neighbor_temp_mean",
    "topo_node_temp_std",
    "topo_temp_spatial_diff",
    "topo_neighbor_power_max",
    "topo_neighbor_power_mean",
    "topo_power_spatial_diff",
    "topo_neighbor_util_mean",
)

ENHANCED_PEER_RELATIVE_FEATURES = (
    "peer_temp_zscore",
    "peer_power_zscore",
    "peer_util_zscore",
    "peer_temp_rank",
    "peer_power_rank",
)

ENHANCED_TEMPORAL_DYNAMICS_FEATURES = (
    "temp_slope",
    "power_slope",
    "thermal_persistence",
)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Compute a bounded ratio so near-zero telemetry cannot dominate models."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=np.abs(denominator) > 1e-4,
        )
    return np.nan_to_num(np.clip(ratio, -100.0, 100.0), nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float32)


class EnhancedTelemetryEngine(base.UnifiedDataEngine):
    """Base engine plus leakage-safe telemetry interactions and Falcon peer-relative features.

    The repository's cached node matrices are arranged in contiguous groups
    of eight GPUs. The ``topology_source`` field is surfaced in the manifest.
    """

    def __init__(self, *args, include_topology: bool = True, include_falcon: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_topology = bool(include_topology)
        self.include_falcon = bool(include_falcon)
        self.topology_source = "contiguous_8_gpu_groups_from_cached_node_matrices"
        
        self.enhanced_feature_names = list(ENHANCED_CROSS_METRIC_FEATURES)
        if self.include_topology:
            self.enhanced_feature_names.extend(ENHANCED_TOPOLOGY_FEATURES)
        if self.include_falcon:
            self.enhanced_feature_names.extend(ENHANCED_PEER_RELATIVE_FEATURES)
            self.enhanced_feature_names.extend(ENHANCED_TEMPORAL_DYNAMICS_FEATURES)

    @staticmethod
    def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
        return frame[name].to_numpy(dtype=np.float32, copy=False)

    def _cross_metric_and_temporal_features(self, frame: pd.DataFrame, l_obs_hours: int) -> dict[str, np.ndarray]:
        util_mean = self._column(frame, f"util_mean_{l_obs_hours}h")
        temp_mean = self._column(frame, f"temp_mean_{l_obs_hours}h")
        power_mean = self._column(frame, f"power_mean_{l_obs_hours}h")
        util_delta = self._column(frame, f"util_delta_{l_obs_hours}h")
        temp_delta = self._column(frame, f"temp_delta_{l_obs_hours}h")
        power_delta = self._column(frame, f"power_delta_{l_obs_hours}h")
        fb_delta = self._column(frame, f"fb_delta_{l_obs_hours}h")
        temp_std = self._column(frame, f"temp_std_{l_obs_hours}h")
        power_std = self._column(frame, f"power_std_{l_obs_hours}h")
        temp_last = self._column(frame, f"temp_last")
        power_last = self._column(frame, f"power_last")

        feats = {
            "thermal_efficiency": _safe_ratio(temp_mean, util_mean),
            "power_temp_ratio": _safe_ratio(power_std, temp_std),
            "util_fb_coupling": (util_delta * fb_delta).astype(np.float32),
            "power_per_util": _safe_ratio(power_mean, util_mean),
            "temp_fb_divergence": np.abs(temp_delta - fb_delta).astype(np.float32),
            "thermal_headroom": (temp_last - temp_mean).astype(np.float32),
            "power_headroom": (power_last - power_mean).astype(np.float32),
        }

        if self.include_falcon:
            # Falcon Temporal Dynamics
            # 1. Slope: rate of change per hour
            h_float = max(float(l_obs_hours), 1.0)
            feats["temp_slope"] = (temp_delta / h_float).astype(np.float32)
            feats["power_slope"] = (power_delta / h_float).astype(np.float32)
            # 2. Persistence: standardized elevation of last temp above window mean
            feats["thermal_persistence"] = _safe_ratio(temp_last - temp_mean, temp_std)

        return feats

    def _topology_and_peer_features(self, bins: np.ndarray, gpus: np.ndarray) -> dict[str, np.ndarray]:
        n = len(bins)
        all_feature_keys = list(ENHANCED_TOPOLOGY_FEATURES)
        if self.include_falcon:
            all_feature_keys.extend(ENHANCED_PEER_RELATIVE_FEATURES)

        result = {name: np.zeros(n, dtype=np.float32) for name in all_feature_keys}
        if not self.include_topology or self.num_gpus % 8 != 0:
            return result

        node_idx = np.asarray(gpus, dtype=np.int64) // 8
        slot_idx = np.asarray(gpus, dtype=np.int64) % 8
        sample_idx = np.arange(n, dtype=np.int64)
        last_bins = np.asarray(bins, dtype=np.int64) - (base.BUFFER_MASK_BINS + 1)

        def neighbours(metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            values = np.asarray(self.matrices[metric][last_bins], dtype=np.float32)
            grouped = values.reshape(n, self.num_gpus // 8, 8)
            node_all = grouped[sample_idx, node_idx, :].copy()
            own = node_all[sample_idx, slot_idx].copy()
            
            # Mask own slot to get peer statistics
            peers_only = node_all.copy()
            peers_only[sample_idx, slot_idx] = np.nan
            with np.errstate(all="ignore"):
                peer_mean = np.nanmean(peers_only, axis=1)
                peer_max = np.nanmax(peers_only, axis=1)
                peer_std = np.nanstd(peers_only, axis=1)
            
            peer_mean = np.nan_to_num(peer_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            peer_max = np.nan_to_num(peer_max, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            peer_std = np.nan_to_num(peer_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            own = np.nan_to_num(own, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            return own, peer_mean, peer_max, peer_std, node_all

        own_temp, neighbour_temp_mean, neighbour_temp_max, temp_spread, temp_node_all = neighbours("temp")
        own_power, neighbour_power_mean, neighbour_power_max, power_spread, power_node_all = neighbours("power")
        own_util, neighbour_util_mean, neighbour_util_max, util_spread, util_node_all = neighbours("util")

        # 1. Base Topology Features
        result.update(
            {
                "topo_neighbor_temp_max": neighbour_temp_max,
                "topo_neighbor_temp_mean": neighbour_temp_mean,
                "topo_node_temp_std": temp_spread,
                "topo_temp_spatial_diff": (own_temp - neighbour_temp_mean).astype(np.float32),
                "topo_neighbor_power_max": neighbour_power_max,
                "topo_neighbor_power_mean": neighbour_power_mean,
                "topo_power_spatial_diff": (own_power - neighbour_power_mean).astype(np.float32),
                "topo_neighbor_util_mean": neighbour_util_mean,
            }
        )

        # 2. Falcon Peer-Relative Features
        if self.include_falcon:
            # Z-scores relative to same-node peers
            result["peer_temp_zscore"] = _safe_ratio(own_temp - neighbour_temp_mean, temp_spread)
            result["peer_power_zscore"] = _safe_ratio(own_power - neighbour_power_mean, power_spread)
            result["peer_util_zscore"] = _safe_ratio(own_util - neighbour_util_mean, util_spread)

            # Peer Ranks within the 8-GPU chassis (1.0 = highest/hottest, 8.0 = lowest/coolest)
            rank_temp = (temp_node_all > own_temp[:, None]).sum(axis=1).astype(np.float32) + 1.0
            rank_power = (power_node_all > own_power[:, None]).sum(axis=1).astype(np.float32) + 1.0
            result["peer_temp_rank"] = rank_temp
            result["peer_power_rank"] = rank_power

        return result

    def extract_branch1_features(self, bins: np.ndarray, gpus: np.ndarray, l_obs_hours: int = 1):
        frame, tensor = super().extract_branch1_features(bins, gpus, l_obs_hours)
        additions = self._cross_metric_and_temporal_features(frame, l_obs_hours)
        additions.update(self._topology_and_peer_features(bins, gpus))
        for name in self.enhanced_feature_names:
            frame[name] = additions[name]
        # Guard the contract explicitly: no history fields may enter B1.
        frame = frame.drop(columns=["xid_count_30d", "days_since_xid"], errors="ignore")
        return frame.fillna(0.0), tensor


def enhanced_feature_manifest(engine: EnhancedTelemetryEngine) -> dict[str, object]:
    return {
        "branch1_feature_contract": "telemetry_only",
        "cross_metric_features": list(ENHANCED_CROSS_METRIC_FEATURES),
        "topology_features": list(ENHANCED_TOPOLOGY_FEATURES) if engine.include_topology else [],
        "peer_relative_features": list(ENHANCED_PEER_RELATIVE_FEATURES) if engine.include_falcon else [],
        "temporal_dynamics_features": list(ENHANCED_TEMPORAL_DYNAMICS_FEATURES) if engine.include_falcon else [],
        "topology_source": engine.topology_source if engine.include_topology else "disabled",
        "history_features_excluded_from_branch1": ["xid_count_30d", "days_since_xid"],
    }


def test_falcon_enhanced_features() -> None:
    """Self-check asserting validity of Falcon feature augmentation."""
    data_dir = Path("../data") if (Path("../data") / "telemetry_5m_util.parquet").exists() else Path("data")
    cache_dir = Path("../outputs/branch1/cache") if (Path("../outputs/branch1/cache") / "telemetry_temp_5m.npy").exists() else Path("outputs/branch1/cache")
    engine = EnhancedTelemetryEngine(data_dir=data_dir, cache_dir=cache_dir, include_topology=True, include_falcon=True)
    
    test_bins = np.array([2000, 2001, 2002], dtype=np.int32)
    test_gpus = np.array([0, 1, 7], dtype=np.int32)
    df, _ = engine.extract_branch1_features(test_bins, test_gpus, l_obs_hours=6)
    
    # Check all new features exist
    for col in ENHANCED_PEER_RELATIVE_FEATURES + ENHANCED_TEMPORAL_DYNAMICS_FEATURES:
        assert col in df.columns, f"Missing feature {col}"
        assert not df[col].isna().any(), f"NaN found in {col}"
        assert not np.isinf(df[col].to_numpy()).any(), f"Inf found in {col}"

    # Check rank bounds
    assert (df["peer_temp_rank"] >= 1.0).all() and (df["peer_temp_rank"] <= 8.0).all(), "Invalid temp rank range"
    assert (df["peer_power_rank"] >= 1.0).all() and (df["peer_power_rank"] <= 8.0).all(), "Invalid power rank range"
    print(f"[Assert Passed] All {len(df.columns)} features generated and validated successfully.")


if __name__ == "__main__":
    test_falcon_enhanced_features()



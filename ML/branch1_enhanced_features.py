"""Telemetry-only Branch 1 feature extension.

This module keeps the GitHub/base telemetry extractor and appends the
report's Cross-Metric and optional intra-node spatial features.  XID history
features are deliberately not added here: ``xid_count_30d`` and
``days_since_xid`` belong to the strict History-only Branch 2 contract.
"""

from __future__ import annotations

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
    """Base engine plus leakage-safe telemetry interactions.

    The repository's cached node matrices are arranged in contiguous groups
    of eight GPUs.  The ``topology_source`` field is surfaced in the manifest
    so the result is not mistaken for a verified physical chassis mapping.
    """

    def __init__(self, *args, include_topology: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_topology = bool(include_topology)
        self.topology_source = "contiguous_8_gpu_groups_from_cached_node_matrices"
        self.enhanced_feature_names = list(ENHANCED_CROSS_METRIC_FEATURES)
        if self.include_topology:
            self.enhanced_feature_names.extend(ENHANCED_TOPOLOGY_FEATURES)

    @staticmethod
    def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
        return frame[name].to_numpy(dtype=np.float32, copy=False)

    def _cross_metric_features(self, frame: pd.DataFrame, l_obs_hours: int) -> dict[str, np.ndarray]:
        util_mean = self._column(frame, f"util_mean_{l_obs_hours}h")
        temp_mean = self._column(frame, f"temp_mean_{l_obs_hours}h")
        power_mean = self._column(frame, f"power_mean_{l_obs_hours}h")
        util_delta = self._column(frame, f"util_delta_{l_obs_hours}h")
        temp_delta = self._column(frame, f"temp_delta_{l_obs_hours}h")
        power_delta = self._column(frame, f"power_delta_{l_obs_hours}h")
        fb_delta = self._column(frame, f"fb_delta_{l_obs_hours}h")
        temp_std = self._column(frame, f"temp_std_{l_obs_hours}h")
        power_std = self._column(frame, f"power_std_{l_obs_hours}h")
        temp_max = self._column(frame, f"temp_last")
        power_max = self._column(frame, f"power_last")
        return {
            "thermal_efficiency": _safe_ratio(temp_mean, util_mean),
            "power_temp_ratio": _safe_ratio(power_std, temp_std),
            "util_fb_coupling": (util_delta * fb_delta).astype(np.float32),
            "power_per_util": _safe_ratio(power_mean, util_mean),
            "temp_fb_divergence": np.abs(temp_delta - fb_delta).astype(np.float32),
            "thermal_headroom": (temp_max - temp_mean).astype(np.float32),
            "power_headroom": (power_max - power_mean).astype(np.float32),
        }

    def _topology_features(self, bins: np.ndarray, gpus: np.ndarray) -> dict[str, np.ndarray]:
        n = len(bins)
        result = {name: np.zeros(n, dtype=np.float32) for name in ENHANCED_TOPOLOGY_FEATURES}
        if not self.include_topology or self.num_gpus % 8 != 0:
            return result

        node_idx = np.asarray(gpus, dtype=np.int64) // 8
        slot_idx = np.asarray(gpus, dtype=np.int64) % 8
        sample_idx = np.arange(n, dtype=np.int64)
        last_bins = np.asarray(bins, dtype=np.int64) - (base.BUFFER_MASK_BINS + 1)

        def neighbours(metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            values = np.asarray(self.matrices[metric][last_bins], dtype=np.float32)
            grouped = values.reshape(n, self.num_gpus // 8, 8)
            selected = grouped[sample_idx, node_idx, :].copy()
            own = selected[sample_idx, slot_idx].copy()
            selected[sample_idx, slot_idx] = np.nan
            with np.errstate(all="ignore"):
                mean = np.nanmean(selected, axis=1)
                maximum = np.nanmax(selected, axis=1)
                spread = np.nanstd(selected, axis=1)
            mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            maximum = np.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            spread = np.nan_to_num(spread, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            own = np.nan_to_num(own, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            return own, mean, maximum if metric != "temp" else spread

        own_temp, neighbour_temp_mean, temp_spread = neighbours("temp")
        own_power, neighbour_power_mean, neighbour_power_max = neighbours("power")
        own_util, neighbour_util_mean, _ = neighbours("util")

        # ``temp_spread`` is the full 7-neighbour spread; the maximum is
        # calculated separately to keep the feature definitions explicit.
        temp_values = np.asarray(self.matrices["temp"][last_bins], dtype=np.float32)
        temp_grouped = temp_values.reshape(n, self.num_gpus // 8, 8)[sample_idx, node_idx, :].copy()
        temp_grouped[sample_idx, slot_idx] = np.nan
        with np.errstate(all="ignore"):
            neighbour_temp_max = np.nanmax(temp_grouped, axis=1)
        neighbour_temp_max = np.nan_to_num(neighbour_temp_max, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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
        return result

    def extract_branch1_features(self, bins: np.ndarray, gpus: np.ndarray, l_obs_hours: int = 1):
        frame, tensor = super().extract_branch1_features(bins, gpus, l_obs_hours)
        additions = self._cross_metric_features(frame, l_obs_hours)
        additions.update(self._topology_features(bins, gpus))
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
        "topology_source": engine.topology_source if engine.include_topology else "disabled",
        "history_features_excluded_from_branch1": ["xid_count_30d", "days_since_xid"],
        "cnn_input": "base_telemetry_sequence_unchanged; enhanced features feed tabular families",
    }


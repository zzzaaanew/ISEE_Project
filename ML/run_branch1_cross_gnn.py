"""
[Branch 1 Enhanced Pipeline] Cross-Metric & Node Topology GNN Spatial Graph Feature Pipeline
- 2D Grid Telemetry Engine with 10-minute Buffer Masking (Leakage-Free)
- Cross-Metric Interaction Features (Thermal Efficiency, Dynamic Power Coupling, Headroom)
- Node Topology GNN Features (Intra-Node 8-GPU Graph Aggregations: Neighbor Spillover, Spatial Imbalance)
- Expanding-Window 4-Fold Out-of-Fold (OOF) Cross-Validation
- Comparison against baseline Branch 1 metrics
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BIN_MINUTES = 5
BIN_NS = BIN_MINUTES * 60 * 1_000_000_000
MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS
DAY_NS = 24 * HOUR_NS

# Leakage-Free Window: [t - 40min, t - 10min] -> Lags 3, 4, 5, 6, 7, 8
FEATURE_GAP_BINS = 2  # 10 min buffer = 2 bins
LAST_FEATURE_LAG = FEATURE_GAP_BINS + 1  # Lag 3 (15m before t)
WINDOW_LAGS = np.arange(LAST_FEATURE_LAG, LAST_FEATURE_LAG + 6)

HORIZON_HOURS = 24
HORIZON_BINS = int(HORIZON_HOURS * 60 / BIN_MINUTES)

FOLDS = [
    ("fold_1", "2023-06-15", "2023-07-01"),
    ("fold_2", "2023-07-01", "2023-07-15"),
    ("fold_3", "2023-07-15", "2023-08-01"),
    ("fold_4", "2023-08-01", "2023-08-18"),
]

METRICS = ["util", "temp", "power", "fb"]

# 1. Base 32 features
BASE_FEATURES = [
    *[
        f"{m}_{stat}"
        for m in METRICS
        for stat in ["last_5m", "mean_30m", "max_30m", "min_30m", "std_30m", "delta_30m"]
    ],
    "util_node_mean_last_5m",
    "temp_node_mean_last_5m",
    "power_node_mean_last_5m",
    "util_diff_node_last_5m",
    "temp_diff_node_last_5m",
    "power_diff_node_last_5m",
    "xid_count_30d",
    "days_since_xid",
]

# 2. Cross-Metric Interaction Features (7 features)
CROSS_METRIC_FEATURES = [
    "thermal_efficiency",     # temp_mean / (util_mean + eps)
    "power_temp_ratio",       # power_std / (temp_std + eps)
    "util_fb_coupling",       # util_delta * fb_delta
    "power_per_util",         # power_mean / (util_mean + eps)
    "temp_fb_divergence",     # abs(temp_delta - fb_delta)
    "thermal_headroom",       # temp_max - temp_mean
    "power_headroom",         # power_max - power_mean
]

# 3. Node Topology GNN Spatial Features (8 features)
GNN_SPATIAL_FEATURES = [
    "gnn_neighbor_temp_max",   # Maximum temperature among 7 sibling GPUs (thermal spillover)
    "gnn_neighbor_temp_mean",  # Average temperature among 7 sibling GPUs
    "gnn_node_temp_std",       # Temperature spread across chassis (cooling imbalance)
    "gnn_temp_spatial_diff",   # Self temp - neighbor temp mean
    "gnn_neighbor_power_max",  # Peak power drawn by sibling GPU on same PSU
    "gnn_neighbor_power_mean", # Average sibling GPU power
    "gnn_power_spatial_diff",  # Self power - neighbor power mean
    "gnn_neighbor_util_mean",  # Average sibling compute utilization
]

FEATURE_NAMES = BASE_FEATURES + CROSS_METRIC_FEATURES + GNN_SPATIAL_FEATURES


def resolve_data_dir(custom_path: str | None = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p.resolve()
        if (PROJECT_ROOT / custom_path).exists():
            return (PROJECT_ROOT / custom_path).resolve()
        if (PROJECT_ROOT.parent / custom_path).exists():
            return (PROJECT_ROOT.parent / custom_path).resolve()

    candidates = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "data",
        Path("data").resolve(),
    ]
    for c in candidates:
        if c.exists() and (c / "telemetry_5m_util.parquet").exists():
            return c.resolve()
    return (PROJECT_ROOT / "data").resolve()


class EnhancedBranch1DataEngine:
    def __init__(self, data_dir: Path, cache_dir: Path):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        candidates_cache = [
            self.cache_dir / "grid_meta.npz",
            PROJECT_ROOT / "outputs" / "branch1" / "cache" / "grid_meta.npz",
            PROJECT_ROOT.parent / "outputs" / "branch1" / "cache" / "grid_meta.npz",
        ]
        meta_cache = None
        for c in candidates_cache:
            if c.exists():
                meta_cache = c
                self.cache_dir = c.parent
                break
        if meta_cache is None:
            raise FileNotFoundError(f"grid_meta.npz not found in any candidate path: {candidates_cache}")

        meta = np.load(meta_cache, allow_pickle=True)
        self.gpu_ids = meta["gpu_ids"]
        self.bin_start_ns = meta["bin_start_ns"]

        self.t0_ns = int(self.bin_start_ns[0])
        self.num_bins = len(self.bin_start_ns)
        self.num_gpus = len(self.gpu_ids)
        self.num_nodes = self.num_gpus // 8
        self.gpu_to_idx = {gpu: i for i, gpu in enumerate(self.gpu_ids)}

        print(
            f"[DataEngine] Grid ready: {self.num_bins:,} bins x {self.num_gpus} GPUs "
            f"({self.num_nodes} nodes). Period: {pd.to_datetime(self.t0_ns, unit='ns', utc=True)} ~ "
            f"{pd.to_datetime(self.bin_start_ns[-1], unit='ns', utc=True)}",
            flush=True,
        )

        self.matrices: dict[str, np.ndarray] = {}
        self.node_matrices: dict[str, np.ndarray] = {}
        self.matrices_3d: dict[str, np.ndarray] = {}  # shape: (num_bins, num_nodes, 8)

    def load_matrices(self) -> None:
        source_cache = self.cache_dir
        if not (source_cache / "telemetry_util_5m.npy").exists():
            source_cache = PROJECT_ROOT / "outputs" / "branch1" / "cache"

        for metric in METRICS:
            mat_path = source_cache / f"telemetry_{metric}_5m.npy"
            node_path = source_cache / f"telemetry_{metric}_node_5m.npy"
            self.matrices[metric] = np.load(mat_path, mmap_mode="r")
            self.node_matrices[metric] = np.load(node_path, mmap_mode="r")
            # 3D view for fast intra-node graph queries: (bins, nodes, 8)
            self.matrices_3d[metric] = self.matrices[metric].reshape(self.num_bins, self.num_nodes, 8)

        print(f"[DataEngine] Loaded cached matrices with 3D intra-node graph views.", flush=True)

    def load_xid_onsets(self) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
        df_xid = pq.read_table(self.data_dir / "xid_onsets_metadata.parquet").to_pandas()
        df_xid = df_xid[df_xid["xid_code"].isin([31, 43])].copy()

        df_xid["onset_time_ns"] = pd.to_datetime(df_xid["onset_time"], utc=True).astype("int64")
        df_xid["gpu_idx"] = df_xid["gpu_id"].map(self.gpu_to_idx)
        df_xid = df_xid[df_xid["gpu_idx"].notna()].copy()
        df_xid["gpu_idx"] = df_xid["gpu_idx"].astype(int)
        df_xid["onset_bin"] = ((df_xid["onset_time_ns"] - self.t0_ns) // BIN_NS).astype(int)

        history = {
            gpu: np.sort(group["onset_time_ns"].to_numpy())
            for gpu, group in df_xid.groupby("gpu_idx")
        }
        return df_xid, history

    def extract_features(self, bins: np.ndarray, gpus: np.ndarray, history: dict[int, np.ndarray]) -> pd.DataFrame:
        n_samples = len(bins)
        feature_dict: dict[str, np.ndarray] = {}

        # 1. Baseline Telemetry Statistics over [t-40m, t-10m]
        for metric in METRICS:
            mat = self.matrices[metric]
            window = np.column_stack([mat[bins - lag, gpus] for lag in WINDOW_LAGS])

            last = window[:, 0].astype(np.float32)
            valid = np.isfinite(window)
            count = valid.sum(axis=1)
            total = np.where(valid, window, 0.0).sum(axis=1)
            mean = np.divide(total, count, out=np.full(n_samples, np.nan, dtype=np.float32), where=count > 0)

            max_val = np.full(n_samples, np.nan, dtype=np.float32)
            min_val = np.full(n_samples, np.nan, dtype=np.float32)
            std_val = np.zeros(n_samples, dtype=np.float32)

            has_valid = count > 0
            if has_valid.any():
                sub_window = np.where(valid[has_valid], window[has_valid], np.nan)
                with np.errstate(all="ignore"):
                    max_val[has_valid] = np.nanmax(sub_window, axis=1)
                    min_val[has_valid] = np.nanmin(sub_window, axis=1)
                    std_val[has_valid] = np.nanstd(sub_window, axis=1)

            feature_dict[f"{metric}_last_5m"] = last
            feature_dict[f"{metric}_mean_30m"] = mean
            feature_dict[f"{metric}_max_30m"] = max_val
            feature_dict[f"{metric}_min_30m"] = min_val
            feature_dict[f"{metric}_std_30m"] = std_val
            feature_dict[f"{metric}_delta_30m"] = last - mean

        # 2. Node Context Relative Differences
        node_indices = gpus // 8
        gpu_slots = gpus % 8
        sample_indices = np.arange(n_samples)
        last_lag_bin = bins - LAST_FEATURE_LAG

        for metric in ["util", "temp", "power"]:
            node_mat = self.node_matrices[metric]
            node_last = node_mat[last_lag_bin, node_indices].astype(np.float32)
            feature_dict[f"{metric}_node_mean_last_5m"] = node_last
            feature_dict[f"{metric}_diff_node_last_5m"] = feature_dict[f"{metric}_last_5m"] - node_last

        # 3. Hardware Aging & Past Failure Frequency
        cutoff_ns = self.bin_start_ns[bins] - 10 * MINUTE_NS
        count_30d = np.zeros(n_samples, dtype=np.float32)
        days_since = np.full(n_samples, 90.0, dtype=np.float32)

        for gpu in np.unique(gpus):
            pos = np.flatnonzero(gpus == gpu)
            events = history.get(int(gpu))
            if events is None or len(events) == 0:
                continue
            gpu_cutoffs = cutoff_ns[pos]
            r = np.searchsorted(events, gpu_cutoffs, side="right")
            l = np.searchsorted(events, gpu_cutoffs - 30 * DAY_NS, side="left")
            count_30d[pos] = r - l

            has_prev = r > 0
            if has_prev.any():
                prev_time = events[r[has_prev] - 1]
                diff_days = (gpu_cutoffs[has_prev] - prev_time) / DAY_NS
                days_since[pos[has_prev]] = np.minimum(diff_days, 90.0)

        feature_dict["xid_count_30d"] = count_30d
        feature_dict["days_since_xid"] = days_since

        # =====================================================================
        # 4. NEW: Cross-Metric Interaction Features
        # =====================================================================
        eps = np.float32(1e-4)
        feature_dict["thermal_efficiency"] = feature_dict["temp_mean_30m"] / (feature_dict["util_mean_30m"] + eps)
        feature_dict["power_temp_ratio"] = feature_dict["power_std_30m"] / (feature_dict["temp_std_30m"] + eps)
        feature_dict["util_fb_coupling"] = feature_dict["util_delta_30m"] * feature_dict["fb_delta_30m"]
        feature_dict["power_per_util"] = feature_dict["power_mean_30m"] / (feature_dict["util_mean_30m"] + eps)
        feature_dict["temp_fb_divergence"] = np.abs(feature_dict["temp_delta_30m"] - feature_dict["fb_delta_30m"])
        feature_dict["thermal_headroom"] = feature_dict["temp_max_30m"] - feature_dict["temp_mean_30m"]
        feature_dict["power_headroom"] = feature_dict["power_max_30m"] - feature_dict["power_mean_30m"]

        # =====================================================================
        # 5. NEW: Node Topology GNN / Spatial Graph Aggregations (8 Intra-Node GPUs)
        # =====================================================================
        # Temperature Spatial Graph Operator
        temp_node_vecs = self.matrices_3d["temp"][last_lag_bin, node_indices, :].copy()  # (N, 8)
        self_temp = temp_node_vecs[sample_indices, gpu_slots]
        node_temp_sum = np.sum(np.nan_to_num(temp_node_vecs, nan=0.0), axis=1)
        neighbor_temp_mean = (node_temp_sum - np.nan_to_num(self_temp, nan=0.0)) / 7.0
        node_temp_std = np.nanstd(temp_node_vecs, axis=1)

        temp_node_vecs[sample_indices, gpu_slots] = -np.inf
        neighbor_temp_max = np.nanmax(temp_node_vecs, axis=1)
        neighbor_temp_max = np.where(np.isfinite(neighbor_temp_max), neighbor_temp_max, self_temp)

        feature_dict["gnn_neighbor_temp_max"] = neighbor_temp_max.astype(np.float32)
        feature_dict["gnn_neighbor_temp_mean"] = neighbor_temp_mean.astype(np.float32)
        feature_dict["gnn_node_temp_std"] = np.nan_to_num(node_temp_std, nan=0.0).astype(np.float32)
        feature_dict["gnn_temp_spatial_diff"] = (self_temp - neighbor_temp_mean).astype(np.float32)

        # Power Spatial Graph Operator
        power_node_vecs = self.matrices_3d["power"][last_lag_bin, node_indices, :].copy()  # (N, 8)
        self_power = power_node_vecs[sample_indices, gpu_slots]
        node_power_sum = np.sum(np.nan_to_num(power_node_vecs, nan=0.0), axis=1)
        neighbor_power_mean = (node_power_sum - np.nan_to_num(self_power, nan=0.0)) / 7.0

        power_node_vecs[sample_indices, gpu_slots] = -np.inf
        neighbor_power_max = np.nanmax(power_node_vecs, axis=1)
        neighbor_power_max = np.where(np.isfinite(neighbor_power_max), neighbor_power_max, self_power)

        feature_dict["gnn_neighbor_power_max"] = neighbor_power_max.astype(np.float32)
        feature_dict["gnn_neighbor_power_mean"] = neighbor_power_mean.astype(np.float32)
        feature_dict["gnn_power_spatial_diff"] = (self_power - neighbor_power_mean).astype(np.float32)

        # Utilization Spatial Graph Operator
        util_node_vecs = self.matrices_3d["util"][last_lag_bin, node_indices, :]
        self_util = util_node_vecs[sample_indices, gpu_slots]
        node_util_sum = np.sum(np.nan_to_num(util_node_vecs, nan=0.0), axis=1)
        neighbor_util_mean = (node_util_sum - np.nan_to_num(self_util, nan=0.0)) / 7.0
        feature_dict["gnn_neighbor_util_mean"] = neighbor_util_mean.astype(np.float32)

        df_out = pd.DataFrame(feature_dict, columns=FEATURE_NAMES).fillna(0.0)
        return df_out

    def build_sample_dataset(
        self, negative_ratio: int = 15, pos_stride: int = 6, seed: int = 20230823
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
        df_xid, history = self.load_xid_onsets()

        print("[DataEngine] Generating 24h onset target pairs...", flush=True)
        min_feature_bin = int(WINDOW_LAGS.max())

        positive_pairs = set()
        for _, row in df_xid.iterrows():
            onset_bin = int(row["onset_bin"])
            gpu = int(row["gpu_idx"])
            start_bin = max(min_feature_bin, onset_bin - HORIZON_BINS)
            end_bin = min(self.num_bins - 1, onset_bin)
            for b in range(start_bin, end_bin + 1, pos_stride):
                positive_pairs.add((b, gpu))

        pos_array = np.array(list(positive_pairs), dtype=np.int32)
        pos_bins = pos_array[:, 0]
        pos_gpus = pos_array[:, 1]
        n_pos = len(pos_bins)
        print(f"[DataEngine] Sampled positive pairs: {n_pos:,}", flush=True)

        n_neg = min(n_pos * negative_ratio, 1_200_000)
        rng = np.random.default_rng(seed)

        neg_bins = rng.integers(min_feature_bin, self.num_bins, size=n_neg, dtype=np.int32)
        neg_gpus = rng.integers(0, self.num_gpus, size=n_neg, dtype=np.int32)

        all_bins = np.concatenate([pos_bins, neg_bins])
        all_gpus = np.concatenate([pos_gpus, neg_gpus])
        labels = np.concatenate([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)])

        print(f"[DataEngine] Extracting Enhanced features ({len(FEATURE_NAMES)} cols) for {len(all_bins):,} samples...", flush=True)
        features_df = self.extract_features(all_bins, all_gpus, history)

        sample_df = pd.DataFrame(
            {
                "decision_time": pd.to_datetime(self.bin_start_ns[all_bins], unit="ns", utc=True),
                "gpu_id": self.gpu_ids[all_gpus],
                "gpu_idx": all_gpus,
                "bin_idx": all_bins,
                "target_24h": labels,
            }
        )
        sample_df = pd.concat([sample_df, features_df], axis=1)

        sample_df["model_fold"] = "warmup"
        for fold_name, start_t, end_t in FOLDS:
            s_ns = pd.Timestamp(start_t, tz="UTC").value
            e_ns = pd.Timestamp(end_t, tz="UTC").value
            mask = (sample_df["decision_time"].astype("int64") >= s_ns) & (sample_df["decision_time"].astype("int64") < e_ns)
            sample_df.loc[mask, "model_fold"] = fold_name

        return sample_df, df_xid, history


def build_model(seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.5,
        class_weight="balanced",
        random_state=seed,
    )


def adjusted_probability(raw_prob: np.ndarray, true_prior: float, sample_prior: float) -> np.ndarray:
    raw_prob = np.clip(raw_prob, 1e-7, 1.0 - 1e-7)
    true_prior = np.clip(true_prior, 1e-7, 1.0 - 1e-7)
    sample_prior = np.clip(sample_prior, 1e-7, 1.0 - 1e-7)

    odds_ratio = (true_prior / (1.0 - true_prior)) / (sample_prior / (1.0 - sample_prior))
    calibrated = (raw_prob * odds_ratio) / (1.0 - raw_prob + raw_prob * odds_ratio)
    return np.clip(calibrated, 0.0, 1.0)


def run_oof_pipeline(
    engine: EnhancedBranch1DataEngine,
    sample_df: pd.DataFrame,
    df_xid: pd.DataFrame,
    history: dict[int, np.ndarray],
    output_dir: Path,
    top_k: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=======================================================", flush=True)
    print("Running Enhanced Branch 1 (Cross-Metric + GNN) 4-Fold OOF Evaluation", flush=True)
    print("=======================================================\n", flush=True)

    gt_matrix = np.zeros((engine.num_bins, engine.num_gpus), dtype=bool)
    for _, row in df_xid.iterrows():
        onset_bin = int(row["onset_bin"])
        gpu = int(row["gpu_idx"])
        start_bin = max(int(WINDOW_LAGS.max()), onset_bin - HORIZON_BINS)
        end_bin = min(engine.num_bins - 1, onset_bin)
        gt_matrix[start_bin : end_bin + 1, gpu] = True

    fold_metrics: list[dict] = []
    prediction_dfs: list[pd.DataFrame] = []
    sample_times_ns = sample_df["decision_time"].astype("int64").to_numpy()

    for fold_idx, (fold_name, val_start_str, val_end_str) in enumerate(FOLDS, start=1):
        val_start_ns = pd.Timestamp(val_start_str, tz="UTC").value
        val_end_ns = pd.Timestamp(val_end_str, tz="UTC").value

        val_start_bin = int(np.searchsorted(engine.bin_start_ns, val_start_ns))
        val_end_bin = int(np.searchsorted(engine.bin_start_ns, val_end_ns))
        val_end_bin = min(val_end_bin, engine.num_bins)

        print(f"\n--- [Fold {fold_idx}/4: {fold_name}] ---", flush=True)
        print(f"Validation: {pd.to_datetime(val_start_ns, unit='ns', utc=True)} ~ {pd.to_datetime(val_end_ns, unit='ns', utc=True)}", flush=True)

        train_mask = sample_times_ns < val_start_ns
        X_train = sample_df.loc[train_mask, FEATURE_NAMES]
        y_train = sample_df.loc[train_mask, "target_24h"].to_numpy()

        train_pos = int(y_train.sum())
        sample_prior = train_pos / max(len(y_train), 1)

        true_train_pos = int(gt_matrix[int(WINDOW_LAGS.max()) : val_start_bin].sum())
        true_train_total = (val_start_bin - int(WINDOW_LAGS.max())) * engine.num_gpus
        true_prior = true_train_pos / max(true_train_total, 1)

        print(f"Train samples: {len(y_train):,} (Pos: {train_pos:,}, Prior: {true_prior:.4f})", flush=True)

        t_fit_start = time.time()
        model = build_model(seed=20230823 + fold_idx)
        model.fit(X_train, y_train)
        print(f"  Model fit in {time.time() - t_fit_start:.1f}s", flush=True)

        # Full Grid OOF Inference in Chunks
        chunk_size = 128
        val_scores = np.zeros((val_end_bin - val_start_bin, engine.num_gpus), dtype=np.float32)
        val_ranks = np.zeros((val_end_bin - val_start_bin, engine.num_gpus), dtype=np.uint16)

        for c_start in range(val_start_bin, val_end_bin, chunk_size):
            c_end = min(c_start + chunk_size, val_end_bin)
            n_bins = c_end - c_start

            batch_bins = np.repeat(np.arange(c_start, c_end, dtype=np.int32), engine.num_gpus)
            batch_gpus = np.tile(np.arange(engine.num_gpus, dtype=np.int32), n_bins)

            feat_chunk = engine.extract_features(batch_bins, batch_gpus, history)
            raw_probs = model.predict_proba(feat_chunk)[:, 1]
            calib_probs = adjusted_probability(raw_probs, true_prior, sample_prior)

            score_grid = calib_probs.reshape(n_bins, engine.num_gpus)
            rel_start, rel_end = c_start - val_start_bin, c_end - val_start_bin
            val_scores[rel_start:rel_end] = score_grid.astype(np.float32)

            order = np.argsort(-score_grid, axis=1, kind="stable")
            ranks = np.empty_like(order, dtype=np.uint16)
            np.put_along_axis(ranks, order, np.arange(1, engine.num_gpus + 1, dtype=np.uint16)[None, :], axis=1)
            val_ranks[rel_start:rel_end] = ranks

        y_val_true = gt_matrix[val_start_bin:val_end_bin]
        val_positives = int(y_val_true.sum())
        val_prevalence = float(y_val_true.mean())

        y_flat, pred_flat = y_val_true.ravel(), val_scores.ravel()
        auc = float(roc_auc_score(y_flat, pred_flat))
        ap = float(average_precision_score(y_flat, pred_flat))
        brier = float(brier_score_loss(y_flat, pred_flat))

        metric_entry = {
            "model_fold": fold_name,
            "validation_start": pd.to_datetime(val_start_ns, unit="ns", utc=True),
            "validation_end": pd.to_datetime(val_end_ns, unit="ns", utc=True),
            "val_bins": val_end_bin - val_start_bin,
            "total_decisions": len(y_flat),
            "positive_decisions": val_positives,
            "prevalence": val_prevalence,
            "roc_auc": auc,
            "average_precision": ap,
            "brier_score": brier,
        }

        for k in [10, 20, 50, 100]:
            hits = int((y_val_true & (val_ranks <= k)).sum())
            recall_k = hits / max(val_positives, 1)
            lift_k = (hits / ((val_end_bin - val_start_bin) * k)) / max(val_prevalence, 1e-12)
            metric_entry[f"recall_at_{k}"] = recall_k
            metric_entry[f"lift_at_{k}"] = lift_k

        fold_metrics.append(metric_entry)
        print(f"[{fold_name}] ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f} | R@100: {metric_entry['recall_at_100']:.1%} (Lift: {metric_entry['lift_at_100']:.1f}x)", flush=True)

        val_times_ns = engine.bin_start_ns[val_start_bin:val_end_bin]
        r_mask = val_ranks <= top_k
        bin_idx_rel, gpu_idx_rel = np.where(r_mask)

        sample_t = val_times_ns[bin_idx_rel]
        fold_df = pd.DataFrame(
            {
                "decision_time": pd.to_datetime(sample_t, unit="ns", utc=True),
                "gpu_id": engine.gpu_ids[gpu_idx_rel],
                "p1_calibrated_risk": val_scores[bin_idx_rel, gpu_idx_rel],
                "risk_rank": val_ranks[bin_idx_rel, gpu_idx_rel],
                "model_fold": fold_name,
                "feature_cutoff_time": pd.to_datetime(sample_t - 10 * MINUTE_NS, unit="ns", utc=True),
                "target_horizon_hours": HORIZON_HOURS,
                "ground_truth_label": y_val_true[bin_idx_rel, gpu_idx_rel].astype(np.uint8),
            }
        ).sort_values(["decision_time", "risk_rank"]).reset_index(drop=True)
        prediction_dfs.append(fold_df)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(output_dir / "branch1_cross_gnn_fold_metrics.csv", index=False)

    final_predictions_df = pd.concat(prediction_dfs, axis=0, ignore_index=True)
    final_predictions_df.to_parquet(output_dir / "branch1_cross_gnn_predictions.parquet", index=False, compression="zstd")
    print(f"[Trainer] Saved {len(final_predictions_df):,} predictions to {output_dir / 'branch1_cross_gnn_predictions.parquet'}", flush=True)

    _generate_report(metrics_df, output_dir)
    return metrics_df, final_predictions_df


def _generate_report(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    # Load original baseline metrics for comparison
    orig_path = PROJECT_ROOT / "outputs" / "branch1" / "branch1_fold_metrics.csv"
    orig_df = pd.read_csv(orig_path) if orig_path.exists() else None

    mean_ap = metrics_df["average_precision"].mean()
    mean_auc = metrics_df["roc_auc"].mean()
    mean_r100 = metrics_df["recall_at_100"].mean()
    mean_lift100 = metrics_df["lift_at_100"].mean()

    orig_mean_ap = orig_df["average_precision"].mean() if orig_df is not None else 0.0163
    orig_mean_auc = orig_df["roc_auc"].mean() if orig_df is not None else 0.6013
    orig_mean_r100 = orig_df["recall_at_100"].mean() if orig_df is not None else 0.1167
    orig_mean_lift100 = orig_df["lift_at_100"].mean() if orig_df is not None else 2.326

    ap_diff = (mean_ap - orig_mean_ap) / orig_mean_ap * 100
    auc_diff = (mean_auc - orig_mean_auc) / orig_mean_auc * 100
    r100_diff = (mean_r100 - orig_mean_r100) / orig_mean_r100 * 100

    report_lines = [
        "# [Branch 1 Enhanced] Cross-Metric + Node Topology GNN 성능 비교 보고서",
        "",
        "## 1. 아키텍처 개요",
        f"- **기본 피처**: 32개 시계열 텔레메트리 통계",
        f"- **Cross-Metric 피처 (7개)**: Thermal Efficiency, Power-Temp Ratio, Workload Coupling 등",
        f"- **Node Topology GNN 피처 (8개)**: Intra-Node 8-GPU Graph Aggregation (Neighbor Spillover, Spatial Imbalance)",
        f"- **총 피처 수**: {len(FEATURE_NAMES)}개",
        "",
        "## 2. 4-Fold OOF 정량적 성능 비교",
        "",
        "| 구분 | 베이스라인 Branch 1 | Enhanced (Cross-Metric + GNN) | 향상도 |",
        "| :--- | :---: | :---: | :---: |",
        f"| **평균 PR-AUC (AP)** | `{orig_mean_ap:.4f}` | **`{mean_ap:.4f}`** | **`{ap_diff:+.1f}%`** |",
        f"| **평균 ROC-AUC** | `{orig_mean_auc:.4f}` | **`{mean_auc:.4f}`** | **`{auc_diff:+.1f}%`** |",
        f"| **Top-100 Recall** | `{orig_mean_r100:.1%}` | **`{mean_r100:.1%}`** | **`{r100_diff:+.1f}%`** |",
        f"| **Top-100 Lift** | `{orig_mean_lift100:.2f}x` | **`{mean_lift100:.2f}x`** | - |",
        "",
        "## 3. Fold별 상세 결과",
        "",
        "| Fold | PR-AUC (AP) | ROC-AUC | Recall@100 | Lift@100 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for _, row in metrics_df.iterrows():
        report_lines.append(
            f"| **{row['model_fold']}** | `{row['average_precision']:.4f}` | `{row['roc_auc']:.4f}` | `{row['recall_at_100']:.1%}` | `{row['lift_at_100']:.2f}x` |"
        )

    (output_dir / "branch1_cross_gnn_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[Report] Saved comparison report to {output_dir / 'branch1_cross_gnn_report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Branch 1 Enhanced Cross-Metric & GNN Pipeline")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/branch1_cross_gnn")
    parser.add_argument("--negative-ratio", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20230823)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "outputs" / "branch1" / "cache"

    engine = EnhancedBranch1DataEngine(data_dir, cache_dir)
    engine.load_matrices()

    sample_df, df_xid, history = engine.build_sample_dataset(negative_ratio=args.negative_ratio, seed=args.seed)
    run_oof_pipeline(engine, sample_df, df_xid, history, output_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()

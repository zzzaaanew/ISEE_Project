"""
[Branch 1 Multi-Model Benchmark] Comprehensive Comparison of ML/DL Architectures for GPU Failure Prediction
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
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# =====================================================================
# Constants & Folds
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BIN_MINUTES = 5
BIN_NS = BIN_MINUTES * 60 * 1_000_000_000
MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS
DAY_NS = 24 * HOUR_NS

FEATURE_GAP_BINS = 2
LAST_FEATURE_LAG = FEATURE_GAP_BINS + 1
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

FEATURE_NAMES = [
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


def resolve_paths(custom_data: str | None = None, custom_cache: str | None = None) -> tuple[Path, Path]:
    candidates = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "data",
        Path("data").resolve(),
    ]
    data_dir = None
    if custom_data and Path(custom_data).exists():
        data_dir = Path(custom_data).resolve()
    else:
        for c in candidates:
            if c.exists() and (c / "telemetry_5m_util.parquet").exists():
                data_dir = c.resolve()
                break
        if data_dir is None:
            data_dir = (PROJECT_ROOT / "data").resolve()

    cache_candidates = [
        PROJECT_ROOT / "outputs" / "branch1" / "cache",
        PROJECT_ROOT.parent / "outputs" / "branch1" / "cache",
        PROJECT_ROOT / "outputs" / "cache",
    ]
    cache_dir = None
    if custom_cache and Path(custom_cache).exists():
        cache_dir = Path(custom_cache).resolve()
    else:
        for c in cache_candidates:
            if c.exists() and (c / "grid_meta.npz").exists():
                cache_dir = c.resolve()
                break
        if cache_dir is None:
            cache_dir = (PROJECT_ROOT / "outputs" / "branch1" / "cache").resolve()

    return data_dir, cache_dir


# =====================================================================
# Data Engine
# =====================================================================
class FastDataEngine:
    def __init__(self, data_dir: Path, cache_dir: Path):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        meta = np.load(self.cache_dir / "grid_meta.npz", allow_pickle=True)
        self.gpu_ids = meta["gpu_ids"]
        self.bin_start_ns = meta["bin_start_ns"]
        self.t0_ns = int(self.bin_start_ns[0])
        self.num_bins = len(self.bin_start_ns)
        self.num_gpus = len(self.gpu_ids)
        self.num_nodes = self.num_gpus // 8
        self.gpu_to_idx = {g: i for i, g in enumerate(self.gpu_ids)}

        self.matrices = {
            m: np.load(self.cache_dir / f"telemetry_{m}_5m.npy", mmap_mode="r")
            for m in METRICS
        }
        self.node_matrices = {
            m: np.load(self.cache_dir / f"telemetry_{m}_node_5m.npy", mmap_mode="r")
            for m in ["util", "temp", "power"]
        }

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

        node_indices = gpus // 8
        for metric in ["util", "temp", "power"]:
            node_mat = self.node_matrices[metric]
            node_last = node_mat[bins - LAST_FEATURE_LAG, node_indices].astype(np.float32)
            feature_dict[f"{metric}_node_mean_last_5m"] = node_last
            feature_dict[f"{metric}_diff_node_last_5m"] = feature_dict[f"{metric}_last_5m"] - node_last

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

        return pd.DataFrame(feature_dict, columns=FEATURE_NAMES)

    def build_sample_dataset(self, negative_ratio: int = 15, pos_stride: int = 6, seed: int = 20230823):
        df_xid, history = self.load_xid_onsets()
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
        pos_flat = pos_bins.astype(np.int64) * self.num_gpus + pos_gpus

        rng = np.random.default_rng(seed)
        fold_boundaries = [
            min_feature_bin,
            *[int(np.searchsorted(self.bin_start_ns, pd.Timestamp(s, tz="UTC").value)) for _, s, _ in FOLDS],
            int(np.searchsorted(self.bin_start_ns, pd.Timestamp(FOLDS[-1][2], tz="UTC").value)),
        ]

        pos_set = set(pos_flat.tolist())
        negatives = set()

        for left, right in zip(fold_boundaries[:-1], fold_boundaries[1:]):
            pos_in_seg = int(((pos_bins >= left) & (pos_bins < right)).sum())
            target_neg = min(negative_ratio * pos_in_seg, 250_000)
            segment_neg = set()

            while len(segment_neg) < target_neg:
                batch_size = max(5000, 2 * (target_neg - len(segment_neg)))
                sample_bins = rng.integers(left, right, size=batch_size, dtype=np.int32)
                sample_gpus = rng.integers(0, self.num_gpus, size=batch_size, dtype=np.int32)
                flats = sample_bins.astype(np.int64) * self.num_gpus + sample_gpus
                for f in flats:
                    if f not in pos_set and f not in segment_neg:
                        segment_neg.add(f)
                    if len(segment_neg) >= target_neg:
                        break
            negatives.update(segment_neg)

        all_flat = np.r_[pos_flat, np.fromiter(negatives, dtype=np.int64)]
        labels = np.r_[np.ones(len(pos_flat), dtype=np.uint8), np.zeros(len(negatives), dtype=np.uint8)]
        all_bins = (all_flat // self.num_gpus).astype(np.int32)
        all_gpus = (all_flat % self.num_gpus).astype(np.int32)

        order = np.lexsort((all_gpus, all_bins))
        all_flat, labels, all_bins, all_gpus = all_flat[order], labels[order], all_bins[order], all_gpus[order]

        features_df = self.extract_features(all_bins, all_gpus, history)
        sample_df = pd.DataFrame(
            {
                "sample_id": all_flat,
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


# =====================================================================
# Model Zoo Definitions
# =====================================================================
def get_model_zoo(seed: int = 42) -> dict[str, object]:
    return {
        "HistGBDT": HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.5,
            class_weight="balanced",
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=30,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=30,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "MLP_NeuralNet": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.01,
                learning_rate_init=0.005,
                max_iter=50,
                early_stopping=True,
                random_state=seed,
            ),
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=200,
                random_state=seed,
            ),
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=80,
            learning_rate=0.08,
            random_state=seed,
        ),
    }


def adjusted_probability(raw_prob: np.ndarray, true_prior: float, sample_prior: float) -> np.ndarray:
    raw_prob = np.clip(raw_prob, 1e-7, 1.0 - 1e-7)
    true_prior = np.clip(true_prior, 1e-7, 1.0 - 1e-7)
    sample_prior = np.clip(sample_prior, 1e-7, 1.0 - 1e-7)
    odds_ratio = (true_prior / (1.0 - true_prior)) / (sample_prior / (1.0 - sample_prior))
    calibrated = (raw_prob * odds_ratio) / (1.0 - raw_prob + raw_prob * odds_ratio)
    return np.clip(calibrated, 0.0, 1.0)


# =====================================================================
# Benchmark Runner
# =====================================================================
def run_benchmark(
    engine: FastDataEngine,
    sample_df: pd.DataFrame,
    df_xid: pd.DataFrame,
    history: dict[int, np.ndarray],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_matrix = np.zeros((engine.num_bins, engine.num_gpus), dtype=bool)
    for _, row in df_xid.iterrows():
        onset_bin = int(row["onset_bin"])
        gpu = int(row["gpu_idx"])
        start_bin = max(int(WINDOW_LAGS.max()), onset_bin - HORIZON_BINS)
        end_bin = min(engine.num_bins - 1, onset_bin)
        gt_matrix[start_bin : end_bin + 1, gpu] = True

    model_zoo = get_model_zoo()
    all_benchmark_rows: list[dict] = []
    sample_times_ns = sample_df["decision_time"].astype("int64").to_numpy()
    imputed_features = sample_df[FEATURE_NAMES].fillna(sample_df[FEATURE_NAMES].median())

    for model_name, model_template in model_zoo.items():
        print(f"\nEvaluating Model: [{model_name}]...", flush=True)
        fold_aucs, fold_aps, fold_r10, fold_r50, fold_r100, fold_lifts, fold_briers = [], [], [], [], [], [], []
        total_train_time, total_infer_time = 0.0, 0.0

        for fold_idx, (fold_name, val_start_str, val_end_str) in enumerate(FOLDS, start=1):
            val_start_ns = pd.Timestamp(val_start_str, tz="UTC").value
            val_end_ns = pd.Timestamp(val_end_str, tz="UTC").value

            val_start_bin = int(np.searchsorted(engine.bin_start_ns, val_start_ns))
            val_end_bin = min(int(np.searchsorted(engine.bin_start_ns, val_end_ns)), engine.num_bins)

            train_mask = sample_times_ns < val_start_ns
            if model_name == "HistGBDT":
                X_train = sample_df.loc[train_mask, FEATURE_NAMES]
            else:
                X_train = imputed_features.loc[train_mask, FEATURE_NAMES]
            
            y_train = sample_df.loc[train_mask, "target_24h"].to_numpy()
            train_pos = int(y_train.sum())
            sample_prior = train_pos / max(len(y_train), 1)

            true_train_pos = int(gt_matrix[int(WINDOW_LAGS.max()) : val_start_bin].sum())
            true_train_total = (val_start_bin - int(WINDOW_LAGS.max())) * engine.num_gpus
            true_prior = true_train_pos / max(true_train_total, 1)

            t_fit_start = time.time()
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_train, y_train)
            t_fit = time.time() - t_fit_start
            total_train_time += t_fit

            chunk_size = 128
            val_scores = np.zeros((val_end_bin - val_start_bin, engine.num_gpus), dtype=np.float32)
            val_ranks = np.zeros((val_end_bin - val_start_bin, engine.num_gpus), dtype=np.uint16)

            t_infer_start = time.time()
            for c_start in range(val_start_bin, val_end_bin, chunk_size):
                c_end = min(c_start + chunk_size, val_end_bin)
                n_bins = c_end - c_start

                batch_bins = np.repeat(np.arange(c_start, c_end, dtype=np.int32), engine.num_gpus)
                batch_gpus = np.tile(np.arange(engine.num_gpus, dtype=np.int32), n_bins)

                feat_chunk = engine.extract_features(batch_bins, batch_gpus, history)
                if model_name != "HistGBDT":
                    feat_chunk = feat_chunk.fillna(sample_df[FEATURE_NAMES].median())

                raw_probs = model.predict_proba(feat_chunk)[:, 1]
                calib_probs = adjusted_probability(raw_probs, true_prior, sample_prior)

                score_grid = calib_probs.reshape(n_bins, engine.num_gpus)
                rel_start, rel_end = c_start - val_start_bin, c_end - val_start_bin
                val_scores[rel_start:rel_end] = score_grid.astype(np.float32)

                order = np.argsort(-score_grid, axis=1, kind="stable")
                ranks = np.empty_like(order, dtype=np.uint16)
                np.put_along_axis(ranks, order, np.arange(1, engine.num_gpus + 1, dtype=np.uint16)[None, :], axis=1)
                val_ranks[rel_start:rel_end] = ranks

            total_infer_time += (time.time() - t_infer_start)

            y_val_true = gt_matrix[val_start_bin:val_end_bin]
            val_positives = int(y_val_true.sum())
            val_prevalence = float(y_val_true.mean())

            y_flat, pred_flat = y_val_true.ravel(), val_scores.ravel()
            auc = float(roc_auc_score(y_flat, pred_flat))
            ap = float(average_precision_score(y_flat, pred_flat))
            brier = float(brier_score_loss(y_flat, pred_flat))

            hits_10 = int((y_val_true & (val_ranks <= 10)).sum())
            hits_50 = int((y_val_true & (val_ranks <= 50)).sum())
            hits_100 = int((y_val_true & (val_ranks <= 100)).sum())

            r10 = hits_10 / max(val_positives, 1)
            r50 = hits_50 / max(val_positives, 1)
            r100 = hits_100 / max(val_positives, 1)
            lift100 = (hits_100 / ((val_end_bin - val_start_bin) * 100)) / max(val_prevalence, 1e-12)

            fold_aucs.append(auc)
            fold_aps.append(ap)
            fold_r10.append(r10)
            fold_r50.append(r50)
            fold_r100.append(r100)
            fold_lifts.append(lift100)
            fold_briers.append(brier)

        mean_auc = np.mean(fold_aucs)
        mean_ap = np.mean(fold_aps)
        mean_r10 = np.mean(fold_r10)
        mean_r50 = np.mean(fold_r50)
        mean_r100 = np.mean(fold_r100)
        mean_lift = np.mean(fold_lifts)
        mean_brier = np.mean(fold_briers)
        total_eval_bins = sum((min(int(np.searchsorted(engine.bin_start_ns, pd.Timestamp(e, tz="UTC").value)), engine.num_bins) - int(np.searchsorted(engine.bin_start_ns, pd.Timestamp(s, tz="UTC").value))) for _, s, e in FOLDS)
        infer_latency_ms = (total_infer_time / total_eval_bins) * 1000.0

        all_benchmark_rows.append({
            "model": model_name,
            "mean_roc_auc": mean_auc,
            "mean_pr_auc": mean_ap,
            "mean_recall_at_10": mean_r10,
            "mean_recall_at_50": mean_r50,
            "mean_recall_at_100": mean_r100,
            "mean_lift_at_100": mean_lift,
            "mean_brier_score": mean_brier,
            "total_train_time_sec": total_train_time,
            "epoch_infer_latency_ms": infer_latency_ms,
        })
        print(f"[{model_name}] AUC: {mean_auc:.4f} | AP: {mean_ap:.4f} | R@100: {mean_r100:.1%} | Lift@100: {mean_lift:.2f}x | Latency: {infer_latency_ms:.2f}ms/epoch", flush=True)

    summary_df = pd.DataFrame(all_benchmark_rows).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
    summary_df.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    return summary_df


def main() -> None:
    data_dir, cache_dir = resolve_paths()
    output_dir = PROJECT_ROOT / "outputs" / "comparison"

    print("=================================================================", flush=True)
    print(" [Multi-Model Benchmark] Starting Evaluation", flush=True)
    print(f" Data Directory   : {data_dir}", flush=True)
    print(f" Output Directory : {output_dir}", flush=True)
    print("=================================================================\n", flush=True)

    engine = FastDataEngine(data_dir=data_dir, cache_dir=cache_dir)
    sample_df, df_xid, history = engine.build_sample_dataset(negative_ratio=15, seed=20230823)

    summary_df = run_benchmark(
        engine=engine,
        sample_df=sample_df,
        df_xid=df_xid,
        history=history,
        output_dir=output_dir,
    )

    print("\n=================================================================", flush=True)
    print(" [Benchmark Completed] Summary Table:")
    print(summary_df[["model", "mean_roc_auc", "mean_pr_auc", "mean_recall_at_100", "mean_lift_at_100", "epoch_infer_latency_ms"]].to_string(index=False))
    print(f"\n Saved to: {output_dir}")
    print("=================================================================\n", flush=True)


if __name__ == "__main__":
    main()

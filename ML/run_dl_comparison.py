"""
[Branch 1 Deep Learning Benchmark] Sequential Deep Learning Models (TCN / 1D-CNN / LSTM / GRU) for GPU Failure Prediction
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
WINDOW_LAGS = np.arange(LAST_FEATURE_LAG, LAST_FEATURE_LAG + 6)  # 6 time steps: Lags 3..8

HORIZON_HOURS = 24
HORIZON_BINS = int(HORIZON_HOURS * 60 / BIN_MINUTES)

FOLDS = [
    ("fold_1", "2023-06-15", "2023-07-01"),
    ("fold_2", "2023-07-01", "2023-07-15"),
    ("fold_3", "2023-07-15", "2023-08-01"),
    ("fold_4", "2023-08-01", "2023-08-18"),
]

METRICS = ["util", "temp", "power", "fb"]


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
# Deep Learning Sequence Architectures
# =====================================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))
        return self.relu(out + residual)


class TinyTCN(nn.Module):
    def __init__(self, in_channels: int = 7, num_channels: list[int] = [32, 64], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.network(x)
        pool = torch.mean(out, dim=2)
        return self.fc(pool).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 7, hidden_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)) + h)
        pooled = self.pool(h).squeeze(-1)
        return self.fc(self.dropout(pooled)).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, in_channels: int = 7, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, in_channels: int = 7, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


# =====================================================================
# Data Engine for 3D Sequence Tensors
# =====================================================================
class DLSequenceEngine:
    def __init__(self, data_dir: Path, cache_dir: Path):
        self.data_dir = data_dir
        self.cache_dir = cache_dir

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

    def extract_3d_tensor(self, bins: np.ndarray, gpus: np.ndarray) -> np.ndarray:
        n_samples = len(bins)
        chronological_lags = WINDOW_LAGS[::-1]
        seq_len = len(chronological_lags)
        channels = 7

        tensor = np.zeros((n_samples, seq_len, channels), dtype=np.float32)
        node_indices = gpus // 8

        for l_idx, lag in enumerate(chronological_lags):
            tensor[:, l_idx, 0] = self.matrices["util"][bins - lag, gpus]
            tensor[:, l_idx, 1] = self.matrices["temp"][bins - lag, gpus]
            tensor[:, l_idx, 2] = self.matrices["power"][bins - lag, gpus]
            tensor[:, l_idx, 3] = self.matrices["fb"][bins - lag, gpus]
            tensor[:, l_idx, 4] = self.node_matrices["util"][bins - lag, node_indices]
            tensor[:, l_idx, 5] = self.node_matrices["temp"][bins - lag, node_indices]
            tensor[:, l_idx, 6] = self.node_matrices["power"][bins - lag, node_indices]

        np.nan_to_num(tensor, copy=False, nan=0.0)
        return tensor

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

        sample_times_ns = self.bin_start_ns[all_bins]
        return all_bins, all_gpus, labels, sample_times_ns, df_xid, history


def adjusted_probability(raw_prob: np.ndarray, true_prior: float, sample_prior: float) -> np.ndarray:
    raw_prob = np.clip(raw_prob, 1e-7, 1.0 - 1e-7)
    true_prior = np.clip(true_prior, 1e-7, 1.0 - 1e-7)
    sample_prior = np.clip(sample_prior, 1e-7, 1.0 - 1e-7)
    odds_ratio = (true_prior / (1.0 - true_prior)) / (sample_prior / (1.0 - sample_prior))
    calibrated = (raw_prob * odds_ratio) / (1.0 - raw_prob + raw_prob * odds_ratio)
    return np.clip(calibrated, 0.0, 1.0)


def run_dl_benchmark(
    engine: DLSequenceEngine,
    all_bins: np.ndarray,
    all_gpus: np.ndarray,
    labels: np.ndarray,
    sample_times_ns: np.ndarray,
    df_xid: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    gt_matrix = np.zeros((engine.num_bins, engine.num_gpus), dtype=bool)
    for _, row in df_xid.iterrows():
        onset_bin = int(row["onset_bin"])
        gpu = int(row["gpu_idx"])
        start_bin = max(int(WINDOW_LAGS.max()), onset_bin - HORIZON_BINS)
        end_bin = min(engine.num_bins - 1, onset_bin)
        gt_matrix[start_bin : end_bin + 1, gpu] = True

    print(f"[DL Engine] Extracting 3D sequence tensors for {len(all_bins):,} training samples...", flush=True)
    X_all_tensor = engine.extract_3d_tensor(all_bins, all_gpus)

    mean_c = X_all_tensor.mean(axis=(0, 1), keepdims=True)
    std_c = X_all_tensor.std(axis=(0, 1), keepdims=True) + 1e-6
    X_all_norm = (X_all_tensor - mean_c) / std_c

    dl_models = {
        "Tiny_TCN": lambda: TinyTCN(in_channels=7, num_channels=[32, 64], kernel_size=3, dropout=0.2),
        "1D_CNN": lambda: CNN1D(in_channels=7, hidden_channels=64, dropout=0.2),
        "LSTM": lambda: LSTMModel(in_channels=7, hidden_size=64, num_layers=2, dropout=0.2),
        "GRU": lambda: GRUModel(in_channels=7, hidden_size=64, num_layers=2, dropout=0.2),
    }

    dl_benchmark_rows: list[dict] = []

    for model_name, model_factory in dl_models.items():
        print(f"\nTraining & Evaluating Deep Learning Model: [{model_name}]...", flush=True)
        fold_aucs, fold_aps, fold_r10, fold_r50, fold_r100, fold_lifts, fold_briers = [], [], [], [], [], [], []
        total_train_time, total_infer_time = 0.0, 0.0

        for fold_idx, (fold_name, val_start_str, val_end_str) in enumerate(FOLDS, start=1):
            val_start_ns = pd.Timestamp(val_start_str, tz="UTC").value
            val_end_ns = pd.Timestamp(val_end_str, tz="UTC").value

            val_start_bin = int(np.searchsorted(engine.bin_start_ns, val_start_ns))
            val_end_bin = min(int(np.searchsorted(engine.bin_start_ns, val_end_ns)), engine.num_bins)

            train_mask = sample_times_ns < val_start_ns
            X_tr = torch.tensor(X_all_norm[train_mask], dtype=torch.float32)
            y_tr = torch.tensor(labels[train_mask], dtype=torch.float32)

            train_pos = int(y_tr.sum().item())
            sample_prior = train_pos / max(len(y_tr), 1)

            true_train_pos = int(gt_matrix[int(WINDOW_LAGS.max()) : val_start_bin].sum())
            true_train_total = (val_start_bin - int(WINDOW_LAGS.max())) * engine.num_gpus
            true_prior = true_train_pos / max(true_train_total, 1)

            pos_weight = torch.tensor([(len(y_tr) - train_pos) / max(train_pos, 1)], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            model = model_factory().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

            dataset = TensorDataset(X_tr, y_tr)
            loader = DataLoader(dataset, batch_size=2048, shuffle=True, drop_last=False)

            t_fit_start = time.time()
            model.train()
            for epoch in range(10):
                for x_b, y_b in loader:
                    optimizer.zero_grad()
                    preds = model(x_b)
                    loss = criterion(preds, y_b)
                    loss.backward()
                    optimizer.step()

            t_fit = time.time() - t_fit_start
            total_train_time += t_fit

            model.eval()
            chunk_size = 128
            val_scores = np.zeros((val_end_bin - val_start_bin, engine.num_gpus), dtype=np.float32)
            val_ranks = np.zeros((val_end_bin - val_start_bin, engine.num_gpus), dtype=np.uint16)

            t_infer_start = time.time()
            with torch.no_grad():
                for c_start in range(val_start_bin, val_end_bin, chunk_size):
                    c_end = min(c_start + chunk_size, val_end_bin)
                    n_bins = c_end - c_start

                    batch_bins = np.repeat(np.arange(c_start, c_end, dtype=np.int32), engine.num_gpus)
                    batch_gpus = np.tile(np.arange(engine.num_gpus, dtype=np.int32), n_bins)

                    raw_tensor = engine.extract_3d_tensor(batch_bins, batch_gpus)
                    norm_tensor = (raw_tensor - mean_c) / std_c
                    x_val = torch.tensor(norm_tensor, dtype=torch.float32)

                    logits = model(x_val)
                    raw_probs = torch.sigmoid(logits).cpu().numpy()
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

        dl_benchmark_rows.append({
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

    return pd.DataFrame(dl_benchmark_rows)


def main() -> None:
    data_dir, cache_dir = resolve_paths()
    output_dir = PROJECT_ROOT / "outputs" / "comparison"

    print("=================================================================", flush=True)
    print(" [Deep Learning Benchmark Runner] Initializing Sequence Engine", flush=True)
    print(f" Data Directory   : {data_dir}", flush=True)
    print(f" Cache Directory  : {cache_dir}", flush=True)
    print(f" Output Directory : {output_dir}", flush=True)
    print("=================================================================\n", flush=True)

    engine = DLSequenceEngine(data_dir=data_dir, cache_dir=cache_dir)
    all_bins, all_gpus, labels, sample_times_ns, df_xid, history = engine.build_sample_dataset(negative_ratio=15, seed=20230823)

    dl_df = run_dl_benchmark(
        engine=engine,
        all_bins=all_bins,
        all_gpus=all_gpus,
        labels=labels,
        sample_times_ns=sample_times_ns,
        df_xid=df_xid,
        output_dir=output_dir,
    )
    print("\nDeep Learning Benchmark Completed!")
    print(dl_df[["model", "mean_roc_auc", "mean_pr_auc", "mean_recall_at_100", "mean_lift_at_100"]].to_string(index=False))


if __name__ == "__main__":
    main()

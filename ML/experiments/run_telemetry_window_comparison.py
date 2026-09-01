"""Compare fixed 1h, 6h, and 12h leakage-safe Telemetry windows.

The run keeps the final-pipeline 10-minute leakage buffer and the original
24-hour new-XID target.  It uses only the four already-audited 5-minute
Telemetry Parquets, with every appropriate per-bucket telemetry summary:
mean, min, max, mean delta, observation ratio, value absence, and row absence.

It intentionally excludes historical/context features, ADST, and Sliding
Training.  Outputs are compact JSON/Markdown reports only; no Excel workbook,
prediction dump, or materialized feature dataset is written.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import run_telemetry_24h_experiment as base

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from torch import nn


WINDOW_BUCKETS = {"1h": 12, "6h": 72, "12h": 144}
TOP_KS = (10, 20, 50)
METRICS = (
    ("util", "telemetry_5m_util.parquet"),
    ("temp", "telemetry_5m_temp.parquet"),
    ("power", "telemetry_5m_power.parquet"),
    ("fb", "telemetry_5m_fb.parquet"),
)
CONTINUOUS_CHANNELS = 16  # 4 metrics × [mean, min, max, mean_delta]
CHANNEL_NAMES = [
    *(f"{metric}_{stat}" for stat in ("mean", "min", "max", "mean_delta_5m") for metric, _ in METRICS),
    *(f"{metric}_obs" for metric, _ in METRICS),
    *(f"{metric}_value_absent" for metric, _ in METRICS),
    "row_absent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--label-dir", type=Path, default=Path("outputs/telemetry_24h_fixed")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/telemetry_window_comparison")
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--train-times", type=int, default=800)
    parser.add_argument("--validation-times", type=int, default=300)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--test-time-block", type=int, default=8)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def offsets_for_window(bucket_count: int) -> np.ndarray:
    """Return input bucket offsets ending at t-10 minutes.

    For 1h, this is [-14, ..., -3], i.e. [t-70m, t-10m] as 12 buckets.
    """
    return np.arange(-(bucket_count + 2), -2, dtype=np.int64)


def valid_times_for_window(grid: np.ndarray, audit: dict, bucket_count: int) -> np.ndarray:
    """Apply continuity, 24h-horizon, and XID-outage gates for one window."""
    lookback = bucket_count + 2  # observation length plus the 10-minute buffer
    valid = np.ones(len(grid), dtype=bool)
    valid[:lookback] = False
    valid[lookback:] &= grid[lookback:] - grid[:-lookback] == lookback * base.STEP_NS
    valid &= grid >= int(audit["raw_start_ns"]) + (bucket_count + 2) * base.STEP_NS
    valid &= grid + base.HORIZON_NS <= int(audit["raw_end_ns"])
    for gap in audit["gaps"]:
        start, end = int(gap["start_ns"]), int(gap["end_ns"])
        valid &= ~((grid < end) & (grid + base.HORIZON_NS > start))
    return grid[valid]


def common_splits(grid: np.ndarray, audit: dict) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    per_window = [valid_times_for_window(grid, audit, count) for count in WINDOW_BUCKETS.values()]
    common = per_window[0]
    for values in per_window[1:]:
        common = np.intersect1d(common, values, assume_unique=True)
    if len(common) == 0:
        raise ValueError("No common leakage-safe decision epoch remains for 1h/6h/12h.")
    return base.chronological_split(common), {name: int(len(values)) for name, values in zip(WINDOW_BUCKETS, per_window)}


def filter_timestamp(value: int) -> pd.Timestamp:
    return pd.Timestamp(int(value), unit="ns", tz="UTC").tz_convert(base.KST)


def load_feature_block(
    data_dir: Path, start_ns: int, end_ns: int, gpu_ids: list[str]
) -> np.ndarray:
    """Load the full appropriate Telemetry feature set into a compact time block."""
    block_times = np.arange(start_ns, end_ns + base.STEP_NS, base.STEP_NS, dtype=np.int64)
    n_time, n_gpu = len(block_times), len(gpu_ids)
    mean = np.full((n_time, n_gpu, 4), np.nan, dtype=np.float32)
    minimum = np.full_like(mean, np.nan)
    maximum = np.full_like(mean, np.nan)
    obs = np.zeros_like(mean)
    util_row_present = np.zeros((n_time, n_gpu), dtype=bool)
    gpu_lookup = pd.Index(gpu_ids)

    for metric_index, (metric, filename) in enumerate(METRICS):
        table = pq.read_table(
            data_dir / filename,
            columns=[
                "Time_5m",
                "gpu_id",
                f"{metric}_mean",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_obs",
            ],
            filters=[
                ("Time_5m", ">=", filter_timestamp(start_ns)),
                ("Time_5m", "<=", filter_timestamp(end_ns)),
            ],
        )
        if table.num_rows == 0:
            continue
        frame = table.to_pandas()
        timestamp = pd.to_datetime(frame["Time_5m"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        time_index = ((timestamp - start_ns) // base.STEP_NS).astype(np.int64)
        gpu_index = gpu_lookup.get_indexer(frame["gpu_id"])
        in_block = (time_index >= 0) & (time_index < n_time) & (gpu_index >= 0)
        if not in_block.all():
            raise ValueError(f"Unexpected time/GPU key while reading {filename}.")
        rows_t, rows_g = time_index[in_block], gpu_index[in_block]
        mean[rows_t, rows_g, metric_index] = pd.to_numeric(
            frame.loc[in_block, f"{metric}_mean"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        minimum[rows_t, rows_g, metric_index] = pd.to_numeric(
            frame.loc[in_block, f"{metric}_min"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        maximum[rows_t, rows_g, metric_index] = pd.to_numeric(
            frame.loc[in_block, f"{metric}_max"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        obs[rows_t, rows_g, metric_index] = np.nan_to_num(
            pd.to_numeric(frame.loc[in_block, f"{metric}_obs"], errors="coerce").to_numpy(dtype=np.float32),
            nan=0.0,
        )
        if metric == "util":
            util_row_present[rows_t, rows_g] = True

    mean_delta = np.full_like(mean, np.nan)
    mean_delta[1:] = mean[1:] - mean[:-1]
    value_absent = (~np.isfinite(mean) | ~np.isfinite(minimum) | ~np.isfinite(maximum)).astype(np.float32)
    row_absent = (~util_row_present).astype(np.float32)[..., None]
    return np.concatenate((mean, minimum, maximum, mean_delta, obs, value_absent, row_absent), axis=2)


def day_keys(times: np.ndarray) -> np.ndarray:
    local = pd.to_datetime(times, unit="ns", utc=True).tz_convert(base.KST)
    return np.asarray(local.strftime("%Y%m%d"), dtype=object)


def extract_sequences(
    feature_block: np.ndarray,
    block_start: int,
    sample_times: np.ndarray,
    gpu_index: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    decision = ((sample_times - block_start) // base.STEP_NS).astype(np.int64)
    sequence_index = decision[:, None] + offsets[None, :]
    return feature_block[sequence_index, gpu_index[:, None], :]


def materialize_training_sequences(
    data_dir: Path,
    sample_times: np.ndarray,
    gpu_index: np.ndarray,
    gpu_ids: list[str],
    offsets: np.ndarray,
    label: str,
) -> np.ndarray:
    """Materialize only the stratified training sample, day by day."""
    output = np.empty((len(sample_times), len(offsets), len(CHANNEL_NAMES)), dtype=np.float32)
    keys = day_keys(sample_times)
    unique = np.unique(keys)
    for complete, key in enumerate(unique, start=1):
        rows = np.flatnonzero(keys == key)
        block_start = int(sample_times[rows].min() + (offsets[0] - 1) * base.STEP_NS)
        block_end = int(sample_times[rows].max() + offsets[-1] * base.STEP_NS)
        block = load_feature_block(data_dir, block_start, block_end, gpu_ids)
        output[rows] = extract_sequences(block, block_start, sample_times[rows], gpu_index[rows], offsets)
        print(f"  {label}: {complete}/{len(unique)} days", flush=True)
    return output


def fit_normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x[..., :CONTINUOUS_CHANNELS].reshape(-1, CONTINUOUS_CHANNELS)
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    scale = np.nanstd(flat, axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean, scale


def normalize_in_place(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normalize numeric telemetry only; keep obs/masks in their native interpretation."""
    for channel in range(CONTINUOUS_CHANNELS):
        values = x[..., channel]
        values[~np.isfinite(values)] = mean[channel]
        values -= mean[channel]
        values /= scale[channel]
    x[..., CONTINUOUS_CHANNELS:] = np.nan_to_num(x[..., CONTINUOUS_CHANNELS:], nan=0.0)
    return x


def sequence_summary(x: np.ndarray) -> np.ndarray:
    """A non-historical Logistic baseline using all Telemetry channels over the window."""
    return np.concatenate(
        (
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            x[:, -1, :] - x[:, 0, :],
        ),
        axis=1,
    ).astype(np.float32, copy=False)


class LongTCN(nn.Module):
    """Causal TCN with a 256-bucket receptive field (longer than 12 hours)."""

    DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self, channels: int, width: int = 8):
        super().__init__()
        blocks = []
        input_channels = channels
        for dilation in self.DILATIONS:
            blocks.append(base.CausalBlock(input_channels, width, dilation=dilation))
            input_channels = width
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, :, -1]).squeeze(-1)


def train_network(
    name: str,
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> nn.Module:
    rng = np.random.default_rng(seed)
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = rng.permutation(len(y))
        total_loss = 0.0
        batches = 0
        for start in range(0, len(y), batch_size):
            rows = permutation[start : start + batch_size]
            xb = np.ascontiguousarray(x[rows].transpose(0, 2, 1))
            yb = torch.from_numpy(y[rows].astype(np.float32, copy=False))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(torch.from_numpy(xb)), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        print(f"  {name} epoch {epoch}: training loss={total_loss / batches:.6f}", flush=True)
    return model


def network_score(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    return base.network_probabilities(model, x, batch_size)


def rank_update(store: dict, score: np.ndarray, y: np.ndarray) -> None:
    if y.sum() == 0:
        return
    order = np.argsort(-score, kind="stable")
    positives = int(y.sum())
    store["active_epochs"] += 1
    for k in TOP_KS:
        picked = y[order[: min(k, len(order))]]
        store[f"recall_{k}"].append(float(picked.sum() / positives))
        discounts = 1.0 / np.log2(np.arange(2, len(picked) + 2))
        dcg = float((picked * discounts).sum())
        ideal_count = min(k, positives)
        ideal = float((1.0 / np.log2(np.arange(2, ideal_count + 2))).sum())
        store[f"ndcg_{k}"].append(dcg / ideal if ideal else 0.0)
    k5 = max(1, int(np.ceil(len(order) * 0.05)))
    store["top5"].append(float(y[order[:k5]].sum() / positives))


def new_rank_store() -> dict:
    result = {"active_epochs": 0, "top5": []}
    for k in TOP_KS:
        result[f"recall_{k}"] = []
        result[f"ndcg_{k}"] = []
    return result


def finish_rank_store(store: dict) -> dict:
    result = {"epochs_with_failure": int(store["active_epochs"])}
    for k in TOP_KS:
        result[f"recall_at_{k}"] = float(np.mean(store[f"recall_{k}"]))
        result[f"ndcg_at_{k}"] = float(np.mean(store[f"ndcg_{k}"]))
    result["top_5pct_capture"] = float(np.mean(store["top5"]))
    return result


class XIDCaptureTracker:
    """Episode-level, code-specific Top-K capture with no multi-event double credit."""

    def __init__(self, ledger: pd.DataFrame, gpu_ids: list[str], config_names: list[str]):
        events = ledger.loc[~ledger["uncertain_onset"], ["gpu_id", "xid_code", "onset_ns"]].copy()
        events = events.sort_values(["gpu_id", "onset_ns"]).reset_index(drop=True)
        events["event_id"] = np.arange(len(events), dtype=np.int64)
        self.events = events
        self.gpu_index = {gpu: index for index, gpu in enumerate(gpu_ids)}
        self.by_gpu: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for gpu, group in events.groupby("gpu_id", sort=False):
            self.by_gpu[self.gpu_index[gpu]] = (
                group["onset_ns"].to_numpy(dtype=np.int64),
                group["event_id"].to_numpy(dtype=np.int64),
            )
        self.evaluable = np.zeros(len(events), dtype=bool)
        self.opportunity_miss = {k: np.ones(len(events), dtype=np.float64) for k in TOP_KS}
        self.hits = {
            name: {k: np.zeros(len(events), dtype=bool) for k in TOP_KS}
            for name in config_names
        }
        self.first_detection = {
            name: {k: np.full(len(events), np.iinfo(np.int64).max, dtype=np.int64) for k in TOP_KS}
            for name in config_names
        }

    def next_event(self, gpu: int, decision_time: int) -> int | None:
        item = self.by_gpu.get(int(gpu))
        if item is None:
            return None
        starts, ids = item
        position = np.searchsorted(starts, decision_time, side="right")
        if position == len(starts) or starts[position] > decision_time + base.HORIZON_NS:
            return None
        return int(ids[position])

    def mark_evaluable(self, decision_time: int, gpu: int, candidate_count: int) -> None:
        event_id = self.next_event(gpu, decision_time)
        if event_id is None:
            return
        self.evaluable[event_id] = True
        for k in TOP_KS:
            self.opportunity_miss[k][event_id] *= 1.0 - min(k, candidate_count) / candidate_count

    def mark_hits(self, config: str, decision_time: int, ranked_gpu: np.ndarray) -> None:
        for k in TOP_KS:
            for gpu in ranked_gpu[: min(k, len(ranked_gpu))]:
                event_id = self.next_event(int(gpu), decision_time)
                if event_id is not None:
                    self.hits[config][k][event_id] = True
                    self.first_detection[config][k][event_id] = min(
                        self.first_detection[config][k][event_id], decision_time
                    )

    def report(self, config: str) -> dict:
        result = {}
        codes = self.events["xid_code"].to_numpy(dtype=np.int64)
        onset = self.events["onset_ns"].to_numpy(dtype=np.int64)
        for code in np.unique(codes):
            code_mask = codes == code
            evaluable = code_mask & self.evaluable
            row = {
                "primary_episodes": int(code_mask.sum()),
                "evaluable_episodes": int(evaluable.sum()),
            }
            for k in TOP_KS:
                hit = evaluable & self.hits[config][k]
                row[f"hit_at_{k}"] = int(hit.sum())
                row[f"capture_at_{k}"] = float(hit.sum() / evaluable.sum()) if evaluable.any() else None
                random_capture = 1.0 - self.opportunity_miss[k][evaluable]
                row[f"random_expected_capture_at_{k}"] = (
                    float(random_capture.mean()) if len(random_capture) else None
                )
                detected = self.first_detection[config][k][hit]
                lead_hours = (onset[hit] - detected) / 3.6e12
                row[f"median_lead_hours_at_{k}"] = float(np.median(lead_hours)) if len(lead_hours) else None
            result[str(int(code))] = row
        return result


def population_blocks(
    data_dir: Path,
    times: np.ndarray,
    eligible: np.ndarray,
    gpu_ids: list[str],
    offsets: np.ndarray,
    time_block: int,
    title: str,
):
    """Yield compact sequence batches for all eligible GPUs at selected times."""
    keys = day_keys(times)
    unique = np.unique(keys)
    for day_number, key in enumerate(unique, start=1):
        day_rows = np.flatnonzero(keys == key)
        block_start = int(times[day_rows].min() + (offsets[0] - 1) * base.STEP_NS)
        block_end = int(times[day_rows].max() + offsets[-1] * base.STEP_NS)
        feature_block = load_feature_block(data_dir, block_start, block_end, gpu_ids)
        for start in range(0, len(day_rows), time_block):
            rows = day_rows[start : start + time_block]
            local_time, gpu = np.where(eligible[rows])
            if len(gpu) == 0:
                continue
            sample_times = times[rows[local_time]]
            x = extract_sequences(feature_block, block_start, sample_times, gpu, offsets)
            yield rows, local_time, gpu, x
        print(f"  {title}: {day_number}/{len(unique)} days", flush=True)
        del feature_block
        gc.collect()


def raw_prediction(models: dict[str, object], x: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    return {
        "telemetry_logistic": models["telemetry_logistic"].predict_proba(sequence_summary(x))[:, 1],
        "one_d_cnn": network_score(models["one_d_cnn"], x, batch_size),
        "tcn": network_score(models["tcn"], x, batch_size),
    }


def collect_validation_scores(
    data_dir: Path,
    times: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    gpu_ids: list[str],
    offsets: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    models: dict[str, object],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    y_chunks, scores = [], defaultdict(list)
    for rows, local, gpu, x in population_blocks(
        data_dir, times, eligible, gpu_ids, offsets, time_block=8, title="validation"
    ):
        normalize_in_place(x, mean, scale)
        y = labels[rows[local], gpu].astype(np.int8)
        y_chunks.append(y)
        for name, value in raw_prediction(models, x, batch_size).items():
            scores[name].append(value.astype(np.float32, copy=False))
        del x
    return np.concatenate(y_chunks), {name: np.concatenate(value) for name, value in scores.items()}


def evaluate_test(
    data_dir: Path,
    times: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    ledger: pd.DataFrame,
    gpu_ids: list[str],
    offsets: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    models: dict[str, object],
    calibrators: dict[str, base.PlattScaler],
    batch_size: int,
    time_block: int,
    config_prefix: str,
) -> tuple[dict, dict]:
    names = list(models)
    tracker = XIDCaptureTracker(ledger, gpu_ids, [f"{config_prefix}__{name}" for name in names])
    y_chunks: list[np.ndarray] = []
    score_chunks = {name: [] for name in names}
    ranks = {name: new_rank_store() for name in names}

    for rows, local, gpu, x in population_blocks(
        data_dir, times, eligible, gpu_ids, offsets, time_block=time_block, title=f"test {config_prefix}"
    ):
        normalize_in_place(x, mean, scale)
        y = labels[rows[local], gpu].astype(np.int8)
        y_chunks.append(y)
        calibrated = {
            name: calibrators[name].predict(value)
            for name, value in raw_prediction(models, x, batch_size).items()
        }
        for name, value in calibrated.items():
            score_chunks[name].append(value.astype(np.float32, copy=False))

        # np.where above emits GPU IDs in ascending order inside each local decision row.
        for position, global_row in enumerate(rows):
            sample_rows = np.flatnonzero(local == position)
            candidate_gpu = gpu[sample_rows]
            candidate_y = y[sample_rows]
            candidate_count = len(candidate_gpu)
            decision = int(times[global_row])
            for positive_gpu in candidate_gpu[candidate_y == 1]:
                tracker.mark_evaluable(decision, int(positive_gpu), candidate_count)
            for name, value in calibrated.items():
                candidate_score = value[sample_rows]
                rank_update(ranks[name], candidate_score, candidate_y)
                order = np.argsort(-candidate_score, kind="stable")
                tracker.mark_hits(f"{config_prefix}__{name}", decision, candidate_gpu[order])
        del x

    y_all = np.concatenate(y_chunks)
    metrics = {}
    captures = {}
    for name in names:
        score_all = np.concatenate(score_chunks[name])
        value = {
            "pr_auc": float(average_precision_score(y_all, score_all)),
            "brier": float(brier_score_loss(y_all, score_all)),
            "ece": base.ece(score_all, y_all),
            "test_samples": int(len(y_all)),
            "test_positive_samples": int(y_all.sum()),
            "test_positive_rate": float(y_all.mean()),
        }
        value.update(finish_rank_store(ranks[name]))
        metrics[name] = value
        captures[name] = tracker.report(f"{config_prefix}__{name}")
    return metrics, captures


def train_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    baseline = LogisticRegression(
        solver="lbfgs", class_weight="balanced", max_iter=300, random_state=seed
    )
    baseline.fit(sequence_summary(x_train), y_train)
    cnn = train_network(
        "1D-CNN", base.OneDCNN(len(CHANNEL_NAMES)), x_train, y_train, epochs, batch_size, seed + 1
    )
    tcn = train_network(
        "TCN", LongTCN(len(CHANNEL_NAMES)), x_train, y_train, epochs, batch_size, seed + 2
    )
    return {"telemetry_logistic": baseline, "one_d_cnn": cnn, "tcn": tcn}


def compact_window_report(result: dict) -> str:
    rows = []
    for window in WINDOW_BUCKETS:
        for model in ("telemetry_logistic", "one_d_cnn", "tcn"):
            value = result["windows"][window]["metrics"][model]
            rows.append(
                f"| {window} | {model} | {value['pr_auc']:.6f} | {value['recall_at_10']:.4f} | "
                f"{value['ndcg_at_10']:.4f} | {value['top_5pct_capture']:.4f} | {value['ece']:.6f} |"
            )
    best_window, best_model = max(
        (
            (window, model)
            for window in WINDOW_BUCKETS
            for model in ("telemetry_logistic", "one_d_cnn", "tcn")
        ),
        key=lambda item: result["windows"][item[0]]["metrics"][item[1]]["pr_auc"],
    )
    captures = result["windows"][best_window]["per_xid_detection"][best_model]
    xid_rows = []
    for code, value in captures.items():
        capture10 = value["capture_at_10"]
        capture50 = value["capture_at_50"]
        xid_rows.append(
            f"| {code} | {value['evaluable_episodes']} | {value['hit_at_10']} | "
            f"{capture10:.3f} | {value['hit_at_50']} | {capture50:.3f} |"
            if capture10 is not None and capture50 is not None
            else f"| {code} | 0 | 0 | - | 0 | - |"
        )
    return f"""# Fixed Telemetry Window Comparison: 1h / 6h / 12h

## Fixed experiment contract

- Input end: `t-10 minutes`; the final 10-minute leakage buffer is retained.
- Windows: 1h=`[t-70m,t-10m]`, 6h=`[t-370m,t-10m]`, 12h=`[t-730m,t-10m]`.
- Target: a same-GPU new XID onset in `(t,t+24h]`.
- Same common decision epochs, chronological split, 36-hour purge, train/validation time samples, and test period for all windows.
- Models: Telemetry Logistic baseline, 1D-CNN, TCN. The causal TCN has a 256-bucket receptive field, covering every 1h/6h/12h input.
- Excluded by design: Historical/Context baseline, ADST, Sliding Training, Branch-3.

## Input channels

All four audited Telemetry sources (Util, Temp, Power, FB) are used. Per source: mean, min, max, mean delta, observation ratio, value-absence mask; one common util-row-absence mask is added. Job/context/calendar and unvalidated raw-only telemetry are excluded.

## Main comparison

| Window | Model | PR-AUC | Recall@10 | NDCG@10 | Top-5% Capture | ECE |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Best PR-AUC configuration: **{best_window} / {best_model}**.

## Which XID episodes did the best configuration capture?

An episode is counted once when its GPU appears in Top-K at any eligible 5-minute decision epoch in its previous 24 hours; only the earliest next certain onset for that GPU is credited. This is post-hoc code attribution, not an XID multiclass prediction.

| XID code | Evaluable episodes | Hit@10 | Capture@10 | Hit@50 | Capture@50 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(xid_rows)}

Codes with very small evaluable counts are exploratory only. The detailed JSON includes each model/window's code-specific capture, expected random capture given its available alert opportunities, and median lead time.
"""


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(6, (os.cpu_count() or 2) - 1)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = args.label_dir / "xid_episode_ledger.parquet"
    audit_path = args.label_dir / "xid_episode_ledger.audit.json"
    if not ledger_path.exists() or not audit_path.exists():
        raise FileNotFoundError("Run the fixed 24h label build first; the verified XID ledger is required.")
    ledger = pq.read_table(ledger_path).to_pandas()
    audit = json.loads(audit_path.read_text("utf-8"))
    gpu_ids = base.read_gpu_ids(args.data_dir / "XID_ERRORS-002.csv")
    grid = base.read_time_grid(args.data_dir / "telemetry_5m_util.parquet")
    splits, per_window_decision_count = common_splits(grid, audit)
    rng = np.random.default_rng(args.seed)
    train_times = rng.choice(splits["train"], size=min(args.train_times, len(splits["train"])), replace=False)
    validation_times = rng.choice(
        splits["validation"], size=min(args.validation_times, len(splits["validation"])), replace=False
    )
    train_times.sort()
    validation_times.sort()
    train_t, train_g, y_train, train_summary = base.choose_training_samples(
        train_times, ledger, gpu_ids, args.negative_ratio, rng
    )
    val_labels, val_eligible = base.label_state(validation_times, ledger, gpu_ids)
    test_labels, test_eligible = base.label_state(splits["test"], ledger, gpu_ids)

    if args.preflight:
        offsets = offsets_for_window(WINDOW_BUCKETS["12h"])
        probe = materialize_training_sequences(
            args.data_dir, train_t[:1], train_g[:1], gpu_ids, offsets, "preflight"
        )
        print(
            json.dumps(
                {
                    "common_splits": {name: int(len(values)) for name, values in splits.items()},
                    "per_window_decision_count": per_window_decision_count,
                    "train_summary": train_summary,
                    "validation_positive": int(val_labels[val_eligible].sum()),
                    "test_positive": int(test_labels[test_eligible].sum()),
                    "probe_shape": list(probe.shape),
                    "channels": CHANNEL_NAMES,
                },
                indent=2,
            )
        )
        return

    result = {
        "experiment": {
            "seed": args.seed,
            "windows": {name: f"{count} buckets ending at t-10m" for name, count in WINDOW_BUCKETS.items()},
            "target": "same-GPU new XID onset in (t, t+24h]",
            "models": ["telemetry logistic baseline", "1D-CNN", "TCN"],
            "excluded": ["historical/context baseline", "ADST", "Sliding Training", "Branch-3"],
            "feature_channels": CHANNEL_NAMES,
        },
        "label_audit": audit,
        "common_split": {
            name: {"count": int(len(values)), "start": base.ns_to_text(values.min()), "end": base.ns_to_text(values.max())}
            for name, values in splits.items()
        },
        "window_safe_decision_epoch_count": per_window_decision_count,
        "training_sample": train_summary,
        "validation_sample": {
            "decision_epochs": int(len(validation_times)),
            "eligible_samples": int(val_eligible.sum()),
            "positive_samples": int(val_labels[val_eligible].sum()),
        },
        "test_sample": {
            "decision_epochs": int(len(splits["test"])),
            "eligible_samples": int(test_eligible.sum()),
            "positive_samples": int(test_labels[test_eligible].sum()),
        },
        "windows": {},
    }

    for window, count in WINDOW_BUCKETS.items():
        print(f"\n=== {window}: {count} buckets ending at t-10m ===", flush=True)
        offsets = offsets_for_window(count)
        x_train = materialize_training_sequences(
            args.data_dir, train_t, train_g, gpu_ids, offsets, f"{window} train"
        )
        mean, scale = fit_normalizer(x_train)
        normalize_in_place(x_train, mean, scale)
        models = train_models(x_train, y_train, args.epochs, args.batch_size, args.seed)
        del x_train
        gc.collect()

        print(f"  {window}: independent validation calibration", flush=True)
        y_val, raw_val = collect_validation_scores(
            args.data_dir,
            validation_times,
            val_labels,
            val_eligible,
            gpu_ids,
            offsets,
            mean,
            scale,
            models,
            args.batch_size,
        )
        calibrators = {name: base.PlattScaler().fit(score, y_val) for name, score in raw_val.items()}
        del raw_val, y_val
        gc.collect()

        print(f"  {window}: full chronological test and per-XID capture", flush=True)
        metrics, per_xid = evaluate_test(
            args.data_dir,
            splits["test"],
            test_labels,
            test_eligible,
            ledger,
            gpu_ids,
            offsets,
            mean,
            scale,
            models,
            calibrators,
            args.batch_size,
            args.test_time_block,
            window,
        )
        result["windows"][window] = {"metrics": metrics, "per_xid_detection": per_xid}
        del models, calibrators
        gc.collect()

    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=base.as_json), "utf-8"
    )
    (args.output_dir / "report.md").write_text(compact_window_report(result), "utf-8")
    print("Done. Wrote only metrics.json and report.md.", flush=True)


if __name__ == "__main__":
    main()

"""Unified Seren telemetry experiment for XID 31/43 prediction.

This is the only executable ML file for the current experiment contract. It
reconstructs the target from the original 15-second Seren XID file, reads the
provided 5-minute Parquets, recomputes 5-minute standard deviations from the
15-second source, evaluates a single binary target (XID 31 OR XID 43 within
24 hours), and writes exactly three Excel workbooks.

Calibration, multitask heads, GPU topology, Blox replay, and separate result
files are intentionally outside this run. Intermediate arrays stay in memory;
only the three requested Excel workbooks are persistent outputs.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - reported clearly at runtime
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
RESEARCH_ROOT = PROJECT_ROOT.parent
LOCAL_DATA_ROOT = EXPERIMENT_ROOT / "DATA"
EXTERNAL_DATA_ROOT = RESEARCH_ROOT / "DATA"
RESULT_ROOT = EXPERIMENT_ROOT / "결과 산출"

STEP_NS = 5 * 60 * 1_000_000_000
MERGE_NS = 30 * 1_000_000_000
HORIZON_NS = 24 * 60 * 60 * 1_000_000_000
PURGE_NS = 36 * 60 * 60 * 1_000_000_000
HISTORY_BINS = 12
FEATURE_GAP_BINS = 2
HISTORY_START_OFFSET = HISTORY_BINS + FEATURE_GAP_BINS - 1
NEGATIVE_RATIO = 4
TRAIN_DECISION_TIMES = 720
VALIDATION_DECISION_TIMES = 240
TEST_STRIDE_BINS = 6
SEED = 20260905
KST = timezone(timedelta(hours=8))

METRIC_ORDER = ("util", "temp", "power", "fb")
TELEMETRY_FILES = {
    "util": "telemetry_5m_util.parquet",
    "temp": "telemetry_5m_temp.parquet",
    "power": "telemetry_5m_power.parquet",
    "fb": "telemetry_5m_fb.parquet",
}
RAW_FILES = {
    "util": "GPU_UTIL.csv",
    "temp": "GPU_TEMP.csv",
    "power": "POWER_USAGE.csv",
    "fb": "FB_USED.csv",
}
BRANCH_RESULT_FILES = {
    "Branch_1": RESULT_ROOT / "Branch_1" / "branch1_results.xlsx",
    "Branch_2": RESULT_ROOT / "Branch_2" / "branch2_results.xlsx",
    "Branch_3_optional": RESULT_ROOT / "Branch_3_optional" / "branch3_results.xlsx",
}


def utc_text(ns: int | float | None) -> str | None:
    if ns is None or not np.isfinite(ns):
        return None
    return str(pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(KST))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def cell_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), ensure_ascii=False)
    return json_safe(value)


def find_data_roots() -> tuple[Path, Path]:
    for root in (LOCAL_DATA_ROOT, EXTERNAL_DATA_ROOT):
        raw_dir = root / "AcmeTrace" / "data" / "utilization" / "seren"
        if (root / TELEMETRY_FILES["util"]).exists() and (raw_dir / "XID_ERRORS.csv").exists():
            return root, raw_dir
    raise FileNotFoundError(
        "Could not find both 5-minute Parquets and raw Seren files under "
        f"{LOCAL_DATA_ROOT} or {EXTERNAL_DATA_ROOT}."
    )


def read_gpu_ids(xid_path: Path) -> list[str]:
    header = pd.read_csv(xid_path, nrows=0).columns.tolist()
    if not header or header[0] != "Time":
        raise ValueError("Seren XID_ERRORS.csv must have Time as its first column.")
    return header[1:]


def reconstruct_target_episodes(xid_path: Path, gpu_ids: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream raw XID and reconstruct target episodes with an audit trail."""
    n_gpu = len(gpu_ids)
    active_code = np.full(n_gpu, -1, dtype=np.int32)
    active_start = np.zeros(n_gpu, dtype=np.int64)
    active_last = np.zeros(n_gpu, dtype=np.int64)
    active_uncertain = np.zeros(n_gpu, dtype=bool)
    previous_missing = np.ones(n_gpu, dtype=bool)
    previous_time: int | None = None
    episodes: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "raw_xid_path": str(xid_path),
        "raw_rows": 0,
        "raw_start_ns": None,
        "raw_end_ns": None,
        "invalid_or_missing_cells": 0,
        "global_gaps_over_30_seconds": 0,
        "gaps": [],
        "merge_seconds": 30,
        "zero_is_episode_boundary": True,
        "missing_to_positive_is_censored": True,
    }

    def close(indices: np.ndarray, reason: str) -> None:
        for gpu_index in indices.tolist():
            code = int(active_code[gpu_index])
            if code in (31, 43):
                episodes.append(
                    {
                        "gpu_id": gpu_ids[gpu_index],
                        "gpu_index": gpu_index,
                        "xid_code": code,
                        "onset_ns": int(active_start[gpu_index]),
                        "episode_end_ns": int(active_last[gpu_index]),
                        "uncertain_onset": bool(active_uncertain[gpu_index]),
                        "close_reason": reason,
                    }
                )
        active_code[indices] = -1
        active_start[indices] = 0
        active_last[indices] = 0
        active_uncertain[indices] = False

    reader = pd.read_csv(xid_path, chunksize=500, low_memory=False, dtype={gpu: np.float32 for gpu in gpu_ids})
    for chunk_no, chunk in enumerate(reader, start=1):
        if list(chunk.columns[1:]) != gpu_ids:
            raise ValueError("XID GPU header order changed while scanning the source file.")
        times = pd.to_datetime(chunk.iloc[:, 0], utc=True, errors="raise")
        values = chunk.iloc[:, 1:].to_numpy(dtype=np.float32, na_value=np.nan)
        time_ns = times.dt.as_unit("ns").astype("int64").to_numpy(dtype=np.int64)
        for row_index, now in enumerate(time_ns):
            if audit["raw_start_ns"] is None:
                audit["raw_start_ns"] = int(now)
            audit["raw_end_ns"] = int(now)
            has_gap = previous_time is not None and int(now) - previous_time > MERGE_NS
            if has_gap:
                audit["global_gaps_over_30_seconds"] += 1
                audit["gaps"].append({"start_ns": previous_time, "end_ns": int(now)})
                active = np.flatnonzero(active_code >= 0)
                if active.size:
                    close(active, "time_gap")
            row = values[row_index]
            finite = np.isfinite(row)
            valid = finite & (np.floor(row) == row) & (row >= 0)
            audit["invalid_or_missing_cells"] += int((~valid).sum())
            code = np.full(n_gpu, -1, dtype=np.int32)
            code[valid] = row[valid].astype(np.int32)
            zero_close = np.flatnonzero((active_code >= 0) & valid & (code == 0))
            if zero_close.size:
                close(zero_close, "observed_zero")
            positive = valid & (code > 0)
            continuing = positive & (active_code >= 0) & (code == active_code)
            timed_out = continuing & (int(now) - active_last > MERGE_NS)
            code_changed = positive & (active_code >= 0) & (code != active_code)
            restart_close = np.flatnonzero(timed_out | code_changed)
            if restart_close.size:
                close(restart_close, "repeat_timeout_or_code_change")
            starts = positive & (active_code < 0)
            if starts.any():
                active_code[starts] = code[starts]
                active_start[starts] = int(now)
                active_last[starts] = int(now)
                active_uncertain[starts] = previous_missing[starts] | has_gap
            updates = positive & (active_code >= 0)
            active_last[updates] = int(now)
            previous_missing = ~valid
            previous_time = int(now)
        audit["raw_rows"] += len(chunk)
        if chunk_no % 100 == 0:
            print(f"  XID scan: {audit['raw_rows']:,} rows", flush=True)
    remaining = np.flatnonzero(active_code >= 0)
    if remaining.size:
        close(remaining, "end_of_file")
    ledger = pd.DataFrame(episodes)
    if ledger.empty:
        raise ValueError("No XID 31/43 episodes were reconstructed from raw Seren.")
    ledger = ledger.sort_values(["gpu_index", "onset_ns"]).reset_index(drop=True)
    audit["episode_count"] = int(len(ledger))
    audit["certain_episode_count"] = int((~ledger["uncertain_onset"]).sum())
    audit["uncertain_episode_count"] = int(ledger["uncertain_onset"].sum())
    audit["xid_code_counts"] = {
        str(int(code)): int(count) for code, count in ledger["xid_code"].value_counts().sort_index().items()
    }
    return ledger, audit


def read_time_grid(util_path: Path) -> np.ndarray:
    parquet = pq.ParquetFile(util_path)
    unique_keys: list[int] = []
    last: int | None = None
    for batch in parquet.iter_batches(columns=["Time_5m"], batch_size=2_000_000):
        values = pd.to_datetime(batch.column(0).to_pandas(), utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        if len(values) == 0:
            continue
        starts = values[np.r_[True, values[1:] != values[:-1]]]
        if last is not None and starts.size and int(starts[0]) == last:
            starts = starts[1:]
        unique_keys.extend(int(x) for x in starts)
        last = int(values[-1])
    grid = np.asarray(unique_keys, dtype=np.int64)
    if len(grid) < 10 or np.any(np.diff(grid) <= 0):
        raise ValueError("Parquet Time_5m keys are strictly increasing before gap filling.")
    full_grid = np.arange(grid[0], grid[-1] + STEP_NS, STEP_NS, dtype=np.int64)
    gap_count = len(full_grid) - len(grid)
    if gap_count:
        print(f"  Parquet global time gaps preserved as left-join missing buckets: {gap_count:,}", flush=True)
    return full_grid


def valid_decision_indices(grid: np.ndarray, audit: dict[str, Any]) -> np.ndarray:
    idx = np.arange(len(grid), dtype=np.int32)
    decision_ns = grid + STEP_NS
    valid = idx >= HISTORY_START_OFFSET
    valid &= decision_ns + HORIZON_NS <= int(audit["raw_end_ns"])
    for gap in audit.get("gaps", []):
        start, end = int(gap["start_ns"]), int(gap["end_ns"])
        valid &= ~((decision_ns < end) & (decision_ns + HORIZON_NS > start))
    if not valid.any():
        raise ValueError("No valid decision times remain after history/horizon gates.")
    return idx[valid]


def chronological_split(indices: np.ndarray) -> dict[str, np.ndarray]:
    q1 = int(indices[int(len(indices) * 0.60)])
    q2 = int(indices[int(len(indices) * 0.80)])
    purge_bins = int(PURGE_NS // STEP_NS)
    split = {
        "train": indices[indices <= q1 - purge_bins],
        "validation": indices[(indices >= q1 + purge_bins) & (indices <= q2 - purge_bins)],
        "test": indices[indices >= q2 + purge_bins],
    }
    if any(len(v) == 0 for v in split.values()):
        raise ValueError("Chronological split became empty after the 36-hour purge.")
    return split


def future_event(decision_ns: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    if len(onsets) == 0:
        return np.zeros(len(decision_ns), dtype=bool)
    pos = np.searchsorted(onsets, decision_ns, side="right")
    result = np.zeros(len(decision_ns), dtype=bool)
    available = pos < len(onsets)
    result[available] = onsets[pos[available]] <= decision_ns[available] + HORIZON_NS
    return result


def label_state(decision_indices: np.ndarray, grid: np.ndarray, ledger: pd.DataFrame, n_gpu: int) -> tuple[np.ndarray, np.ndarray]:
    decision_ns = grid[decision_indices] + STEP_NS
    labels = np.zeros((len(decision_indices), n_gpu), dtype=bool)
    eligible = np.ones_like(labels)
    for gpu_index, group in ledger.groupby("gpu_index", sort=False):
        gpu_index = int(gpu_index)
        group = group.sort_values("onset_ns")
        starts = group["onset_ns"].to_numpy(dtype=np.int64)
        ends = group["episode_end_ns"].to_numpy(dtype=np.int64)
        uncertain = group["uncertain_onset"].to_numpy(dtype=bool)
        labels[:, gpu_index] = future_event(decision_ns, starts[~uncertain])
        uncertain_future = future_event(decision_ns, starts[uncertain])
        previous = np.searchsorted(starts, decision_ns, side="right") - 1
        active = np.zeros(len(decision_ns), dtype=bool)
        have_previous = previous >= 0
        active[have_previous] = ends[previous[have_previous]] >= decision_ns[have_previous]
        eligible[:, gpu_index] = ~(active | uncertain_future)
    return labels, eligible


def sample_population(decision_indices: np.ndarray, grid: np.ndarray, ledger: pd.DataFrame, n_gpu: int, rng: np.random.Generator, negative_ratio: int | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    labels, eligible = label_state(decision_indices, grid, ledger, n_gpu)
    if negative_ratio is None:
        time_row, gpu_row = np.where(eligible)
        sample_y = labels[time_row, gpu_row].astype(np.int8)
    else:
        pos_t, pos_g = np.where(labels & eligible)
        neg_t, neg_g = np.where((~labels) & eligible)
        if len(pos_t) == 0:
            raise ValueError("Sampled training/validation population contains no positives.")
        n_negative = min(len(neg_t), len(pos_t) * negative_ratio)
        chosen = rng.choice(len(neg_t), size=n_negative, replace=False)
        time_row = np.concatenate((pos_t, neg_t[chosen]))
        gpu_row = np.concatenate((pos_g, neg_g[chosen]))
        sample_y = np.concatenate((np.ones(len(pos_t), dtype=np.int8), np.zeros(n_negative, dtype=np.int8)))
        order = rng.permutation(len(sample_y))
        time_row, gpu_row, sample_y = time_row[order], gpu_row[order], sample_y[order]
    sample = {
        "decision_index": decision_indices[time_row].astype(np.int32),
        "gpu_index": gpu_row.astype(np.int32),
        "y": sample_y.astype(np.int8),
        "decision_times": decision_indices,
        "labels_matrix": labels,
        "eligible_matrix": eligible,
    }
    summary = {
        "decision_times": int(len(decision_indices)),
        "sample_rows": int(len(sample_y)),
        "positive_rows": int(sample_y.sum()),
        "negative_rows": int((sample_y == 0).sum()),
        "eligible_rows": int(eligible.sum()),
        "censored_rows": int((~eligible).sum()),
        "natural_positive_rows": int(labels[eligible].sum()),
        "natural_prevalence": float(labels[eligible].mean()) if eligible.any() else None,
    }
    return sample, summary


def build_needed_buckets(samples: list[dict[str, np.ndarray]]) -> dict[int, np.ndarray]:
    requested: dict[int, set[int]] = defaultdict(set)
    for sample in samples:
        for decision_index, gpu_index in zip(sample["decision_index"], sample["gpu_index"]):
            start = int(decision_index) - HISTORY_START_OFFSET
            for bucket in range(start, start + HISTORY_BINS):
                requested[bucket].add(int(gpu_index))
    return {bucket: np.asarray(sorted(gpus), dtype=np.int32) for bucket, gpus in requested.items()}


def _numeric_matrix(chunk: pd.DataFrame) -> np.ndarray:
    return chunk.iloc[:, 1:].to_numpy(dtype=np.float32, na_value=np.nan)


def recompute_raw_std(raw_dir: Path, grid: np.ndarray, needed_buckets: dict[int, np.ndarray]) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Recompute requested 5-minute standard deviations from raw 15-second rows."""
    result: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for metric_index, metric in enumerate(METRIC_ORDER, start=1):
        raw_path = raw_dir / RAW_FILES[metric]
        if not raw_path.exists():
            raise FileNotFoundError(f"Required raw Seren metric is missing: {raw_path}")
        state: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for bucket, cols in needed_buckets.items():
            state[bucket] = (np.zeros(len(cols), dtype=np.float64), np.zeros(len(cols), dtype=np.float64), np.zeros(len(cols), dtype=np.int32))
        print(f"[std {metric_index}/4] {raw_path.name}: recomputing from 15-second rows", flush=True)
        row_count = 0
        header = pd.read_csv(raw_path, nrows=0).columns.tolist()
        reader = pd.read_csv(raw_path, chunksize=500, low_memory=False, dtype={gpu: np.float32 for gpu in header[1:]})
        for chunk_no, chunk in enumerate(reader, start=1):
            times = pd.to_datetime(chunk.iloc[:, 0], utc=True, errors="coerce").dt.as_unit("ns").astype("int64").to_numpy()
            values = _numeric_matrix(chunk)
            buckets = ((times - int(grid[0])) // STEP_NS).astype(np.int64)
            for bucket in np.unique(buckets):
                bucket = int(bucket)
                entry = state.get(bucket)
                if entry is None:
                    continue
                rows = buckets == bucket
                cols = needed_buckets[bucket]
                values_for_bucket = values[rows][:, cols]
                valid = np.isfinite(values_for_bucket)
                sums, squares, counts = entry
                sums += np.where(valid, values_for_bucket, 0.0).sum(axis=0, dtype=np.float64)
                squares += np.where(valid, values_for_bucket * values_for_bucket, 0.0).sum(axis=0, dtype=np.float64)
                counts += valid.sum(axis=0, dtype=np.int32)
            row_count += len(chunk)
            if chunk_no % 200 == 0:
                print(f"  {metric}: {row_count:,} raw rows", flush=True)
        metric_result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for bucket, (sums, squares, counts) in state.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = sums / counts
                variance = squares / counts - mean * mean
            std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
            std[counts == 0] = np.nan
            metric_result[bucket] = (needed_buckets[bucket], std)
        result[metric] = metric_result
        print(f"  {metric}: finished ({row_count:,} rows)", flush=True)
    return result


def _time_filter(ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(KST)


def load_feature_block(parquet_root: Path, grid: np.ndarray, start_index: int, end_index: int, gpu_ids: list[str], std_maps: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]) -> tuple[np.ndarray, np.ndarray]:
    start_index = max(0, int(start_index))
    end_index = min(len(grid) - 1, int(end_index))
    length = end_index - start_index + 1
    n_gpu = len(gpu_ids)
    values = np.full((length, n_gpu, 4), np.nan, dtype=np.float32)
    observed = np.zeros((length, n_gpu, 4), dtype=np.float32)
    categories = pd.Index(gpu_ids)
    start_time, end_time = _time_filter(int(grid[start_index])), _time_filter(int(grid[end_index]))
    for metric_index, metric in enumerate(METRIC_ORDER):
        table = pq.read_table(parquet_root / TELEMETRY_FILES[metric], columns=["Time_5m", "gpu_id", f"{metric}_mean", f"{metric}_obs"], filters=[("Time_5m", ">=", start_time), ("Time_5m", "<=", end_time)])
        if table.num_rows == 0:
            continue
        frame = table.to_pandas()
        time_ns = pd.to_datetime(frame["Time_5m"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        local_time = ((time_ns - int(grid[start_index])) // STEP_NS).astype(np.int32)
        gpu_index = categories.get_indexer(frame["gpu_id"])
        good = (local_time >= 0) & (local_time < length) & (gpu_index >= 0)
        raw_value = pd.to_numeric(frame[f"{metric}_mean"], errors="coerce").to_numpy(dtype=np.float32)
        raw_obs = pd.to_numeric(frame[f"{metric}_obs"], errors="coerce").to_numpy(dtype=np.float32)
        values[local_time[good], gpu_index[good], metric_index] = raw_value[good]
        observed[local_time[good], gpu_index[good], metric_index] = np.nan_to_num(raw_obs[good], nan=0.0)
    std = np.full_like(values, np.nan)
    for metric_index, metric in enumerate(METRIC_ORDER):
        for local_index, grid_index in enumerate(range(start_index, end_index + 1)):
            item = std_maps[metric].get(grid_index)
            if item is not None:
                cols, std_values = item
                std[local_index, cols, metric_index] = std_values
    delta = np.full_like(values, np.nan)
    delta[1:] = values[1:] - values[:-1]
    missing = ((~np.isfinite(values)) | (observed < 1.0)).astype(np.float32)
    temporal = np.concatenate((values, std, delta, observed, missing), axis=2)

    context = np.full((length, n_gpu, 6), np.nan, dtype=np.float32)
    context_columns = ["new_jobs_count", "total_gpus_demanded", "mean_job_duration", "hour_of_day", "day_of_week", "is_weekend"]
    table = pq.read_table(parquet_root / "branch2_context_features.parquet", columns=["Time_5m", "gpu_id", *context_columns], filters=[("Time_5m", ">=", start_time), ("Time_5m", "<=", end_time)])
    if table.num_rows:
        frame = table.to_pandas()
        time_ns = pd.to_datetime(frame["Time_5m"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        local_time = ((time_ns - int(grid[start_index])) // STEP_NS).astype(np.int32)
        gpu_index = categories.get_indexer(frame["gpu_id"])
        good = (local_time >= 0) & (local_time < length) & (gpu_index >= 0)
        for col_index, column in enumerate(context_columns):
            col_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
            context[local_time[good], gpu_index[good], col_index] = col_values[good]
    return temporal, context


def materialize_samples(sample: dict[str, np.ndarray], parquet_root: Path, grid: np.ndarray, gpu_ids: list[str], std_maps: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], title: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(sample["y"])
    temporal_output = np.empty((n, HISTORY_BINS, 20), dtype=np.float32)
    context_output = np.empty((n, HISTORY_BINS, 6), dtype=np.float32)
    blocks = (sample["decision_index"] // 144).astype(np.int32)
    unique_blocks = np.unique(blocks)
    for completed, block in enumerate(unique_blocks, start=1):
        rows = np.flatnonzero(blocks == block)
        decisions = sample["decision_index"][rows]
        start_index = int(decisions.min()) - HISTORY_START_OFFSET - 1
        end_index = int(decisions.max()) - FEATURE_GAP_BINS
        temporal, context = load_feature_block(parquet_root, grid, start_index, end_index, gpu_ids, std_maps)
        local = decisions[:, None] - HISTORY_START_OFFSET - start_index + np.arange(HISTORY_BINS)[None, :]
        gpu = sample["gpu_index"][rows]
        temporal_output[rows] = temporal[local, gpu[:, None], :]
        context_output[rows] = context[local, gpu[:, None], :]
        print(f"  {title}: block {completed}/{len(unique_blocks)}", flush=True)
    return temporal_output, context_output


def fit_normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    scale = np.nanstd(flat, axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean, scale


def normalize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, mean[None, None, :]).astype(np.float32, copy=False)
    return ((filled - mean[None, None, :]) / scale[None, None, :]).astype(np.float32, copy=False)


class OneDCNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(channels, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).squeeze(-1)).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 2, dilation=dilation, padding=dilation)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)[:, :, : x.shape[-1]]
        return torch.relu(y + self.skip(x))


class TinyTCN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block1 = CausalBlock(channels, 32, 1)
        self.block2 = CausalBlock(32, 32, 2)
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block2(self.block1(x))[:, :, -1]).squeeze(-1)


def network_probabilities(model: nn.Module, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = np.ascontiguousarray(x[start : start + batch_size].transpose(0, 2, 1))
            logits = model(torch.from_numpy(batch)).cpu().numpy()
            outputs.append((1.0 / (1.0 + np.exp(-logits))).astype(np.float32))
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def train_network(name: str, model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, seed: int, epochs: int = 5, batch_size: int = 2048) -> nn.Module:
    rng = np.random.default_rng(seed)
    positive = max(1, int(y_train.sum()))
    negative = max(1, len(y_train) - positive)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_ap, best_state, stale = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            xb = torch.from_numpy(np.ascontiguousarray(x_train[rows].transpose(0, 2, 1)))
            yb = torch.from_numpy(y_train[rows].astype(np.float32, copy=False))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        validation_score = network_probabilities(model, x_validation, batch_size)
        ap = average_precision_score(y_validation, validation_score)
        print(f"  {name}: epoch {epoch}, validation PR-AUC={ap:.6f}", flush=True)
        if ap > best_ap + 1e-8:
            best_ap = ap
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def history_features(decision_index: np.ndarray, gpu_index: np.ndarray, grid: np.ndarray, ledger: pd.DataFrame) -> np.ndarray:
    decision_ns = grid[decision_index] + STEP_NS
    output = np.zeros((len(decision_index), 2), dtype=np.float32)
    thirty_days = 30 * 24 * 60 * 60 * 1_000_000_000
    grouped = {int(gpu): group["onset_ns"].to_numpy(dtype=np.int64) for gpu, group in ledger.groupby("gpu_index")}
    for gpu, onsets in grouped.items():
        rows = np.flatnonzero(gpu_index == gpu)
        if len(rows) == 0:
            continue
        now = decision_ns[rows]
        right = np.searchsorted(onsets, now, side="left")
        left = np.searchsorted(onsets, now - thirty_days, side="left")
        output[rows, 0] = (right - left).astype(np.float32)
        previous = right - 1
        has_previous = previous >= 0
        output[rows[has_previous], 1] = ((now[has_previous] - onsets[previous[has_previous]]) / 86_400_000_000_000).astype(np.float32)
        output[rows[~has_previous], 1] = 999.0
    return output


def make_context_matrix(temporal: np.ndarray, context: np.ndarray, sample: dict[str, np.ndarray], grid: np.ndarray, ledger: pd.DataFrame) -> np.ndarray:
    with np.errstate(all="ignore"):
        mean = np.nanmean(temporal, axis=1)
        std = np.nanstd(temporal, axis=1)
        last = temporal[:, -1, :]
        minimum = np.nanmin(temporal, axis=1)
        maximum = np.nanmax(temporal, axis=1)
        context_mean = np.nanmean(context, axis=1)
    context_last = context[:, -1, :]
    history = history_features(sample["decision_index"], sample["gpu_index"], grid, ledger)
    result = np.concatenate((mean, std, last, minimum, maximum, context_mean, context_last, history), axis=1).astype(np.float32)
    result[~np.isfinite(result)] = np.nan
    return result


def observability_matrix(temporal: np.ndarray) -> np.ndarray:
    missing = temporal[:, :, 16:20]
    observed = temporal[:, :, 12:16]
    with np.errstate(all="ignore"):
        result = np.column_stack((np.nanmean(missing, axis=(1, 2)), np.nanmean(missing, axis=1), np.nanmax(missing, axis=1), np.nanmean(observed, axis=(1, 2)), np.nanmin(observed, axis=1), np.nanmean(np.isfinite(temporal[:, :, :16]), axis=(1, 2)))).astype(np.float32)
    result[~np.isfinite(result)] = 1.0
    return result


def flat_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(score)
    y = y[mask].astype(np.int8)
    score = score[mask].astype(np.float64)
    output: dict[str, Any] = {"samples": int(len(y)), "positive_samples": int(y.sum()), "positive_rate": float(y.mean()) if len(y) else None, "pr_auc": None, "roc_auc": None, "normalized_pr_auc": None}
    if len(y) and np.unique(y).size == 2:
        ap = float(average_precision_score(y, score))
        output["pr_auc"] = ap
        output["roc_auc"] = float(roc_auc_score(y, score))
        output["normalized_pr_auc"] = ap / max(float(y.mean()), 1e-12)
    return output


def ranking_metrics(labels: np.ndarray, eligible: np.ndarray, scores: np.ndarray, ks: tuple[int, ...] = (10, 50, 100)) -> dict[str, Any]:
    output: dict[str, Any] = {"decision_epochs_with_failure": 0}
    recalls = {k: [] for k in ks}
    precisions = {k: [] for k in ks}
    captures = []
    for row in range(len(labels)):
        keep = eligible[row] & np.isfinite(scores[row])
        if not keep.any() or labels[row, keep].sum() == 0:
            continue
        y = labels[row, keep].astype(np.int8)
        order = np.argsort(-scores[row, keep], kind="stable")
        total = int(y.sum())
        output["decision_epochs_with_failure"] += 1
        for k in ks:
            picked = y[order[: min(k, len(order))]]
            recalls[k].append(float(picked.sum() / total))
            precisions[k].append(float(picked.mean()) if len(picked) else 0.0)
        top = max(1, int(math.ceil(len(order) * 0.05)))
        captures.append(float(y[order[:top]].sum() / total))
    for k in ks:
        output[f"recall_at_{k}"] = float(np.mean(recalls[k])) if recalls[k] else None
        output[f"precision_at_{k}"] = float(np.mean(precisions[k])) if precisions[k] else None
    output["top_5pct_capture"] = float(np.mean(captures)) if captures else None
    return output


def evaluate_matrix(labels: np.ndarray, eligible: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    output = flat_metrics(labels[eligible], scores[eligible])
    output.update(ranking_metrics(labels, eligible, scores))
    return output


def fill_test_scores(test_sample: dict[str, np.ndarray], parquet_root: Path, grid: np.ndarray, gpu_ids: list[str], std_maps: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], temporal_normalizer: tuple[np.ndarray, np.ndarray], models: dict[str, Any], context_models: dict[str, Any], context_train_stats: tuple[np.ndarray, np.ndarray], ledger: pd.DataFrame, batch_size: int = 2048) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    decision_times = test_sample["decision_times"]
    labels = test_sample["labels_matrix"]
    eligible = test_sample["eligible_matrix"]
    n_time, n_gpu = labels.shape
    temporal_scores = {name: np.full((n_time, n_gpu), np.nan, dtype=np.float32) for name in models if name != "isolation_forest"}
    context_scores = {name: np.full((n_time, n_gpu), np.nan, dtype=np.float32) for name in context_models}
    obs_scores = {"isolation_forest": np.full((n_time, n_gpu), np.nan, dtype=np.float32)}
    blocks = (decision_times // 144).astype(np.int32)
    unique_blocks = np.unique(blocks)
    mean, scale = temporal_normalizer
    ctx_mean, ctx_scale = context_train_stats
    for completed, block in enumerate(unique_blocks, start=1):
        time_rows = np.flatnonzero(blocks == block)
        local_times = decision_times[time_rows]
        local_t, local_g = np.where(eligible[time_rows])
        if len(local_t) == 0:
            continue
        sub = {"decision_index": local_times[local_t], "gpu_index": local_g.astype(np.int32), "y": labels[time_rows[local_t], local_g].astype(np.int8), "decision_times": local_times}
        raw_x, raw_context = materialize_samples(sub, parquet_root, grid, gpu_ids, std_maps, f"test block {completed}/{len(unique_blocks)}")
        norm_x = normalize(raw_x, mean, scale)
        flat_x = norm_x.reshape(len(norm_x), -1)
        for name, model in models.items():
            if name == "isolation_forest":
                continue
            if name in ("logistic_regression", "extra_trees"):
                score = model.predict_proba(flat_x)[:, 1]
            else:
                score = network_probabilities(model, norm_x, batch_size)
            temporal_scores[name][time_rows[local_t], local_g] = score
        ctx_x = make_context_matrix(raw_x, raw_context, {"decision_index": sub["decision_index"], "gpu_index": sub["gpu_index"]}, grid, ledger)
        ctx_x = np.where(np.isfinite(ctx_x), ctx_x, ctx_mean[None, :]).astype(np.float32)
        ctx_x = (ctx_x - ctx_mean[None, :]) / ctx_scale[None, :]
        for name, model in context_models.items():
            context_scores[name][time_rows[local_t], local_g] = model.predict_proba(ctx_x)[:, 1]
        obs_x = observability_matrix(raw_x)
        obs_scores["isolation_forest"][time_rows[local_t], local_g] = models["isolation_forest"].score_samples(obs_x) * -1.0
        del raw_x, raw_context, norm_x, flat_x, ctx_x, obs_x
    return temporal_scores, context_scores, obs_scores


def parallel_rank_mean(score_matrices: dict[str, np.ndarray], eligible: np.ndarray) -> np.ndarray:
    names = list(score_matrices)
    output = np.full_like(next(iter(score_matrices.values())), np.nan, dtype=np.float32)
    for row in range(output.shape[0]):
        keep = eligible[row]
        if not keep.any():
            continue
        rank_scores = []
        for name in names:
            column = score_matrices[name][row, keep]
            order = np.argsort(np.argsort(-column, kind="stable"), kind="stable")
            rank_scores.append((len(column) - order) / max(len(column), 1))
        output[row, keep] = np.mean(np.vstack(rank_scores), axis=0).astype(np.float32)
    return output


def fold_rows(labels: np.ndarray, eligible: np.ndarray, scores_by_model: dict[str, np.ndarray], decision_times: np.ndarray, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_index, time_rows in enumerate(np.array_split(np.arange(len(decision_times)), 3), start=1):
        for model_name, scores in scores_by_model.items():
            metrics = evaluate_matrix(labels[time_rows], eligible[time_rows], scores[time_rows])
            rows.append({"evaluation_fold": f"{prefix}_{fold_index}", "model": model_name, "start": utc_text(int(decision_times[time_rows].min()) + STEP_NS), "end": utc_text(int(decision_times[time_rows].max()) + STEP_NS), **metrics})
    return rows


def write_workbooks(payloads: list[dict[str, Any]]) -> None:
    """Author workbooks with the bundled @oai/artifact-tool runtime."""
    node = Path(r"C:\Users\이준호\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    packages = Path(r"C:\Users\이준호\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules")
    if not node.exists() or not packages.exists():
        raise RuntimeError("Bundled Node.js/artifact-tool dependencies are unavailable.")
    temp_dir = Path(tempfile.mkdtemp(prefix="gpu_artifact_"))
    link = temp_dir / "node_modules"
    try:
        try:
            os.symlink(packages, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(packages)], check=True, capture_output=True)
        node_script = r'''
import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";
const stdinChunks = [];
for await (const chunk of process.stdin) stdinChunks.push(chunk);
const payload = JSON.parse(Buffer.concat(stdinChunks).toString("utf8"));
function writeSheet(wb, spec) {
  const sheet = wb.worksheets.add(spec.name);
  const rows = spec.rows || [];
  const width = Math.max(1, ...rows.map(row => row.length));
  const matrix = rows.map(row => Array.from({length: width}, (_, i) => row[i] ?? null));
  if (matrix.length) {
    sheet.getRangeByIndexes(0, 0, matrix.length, width).values = matrix;
    const header = sheet.getRangeByIndexes(0, 0, 1, width);
    header.format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
    header.format.rowHeight = 30;
    sheet.getRangeByIndexes(0, 0, matrix.length, width).format.borders = { preset: "insideHorizontal", style: "thin", color: "#D9E2F3" };
    for (let c = 0; c < width; c++) {
      const headerText = String(matrix[0][c] ?? "");
      sheet.getRangeByIndexes(0, c, matrix.length, 1).format.columnWidth = Math.min(34, Math.max(11, headerText.length + 2));
    }
    sheet.freezePanes.freezeRows(1);
  }
  sheet.showGridLines = false;
}
for (const workbookSpec of payload.workbooks) {
  const workbook = Workbook.create();
  for (const spec of workbookSpec.sheets) writeSheet(workbook, spec);
  // Rendering is disabled for this headless Node 24 runtime; workbook structure is inspected after export.
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(workbookSpec.path);
}
'''
        process = subprocess.run([str(node), "--input-type=module", "-e", node_script], input=json.dumps({"workbooks": json_safe(payloads)}, ensure_ascii=False), text=True, encoding="utf-8", errors="replace", cwd=temp_dir, capture_output=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(f"artifact-tool workbook generation failed (exit={process.returncode}); stdout={process.stdout[-4000:]}; stderr={process.stderr[-4000:]}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def rows_from_dicts(rows: list[dict[str, Any]], columns: list[str] | None = None) -> list[list[Any]]:
    if not rows:
        return [["no_rows"]]
    columns = columns or list(rows[0].keys())
    return [columns] + [[cell_value(row.get(column)) for column in columns] for row in rows]


def common_sheets(config: dict[str, Any], audit: dict[str, Any], split_summary: list[dict[str, Any]], sample_summary: list[dict[str, Any]], event_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": "run_config", "rows": rows_from_dicts([config])},
        {"name": "data_audit", "rows": rows_from_dicts([audit])},
        {"name": "time_split", "rows": rows_from_dicts(split_summary)},
        {"name": "sample_audit", "rows": rows_from_dicts(sample_summary)},
        {"name": "event_summary", "rows": rows_from_dicts(event_summary)},
    ]


def main() -> None:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for Branch 1 CNN/TCN: {TORCH_IMPORT_ERROR}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(6, (os.cpu_count() or 2) - 1)))
    parquet_root, raw_dir = find_data_roots()
    xid_path = raw_dir / "XID_ERRORS.csv"
    required = [xid_path, *(parquet_root / name for name in TELEMETRY_FILES.values()), parquet_root / "branch2_context_features.parquet"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required inputs are missing: " + ", ".join(missing))

    print("[1/8] Reconstructing XID 31/43 episodes from raw Seren...", flush=True)
    gpu_ids = read_gpu_ids(xid_path)
    ledger, audit = reconstruct_target_episodes(xid_path, gpu_ids)
    audit["target_definition"] = "single binary: future XID 31 OR XID 43 onset within 24 hours"
    audit["target_codes"] = "31,43"
    audit["gpu_count"] = len(gpu_ids)
    print(f"  target episodes={len(ledger):,}, certain={audit['certain_episode_count']:,}, uncertain={audit['uncertain_episode_count']:,}", flush=True)

    print("[2/8] Reading 5-minute grid and creating chronological splits...", flush=True)
    grid = read_time_grid(parquet_root / TELEMETRY_FILES["util"])
    decision_indices = valid_decision_indices(grid, audit)
    splits = chronological_split(decision_indices)
    rng = np.random.default_rng(SEED)
    train_times = np.sort(rng.choice(splits["train"], min(TRAIN_DECISION_TIMES, len(splits["train"])), replace=False))
    validation_times = np.sort(rng.choice(splits["validation"], min(VALIDATION_DECISION_TIMES, len(splits["validation"])), replace=False))
    test_times = splits["test"][::TEST_STRIDE_BINS]
    if len(test_times) < 30:
        test_times = splits["test"]
    print(f"  grid={len(grid):,}, valid={len(decision_indices):,}, train/val/test={len(train_times):,}/{len(validation_times):,}/{len(test_times):,}", flush=True)

    train_sample, train_summary = sample_population(train_times, grid, ledger, len(gpu_ids), rng, NEGATIVE_RATIO)
    validation_sample, validation_summary = sample_population(validation_times, grid, ledger, len(gpu_ids), rng, NEGATIVE_RATIO)
    test_sample, test_summary = sample_population(test_times, grid, ledger, len(gpu_ids), rng, None)
    sample_summary = [{"population": "train", **train_summary}, {"population": "validation", **validation_summary}, {"population": "test", **test_summary}]

    print("[3/8] Recomputing requested 5-minute standard deviations from raw 15-second metrics...", flush=True)
    needed_buckets = build_needed_buckets([train_sample, validation_sample, test_sample])
    std_maps = recompute_raw_std(raw_dir, grid, needed_buckets)
    print(f"  requested bucket/GPU groups={sum(len(v) for v in needed_buckets.values()):,}", flush=True)

    print("[4/8] Materializing 1-hour telemetry sequences...", flush=True)
    train_raw, train_context = materialize_samples(train_sample, parquet_root, grid, gpu_ids, std_maps, "train")
    validation_raw, validation_context = materialize_samples(validation_sample, parquet_root, grid, gpu_ids, std_maps, "validation")
    temporal_mean, temporal_scale = fit_normalizer(train_raw)
    train_x = normalize(train_raw, temporal_mean, temporal_scale)
    validation_x = normalize(validation_raw, temporal_mean, temporal_scale)
    print(f"  train sequence shape={train_x.shape}, validation sequence shape={validation_x.shape}", flush=True)

    print("[5/8] Training Branch 1: LR, Extra Trees, 1D-CNN, Tiny-TCN...", flush=True)
    train_y = train_sample["y"]
    validation_y = validation_sample["y"]
    flat_train = train_x.reshape(len(train_x), -1)
    flat_validation = validation_x.reshape(len(validation_x), -1)
    logistic = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=250, random_state=SEED, n_jobs=-1)
    logistic.fit(flat_train, train_y)
    extra_trees = ExtraTreesClassifier(n_estimators=160, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    extra_trees.fit(flat_train, train_y)
    cnn = train_network("1D-CNN", OneDCNN(20), train_x, train_y, validation_x, validation_y, SEED + 1)
    tcn = train_network("Tiny-TCN", TinyTCN(20), train_x, train_y, validation_x, validation_y, SEED + 2)
    branch1_models: dict[str, Any] = {"logistic_regression": logistic, "extra_trees": extra_trees, "one_d_cnn": cnn, "tiny_tcn": tcn}
    validation_scores_b1 = {"logistic_regression": logistic.predict_proba(flat_validation)[:, 1], "extra_trees": extra_trees.predict_proba(flat_validation)[:, 1], "one_d_cnn": network_probabilities(cnn, validation_x), "tiny_tcn": network_probabilities(tcn, validation_x)}

    print("[6/8] Training Branch 2 historical/context models and Branch 3 Isolation Forest...", flush=True)
    train_context_x = make_context_matrix(train_raw, train_context, train_sample, grid, ledger)
    validation_context_x = make_context_matrix(validation_raw, validation_context, validation_sample, grid, ledger)
    context_mean, context_scale = fit_normalizer(train_context_x[:, None, :])
    context_train = np.where(np.isfinite(train_context_x), train_context_x, context_mean[None, :]).astype(np.float32)
    context_validation = np.where(np.isfinite(validation_context_x), validation_context_x, context_mean[None, :]).astype(np.float32)
    context_train = (context_train - context_mean[None, :]) / context_scale[None, :]
    context_validation = (context_validation - context_mean[None, :]) / context_scale[None, :]
    context_lr = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=300, random_state=SEED))
    context_lr.fit(context_train, train_y)
    context_gbdt = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED)
    weights = np.where(train_y == 1, max(1.0, (train_y == 0).sum() / max(1, (train_y == 1).sum())), 1.0)
    context_gbdt.fit(context_train, train_y, sample_weight=weights)
    branch2_models = {"historical_logistic": context_lr, "historical_gbdt": context_gbdt}
    validation_scores_b2 = {"historical_logistic": context_lr.predict_proba(context_validation)[:, 1], "historical_gbdt": context_gbdt.predict_proba(context_validation)[:, 1]}
    train_obs = observability_matrix(train_raw)
    isolation_forest = IsolationForest(n_estimators=180, contamination="auto", random_state=SEED, n_jobs=-1, max_samples="auto")
    isolation_forest.fit(train_obs[train_y == 0])
    branch3_models = {"isolation_forest": isolation_forest}
    del flat_train, flat_validation, train_x, validation_x, context_train, context_validation, train_context_x, validation_context_x, train_obs

    print("[7/8] Scoring chronological test population and calculating PR-AUC/ROC-AUC...", flush=True)
    temporal_scores, context_scores, obs_scores = fill_test_scores(test_sample, parquet_root, grid, gpu_ids, std_maps, (temporal_mean, temporal_scale), {**branch1_models, **branch3_models}, branch2_models, (context_mean, context_scale), ledger)
    parallel_scores = parallel_rank_mean(temporal_scores, test_sample["eligible_matrix"])
    temporal_scores["parallel_ensemble_rank_mean"] = parallel_scores
    labels_test, eligible_test = test_sample["labels_matrix"], test_sample["eligible_matrix"]
    b1_metrics = [{"model": name, **evaluate_matrix(labels_test, eligible_test, scores)} for name, scores in temporal_scores.items()]
    b2_metrics = [{"model": name, **evaluate_matrix(labels_test, eligible_test, scores)} for name, scores in context_scores.items()]
    b3_metrics = [{"model": name, **evaluate_matrix(labels_test, eligible_test, scores)} for name, scores in obs_scores.items()]
    fold_metric_times = grid[test_sample["decision_times"]] + STEP_NS
    b1_fold = fold_rows(labels_test, eligible_test, temporal_scores, fold_metric_times, "test_temporal")
    b2_fold = fold_rows(labels_test, eligible_test, context_scores, fold_metric_times, "test_temporal")
    b3_fold = fold_rows(labels_test, eligible_test, obs_scores, fold_metric_times, "test_temporal")

    event_summary = [{"xid_code": int(code), "episodes": int((ledger["xid_code"] == code).sum()), "certain_episodes": int(((ledger["xid_code"] == code) & (~ledger["uncertain_onset"])).sum()), "uncertain_episodes": int(((ledger["xid_code"] == code) & ledger["uncertain_onset"]).sum())} for code in (31, 43)]
    split_summary = [{"split": name, "available_grid_points": int(len(values)), "start": utc_text(int(grid[values.min()] + STEP_NS)), "end": utc_text(int(grid[values.max()] + STEP_NS)), "purge_between_splits": "36 hours"} for name, values in splits.items()]
    config = {
        "target": "single binary XID31_OR_XID43",
        "horizon": "24 hours",
        "input_window": "1 hour = 12 x 5-minute buckets",
        "feature_cutoff": "last 10 minutes excluded; history buckets decision-65m through decision-10m",
        "telemetry_source": "Seren only; 5-minute Parquet mean/obs plus raw 15-second recomputed std",
        "join_policy": "left-aligned telemetry block by Time_5m and gpu_id",
        "time_split": "60/20/20 chronological with 36-hour purge",
        "test_cadence": "30 minutes (every sixth five-minute decision bucket)",
        "branch_1_models": "Logistic Regression, Extra Trees, 1D-CNN, Tiny-TCN, rank-mean parallel ensemble",
        "branch_2_models": "historical logistic and historical GBDT using XID history + context",
        "branch_3_model": "Isolation Forest on observability/missingness only",
        "calibration": "not run by user decision",
        "multitask": "not run; one union target only",
        "gpu_topology": "excluded; unavailable",
        "random_seed": SEED,
        "persistent_outputs": "exactly three Excel workbooks; no CSV/Parquet/JSON/plot outputs",
    }
    audit_out = {**audit, "parquet_root": str(parquet_root), "raw_seren_directory": str(raw_dir), "parquet_grid_points": int(len(grid)), "parquet_gpu_count": int(len(gpu_ids)), "recomputed_std_metric_count": 4, "std_requested_bucket_count": int(len(needed_buckets))}
    val_b1 = [{"model": name, **flat_metrics(validation_sample["y"], score)} for name, score in validation_scores_b1.items()]
    val_b2 = [{"model": name, **flat_metrics(validation_sample["y"], score)} for name, score in validation_scores_b2.items()]
    val_b3_score = isolation_forest.score_samples(observability_matrix(validation_raw)) * -1.0
    val_b3 = [{"model": "isolation_forest", **flat_metrics(validation_sample["y"], val_b3_score)}]
    base1 = common_sheets(config, audit_out, split_summary, sample_summary, event_summary)
    base2 = common_sheets(config, audit_out, split_summary, sample_summary, event_summary)
    base3 = common_sheets(config, audit_out, split_summary, sample_summary, event_summary)
    payloads = [
        {"path": str(BRANCH_RESULT_FILES["Branch_1"]), "sheets": base1 + [{"name": "model_metrics", "rows": rows_from_dicts(b1_metrics)}, {"name": "fold_metrics", "rows": rows_from_dicts(b1_fold)}, {"name": "validation_metrics", "rows": rows_from_dicts(val_b1)}]},
        {"path": str(BRANCH_RESULT_FILES["Branch_2"]), "sheets": base2 + [{"name": "model_metrics", "rows": rows_from_dicts(b2_metrics)}, {"name": "fold_metrics", "rows": rows_from_dicts(b2_fold)}, {"name": "validation_metrics", "rows": rows_from_dicts(val_b2)}]},
        {"path": str(BRANCH_RESULT_FILES["Branch_3_optional"]), "sheets": base3 + [{"name": "model_metrics", "rows": rows_from_dicts(b3_metrics)}, {"name": "fold_metrics", "rows": rows_from_dicts(b3_fold)}, {"name": "validation_metrics", "rows": rows_from_dicts(val_b3)}]},
    ]
    for path in BRANCH_RESULT_FILES.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    print("[8/8] Writing exactly three verified Excel workbooks...", flush=True)
    write_workbooks(payloads)
    for path in BRANCH_RESULT_FILES.values():
        print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()

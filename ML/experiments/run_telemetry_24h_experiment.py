"""Run the first leakage-aware Telemetry -> 24h XID risk experiment.

This script intentionally uses only the four 5-minute telemetry Parquets for
model inputs.  It reads the original XID file once to reconstruct exact
30-second episodes; the supplied 5-minute XID metadata is not precise enough
to be the official target.

Outputs are deliberately small:
  - xid_episode_ledger.parquet  (reproducible target audit)
  - metrics.json
  - report.md
No Excel files, full prediction dump, or sampled feature dump is written.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import timedelta, timezone
from pathlib import Path

# Dependencies are installed locally so the project remains self-contained.
VENDOR = Path(__file__).with_name("_vendor")
if VENDOR.exists():
    import sys

    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from torch import nn


KST = timezone(timedelta(hours=8))
STEP_NS = int(pd.Timedelta(minutes=5).value)
MERGE_NS = int(pd.Timedelta(seconds=30).value)
HORIZON_NS = int(pd.Timedelta(hours=24).value)
PURGE_NS = int(pd.Timedelta(hours=36).value)
FEATURE_OFFSETS = np.arange(-8, -2, dtype=np.int64)  # t-40m ... t-15m
CHANNEL_NAMES = [
    "util_mean",
    "temp_mean",
    "power_mean",
    "fb_mean",
    "util_delta",
    "temp_delta",
    "power_delta",
    "fb_delta",
    "util_obs",
    "temp_obs",
    "power_obs",
    "fb_obs",
    "util_missing",
    "temp_missing",
    "power_missing",
    "fb_missing",
]
METRICS = (
    ("util", "telemetry_5m_util.parquet"),
    ("temp", "telemetry_5m_temp.parquet"),
    ("power", "telemetry_5m_power.parquet"),
    ("fb", "telemetry_5m_fb.parquet"),
)


def ns_to_text(value: int) -> str:
    return str(pd.Timestamp(int(value), unit="ns", tz="UTC").tz_convert(KST))


def as_json(value):
    """Convert NumPy/Pandas values used in the compact report to JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/telemetry_24h_fixed")
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--train-times", type=int, default=800)
    parser.add_argument("--validation-times", type=int, default=300)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rebuild-labels", action="store_true")
    return parser.parse_args()


def read_gpu_ids(xid_path: Path) -> list[str]:
    header = pd.read_csv(xid_path, nrows=0)
    if header.columns[0] != "Time":
        raise ValueError("XID file must have Time as its first column.")
    return list(header.columns[1:])


def build_episode_ledger(
    xid_path: Path, gpu_ids: list[str], cache_path: Path, rebuild: bool
) -> tuple[pd.DataFrame, dict]:
    """Stream the wide XID file with an explicit per-GPU episode state machine.

    Policy fixed for this run:
      * a valid zero closes an episode;
      * same code repeats within 30 seconds remain one episode;
      * a positive immediately after missing data/global time gap is flagged
        uncertain and is censored from the primary future-onset target.
    """
    audit_path = cache_path.with_suffix(".audit.json")
    if cache_path.exists() and audit_path.exists() and not rebuild:
        return pq.read_table(cache_path).to_pandas(), json.loads(audit_path.read_text("utf-8"))

    n_gpu = len(gpu_ids)
    active_code = np.full(n_gpu, -1, dtype=np.int32)
    active_start = np.zeros(n_gpu, dtype=np.int64)
    active_last = np.zeros(n_gpu, dtype=np.int64)
    active_uncertain = np.zeros(n_gpu, dtype=bool)
    previous_missing = np.ones(n_gpu, dtype=bool)
    previous_time = None
    episodes: list[dict] = []
    gaps: list[dict] = []
    audit = {
        "raw_rows": 0,
        "raw_xid_path": str(xid_path),
        "merge_seconds": 30,
        "zero_is_episode_boundary": True,
        "missing_to_positive_is_censored": True,
        "invalid_or_missing_cells": 0,
        "global_gaps_over_30_seconds": 0,
        "raw_start_ns": None,
        "raw_end_ns": None,
    }

    def close(indices: np.ndarray, reason: str) -> None:
        for gpu_idx in indices:
            episodes.append(
                {
                    "gpu_id": gpu_ids[int(gpu_idx)],
                    "xid_code": int(active_code[gpu_idx]),
                    "onset_ns": int(active_start[gpu_idx]),
                    "episode_end_ns": int(active_last[gpu_idx]),
                    "uncertain_onset": bool(active_uncertain[gpu_idx]),
                    "close_reason": reason,
                }
            )
        active_code[indices] = -1
        active_start[indices] = 0
        active_last[indices] = 0
        active_uncertain[indices] = False

    reader = pd.read_csv(xid_path, chunksize=500, low_memory=False)
    for chunk_number, chunk in enumerate(reader, start=1):
        if list(chunk.columns[1:]) != gpu_ids:
            raise ValueError("XID GPU header order changed inside the source file.")
        times = pd.to_datetime(chunk.iloc[:, 0], utc=True, errors="raise")
        values = chunk.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64
        )
        time_ns = times.dt.as_unit("ns").astype("int64").to_numpy()

        for row_index, now in enumerate(time_ns):
            row = values[row_index]
            if audit["raw_start_ns"] is None:
                audit["raw_start_ns"] = int(now)
            audit["raw_end_ns"] = int(now)

            has_gap = previous_time is not None and now - previous_time > MERGE_NS
            if has_gap:
                active = np.flatnonzero(active_code >= 0)
                if active.size:
                    close(active, "time_gap")
                gaps.append(
                    {"start_ns": int(previous_time), "end_ns": int(now), "seconds": float((now - previous_time) / 1e9)}
                )

            finite = np.isfinite(row)
            integer = finite & (np.floor(row) == row)
            valid = integer & (row >= 0)
            audit["invalid_or_missing_cells"] += int((~valid).sum())
            code = np.full(n_gpu, -1, dtype=np.int32)
            code[valid] = row[valid].astype(np.int32)

            # A real zero is an observed recovery boundary, never bridge it.
            zero_close = np.flatnonzero((active_code >= 0) & valid & (code == 0))
            if zero_close.size:
                close(zero_close, "observed_zero")

            positive = valid & (code > 0)
            continuing = positive & (active_code >= 0) & (code == active_code)
            timed_out = continuing & (now - active_last > MERGE_NS)
            code_changed = positive & (active_code >= 0) & (code != active_code)
            restart_close = np.flatnonzero(timed_out | code_changed)
            if restart_close.size:
                close(restart_close, "repeat_timeout_or_code_change")

            starts = positive & (active_code < 0)
            if starts.any():
                active_code[starts] = code[starts]
                active_start[starts] = now
                active_last[starts] = now
                active_uncertain[starts] = previous_missing[starts]

            updates = positive & (active_code >= 0)
            active_last[updates] = now
            previous_missing = ~valid
            previous_time = now

        audit["raw_rows"] += len(chunk)
        if chunk_number % 100 == 0:
            print(f"  XID scan: {audit['raw_rows']:,} rows", flush=True)

    remaining = np.flatnonzero(active_code >= 0)
    if remaining.size:
        close(remaining, "end_of_file")

    ledger = pd.DataFrame(episodes).sort_values(["gpu_id", "onset_ns"]).reset_index(drop=True)
    audit["global_gaps_over_30_seconds"] = len(gaps)
    audit["gaps"] = gaps
    audit["episode_count"] = int(len(ledger))
    audit["certain_episode_count"] = int((~ledger["uncertain_onset"]).sum())
    audit["uncertain_episode_count"] = int(ledger["uncertain_onset"].sum())
    audit["xid_code_counts"] = {
        str(int(code)): int(count)
        for code, count in ledger["xid_code"].value_counts().sort_index().items()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(cache_path, index=False)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=as_json), "utf-8")
    return ledger, audit


def read_time_grid(util_path: Path) -> np.ndarray:
    """Read only the duplicated time column and return the unique 5-minute grid."""
    unique_chunks = []
    parquet = pq.ParquetFile(util_path)
    for batch in parquet.iter_batches(columns=["Time_5m"], batch_size=2_000_000):
        series = pd.to_datetime(batch.column(0).to_pandas(), utc=True)
        unique_chunks.append(np.unique(series.dt.as_unit("ns").astype("int64").to_numpy()))
    grid = np.unique(np.concatenate(unique_chunks))
    if grid.size < 10 or np.any(np.diff(grid) <= 0):
        raise ValueError("Telemetry time grid is invalid.")
    return grid.astype(np.int64, copy=False)


def valid_decision_times(grid: np.ndarray, audit: dict) -> np.ndarray:
    """Enforce feature continuity, complete label horizon, and outage censoring."""
    valid = np.ones(grid.size, dtype=bool)
    valid[:8] = False
    valid[8:] &= grid[8:] - grid[:-8] == 8 * STEP_NS
    valid &= grid >= int(audit["raw_start_ns"]) + 40 * 60 * 1_000_000_000
    valid &= grid + HORIZON_NS <= int(audit["raw_end_ns"])
    for gap in audit["gaps"]:
        start, end = int(gap["start_ns"]), int(gap["end_ns"])
        # Censor a label horizon that contains a period with no XID observations.
        valid &= ~((grid < end) & (grid + HORIZON_NS > start))
    result = grid[valid]
    if result.size == 0:
        raise ValueError("No leakage-safe decision times remain after quality gates.")
    return result


def chronological_split(times: np.ndarray) -> dict[str, np.ndarray]:
    """60/20/20 chronological split with a 36-hour gap on both boundaries."""
    q1 = times[int(len(times) * 0.60)]
    q2 = times[int(len(times) * 0.80)]
    split = {
        "train": times[times <= q1 - PURGE_NS],
        "validation": times[(times >= q1 + PURGE_NS) & (times <= q2 - PURGE_NS)],
        "test": times[times >= q2 + PURGE_NS],
    }
    if min(len(values) for values in split.values()) == 0:
        raise ValueError("A chronological split became empty after the temporal purge.")
    return split


def future_event(times: np.ndarray, onset_times: np.ndarray) -> np.ndarray:
    """Whether an onset exists in the open/closed interval (t, t + 24h]."""
    if onset_times.size == 0:
        return np.zeros(times.size, dtype=bool)
    position = np.searchsorted(onset_times, times, side="right")
    answer = np.zeros(times.size, dtype=bool)
    available = position < onset_times.size
    answer[available] = onset_times[position[available]] <= times[available] + HORIZON_NS
    return answer


def label_state(
    times: np.ndarray, ledger: pd.DataFrame, gpu_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return primary labels and the policy-driven eligibility mask.

    A sample is censored when it is already in an active XID episode or when
    its future 24-hour horizon contains an uncertain (missing->positive) onset.
    """
    gpu_to_index = {gpu: index for index, gpu in enumerate(gpu_ids)}
    labels = np.zeros((len(times), len(gpu_ids)), dtype=bool)
    eligible = np.ones_like(labels)

    for gpu_id, group in ledger.groupby("gpu_id", sort=False):
        gpu_index = gpu_to_index[gpu_id]
        group = group.sort_values("onset_ns")
        starts = group["onset_ns"].to_numpy(dtype=np.int64)
        ends = group["episode_end_ns"].to_numpy(dtype=np.int64)
        uncertain = group["uncertain_onset"].to_numpy(dtype=bool)

        labels[:, gpu_index] = future_event(times, starts[~uncertain])
        uncertain_future = future_event(times, starts[uncertain])

        previous = np.searchsorted(starts, times, side="right") - 1
        active = np.zeros(len(times), dtype=bool)
        has_previous = previous >= 0
        active[has_previous] = ends[previous[has_previous]] >= times[has_previous]
        eligible[:, gpu_index] = ~(active | uncertain_future)

    return labels, eligible


def choose_training_samples(
    times: np.ndarray,
    ledger: pd.DataFrame,
    gpu_ids: list[str],
    negative_ratio: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    labels, eligible = label_state(times, ledger, gpu_ids)
    pos_t, pos_g = np.where(labels & eligible)
    neg_t, neg_g = np.where((~labels) & eligible)
    if len(pos_t) == 0:
        raise ValueError("The training time sample contains no positive labels.")
    wanted_negative = min(len(neg_t), len(pos_t) * negative_ratio)
    selected_negative = rng.choice(len(neg_t), size=wanted_negative, replace=False)
    time_rows = np.concatenate((pos_t, neg_t[selected_negative]))
    gpu_rows = np.concatenate((pos_g, neg_g[selected_negative]))
    y = np.concatenate((np.ones(len(pos_t), dtype=np.int8), np.zeros(wanted_negative, dtype=np.int8)))
    order = rng.permutation(len(y))
    stats = {
        "candidate_times": int(len(times)),
        "positive_samples": int(len(pos_t)),
        "negative_samples": int(wanted_negative),
        "censored_samples": int((~eligible).sum()),
    }
    return times[time_rows[order]], gpu_rows[order], y[order], stats


def choose_population_samples(
    times: np.ndarray, ledger: pd.DataFrame, gpu_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    labels, eligible = label_state(times, ledger, gpu_ids)
    time_rows, gpu_rows = np.where(eligible)
    stats = {
        "candidate_times": int(len(times)),
        "samples": int(len(time_rows)),
        "positive_samples": int(labels[eligible].sum()),
        "censored_samples": int((~eligible).sum()),
    }
    return times[time_rows], gpu_rows, labels[eligible].astype(np.int8), stats


def ns_to_filter_timestamp(value: int) -> pd.Timestamp:
    return pd.Timestamp(int(value), unit="ns", tz="UTC").tz_convert(KST)


def load_feature_block(
    data_dir: Path,
    start_ns: int,
    end_ns: int,
    gpu_ids: list[str],
) -> np.ndarray:
    """Load a compact dense [time, GPU, channel] block from the four Parquets."""
    block_times = np.arange(start_ns, end_ns + STEP_NS, STEP_NS, dtype=np.int64)
    n_time, n_gpu = len(block_times), len(gpu_ids)
    values = np.full((n_time, n_gpu, 4), np.nan, dtype=np.float32)
    observed = np.zeros((n_time, n_gpu, 4), dtype=np.float32)
    categories = pd.Index(gpu_ids)

    for metric_index, (metric, filename) in enumerate(METRICS):
        table = pq.read_table(
            data_dir / filename,
            columns=["Time_5m", "gpu_id", f"{metric}_mean", f"{metric}_obs"],
            filters=[
                ("Time_5m", ">=", ns_to_filter_timestamp(start_ns)),
                ("Time_5m", "<=", ns_to_filter_timestamp(end_ns)),
            ],
        )
        if table.num_rows == 0:
            continue
        frame = table.to_pandas()
        time_values = pd.to_datetime(frame["Time_5m"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        time_index = ((time_values - start_ns) // STEP_NS).astype(np.int64)
        gpu_index = categories.get_indexer(frame["gpu_id"])
        if (gpu_index < 0).any():
            raise ValueError(f"Unexpected GPU ID in {filename}.")
        in_block = (time_index >= 0) & (time_index < n_time)
        value = pd.to_numeric(frame[f"{metric}_mean"], errors="coerce").to_numpy(dtype=np.float32)
        obs = pd.to_numeric(frame[f"{metric}_obs"], errors="coerce").to_numpy(dtype=np.float32)
        values[time_index[in_block], gpu_index[in_block], metric_index] = value[in_block]
        observed[time_index[in_block], gpu_index[in_block], metric_index] = np.nan_to_num(
            obs[in_block], nan=0.0
        )

    delta = np.full_like(values, np.nan)
    delta[1:] = values[1:] - values[:-1]
    missing = ((~np.isfinite(values)) | (observed < 1.0)).astype(np.float32)
    return np.concatenate((values, delta, observed, missing), axis=2)


def half_day_keys(times: np.ndarray) -> np.ndarray:
    local = pd.to_datetime(times, unit="ns", utc=True).tz_convert(KST)
    return np.asarray([f"{day:%Y%m%d}-{day.hour // 12}" for day in local], dtype=object)


def materialize_sequences(
    data_dir: Path,
    sample_times: np.ndarray,
    gpu_index: np.ndarray,
    gpu_ids: list[str],
    title: str,
) -> np.ndarray:
    """Materialize only requested sequences, grouping work in 12-hour blocks."""
    output = np.empty((len(sample_times), len(FEATURE_OFFSETS), len(CHANNEL_NAMES)), dtype=np.float32)
    keys = half_day_keys(sample_times)
    for completed, key in enumerate(np.unique(keys), start=1):
        rows = np.flatnonzero(keys == key)
        start = int(sample_times[rows].min() + FEATURE_OFFSETS[0] * STEP_NS)
        end = int(sample_times[rows].max() + FEATURE_OFFSETS[-1] * STEP_NS)
        feature_block = load_feature_block(data_dir, start, end, gpu_ids)
        decision = ((sample_times[rows] - start) // STEP_NS).astype(np.int64)
        offsets = decision[:, None] + FEATURE_OFFSETS[None, :]
        output[rows] = feature_block[offsets, gpu_index[rows, None], :]
        print(f"  {title}: {completed}/{len(np.unique(keys))} blocks", flush=True)
    return output


def fit_normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    scale = np.nanstd(flat, axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean, scale


def normalize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = np.where(np.isfinite(x), x, mean[None, None, :]).astype(np.float32, copy=False)
    return ((result - mean[None, None, :]) / scale[None, None, :]).astype(np.float32, copy=False)


class OneDCNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).squeeze(-1)).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2, dilation=dilation, padding=dilation
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)[:, :, : x.shape[-1]]  # remove right-padding: causal output
        return torch.relu(y + self.skip(x))


class TinyTCN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block1 = CausalBlock(channels, 32, dilation=1)
        self.block2 = CausalBlock(32, 32, dilation=2)
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(x))
        return self.head(x[:, :, -1]).squeeze(-1)


def network_probabilities(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            block = np.ascontiguousarray(x[start : start + batch_size].transpose(0, 2, 1))
            logits = model(torch.from_numpy(block)).cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-logits)))
    return np.concatenate(scores).astype(np.float32, copy=False)


def train_network(
    name: str,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> nn.Module:
    rng = np.random.default_rng(seed)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_ap, best_state, stale = -np.inf, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for start in range(0, len(y_train), batch_size):
            rows = rng.permutation(len(y_train))[start : start + batch_size] if start == 0 else order[start : start + batch_size]
            # Construct one permutation per epoch without retaining a giant tensor.
            if start == 0:
                order = rng.permutation(len(y_train))
                rows = order[:batch_size]
            xb = np.ascontiguousarray(x_train[rows].transpose(0, 2, 1))
            yb = torch.from_numpy(y_train[rows].astype(np.float32, copy=False))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(torch.from_numpy(xb)), yb)
            loss.backward()
            optimizer.step()

        validation_score = network_probabilities(model, x_validation, batch_size)
        validation_ap = average_precision_score(y_validation, validation_score)
        print(f"  {name} epoch {epoch}: validation PR-AUC={validation_ap:.6f}", flush=True)
        if validation_ap > best_ap + 1e-8:
            best_ap = validation_ap
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break

    model.load_state_dict(best_state)
    return model


class PlattScaler:
    """Independent probability calibration trained only on the chronological validation set."""

    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)

    def fit(self, score: np.ndarray, y: np.ndarray) -> "PlattScaler":
        logit = np.log(np.clip(score, 1e-6, 1 - 1e-6) / np.clip(1 - score, 1e-6, 1))
        self.model.fit(logit.reshape(-1, 1), y)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        logit = np.log(np.clip(score, 1e-6, 1 - 1e-6) / np.clip(1 - score, 1e-6, 1))
        return self.model.predict_proba(logit.reshape(-1, 1))[:, 1].astype(np.float32)


def ece(score: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (score >= left) & (score < right if right < 1 else score <= right)
        if in_bin.any():
            value += abs(float(score[in_bin].mean()) - float(y[in_bin].mean())) * (in_bin.sum() / total)
    return float(value)


def ranking_metrics(score: np.ndarray, labels: np.ndarray, eligible: np.ndarray) -> dict:
    result: dict[str, float] = {}
    active_epochs = 0
    recalls = {k: [] for k in (10, 20, 50)}
    ndcgs = {k: [] for k in (10, 20, 50)}
    top5 = []
    for row in range(labels.shape[0]):
        y = labels[row, eligible[row]].astype(np.int8)
        if y.sum() == 0:
            continue
        s = score[row, eligible[row]]
        order = np.argsort(-s, kind="stable")
        total_positive = int(y.sum())
        active_epochs += 1
        for k in recalls:
            picked = y[order[: min(k, len(order))]]
            recalls[k].append(float(picked.sum() / total_positive))
            discounts = 1.0 / np.log2(np.arange(2, len(picked) + 2))
            dcg = float((picked * discounts).sum())
            ideal_count = min(k, total_positive)
            ideal = float((1.0 / np.log2(np.arange(2, ideal_count + 2))).sum())
            ndcgs[k].append(dcg / ideal if ideal else 0.0)
        k5 = max(1, int(np.ceil(len(order) * 0.05)))
        top5.append(float(y[order[:k5]].sum() / total_positive))
    result["epochs_with_failure"] = active_epochs
    for k in recalls:
        result[f"recall_at_{k}"] = float(np.mean(recalls[k])) if recalls[k] else float("nan")
        result[f"ndcg_at_{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else float("nan")
    result["top_5pct_capture"] = float(np.mean(top5)) if top5 else float("nan")
    return result


def evaluate_test(
    data_dir: Path,
    test_times: np.ndarray,
    ledger: pd.DataFrame,
    gpu_ids: list[str],
    normalizer: tuple[np.ndarray, np.ndarray],
    baseline: LogisticRegression,
    cnn: nn.Module,
    tcn: nn.Module,
    calibrators: dict[str, PlattScaler],
    batch_size: int,
) -> tuple[dict, dict]:
    labels, eligible = label_state(test_times, ledger, gpu_ids)
    score_matrix = {
        "baseline_logistic": np.full(labels.shape, np.nan, dtype=np.float32),
        "one_d_cnn": np.full(labels.shape, np.nan, dtype=np.float32),
        "tiny_tcn": np.full(labels.shape, np.nan, dtype=np.float32),
    }
    keys = half_day_keys(test_times)
    mean, scale = normalizer

    for completed, key in enumerate(np.unique(keys), start=1):
        time_rows = np.flatnonzero(keys == key)
        local_t, gpu_rows = np.where(eligible[time_rows])
        sample_times = test_times[time_rows[local_t]]
        x = materialize_sequences(data_dir, sample_times, gpu_rows, gpu_ids, f"test {key}")
        x = normalize(x, mean, scale)

        raw_scores = {
            "baseline_logistic": baseline.predict_proba(x.reshape(len(x), -1))[:, 1],
            "one_d_cnn": network_probabilities(cnn, x, batch_size),
            "tiny_tcn": network_probabilities(tcn, x, batch_size),
        }
        for name, raw in raw_scores.items():
            score_matrix[name][time_rows[local_t], gpu_rows] = calibrators[name].predict(raw)
        print(f"  test evaluation: {completed}/{len(np.unique(keys))} blocks", flush=True)

    metrics = {}
    y_all = labels[eligible].astype(np.int8)
    for name, matrix in score_matrix.items():
        score_all = matrix[eligible]
        if not np.isfinite(score_all).all():
            raise ValueError(f"Non-finite test prediction found for {name}.")
        model_metrics = {
            "pr_auc": float(average_precision_score(y_all, score_all)),
            "brier": float(brier_score_loss(y_all, score_all)),
            "ece": ece(score_all, y_all),
            "test_samples": int(len(y_all)),
            "test_positive_samples": int(y_all.sum()),
            "test_positive_rate": float(y_all.mean()),
        }
        model_metrics.update(ranking_metrics(matrix, labels, eligible))
        metrics[name] = model_metrics

    test_summary = {
        "decision_epochs": int(len(test_times)),
        "eligible_samples": int(eligible.sum()),
        "censored_samples": int((~eligible).sum()),
        "positive_samples": int(labels[eligible].sum()),
        "test_start": ns_to_text(test_times.min()),
        "test_end": ns_to_text(test_times.max()),
    }
    return metrics, test_summary


def render_report(result: dict) -> str:
    metrics = result["metrics"]
    rows = []
    for name, value in metrics.items():
        rows.append(
            "| {name} | {pr:.6f} | {r10:.4f} | {n10:.4f} | {capture:.4f} | {ece:.6f} |".format(
                name=name,
                pr=value["pr_auc"],
                r10=value["recall_at_10"],
                n10=value["ndcg_at_10"],
                capture=value["top_5pct_capture"],
                ece=value["ece"],
            )
        )
    return f"""# Telemetry 5분 → 24시간 XID 발생확률: 고정 윈도우 1차 실험

## 실행 범위

- 입력: 팀 제공 5분 Telemetry Parquet의 Util / Temp / Power / FB
- 입력 시퀀스: 의사결정 시점 `t`의 `[t-40분, t-10분]` (6개 5분 bucket)
- 타깃: `(t, t+24시간]`의 동일 GPU **신규** XID onset 여부
- 모델: Logistic Regression baseline, 1D-CNN, Tiny-TCN
- 이번 실험에서 제외: ADST, Sliding Training, Branch-2 Context, Branch-3 Observability 모델
- 시간 분할: 60/20/20 chronological split, 경계마다 36시간 purge

## 라벨·누수 정책

- 동일 GPU·동일 XID가 30초 이내 반복되면 하나의 episode로 병합했습니다.
- 관측된 `0`은 episode 종료로 처리했습니다.
- 결측 직후 양성인 onset은 primary target에서 censor 했습니다.
- 진행 중인 episode의 decision time 및 불확실 onset이 미래 24시간 안에 있는 sample은 평가에서 제외했습니다.
- 원본 XID의 관측 공백을 가로지르는 24시간 horizon도 제외했습니다.

## 데이터 요약

- Raw XID episode: {result['label_audit']['episode_count']:,}건
- Primary(확실) episode: {result['label_audit']['certain_episode_count']:,}건
- Censor된 missing→positive episode: {result['label_audit']['uncertain_episode_count']:,}건
- XID 30초 초과 관측 공백: {result['label_audit']['global_gaps_over_30_seconds']}개
- Telemetry feature channel: {', '.join(CHANNEL_NAMES)}
- 참고: 팀 Parquet에는 5분 표준편차가 없어, 이번 1차 입력은 mean / 5분 변화량 / 관측률 / missing mask로 구성했습니다.

## 평가 데이터

- 테스트 기간: {result['test_summary']['test_start']} ~ {result['test_summary']['test_end']}
- 테스트 decision epoch: {result['test_summary']['decision_epochs']:,}
- 평가 sample: {result['test_summary']['eligible_samples']:,}
- 양성 sample: {result['test_summary']['positive_samples']:,}
- Censor sample: {result['test_summary']['censored_samples']:,}

## 테스트 지표

| Model | PR-AUC | Recall@10 | NDCG@10 | Top-5% Capture | ECE |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`Recall@K`, `NDCG@K`, Top-5% Capture는 실제 오류가 하나 이상 있는 decision epoch에서 macro-average했습니다. PR-AUC, Brier score, ECE는 전체 leakage-safe test sample에서 계산했습니다.
"""


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(6, (os.cpu_count() or 2) - 1)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    xid_path = args.data_dir / "XID_ERRORS-002.csv"
    util_path = args.data_dir / "telemetry_5m_util.parquet"
    required = [xid_path, util_path] + [args.data_dir / filename for _, filename in METRICS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input is missing: " + ", ".join(missing))

    print("[1/6] Reconstructing exact XID episodes...", flush=True)
    gpu_ids = read_gpu_ids(xid_path)
    ledger, audit = build_episode_ledger(
        xid_path,
        gpu_ids,
        args.output_dir / "xid_episode_ledger.parquet",
        args.rebuild_labels,
    )
    print(
        f"  episodes={len(ledger):,}, certain={(~ledger['uncertain_onset']).sum():,}, "
        f"uncertain={ledger['uncertain_onset'].sum():,}",
        flush=True,
    )

    print("[2/6] Building leakage-safe chronological splits...", flush=True)
    grid = read_time_grid(util_path)
    decision_times = valid_decision_times(grid, audit)
    splits = chronological_split(decision_times)
    rng = np.random.default_rng(args.seed)
    train_times = rng.choice(splits["train"], size=min(args.train_times, len(splits["train"])), replace=False)
    validation_times = rng.choice(
        splits["validation"], size=min(args.validation_times, len(splits["validation"])), replace=False
    )
    train_times.sort()
    validation_times.sort()

    print("[3/6] Materializing real training and validation sequences...", flush=True)
    train_t, train_g, y_train, train_summary = choose_training_samples(
        train_times, ledger, gpu_ids, args.negative_ratio, rng
    )
    val_t, val_g, y_val, validation_summary = choose_population_samples(validation_times, ledger, gpu_ids)
    x_train = materialize_sequences(args.data_dir, train_t, train_g, gpu_ids, "train")
    x_val = materialize_sequences(args.data_dir, val_t, val_g, gpu_ids, "validation")
    mean, scale = fit_normalizer(x_train)
    x_train = normalize(x_train, mean, scale)
    x_val = normalize(x_val, mean, scale)

    print("[4/6] Training baseline, 1D-CNN, and Tiny-TCN...", flush=True)
    baseline = LogisticRegression(
        solver="lbfgs", class_weight="balanced", max_iter=300, random_state=args.seed
    )
    baseline.fit(x_train.reshape(len(x_train), -1), y_train)
    cnn = train_network(
        "1D-CNN", OneDCNN(len(CHANNEL_NAMES)), x_train, y_train, x_val, y_val,
        args.epochs, args.batch_size, args.seed + 1,
    )
    tcn = train_network(
        "Tiny-TCN", TinyTCN(len(CHANNEL_NAMES)), x_train, y_train, x_val, y_val,
        args.epochs, args.batch_size, args.seed + 2,
    )

    raw_validation = {
        "baseline_logistic": baseline.predict_proba(x_val.reshape(len(x_val), -1))[:, 1],
        "one_d_cnn": network_probabilities(cnn, x_val, args.batch_size),
        "tiny_tcn": network_probabilities(tcn, x_val, args.batch_size),
    }
    calibrators = {name: PlattScaler().fit(score, y_val) for name, score in raw_validation.items()}

    print("[5/6] Evaluating the full chronological test period...", flush=True)
    metrics, test_summary = evaluate_test(
        args.data_dir,
        splits["test"],
        ledger,
        gpu_ids,
        (mean, scale),
        baseline,
        cnn,
        tcn,
        calibrators,
        args.batch_size,
    )

    result = {
        "experiment": {
            "seed": args.seed,
            "input_window": "[t-40m, t-10m] / 6 five-minute buckets",
            "label_window": "(t, t+24h]",
            "models": ["logistic baseline", "1D-CNN", "Tiny-TCN"],
            "excluded_for_this_run": ["ADST", "Sliding Training", "Branch-2 Context", "Branch-3"],
            "feature_channels": CHANNEL_NAMES,
        },
        "label_audit": audit,
        "splits": {
            name: {"count": int(len(values)), "start": ns_to_text(values.min()), "end": ns_to_text(values.max())}
            for name, values in splits.items()
        },
        "training_sample": train_summary,
        "validation_sample": validation_summary,
        "test_summary": test_summary,
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=as_json), "utf-8"
    )
    (args.output_dir / "report.md").write_text(render_report(result), "utf-8")
    print("[6/6] Done. Wrote metrics.json and report.md only (plus the small label ledger).", flush=True)


if __name__ == "__main__":
    main()

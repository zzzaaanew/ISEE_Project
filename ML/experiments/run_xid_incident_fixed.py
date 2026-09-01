"""Run the fixed-split XID incident experiments agreed for P0/E0/E1.

P0 freezes the raw-XID episode audit and rolling-window feasibility table.
E0 predicts XID 43 with GBDT, MLP, 1D-CNN, and Tiny-TCN.
E1 predicts any XID incident and compares binary against three-head
multi-task learning (any XID, XID 31, XID 43).  E1 adds only leakage-safe
XID history available strictly before the decision time; topology and the
unverified job-context columns are excluded.

The code intentionally reuses the existing audited episode reconstruction,
temporal censoring, parquet loader, and 60/20/20 split implementation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from torch import nn

import run_nie_xid43_reproduction as nie
import run_telemetry_24h_experiment as base


HEADS = ("any", "xid31", "xid43")
MODELS = ("gbdt", "mlp", "cnn", "tcn")
DAY_NS = 24 * 60 * 60 * 1_000_000_000
HISTORY_FEATURE_NAMES = tuple(
    f"{scope}_{feature}"
    for scope in HEADS
    for feature in ("count_7d", "count_30d", "count_lifetime", "recency_days")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--label-dir", type=Path, default=Path("outputs/telemetry_24h_fixed")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/xid_incident_fixed")
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--train-times", type=int, default=800)
    parser.add_argument("--validation-times", type=int, default=300)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-model-samples", type=int, default=120_000)
    parser.add_argument("--p0-only", action="store_true")
    return parser.parse_args()


def robust_summary_features(x: np.ndarray) -> np.ndarray:
    """Leakage-safe summaries of the supplied six-bucket history."""
    levels = x[..., :4].astype(np.float32, copy=False)
    deltas = x[..., 4:8].astype(np.float32, copy=False)
    obs = np.nan_to_num(x[..., 8:12], nan=0.0).astype(np.float32, copy=False)
    missing = np.nan_to_num(x[..., 12:16], nan=1.0).astype(np.float32, copy=False)
    finite = np.isfinite(levels)
    counts = finite.sum(axis=1)
    mean = np.nansum(levels, axis=1) / np.maximum(counts, 1)
    filled = np.where(finite, levels, mean[:, None, :])
    filled = np.where((counts == 0)[:, None, :], 0.0, filled)
    # Some GPU/metric histories are entirely absent.  Their missing-mask
    # channels retain that information; suppress NumPy's expected warning
    # while assigning the neutral numeric fill used by the summary model.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nan_to_num(np.nanmedian(levels, axis=1), nan=0.0)
    mad = np.median(np.abs(filled - median[:, None, :]), axis=1)
    robust_scale = np.maximum(1.4826 * mad, 1e-3)
    delta_filled = np.nan_to_num(deltas, nan=0.0)
    pieces = (
        filled[:, -1, :],
        mean,
        filled.min(axis=1),
        filled.max(axis=1),
        filled.std(axis=1),
        filled.max(axis=1) - filled.min(axis=1),
        (filled[:, -1, :] - filled[:, 0, :]) / max(1, x.shape[1] - 1),
        delta_filled.mean(axis=1),
        np.abs(delta_filled).max(axis=1),
        (filled[:, -1, :] - median) / robust_scale,
        obs.mean(axis=1),
        missing.mean(axis=1),
    )
    return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)


def multitask_label_state(
    times: np.ndarray, ledger: pd.DataFrame, gpu_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return [any, XID31, XID43] labels with common any-XID censoring."""
    any_label, eligible = base.label_state(times, ledger, gpu_ids)
    labels = np.zeros((len(times), len(gpu_ids), len(HEADS)), dtype=bool)
    labels[..., 0] = any_label
    gpu_to_index = {gpu: index for index, gpu in enumerate(gpu_ids)}
    certain = ledger.loc[~ledger["uncertain_onset"]]
    for code, head_index in ((31, 1), (43, 2)):
        selected = certain.loc[certain["xid_code"] == code]
        for gpu_id, group in selected.groupby("gpu_id", sort=False):
            starts = np.sort(group["onset_ns"].to_numpy(dtype=np.int64))
            labels[:, gpu_to_index[gpu_id], head_index] = base.future_event(times, starts)
    return labels, eligible


def choose_multitask_training_samples(
    times: np.ndarray,
    ledger: pd.DataFrame,
    gpu_ids: list[str],
    negative_ratio: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    labels, eligible = multitask_label_state(times, ledger, gpu_ids)
    any_label = labels[..., 0]
    pos_t, pos_g = np.where(any_label & eligible)
    neg_t, neg_g = np.where((~any_label) & eligible)
    wanted_negative = min(len(neg_t), len(pos_t) * negative_ratio)
    selected_negative = rng.choice(len(neg_t), wanted_negative, replace=False)
    row_t = np.concatenate((pos_t, neg_t[selected_negative]))
    row_g = np.concatenate((pos_g, neg_g[selected_negative]))
    y = labels[row_t, row_g].astype(np.int8)
    order = rng.permutation(len(row_t))
    return (
        times[row_t[order]],
        row_g[order],
        y[order],
        {
            "candidate_times": int(len(times)),
            "positive_any": int(y[:, 0].sum()),
            "positive_xid31": int(y[:, 1].sum()),
            "positive_xid43": int(y[:, 2].sum()),
            "negative_samples": int(wanted_negative),
            "censored_samples": int((~eligible).sum()),
        },
    )


def choose_multitask_population_samples(
    times: np.ndarray, ledger: pd.DataFrame, gpu_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    labels, eligible = multitask_label_state(times, ledger, gpu_ids)
    row_t, row_g = np.where(eligible)
    y = labels[row_t, row_g].astype(np.int8)
    return (
        times[row_t],
        row_g,
        y,
        {
            "candidate_times": int(len(times)),
            "samples": int(len(row_t)),
            "positive_any": int(y[:, 0].sum()),
            "positive_xid31": int(y[:, 1].sum()),
            "positive_xid43": int(y[:, 2].sum()),
            "censored_samples": int((~eligible).sum()),
        },
    )


def history_index(ledger: pd.DataFrame, gpu_ids: list[str]) -> dict[str, list[np.ndarray]]:
    certain = ledger.loc[~ledger["uncertain_onset"]]
    result: dict[str, list[np.ndarray]] = {}
    for scope in HEADS:
        if scope == "any":
            selected = certain
        else:
            selected = certain.loc[certain["xid_code"] == int(scope[3:])]
        by_gpu = {
            gpu: np.sort(group["onset_ns"].to_numpy(dtype=np.int64))
            for gpu, group in selected.groupby("gpu_id", sort=False)
        }
        result[scope] = [by_gpu.get(gpu, np.empty(0, dtype=np.int64)) for gpu in gpu_ids]
    return result


def history_features(
    sample_times: np.ndarray,
    gpu_indices: np.ndarray,
    index: dict[str, list[np.ndarray]],
) -> np.ndarray:
    result = np.zeros((len(sample_times), len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
    for gpu_index in np.unique(gpu_indices):
        rows = np.flatnonzero(gpu_indices == gpu_index)
        t = sample_times[rows]
        for scope_index, scope in enumerate(HEADS):
            starts = index[scope][int(gpu_index)]
            right = np.searchsorted(starts, t, side="left")
            left7 = np.searchsorted(starts, t - 7 * DAY_NS, side="left")
            left30 = np.searchsorted(starts, t - 30 * DAY_NS, side="left")
            recency = np.full(len(rows), 90.0, dtype=np.float64)
            seen = right > 0
            if seen.any():
                recency[seen] = (t[seen] - starts[right[seen] - 1]) / DAY_NS
            offset = scope_index * 4
            result[rows, offset] = np.log1p(right - left7)
            result[rows, offset + 1] = np.log1p(right - left30)
            result[rows, offset + 2] = np.log1p(right)
            result[rows, offset + 3] = np.minimum(recency, 90.0)
    return result


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    scale = np.nanstd(x, axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean, scale


def apply_standardizer(x: np.ndarray, state: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, scale = state
    return ((np.nan_to_num(x, nan=0.0) - mean) / scale).astype(np.float32, copy=False)


class MLP(nn.Module):
    def __init__(self, channels: int, steps: int, context_dim: int, outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels * steps + context_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, outputs),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x.flatten(1), context), dim=1))


class CNN(nn.Module):
    def __init__(self, channels: int, context_dim: int, outputs: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16 + context_dim, outputs)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoded = self.features(x).squeeze(-1)
        return self.head(torch.cat((encoded, context), dim=1))


class TCN(nn.Module):
    def __init__(self, channels: int, context_dim: int, outputs: int):
        super().__init__()
        self.block1 = base.CausalBlock(channels, 32, dilation=1)
        self.block2 = base.CausalBlock(32, 32, dilation=2)
        self.head = nn.Linear(32 + context_dim, outputs)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoded = self.block2(self.block1(x))[:, :, -1]
        return self.head(torch.cat((encoded, context), dim=1))


def network_probabilities(
    model: nn.Module, x: np.ndarray, context: np.ndarray, batch_size: int
) -> np.ndarray:
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = np.ascontiguousarray(x[start : start + batch_size].transpose(0, 2, 1))
            cb = np.ascontiguousarray(context[start : start + batch_size])
            logits = model(torch.from_numpy(xb), torch.from_numpy(cb)).cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-logits)))
    answer = np.concatenate(scores).astype(np.float32, copy=False)
    return answer[:, None] if answer.ndim == 1 else answer


def validation_subset(y: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(y) <= limit:
        return np.arange(len(y))
    positive = np.flatnonzero(y[:, 0] > 0)
    negative = np.flatnonzero(y[:, 0] == 0)
    wanted_negative = max(0, limit - len(positive))
    picked_negative = rng.choice(negative, min(wanted_negative, len(negative)), replace=False)
    return np.sort(np.concatenate((positive, picked_negative)))


def train_network(
    name: str,
    model: nn.Module,
    x_train: np.ndarray,
    context_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    context_validation: np.ndarray,
    y_validation: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> nn.Module:
    rng = np.random.default_rng(seed)
    positives = np.maximum(y_train.sum(axis=0), 1)
    negatives = np.maximum(len(y_train) - positives, 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy((negatives / positives).astype(np.float32))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_metric, best_state, stale = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            xb = np.ascontiguousarray(x_train[rows].transpose(0, 2, 1))
            cb = np.ascontiguousarray(context_train[rows])
            yb = np.ascontiguousarray(y_train[rows].astype(np.float32, copy=False))
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(xb), torch.from_numpy(cb))
            loss = loss_fn(logits, torch.from_numpy(yb))
            loss.backward()
            optimizer.step()
        score = network_probabilities(model, x_validation, context_validation, batch_size)
        aps = []
        for head in range(y_validation.shape[1]):
            aps.append(average_precision_score(y_validation[:, head], score[:, head]))
        selection_metric = aps[0] if len(aps) == 1 else aps[0] + 0.25 * float(np.mean(aps[1:]))
        print(
            f"  {name} epoch {epoch}: validation AP="
            + ", ".join(f"{value:.6f}" for value in aps),
            flush=True,
        )
        if selection_metric > best_metric + 1e-8:
            best_metric = selection_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a model state")
    model.load_state_dict(best_state)
    return model


def balanced_weights(y: np.ndarray) -> np.ndarray:
    positive = max(1, int(y.sum()))
    negative = max(1, len(y) - positive)
    return np.where(y > 0, len(y) / (2 * positive), len(y) / (2 * negative)).astype(
        np.float32
    )


def fit_gbdt(x: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=balanced_weights(y))
    return model


def fit_gbdt_heads(x: np.ndarray, y: np.ndarray, seed: int) -> list[HistGradientBoostingClassifier]:
    return [fit_gbdt(x, y[:, head], seed + head) for head in range(y.shape[1])]


def gbdt_probabilities(models: list[HistGradientBoostingClassifier], x: np.ndarray) -> np.ndarray:
    return np.stack([model.predict_proba(x)[:, 1] for model in models], axis=1).astype(
        np.float32
    )


def fit_calibrators(score: np.ndarray, y: np.ndarray) -> list[base.PlattScaler]:
    return [base.PlattScaler().fit(score[:, head], y[:, head]) for head in range(y.shape[1])]


def calibrate(score: np.ndarray, calibrators: list[base.PlattScaler]) -> np.ndarray:
    return np.stack(
        [calibrators[head].predict(score[:, head]) for head in range(score.shape[1])], axis=1
    ).astype(np.float32)


def score_metrics(score: np.ndarray, labels: np.ndarray, eligible: np.ndarray) -> dict:
    y = labels[eligible].astype(np.int8)
    s = score[eligible]
    prevalence = float(y.mean())
    result = {
        "pr_auc": float(average_precision_score(y, s)),
        "prevalence": prevalence,
        "normalized_pr_auc": float(average_precision_score(y, s) / prevalence),
        "brier": float(brier_score_loss(y, s)),
        "ece": base.ece(s, y),
        "samples": int(len(y)),
        "positives": int(y.sum()),
    }
    for fraction in (0.02, 0.05):
        precisions, recalls = [], []
        pooled_tp = pooled_alerts = pooled_positive = 0
        for row in range(labels.shape[0]):
            mask = eligible[row]
            yr = labels[row, mask]
            sr = score[row, mask]
            k = max(1, int(math.ceil(len(sr) * fraction)))
            picked = np.argpartition(sr, -k)[-k:]
            tp = int(yr[picked].sum())
            pooled_tp += tp
            pooled_alerts += k
            pooled_positive += int(yr.sum())
            if yr.any():
                precisions.append(tp / k)
                recalls.append(tp / int(yr.sum()))
        key = int(fraction * 100)
        result[f"precision_at_{key}pct_pooled"] = pooled_tp / pooled_alerts
        result[f"recall_at_{key}pct_pooled"] = pooled_tp / max(1, pooled_positive)
        result[f"precision_at_{key}pct_active_epoch_macro"] = float(np.mean(precisions))
        result[f"recall_at_{key}pct_active_epoch_macro"] = float(np.mean(recalls))
    return result


def strict_parallel_metrics(
    score_by_model: dict[str, np.ndarray], labels: np.ndarray, eligible: np.ndarray
) -> dict:
    result = {}
    for fraction in (0.02, 0.05):
        tp_total = alerts_total = positives_total = empty = 0
        active_precision, active_recall = [], []
        for row in range(labels.shape[0]):
            mask = eligible[row]
            y = labels[row, mask]
            k = max(1, int(math.ceil(mask.sum() * fraction)))
            intersection = np.ones(mask.sum(), dtype=bool)
            for score in score_by_model.values():
                s = score[row, mask]
                picked = np.argpartition(s, -k)[-k:]
                top = np.zeros(len(s), dtype=bool)
                top[picked] = True
                intersection &= top
            chosen = np.flatnonzero(intersection)
            tp = int(y[chosen].sum()) if len(chosen) else 0
            tp_total += tp
            alerts_total += len(chosen)
            positives_total += int(y.sum())
            empty += int(len(chosen) == 0)
            if y.any():
                active_precision.append(tp / len(chosen) if len(chosen) else 0.0)
                active_recall.append(tp / int(y.sum()))
        key = int(fraction * 100)
        result[f"precision_at_{key}pct_pooled"] = tp_total / max(1, alerts_total)
        result[f"recall_at_{key}pct_pooled"] = tp_total / max(1, positives_total)
        result[f"alert_rate_at_{key}pct"] = alerts_total / max(1, int(eligible.sum()))
        result[f"empty_epoch_rate_at_{key}pct"] = empty / labels.shape[0]
        result[f"precision_at_{key}pct_active_epoch_macro"] = float(np.mean(active_precision))
        result[f"recall_at_{key}pct_active_epoch_macro"] = float(np.mean(active_recall))
    return result


def p0_audit(ledger: pd.DataFrame, audit: dict) -> dict:
    certain = ledger.loc[~ledger["uncertain_onset"]].sort_values(["gpu_id", "onset_ns"]).copy()
    certain["previous_onset"] = certain.groupby("gpu_id")["onset_ns"].shift()
    certain["previous_code"] = certain.groupby("gpu_id")["xid_code"].shift()
    certain["gap_seconds"] = (certain["onset_ns"] - certain["previous_onset"]) / 1e9
    pairs = certain.loc[certain["previous_onset"].notna()].copy()
    different = pairs["xid_code"] != pairs["previous_code"]
    gap_counts = {}
    for seconds in (300, 900, 1800, 3600, 21600, 86400):
        within = pairs["gap_seconds"] <= seconds
        gap_counts[str(seconds)] = {
            "all_pairs": int(within.sum()),
            "different_code_pairs": int((within & different).sum()),
        }
    onset = pd.to_datetime(certain["onset_ns"], unit="ns", utc=True)
    retrain = pd.date_range(onset.min() + pd.Timedelta(days=16), onset.max() - pd.Timedelta(days=1), freq="3D")
    rolling = {}
    for days in (9, 12, 15):
        rows = []
        for decision in retrain:
            cutoff = decision - pd.Timedelta(hours=36)
            selected = certain.loc[
                (onset >= cutoff - pd.Timedelta(days=days)) & (onset < cutoff)
            ]
            rows.append(
                {
                    "all": int(len(selected)),
                    "xid31": int((selected["xid_code"] == 31).sum()),
                    "xid43": int((selected["xid_code"] == 43).sum()),
                    "unique_gpus": int(selected["gpu_id"].nunique()),
                }
            )
        rolling[str(days)] = {
            key: {
                "min": int(min(row[key] for row in rows)),
                "median": float(np.median([row[key] for row in rows])),
                "max": int(max(row[key] for row in rows)),
                "windows_below_20": int(sum(row[key] < 20 for row in rows)),
            }
            for key in ("all", "xid31", "xid43", "unique_gpus")
        }
    return {
        "label_audit": audit,
        "certain_counts": {
            str(code): int(count)
            for code, count in certain["xid_code"].value_counts().sort_index().items()
        },
        "cross_code_gap_counts": gap_counts,
        "rolling_training_feasibility": rolling,
        "decision": {
            "same_code_episode_merge_seconds": 30,
            "cross_code_default_minutes": 30,
            "cross_code_sensitivity_minutes": [15, 30, 60],
            "primary_target": "any certain XID onset in (t, t+24h]",
            "multi_task_heads": list(HEADS),
            "rare_codes": "included in any head only",
        },
    }


def render_report(result: dict) -> str:
    rows = []
    for stage in ("e0_xid43", "e1_binary", "e1_multitask_any"):
        for model, metric in result["metrics"][stage].items():
            rows.append(
                f"| {stage} | {model} | {metric['pr_auc']:.6f} | "
                f"{metric['normalized_pr_auc']:.2f} | "
                f"{metric['precision_at_2pct_pooled']:.4f} | "
                f"{metric['recall_at_2pct_pooled']:.4f} | "
                f"{metric['recall_at_5pct_pooled']:.4f} |"
            )
    parallel_rows = []
    for name, metric in result["parallel"].items():
        parallel_rows.append(
            f"| {name} | {metric['precision_at_2pct_pooled']:.4f} | "
            f"{metric['recall_at_2pct_pooled']:.4f} | "
            f"{metric['precision_at_5pct_pooled']:.4f} | "
            f"{metric['recall_at_5pct_pooled']:.4f} |"
        )
    return f"""# XID Incident Fixed-Split Experiment

## Protocol

- Horizon: `(t, t+24h]`
- Input: `[t-40m, t-10m]`, six 5-minute buckets
- E0 target: XID 43
- E1 binary target: any certain XID incident
- E1 multi-task heads: any, XID 31, XID 43
- Models: GBDT, MLP, 1D-CNN, Tiny-TCN
- Split: chronological 60/20/20 with 36-hour purge
- Features: telemetry plus leakage-safe past-XID history in E1
- Excluded: GPU topology and unverified job-context columns

## Individual models

| Stage | Model | PR-AUC | AP/prevalence | Precision@2% | Recall@2% | Recall@5% |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Strict Liu-style Parallel Ensemble

| Ensemble | Precision@2% | Recall@2% | Precision@5% | Recall@5% |
|---|---:|---:|---:|---:|
{chr(10).join(parallel_rows)}

PR-AUC is the primary metric. Strict Parallel is the intersection of each
component model's top-K set and is therefore reported with operating-point
precision/recall rather than being treated as a continuous probability model.
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(6, (os.cpu_count() or 2) - 1)))
    rng = np.random.default_rng(args.seed)

    xid_path = args.data_dir / "XID_ERRORS-002.csv"
    util_path = args.data_dir / "telemetry_5m_util.parquet"
    gpu_ids = base.read_gpu_ids(xid_path)
    ledger, audit = base.build_episode_ledger(
        xid_path, gpu_ids, args.label_dir / "xid_episode_ledger.parquet", False
    )
    p0 = p0_audit(ledger, audit)
    (args.output_dir / "p0_audit.json").write_text(
        json.dumps(p0, ensure_ascii=False, indent=2, default=base.as_json), "utf-8"
    )
    print(f"[P0] Wrote audit: {args.output_dir / 'p0_audit.json'}", flush=True)
    if args.p0_only:
        return

    grid = base.read_time_grid(util_path)
    decision_times = base.valid_decision_times(grid, audit)
    splits = base.chronological_split(decision_times)
    train_times = np.sort(
        rng.choice(splits["train"], min(args.train_times, len(splits["train"])), replace=False)
    )
    validation_times = np.sort(
        rng.choice(
            splits["validation"],
            min(args.validation_times, len(splits["validation"])),
            replace=False,
        )
    )
    history = history_index(ledger, gpu_ids)

    # E0: XID 43 only.
    print("[E0] Materializing XID43 train/validation data...", flush=True)
    xid43_ledger = nie.target_ledger(ledger)
    e0_t, e0_g, e0_y_flat, e0_train_stats = base.choose_training_samples(
        train_times, xid43_ledger, gpu_ids, args.negative_ratio, rng
    )
    e0_vt, e0_vg, e0_vy_flat, e0_val_stats = base.choose_population_samples(
        validation_times, xid43_ledger, gpu_ids
    )
    e0_y = e0_y_flat[:, None]
    e0_vy = e0_vy_flat[:, None]
    e0_x_raw = base.materialize_sequences(args.data_dir, e0_t, e0_g, gpu_ids, "E0 train")
    e0_vx_raw = base.materialize_sequences(args.data_dir, e0_vt, e0_vg, gpu_ids, "E0 validation")
    e0_normalizer = base.fit_normalizer(e0_x_raw)
    e0_x = base.normalize(e0_x_raw, *e0_normalizer)
    e0_vx = base.normalize(e0_vx_raw, *e0_normalizer)
    empty_e0 = np.empty((len(e0_x), 0), dtype=np.float32)
    empty_e0_val = np.empty((len(e0_vx), 0), dtype=np.float32)
    e0_val_rows = validation_subset(e0_vy, args.validation_model_samples, rng)
    e0_gbdt = fit_gbdt_heads(robust_summary_features(e0_x_raw), e0_y, args.seed + 10)
    e0_networks = {
        "mlp": MLP(len(base.CHANNEL_NAMES), len(base.FEATURE_OFFSETS), 0, 1),
        "cnn": CNN(len(base.CHANNEL_NAMES), 0, 1),
        "tcn": TCN(len(base.CHANNEL_NAMES), 0, 1),
    }
    for offset, (name, model) in enumerate(e0_networks.items(), start=1):
        e0_networks[name] = train_network(
            f"E0 {name}", model, e0_x, empty_e0, e0_y,
            e0_vx[e0_val_rows], empty_e0_val[e0_val_rows], e0_vy[e0_val_rows],
            args.epochs, args.batch_size, args.seed + 10 + offset,
        )
    e0_val_raw = {
        "gbdt": gbdt_probabilities(e0_gbdt, robust_summary_features(e0_vx_raw)),
        **{
            name: network_probabilities(model, e0_vx, empty_e0_val, args.batch_size)
            for name, model in e0_networks.items()
        },
    }
    e0_calibrators = {
        name: fit_calibrators(score, e0_vy) for name, score in e0_val_raw.items()
    }
    del e0_x_raw, e0_vx_raw, e0_x, e0_vx, empty_e0, empty_e0_val, e0_val_raw
    gc.collect()

    # E1: any-XID binary and multi-task, both with the same telemetry/history input.
    print("[E1] Materializing any-XID train/validation data...", flush=True)
    e1_t, e1_g, e1_y, e1_train_stats = choose_multitask_training_samples(
        train_times, ledger, gpu_ids, args.negative_ratio, rng
    )
    e1_vt, e1_vg, e1_vy, e1_val_stats = choose_multitask_population_samples(
        validation_times, ledger, gpu_ids
    )
    e1_x_raw = base.materialize_sequences(args.data_dir, e1_t, e1_g, gpu_ids, "E1 train")
    e1_vx_raw = base.materialize_sequences(args.data_dir, e1_vt, e1_vg, gpu_ids, "E1 validation")
    e1_h_raw = history_features(e1_t, e1_g, history)
    e1_vh_raw = history_features(e1_vt, e1_vg, history)
    e1_normalizer = base.fit_normalizer(e1_x_raw)
    e1_history_standardizer = fit_standardizer(e1_h_raw)
    e1_x = base.normalize(e1_x_raw, *e1_normalizer)
    e1_vx = base.normalize(e1_vx_raw, *e1_normalizer)
    e1_h = apply_standardizer(e1_h_raw, e1_history_standardizer)
    e1_vh = apply_standardizer(e1_vh_raw, e1_history_standardizer)
    e1_train_table = np.concatenate((robust_summary_features(e1_x_raw), e1_h_raw), axis=1)
    e1_val_table = np.concatenate((robust_summary_features(e1_vx_raw), e1_vh_raw), axis=1)
    e1_val_rows = validation_subset(e1_vy, args.validation_model_samples, rng)

    e1_binary_gbdt = fit_gbdt_heads(e1_train_table, e1_y[:, :1], args.seed + 20)
    e1_multi_gbdt = fit_gbdt_heads(e1_train_table, e1_y, args.seed + 30)
    e1_binary_networks = {
        "mlp": MLP(len(base.CHANNEL_NAMES), len(base.FEATURE_OFFSETS), e1_h.shape[1], 1),
        "cnn": CNN(len(base.CHANNEL_NAMES), e1_h.shape[1], 1),
        "tcn": TCN(len(base.CHANNEL_NAMES), e1_h.shape[1], 1),
    }
    e1_multi_networks = {
        "mlp": MLP(len(base.CHANNEL_NAMES), len(base.FEATURE_OFFSETS), e1_h.shape[1], 3),
        "cnn": CNN(len(base.CHANNEL_NAMES), e1_h.shape[1], 3),
        "tcn": TCN(len(base.CHANNEL_NAMES), e1_h.shape[1], 3),
    }
    for offset, (name, model) in enumerate(e1_binary_networks.items(), start=1):
        e1_binary_networks[name] = train_network(
            f"E1 binary {name}", model, e1_x, e1_h, e1_y[:, :1],
            e1_vx[e1_val_rows], e1_vh[e1_val_rows], e1_vy[e1_val_rows, :1],
            args.epochs, args.batch_size, args.seed + 40 + offset,
        )
    for offset, (name, model) in enumerate(e1_multi_networks.items(), start=1):
        e1_multi_networks[name] = train_network(
            f"E1 multitask {name}", model, e1_x, e1_h, e1_y,
            e1_vx[e1_val_rows], e1_vh[e1_val_rows], e1_vy[e1_val_rows],
            args.epochs, args.batch_size, args.seed + 50 + offset,
        )
    e1_binary_val_raw = {
        "gbdt": gbdt_probabilities(e1_binary_gbdt, e1_val_table),
        **{
            name: network_probabilities(model, e1_vx, e1_vh, args.batch_size)
            for name, model in e1_binary_networks.items()
        },
    }
    e1_multi_val_raw = {
        "gbdt": gbdt_probabilities(e1_multi_gbdt, e1_val_table),
        **{
            name: network_probabilities(model, e1_vx, e1_vh, args.batch_size)
            for name, model in e1_multi_networks.items()
        },
    }
    e1_binary_calibrators = {
        name: fit_calibrators(score, e1_vy[:, :1])
        for name, score in e1_binary_val_raw.items()
    }
    e1_multi_calibrators = {
        name: fit_calibrators(score, e1_vy) for name, score in e1_multi_val_raw.items()
    }
    del (
        e1_x_raw, e1_vx_raw, e1_h_raw, e1_vh_raw, e1_x, e1_vx, e1_h, e1_vh,
        e1_train_table, e1_val_table, e1_binary_val_raw, e1_multi_val_raw,
    )
    gc.collect()

    print("[E0/E1] Evaluating the full chronological test period...", flush=True)
    e0_labels, e0_eligible = nie.label_state_for_code(
        splits["test"], xid43_ledger, gpu_ids
    )
    e1_labels, e1_eligible = multitask_label_state(splits["test"], ledger, gpu_ids)
    e0_scores = {
        name: np.full(e0_labels.shape, np.nan, dtype=np.float32) for name in MODELS
    }
    e1_binary_scores = {
        name: np.full(e1_labels.shape[:2], np.nan, dtype=np.float32) for name in MODELS
    }
    e1_multi_scores = {
        name: np.full(e1_labels.shape, np.nan, dtype=np.float32) for name in MODELS
    }
    keys = base.half_day_keys(splits["test"])
    unique_keys = np.unique(keys)
    for completed, key in enumerate(unique_keys, start=1):
        time_rows = np.flatnonzero(keys == key)
        union = e0_eligible[time_rows] | e1_eligible[time_rows]
        local_t, gpu_rows = np.where(union)
        sample_times = splits["test"][time_rows[local_t]]
        x_raw = base.materialize_sequences(
            args.data_dir, sample_times, gpu_rows, gpu_ids, f"fixed test {key}"
        )
        empty_context = np.empty((len(x_raw), 0), dtype=np.float32)
        e0_x_block = base.normalize(x_raw, *e0_normalizer)
        e0_raw = {
            "gbdt": gbdt_probabilities(e0_gbdt, robust_summary_features(x_raw)),
            **{
                name: network_probabilities(model, e0_x_block, empty_context, args.batch_size)
                for name, model in e0_networks.items()
            },
        }
        h_raw = history_features(sample_times, gpu_rows, history)
        h = apply_standardizer(h_raw, e1_history_standardizer)
        e1_x_block = base.normalize(x_raw, *e1_normalizer)
        table = np.concatenate((robust_summary_features(x_raw), h_raw), axis=1)
        binary_raw = {
            "gbdt": gbdt_probabilities(e1_binary_gbdt, table),
            **{
                name: network_probabilities(model, e1_x_block, h, args.batch_size)
                for name, model in e1_binary_networks.items()
            },
        }
        multi_raw = {
            "gbdt": gbdt_probabilities(e1_multi_gbdt, table),
            **{
                name: network_probabilities(model, e1_x_block, h, args.batch_size)
                for name, model in e1_multi_networks.items()
            },
        }
        for name in MODELS:
            e0_value = calibrate(e0_raw[name], e0_calibrators[name])[:, 0]
            binary_value = calibrate(binary_raw[name], e1_binary_calibrators[name])[:, 0]
            multi_value = calibrate(multi_raw[name], e1_multi_calibrators[name])
            e0_scores[name][time_rows[local_t], gpu_rows] = e0_value
            e1_binary_scores[name][time_rows[local_t], gpu_rows] = binary_value
            e1_multi_scores[name][time_rows[local_t], gpu_rows] = multi_value
        print(f"  full test prediction: {completed}/{len(unique_keys)} blocks", flush=True)
        del x_raw, empty_context, e0_x_block, e0_raw, h_raw, h, e1_x_block, table, binary_raw, multi_raw
        gc.collect()

    metrics = {
        "e0_xid43": {
            name: score_metrics(score, e0_labels, e0_eligible)
            for name, score in e0_scores.items()
        },
        "e1_binary": {
            name: score_metrics(score, e1_labels[..., 0], e1_eligible)
            for name, score in e1_binary_scores.items()
        },
        "e1_multitask_any": {
            name: score_metrics(score[..., 0], e1_labels[..., 0], e1_eligible)
            for name, score in e1_multi_scores.items()
        },
        "e1_multitask_heads": {
            name: {
                head: score_metrics(score[..., index], e1_labels[..., index], e1_eligible)
                for index, head in enumerate(HEADS)
            }
            for name, score in e1_multi_scores.items()
        },
    }
    parallel = {}
    for branch, source in (("binary", e1_binary_scores), ("multitask", {k: v[..., 0] for k, v in e1_multi_scores.items()})):
        parallel[f"{branch}_parallel3"] = strict_parallel_metrics(
            {name: source[name] for name in ("gbdt", "mlp", "cnn")},
            e1_labels[..., 0], e1_eligible,
        )
        parallel[f"{branch}_parallel4"] = strict_parallel_metrics(
            source, e1_labels[..., 0], e1_eligible
        )

    result = {
        "experiment": {
            "seed": args.seed,
            "horizon": "(t, t+24h]",
            "input_window": "[t-40m, t-10m] / six 5-minute buckets",
            "models": list(MODELS),
            "multi_task_heads": list(HEADS),
            "history_features": list(HISTORY_FEATURE_NAMES),
            "excluded": ["GPU topology", "unverified job context"],
        },
        "splits": {
            name: {
                "count": int(len(value)),
                "start": base.ns_to_text(value.min()),
                "end": base.ns_to_text(value.max()),
            }
            for name, value in splits.items()
        },
        "samples": {
            "e0_train": e0_train_stats,
            "e0_validation": e0_val_stats,
            "e1_train": e1_train_stats,
            "e1_validation": e1_val_stats,
        },
        "metrics": metrics,
        "parallel": parallel,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=base.as_json), "utf-8"
    )
    (args.output_dir / "report.md").write_text(render_report(result), "utf-8")
    print(f"[DONE] Wrote {args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

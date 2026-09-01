"""Run leakage-safe Liu-style sliding training for the XID incident task.

The fixed chronological test period is divided into consecutive three-day
windows.  Before each window, 9/12/15-day histories are tried.  Each history
is split chronologically into training and validation portions with a
36-hour purge.  The best history length is selected separately for every
model/branch using validation PR-AUC only; the current test window is never
used for model or history-length selection.

Each completed window is checkpointed as JSON plus compressed predictions,
so this CPU-only experiment can resume without repeating completed windows.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss

import run_telemetry_24h_experiment as base
import run_xid_incident_fixed as fixed


PURGE_NS = 36 * 60 * 60 * 1_000_000_000
WINDOW_NS = 3 * fixed.DAY_NS
LENGTHS = (9, 12, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--label-dir", type=Path, default=Path("outputs/telemetry_24h_fixed")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/xid_incident_sliding")
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--train-times", type=int, default=300)
    parser.add_argument("--validation-times", type=int, default=120)
    parser.add_argument("--validation-model-samples", type=int, default=80_000)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_times(pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(pool) == 0:
        raise ValueError("A rolling time pool is empty")
    if len(pool) <= count:
        return pool.copy()
    return np.sort(rng.choice(pool, count, replace=False))


def candidate_time_pools(
    decision_times: np.ndarray, test_start: int, days: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    history_end = test_start - PURGE_NS
    validation_span = 3 * fixed.DAY_NS
    validation_start = history_end - validation_span
    train_end = validation_start - PURGE_NS
    history_start = train_end - days * fixed.DAY_NS
    train_pool = decision_times[
        (decision_times >= history_start) & (decision_times <= train_end)
    ]
    validation_pool = decision_times[
        (decision_times >= validation_start) & (decision_times <= history_end)
    ]
    return train_pool, validation_pool, {
        "history_days": days,
        "training_start": base.ns_to_text(history_start),
        "train_end": base.ns_to_text(train_end),
        "validation_start": base.ns_to_text(validation_start),
        "history_end": base.ns_to_text(history_end),
        "train_pool_times": int(len(train_pool)),
        "validation_pool_times": int(len(validation_pool)),
    }


def build_network(model_name: str, context_dim: int, outputs: int):
    channels = len(base.CHANNEL_NAMES)
    if model_name == "mlp":
        return fixed.MLP(channels, len(base.FEATURE_OFFSETS), context_dim, outputs)
    if model_name == "cnn":
        return fixed.CNN(channels, context_dim, outputs)
    if model_name == "tcn":
        return fixed.TCN(channels, context_dim, outputs)
    raise KeyError(model_name)


def fit_candidate(
    args: argparse.Namespace,
    days: int,
    train_times: np.ndarray,
    validation_times: np.ndarray,
    ledger,
    gpu_ids: list[str],
    history,
    seed: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    train_t, train_g, y_train, train_stats = fixed.choose_multitask_training_samples(
        train_times, ledger, gpu_ids, args.negative_ratio, rng
    )
    val_t, val_g, y_val, val_stats = fixed.choose_multitask_population_samples(
        validation_times, ledger, gpu_ids
    )
    print(f"    L={days}: materializing train and validation sequences", flush=True)
    x_train_raw = base.materialize_sequences(
        args.data_dir, train_t, train_g, gpu_ids, f"sliding L{days} train"
    )
    x_val_raw = base.materialize_sequences(
        args.data_dir, val_t, val_g, gpu_ids, f"sliding L{days} validation"
    )
    h_train_raw = fixed.history_features(train_t, train_g, history)
    h_val_raw = fixed.history_features(val_t, val_g, history)
    normalizer = base.fit_normalizer(x_train_raw)
    history_standardizer = fixed.fit_standardizer(h_train_raw)
    x_train = base.normalize(x_train_raw, *normalizer)
    x_val = base.normalize(x_val_raw, *normalizer)
    h_train = fixed.apply_standardizer(h_train_raw, history_standardizer)
    h_val = fixed.apply_standardizer(h_val_raw, history_standardizer)
    table_train = np.concatenate(
        (fixed.robust_summary_features(x_train_raw), h_train_raw), axis=1
    )
    table_val = np.concatenate(
        (fixed.robust_summary_features(x_val_raw), h_val_raw), axis=1
    )
    early_rows = fixed.validation_subset(y_val, args.validation_model_samples, rng)

    bundles: dict[str, dict[str, dict]] = {"binary": {}, "multitask": {}}
    candidate_metrics: dict[str, dict[str, dict]] = {"binary": {}, "multitask": {}}

    binary_gbdt = fixed.fit_gbdt_heads(table_train, y_train[:, :1], seed + 1)
    multi_gbdt = fixed.fit_gbdt_heads(table_train, y_train, seed + 10)
    for branch, models, target in (
        ("binary", binary_gbdt, y_val[:, :1]),
        ("multitask", multi_gbdt, y_val),
    ):
        raw = fixed.gbdt_probabilities(models, table_val)
        calibrators = fixed.fit_calibrators(raw, target)
        calibrated = fixed.calibrate(raw, calibrators)
        aps = [
            float(average_precision_score(target[:, head], calibrated[:, head]))
            for head in range(target.shape[1])
        ]
        bundles[branch]["gbdt"] = {
            "kind": "gbdt",
            "model": models,
            "normalizer": normalizer,
            "history_standardizer": history_standardizer,
            "calibrators": calibrators,
            "history_days": days,
            "validation_ap": aps[0],
        }
        candidate_metrics[branch]["gbdt"] = {"head_ap": aps, "selection_ap": aps[0]}

    for model_offset, model_name in enumerate(("mlp", "cnn", "tcn"), start=1):
        for branch, outputs, target, branch_offset in (
            ("binary", 1, y_train[:, :1], 0),
            ("multitask", 3, y_train, 100),
        ):
            model = build_network(model_name, h_train.shape[1], outputs)
            model = fixed.train_network(
                f"sliding L{days} {branch} {model_name}",
                model,
                x_train,
                h_train,
                target,
                x_val[early_rows],
                h_val[early_rows],
                (y_val[:, :outputs])[early_rows],
                args.epochs,
                args.batch_size,
                seed + branch_offset + 20 + model_offset,
            )
            raw = fixed.network_probabilities(model, x_val, h_val, args.batch_size)
            val_target = y_val[:, :outputs]
            calibrators = fixed.fit_calibrators(raw, val_target)
            calibrated = fixed.calibrate(raw, calibrators)
            aps = [
                float(average_precision_score(val_target[:, head], calibrated[:, head]))
                for head in range(outputs)
            ]
            bundles[branch][model_name] = {
                "kind": "network",
                "model": model,
                "normalizer": normalizer,
                "history_standardizer": history_standardizer,
                "calibrators": calibrators,
                "history_days": days,
                "validation_ap": aps[0],
            }
            candidate_metrics[branch][model_name] = {
                "head_ap": aps,
                "selection_ap": aps[0],
            }

    stats = {
        "history_days": days,
        "train": train_stats,
        "validation": val_stats,
        "validation_metrics": candidate_metrics,
    }
    del (
        x_train_raw,
        x_val_raw,
        h_train_raw,
        h_val_raw,
        x_train,
        x_val,
        h_train,
        h_val,
        table_train,
        table_val,
    )
    gc.collect()
    return bundles, stats


def select_bundles(candidate_bundles: dict[int, dict]) -> tuple[dict, dict]:
    selected: dict[str, dict[str, dict]] = {"binary": {}, "multitask": {}}
    selection: dict[str, dict[str, dict]] = {"binary": {}, "multitask": {}}
    for branch in selected:
        for model_name in fixed.MODELS:
            ranked = sorted(
                (
                    (
                        bundle[branch][model_name]["validation_ap"],
                        days,
                        bundle[branch][model_name],
                    )
                    for days, bundle in candidate_bundles.items()
                ),
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
            validation_ap, days, winner = ranked[0]
            selected[branch][model_name] = winner
            selection[branch][model_name] = {
                "history_days": int(days),
                "validation_ap": float(validation_ap),
                "all_candidates": {
                    str(candidate_days): float(candidate_bundles[candidate_days][branch][model_name]["validation_ap"])
                    for candidate_days in LENGTHS
                },
            }
    return selected, selection


def predict_bundle(
    bundle: dict,
    x_raw: np.ndarray,
    summary: np.ndarray,
    h_raw: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if bundle["kind"] == "gbdt":
        table = np.concatenate((summary, h_raw), axis=1)
        raw = fixed.gbdt_probabilities(bundle["model"], table)
    else:
        x = base.normalize(x_raw, *bundle["normalizer"])
        h = fixed.apply_standardizer(h_raw, bundle["history_standardizer"])
        raw = fixed.network_probabilities(bundle["model"], x, h, batch_size)
    return fixed.calibrate(raw, bundle["calibrators"])


def evaluate_window(
    args: argparse.Namespace,
    test_times: np.ndarray,
    selected: dict,
    ledger,
    gpu_ids: list[str],
    history,
) -> tuple[dict, dict, dict]:
    labels, eligible = fixed.multitask_label_state(test_times, ledger, gpu_ids)
    binary_scores = {
        name: np.full(labels.shape[:2], np.nan, dtype=np.float32) for name in fixed.MODELS
    }
    multi_scores = {
        name: np.full(labels.shape, np.nan, dtype=np.float32) for name in fixed.MODELS
    }
    keys = base.half_day_keys(test_times)
    unique_keys = np.unique(keys)
    for completed, key in enumerate(unique_keys, start=1):
        time_rows = np.flatnonzero(keys == key)
        local_t, gpu_rows = np.where(eligible[time_rows])
        sample_times = test_times[time_rows[local_t]]
        x_raw = base.materialize_sequences(
            args.data_dir, sample_times, gpu_rows, gpu_ids, f"sliding test {key}"
        )
        summary = fixed.robust_summary_features(x_raw)
        h_raw = fixed.history_features(sample_times, gpu_rows, history)
        for branch, destination in (("binary", binary_scores), ("multitask", multi_scores)):
            for model_name, bundle in selected[branch].items():
                value = predict_bundle(bundle, x_raw, summary, h_raw, args.batch_size)
                if branch == "binary":
                    destination[model_name][time_rows[local_t], gpu_rows] = value[:, 0]
                else:
                    destination[model_name][time_rows[local_t], gpu_rows] = value
        print(f"    test prediction {completed}/{len(unique_keys)} blocks", flush=True)
        del x_raw, summary, h_raw
        gc.collect()

    metrics = {
        "binary": {
            name: fixed.score_metrics(score, labels[..., 0], eligible)
            for name, score in binary_scores.items()
        },
        "multitask_any": {
            name: fixed.score_metrics(score[..., 0], labels[..., 0], eligible)
            for name, score in multi_scores.items()
        },
        "multitask_heads": {
            name: {
                head: fixed.score_metrics(score[..., index], labels[..., index], eligible)
                for index, head in enumerate(fixed.HEADS)
            }
            for name, score in multi_scores.items()
        },
    }
    parallel = {}
    for branch, source in (
        ("binary", binary_scores),
        ("multitask", {name: value[..., 0] for name, value in multi_scores.items()}),
    ):
        parallel[f"{branch}_parallel3"] = fixed.strict_parallel_metrics(
            {name: source[name] for name in ("gbdt", "mlp", "cnn")},
            labels[..., 0],
            eligible,
        )
        parallel[f"{branch}_parallel4"] = fixed.strict_parallel_metrics(
            source, labels[..., 0], eligible
        )
    flattened = {"labels": labels[eligible].astype(np.uint8)}
    for name, score in binary_scores.items():
        flattened[f"binary_{name}"] = score[eligible].astype(np.float32)
    for name, score in multi_scores.items():
        for index, head in enumerate(fixed.HEADS):
            flattened[f"multitask_{name}_{head}"] = score[..., index][eligible].astype(
                np.float32
            )
    return metrics, parallel, flattened


def flat_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, score))
    return {
        "pr_auc": ap,
        "prevalence": prevalence,
        "normalized_pr_auc": ap / prevalence,
        "brier": float(brier_score_loss(y, score)),
        "ece": base.ece(score, y),
        "samples": int(len(y)),
        "positives": int(y.sum()),
    }


def aggregate_results(output_dir: Path, window_count: int) -> dict:
    summaries = [
        json.loads((output_dir / f"window_{index:02d}.json").read_text("utf-8"))
        for index in range(window_count)
    ]
    prediction_paths = [output_dir / f"window_{index:02d}.npz" for index in range(window_count)]
    y_parts = []
    for path in prediction_paths:
        with np.load(path) as data:
            y_parts.append(data["labels"].copy())
    y = np.concatenate(y_parts, axis=0)
    pooled = {"binary": {}, "multitask_heads": {}}
    for model_name in fixed.MODELS:
        parts = []
        for path in prediction_paths:
            with np.load(path) as data:
                parts.append(data[f"binary_{model_name}"].copy())
        pooled["binary"][model_name] = flat_metrics(y[:, 0], np.concatenate(parts))
        pooled["multitask_heads"][model_name] = {}
        for head_index, head in enumerate(fixed.HEADS):
            parts = []
            for path in prediction_paths:
                with np.load(path) as data:
                    parts.append(data[f"multitask_{model_name}_{head}"].copy())
            pooled["multitask_heads"][model_name][head] = flat_metrics(
                y[:, head_index], np.concatenate(parts)
            )

    stability = {"binary": {}, "multitask_any": {}}
    for branch in stability:
        for model_name in fixed.MODELS:
            values = np.asarray(
                [summary["metrics"][branch][model_name]["pr_auc"] for summary in summaries],
                dtype=float,
            )
            stability[branch][model_name] = {
                "mean_window_pr_auc": float(values.mean()),
                "std_window_pr_auc": float(values.std()),
                "min_window_pr_auc": float(values.min()),
                "max_window_pr_auc": float(values.max()),
            }
    selection_counts = {"binary": {}, "multitask": {}}
    for branch in selection_counts:
        for model_name in fixed.MODELS:
            counts = {str(days): 0 for days in LENGTHS}
            for summary in summaries:
                days = str(summary["selection"][branch][model_name]["history_days"])
                counts[days] += 1
            selection_counts[branch][model_name] = counts

    parallel = {}
    for name in summaries[0]["parallel"]:
        parallel[name] = {}
        for metric_name in summaries[0]["parallel"][name]:
            values = [summary["parallel"][name][metric_name] for summary in summaries]
            parallel[name][f"mean_{metric_name}"] = float(np.mean(values))
            parallel[name][f"min_{metric_name}"] = float(np.min(values))
            parallel[name][f"max_{metric_name}"] = float(np.max(values))
    return {
        "window_count": window_count,
        "pooled": pooled,
        "stability": stability,
        "selection_counts": selection_counts,
        "parallel_window_summary": parallel,
        "windows": summaries,
    }


def render_report(result: dict) -> str:
    rows = []
    pooled = result["aggregate"]["pooled"]
    stability = result["aggregate"]["stability"]
    for branch in ("binary", "multitask_any"):
        for model_name in fixed.MODELS:
            metric = (
                pooled["binary"][model_name]
                if branch == "binary"
                else pooled["multitask_heads"][model_name]["any"]
            )
            stable = stability[branch][model_name]
            rows.append(
                f"| {branch} | {model_name} | {metric['pr_auc']:.6f} | "
                f"{metric['normalized_pr_auc']:.2f} | {stable['mean_window_pr_auc']:.6f} | "
                f"{stable['std_window_pr_auc']:.6f} | {stable['min_window_pr_auc']:.6f} |"
            )
    selection_rows = []
    for branch, models in result["aggregate"]["selection_counts"].items():
        for model_name, counts in models.items():
            selection_rows.append(
                f"| {branch} | {model_name} | {counts['9']} | {counts['12']} | {counts['15']} |"
            )
    return f"""# XID Incident Sliding Training Experiment

## Protocol

- Retrain/test period: 3 days
- Candidate history lengths: 9, 12, 15 days
- Horizon: `(t, t+24h]`
- Purge: 36 hours before test and between candidate train/validation sections
- Selection: validation PR-AUC only, separately per model and binary/multi-task branch
- Fair selection: all 9/12/15-day candidates share the same natural-prevalence 3-day validation set
- Training span: the candidate train blocks are exactly 9/12/15 days, with a 36-hour purge before validation
- Models: GBDT, MLP, 1D-CNN, Tiny-TCN

## Pooled and stability results

| Branch | Model | Pooled PR-AUC | AP/prevalence | Window mean | Window std | Worst window |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Selected history lengths

| Branch | Model | 9d | 12d | 15d |
|---|---|---:|---:|---:|
{chr(10).join(selection_rows)}

Strict Parallel-3/4 results are reported in `metrics.json` as operating-point
precision/recall summaries.  Completed rolling windows are independently
checkpointed and can be reproduced or inspected without retraining.
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(6, (os.cpu_count() or 2) - 1)))

    xid_path = args.data_dir / "XID_ERRORS-002.csv"
    gpu_ids = base.read_gpu_ids(xid_path)
    ledger, audit = base.build_episode_ledger(
        xid_path, gpu_ids, args.label_dir / "xid_episode_ledger.parquet", False
    )
    grid = base.read_time_grid(args.data_dir / "telemetry_5m_util.parquet")
    decision_times = base.valid_decision_times(grid, audit)
    splits = base.chronological_split(decision_times)
    history = fixed.history_index(ledger, gpu_ids)

    test_start = int(splits["test"].min())
    test_end = int(splits["test"].max())
    boundaries = list(range(test_start, test_end + 1, WINDOW_NS))
    windows = []
    for start in boundaries:
        values = splits["test"][(splits["test"] >= start) & (splits["test"] < start + WINDOW_NS)]
        if len(values):
            windows.append(values)
    if args.max_windows > 0:
        windows = windows[: args.max_windows]

    for window_index, test_times in enumerate(windows):
        json_path = args.output_dir / f"window_{window_index:02d}.json"
        prediction_path = args.output_dir / f"window_{window_index:02d}.npz"
        if json_path.exists() and prediction_path.exists() and not args.overwrite:
            print(f"[window {window_index + 1}/{len(windows)}] checkpoint exists; skipping", flush=True)
            continue
        start = int(test_times.min())
        print(
            f"[window {window_index + 1}/{len(windows)}] "
            f"{base.ns_to_text(test_times.min())} .. {base.ns_to_text(test_times.max())}",
            flush=True,
        )
        candidate_bundles = {}
        candidate_stats = {}
        pool_stats = {}
        _, common_val_pool, _ = candidate_time_pools(decision_times, start, LENGTHS[0])
        validation_rng = np.random.default_rng(args.seed + window_index * 1000 + 777)
        common_validation_times = sample_times(
            common_val_pool, args.validation_times, validation_rng
        )
        for days in LENGTHS:
            train_pool, val_pool, time_stats = candidate_time_pools(decision_times, start, days)
            if not np.array_equal(val_pool, common_val_pool):
                raise AssertionError("Candidate history lengths must share one validation pool")
            train_rng = np.random.default_rng(args.seed + window_index * 1000 + days)
            train_times = sample_times(train_pool, args.train_times, train_rng)
            bundles, stats = fit_candidate(
                args,
                days,
                train_times,
                common_validation_times,
                ledger,
                gpu_ids,
                history,
                args.seed + window_index * 1000 + days * 10,
            )
            candidate_bundles[days] = bundles
            candidate_stats[str(days)] = stats
            pool_stats[str(days)] = time_stats
        selected, selection = select_bundles(candidate_bundles)
        print("  selected history lengths: " + json.dumps(selection, ensure_ascii=False), flush=True)
        metrics, parallel, flattened = evaluate_window(
            args, test_times, selected, ledger, gpu_ids, history
        )
        np.savez_compressed(prediction_path, **flattened)
        summary = {
            "window_index": window_index,
            "test_start": base.ns_to_text(test_times.min()),
            "test_end": base.ns_to_text(test_times.max()),
            "test_decision_times": int(len(test_times)),
            "selection": selection,
            "candidate_time_pools": pool_stats,
            "candidate_samples": candidate_stats,
            "metrics": metrics,
            "parallel": parallel,
            "prediction_file": str(prediction_path),
        }
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=base.as_json), "utf-8"
        )
        print(f"  checkpointed {json_path.name}", flush=True)
        del candidate_bundles, selected, flattened
        gc.collect()

    aggregate = aggregate_results(args.output_dir, len(windows))
    result = {
        "experiment": {
            "seed": args.seed,
            "retrain_days": 3,
            "candidate_history_days": list(LENGTHS),
            "horizon": "(t, t+24h]",
            "purge_hours": 36,
            "models": list(fixed.MODELS),
            "heads": list(fixed.HEADS),
            "test_start": base.ns_to_text(splits["test"].min()),
            "test_end": base.ns_to_text(splits["test"].max()),
        },
        "aggregate": aggregate,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=base.as_json), "utf-8"
    )
    (args.output_dir / "report.md").write_text(render_report(result), "utf-8")
    print(f"[DONE] Wrote {args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Approximate reproduction of Nie et al. (DSN 2018) for XID 43.

The paper predicts SBE occurrence for application/node samples using temporal,
spatial, workload, and SBE-history features plus a TwoStage offender-node gate.
This implementation keeps the current project's leakage-safe 24-hour onset
target and adapts only the unavailable parts:

* XID 43 is the sole target event.
* GPU topology and workload/application features are omitted.
* Nie temporal features are computed over 5/15/30/60 minutes ending at the
  existing t-10-minute availability buffer.
* XID 43 history is retained as target-history features at GPU and fleet level.
* The paper's TwoStage gate is implemented using GPUs that had XID 43 during
  the training period; the second-stage model is trained only on those GPUs.

The four compared models are Logistic Regression, GBDT, RBF SVM, and MLP.
Outputs are compact metrics and a report only.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import run_telemetry_24h_experiment as base

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TARGET_CODE = 43
LOOKBACK_BUCKETS = 12  # 60 minutes of telemetry, ending at t-10m buffer
WINDOWS = (("5m", 1), ("15m", 3), ("30m", 6), ("60m", 12))
DAY_NS = int(pd.Timedelta(days=1).value)
HISTORY_DAYS = (1, 7, 30)
METRICS = ("util", "temp", "power", "fb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--label-dir", type=Path, default=Path("outputs/telemetry_24h_fixed")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/nie_xid43_reproduction")
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--train-times", type=int, default=800)
    parser.add_argument("--validation-times", type=int, default=300)
    parser.add_argument("--negative-ratio", type=int, default=2)
    parser.add_argument("--svm-max-samples", type=int, default=60000)
    return parser.parse_args()


def target_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    selected = ledger[
        (ledger["xid_code"] == TARGET_CODE) & (~ledger["uncertain_onset"])
    ].copy()
    return selected.sort_values(["gpu_id", "onset_ns"]).reset_index(drop=True)


def label_state_for_code(
    times: np.ndarray, ledger: pd.DataFrame, gpu_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Create labels and eligibility for only XID 43.

    Other XID codes do not censor an observation because they are not the
    target event. Target-code active episodes and uncertain target onsets do.
    """
    labels = np.zeros((len(times), len(gpu_ids)), dtype=bool)
    eligible = np.ones_like(labels)
    gpu_to_index = {gpu: index for index, gpu in enumerate(gpu_ids)}
    for gpu_id, group in ledger.groupby("gpu_id", sort=False):
        index = gpu_to_index[gpu_id]
        starts = group["onset_ns"].to_numpy(dtype=np.int64)
        ends = group["episode_end_ns"].to_numpy(dtype=np.int64)
        uncertain = group["uncertain_onset"].to_numpy(dtype=bool)
        labels[:, index] = base.future_event(times, starts[~uncertain])
        uncertain_future = base.future_event(times, starts[uncertain])
        previous = np.searchsorted(starts, times, side="right") - 1
        active = np.zeros(len(times), dtype=bool)
        has_previous = previous >= 0
        active[has_previous] = ends[previous[has_previous]] >= times[has_previous]
        eligible[:, index] = ~(active | uncertain_future)
    return labels, eligible


def choose_stage2_training_samples(
    times: np.ndarray,
    ledger: pd.DataFrame,
    gpu_ids: list[str],
    offender_indices: np.ndarray,
    negative_ratio: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    labels, eligible = label_state_for_code(times, ledger, gpu_ids)
    offender_mask = np.zeros(len(gpu_ids), dtype=bool)
    offender_mask[offender_indices] = True
    stage2 = eligible & offender_mask[None, :]
    pos_t, pos_g = np.where(labels & stage2)
    neg_t, neg_g = np.where((~labels) & stage2)
    if len(pos_t) == 0:
        raise ValueError("No XID 43 positives remain in the TwoStage training set.")
    wanted_negative = min(len(neg_t), len(pos_t) * negative_ratio)
    selected = rng.choice(len(neg_t), size=wanted_negative, replace=False)
    sample_t = np.concatenate((pos_t, neg_t[selected]))
    sample_g = np.concatenate((pos_g, neg_g[selected]))
    y = np.concatenate(
        (np.ones(len(pos_t), dtype=np.int8), np.zeros(wanted_negative, dtype=np.int8))
    )
    order = rng.permutation(len(y))
    return (
        times[sample_t[order]],
        sample_g[order],
        y[order],
        {
            "candidate_times": int(len(times)),
            "offender_gpu_count": int(offender_mask.sum()),
            "positive_samples": int(len(pos_t)),
            "negative_samples": int(wanted_negative),
            "censored_samples": int((~eligible).sum()),
        },
    )


def choose_population_samples(
    times: np.ndarray, ledger: pd.DataFrame, gpu_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    labels, eligible = label_state_for_code(times, ledger, gpu_ids)
    time_rows, gpu_rows = np.where(eligible)
    return (
        times[time_rows],
        gpu_rows,
        labels[eligible].astype(np.int8),
        {
            "candidate_times": int(len(times)),
            "samples": int(len(time_rows)),
            "positive_samples": int(labels[eligible].sum()),
            "censored_samples": int((~eligible).sum()),
        },
    )


def materialize_sequences(
    data_dir: Path,
    sample_times: np.ndarray,
    gpu_index: np.ndarray,
    gpu_ids: list[str],
    offsets: np.ndarray,
    title: str,
) -> np.ndarray:
    output = np.empty(
        (len(sample_times), len(offsets), len(base.CHANNEL_NAMES)), dtype=np.float32
    )
    keys = base.half_day_keys(sample_times)
    unique = np.unique(keys)
    for complete, key in enumerate(unique, start=1):
        rows = np.flatnonzero(keys == key)
        block_start = int(sample_times[rows].min() + offsets[0] * base.STEP_NS)
        block_end = int(sample_times[rows].max() + offsets[-1] * base.STEP_NS)
        block = base.load_feature_block(data_dir, block_start, block_end, gpu_ids)
        decision = ((sample_times[rows] - block_start) // base.STEP_NS).astype(np.int64)
        sequence_index = decision[:, None] + offsets[None, :]
        output[rows] = block[sequence_index, gpu_index[rows, None], :]
        print(f"  {title}: {complete}/{len(unique)} blocks", flush=True)
    return output


def feature_names() -> list[str]:
    names = []
    for window, _ in WINDOWS:
        for metric in METRICS:
            for statistic in ("mean", "std", "delta_mean", "delta_std"):
                names.append(f"{metric}_{statistic}_{window}")
    for level in ("local", "global"):
        for days in HISTORY_DAYS:
            names.append(f"xid43_{level}_count_{days}d")
    names.append("xid43_local_days_since_last")
    return names


def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    safe = np.maximum(count, 1)
    mean = np.nansum(values, axis=1) / safe
    filled = np.where(finite, values, mean[:, None])
    all_missing = count == 0
    filled = np.where(all_missing[:, None], 0.0, filled)
    return filled.mean(axis=1), filled.std(axis=1)


def history_features(
    times: np.ndarray,
    gpu_index: np.ndarray,
    gpu_ids: list[str],
    target: pd.DataFrame,
) -> np.ndarray:
    onset_by_gpu = {
        gpu: group["onset_ns"].to_numpy(dtype=np.int64)
        for gpu, group in target.groupby("gpu_id", sort=False)
    }
    all_onsets = np.sort(target["onset_ns"].to_numpy(dtype=np.int64))
    result = np.zeros((len(times), 7), dtype=np.float32)
    for index, gpu_id in enumerate(gpu_ids):
        rows = np.flatnonzero(gpu_index == index)
        if len(rows) == 0:
            continue
        onsets = onset_by_gpu.get(gpu_id, np.empty(0, dtype=np.int64))
        query = times[rows]
        position = np.searchsorted(onsets, query, side="left")
        global_position = np.searchsorted(all_onsets, query, side="left")
        for column, days in enumerate(HISTORY_DAYS):
            cutoff = query - days * DAY_NS
            result[rows, column] = (
                position - np.searchsorted(onsets, cutoff, side="left")
            )
            result[rows, 3 + column] = (
                global_position - np.searchsorted(all_onsets, cutoff, side="left")
            )
        last_position = position - 1
        has_last = last_position >= 0
        days_since = np.full(len(rows), 999.0, dtype=np.float32)
        if has_last.any():
            days_since[has_last] = (
                query[has_last] - onsets[last_position[has_last]]
            ) / DAY_NS
        result[rows, 6] = days_since
    return result


def nie_features(
    x: np.ndarray,
    times: np.ndarray,
    gpu_index: np.ndarray,
    gpu_ids: list[str],
    target: pd.DataFrame,
) -> np.ndarray:
    """Build Nie-style temporal features plus target-history features."""
    levels = x[..., :4].astype(np.float32, copy=False)
    pieces = []
    for _, count in WINDOWS:
        window_values = levels[:, -count:, :]
        level_mean, level_std = _stats(window_values.reshape(len(x), count, 4))
        if count > 1:
            differences = np.diff(window_values, axis=1)
        else:
            differences = np.zeros((len(x), 1, 4), dtype=np.float32)
        delta_mean, delta_std = _stats(differences)
        for values in (level_mean, level_std, delta_mean, delta_std):
            pieces.append(values.astype(np.float32, copy=False))
    pieces.append(history_features(times, gpu_index, gpu_ids, target))
    result = np.concatenate(pieces, axis=1)
    return np.nan_to_num(result, nan=0.0, posinf=999.0, neginf=-999.0).astype(
        np.float32, copy=False
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def fit_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    svm_max_samples: int,
) -> tuple[dict, StandardScaler, dict]:
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train).astype(np.float32, copy=False)
    models = {}
    train_info = {}

    models["logistic"] = LogisticRegression(
        solver="lbfgs", max_iter=500, C=1.0, random_state=seed
    ).fit(x_scaled, y_train)
    train_info["logistic_samples"] = int(len(y_train))

    models["gbdt"] = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=seed + 1,
    ).fit(x_train, y_train)
    train_info["gbdt_samples"] = int(len(y_train))

    svm_rows = np.arange(len(y_train))
    if len(svm_rows) > svm_max_samples:
        sample_rng = np.random.default_rng(seed + 2)
        positive = svm_rows[y_train > 0]
        negative = svm_rows[y_train == 0]
        positive_keep = min(len(positive), svm_max_samples // 3)
        negative_keep = min(len(negative), svm_max_samples - positive_keep)
        svm_rows = np.concatenate(
            (
                sample_rng.choice(positive, size=positive_keep, replace=False),
                sample_rng.choice(negative, size=negative_keep, replace=False),
            )
        )
        svm_rows = sample_rng.permutation(svm_rows)
    models["rbf_svm"] = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        cache_size=4096,
        probability=False,
        random_state=seed + 2,
    ).fit(x_scaled[svm_rows], y_train[svm_rows])
    train_info["svm_samples"] = int(len(svm_rows))

    models["mlp"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=120,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        random_state=seed + 3,
    ).fit(x_scaled, y_train)
    train_info["mlp_samples"] = int(len(y_train))
    return models, scaler, train_info


def model_scores(models: dict, scaler: StandardScaler, x: np.ndarray) -> dict[str, np.ndarray]:
    x_scaled = scaler.transform(x).astype(np.float32, copy=False)
    result = {
        "logistic": models["logistic"].predict_proba(x_scaled)[:, 1].astype(np.float32),
        "gbdt": models["gbdt"].predict_proba(x)[:, 1].astype(np.float32),
        "rbf_svm": sigmoid(models["rbf_svm"].decision_function(x_scaled)),
        "mlp": models["mlp"].predict_proba(x_scaled)[:, 1].astype(np.float32),
    }
    return result


def gate_scores(
    scores: dict[str, np.ndarray], gpu_index: np.ndarray, offender_indices: np.ndarray
) -> dict[str, np.ndarray]:
    gate = np.isin(gpu_index, offender_indices)
    return {
        name: np.where(gate, values, 0.0).astype(np.float32)
        for name, values in scores.items()
    }


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, dict]:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[index]) if len(thresholds) else 0.5
    return threshold, {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
    }


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    predicted = score >= threshold
    true_positive = int(np.sum(predicted & (y > 0)))
    false_positive = int(np.sum(predicted & (y == 0)))
    false_negative = int(np.sum((~predicted) & (y > 0)))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive": int(predicted.sum()),
    }


def evaluate(
    score_matrix: dict[str, np.ndarray],
    labels: np.ndarray,
    eligible: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, dict]:
    y_all = labels[eligible].astype(np.int8)
    result = {}
    for name, matrix in score_matrix.items():
        score_all = matrix[eligible]
        metrics = {
            "pr_auc": float(average_precision_score(y_all, score_all)),
            "test_samples": int(len(y_all)),
            "test_positive_samples": int(y_all.sum()),
            "test_positive_rate": float(y_all.mean()),
        }
        metrics.update(threshold_metrics(y_all, score_all, thresholds[name]))
        metrics.update(base.ranking_metrics(matrix, labels, eligible))
        result[name] = metrics
    return result


def render_report(result: dict) -> str:
    rows = []
    for name, value in result["metrics"].items():
        rows.append(
            "| {name} | {pr:.6f} | {p:.4f} | {r:.4f} | {f1:.4f} | {r10:.4f} | {capture:.4f} |".format(
                name=name,
                pr=value["pr_auc"],
                p=value["precision"],
                r=value["recall"],
                f1=value["f1"],
                r10=value["recall_at_10"],
                capture=value["top_5pct_capture"],
            )
        )
    return f"""# Nie et al. 방법론의 XID 43 재현 실험

## 고정한 조건

- Target: XID 43만 사용
- Horizon: `(t, t+24h]`
- Telemetry: Util / Temp / Power / FB 5분 Parquet
- Temporal feature windows: 5/15/30/60분, 기존 t-10분 availability buffer 적용
- Spatial features: GPU topology 부재로 제외
- Workload/application features: 검증된 as-of 입력 부재로 제외
- History features: 과거 XID 43 GPU-level 및 fleet-level count 포함
- TwoStage: train period 동안 XID 43이 있었던 GPU만 2단계 모델에 통과
- Models: Logistic Regression, GBDT, RBF SVM, MLP
- Sampling: TwoStage 이후 positive:negative 약 1:{result['experiment']['negative_ratio']}로 학습
- Split: chronological 60/20/20, 36시간 purge

## Nie 논문과의 재현 차이

원 논문은 SBE를 application/node 실행 단위로 예측하고, node topology·workload/application feature를 사용합니다. 현재 데이터에는 해당 정보가 없어 GPU×decision epoch과 XID 43으로 대응했습니다. 따라서 원 논문 수치와 직접 비교하지 않고, 기존 24시간 telemetry baseline 대비 PR-AUC 변화를 비교합니다.

## 결과

| Model | PR-AUC | Precision | Recall | F1 | Recall@10 | Top-5% Capture |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Threshold는 validation에서 F1 최대가 되도록 선택했으며 test에는 고정 적용했습니다. PR-AUC와 ranking 지표는 threshold와 무관하게 계산했습니다.
"""


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir
    xid_path = data_dir / "XID_ERRORS-002.csv"
    util_path = data_dir / "telemetry_5m_util.parquet"
    required = [xid_path, util_path]
    required.extend(data_dir / name for _, name in base.METRICS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input is missing: " + ", ".join(missing))

    print("[1/8] Loading the existing exact XID episode ledger...", flush=True)
    gpu_ids = base.read_gpu_ids(xid_path)
    ledger, audit = base.build_episode_ledger(
        xid_path,
        gpu_ids,
        args.label_dir / "xid_episode_ledger.parquet",
        rebuild=False,
    )
    target = target_ledger(ledger)
    if len(target) == 0:
        raise ValueError("No certain XID 43 episodes were found.")
    print(f"  target XID 43 episodes={len(target):,}", flush=True)

    print("[2/8] Building the common leakage-safe chronological split...", flush=True)
    grid = base.read_time_grid(util_path)
    decision_times = base.valid_decision_times(grid, audit)
    splits = base.chronological_split(decision_times)
    train_end = int(splits["train"][-1])
    offender_gpu_ids = set(target.loc[target["onset_ns"] <= train_end, "gpu_id"])
    offender_indices = np.asarray(
        [index for index, gpu in enumerate(gpu_ids) if gpu in offender_gpu_ids], dtype=np.int64
    )
    rng = np.random.default_rng(args.seed)
    train_times = rng.choice(
        splits["train"], size=min(args.train_times, len(splits["train"])), replace=False
    )
    validation_times = rng.choice(
        splits["validation"],
        size=min(args.validation_times, len(splits["validation"])),
        replace=False,
    )
    train_times.sort()
    validation_times.sort()
    print(f"  TwoStage offender GPUs={len(offender_indices):,}", flush=True)

    print("[3/8] Selecting TwoStage training and population validation samples...", flush=True)
    train_t, train_g, y_train, train_summary = choose_stage2_training_samples(
        train_times,
        target,
        gpu_ids,
        offender_indices,
        args.negative_ratio,
        rng,
    )
    val_t, val_g, y_val, validation_summary = choose_population_samples(
        validation_times, target, gpu_ids
    )
    offsets = np.arange(-(LOOKBACK_BUCKETS + 2), -2, dtype=np.int64)
    x_train_raw = materialize_sequences(
        data_dir, train_t, train_g, gpu_ids, offsets, "train telemetry"
    )
    x_val_raw = materialize_sequences(
        data_dir, val_t, val_g, gpu_ids, offsets, "validation telemetry"
    )
    x_train = nie_features(x_train_raw, train_t, train_g, gpu_ids, target)
    x_val = nie_features(x_val_raw, val_t, val_g, gpu_ids, target)

    print("[4/8] Training Nie four-model comparison...", flush=True)
    models, scaler, train_info = fit_models(
        x_train, y_train, args.seed, args.svm_max_samples
    )
    val_raw_scores = model_scores(models, scaler, x_val)
    val_gated_scores = gate_scores(val_raw_scores, val_g, offender_indices)
    validation_labels, validation_eligible = label_state_for_code(
        validation_times, target, gpu_ids
    )
    val_y_all = validation_labels[validation_eligible].astype(np.int8)
    val_gate = np.isin(val_g, offender_indices)
    val_basic_a = val_gate.astype(np.float32)
    thresholds = {}
    validation_model_metrics = {}
    for name, score in val_gated_scores.items():
        threshold, best = best_f1_threshold(val_y_all, score)
        thresholds[name] = threshold
        validation_model_metrics[name] = best
    basic_threshold, basic_best = best_f1_threshold(val_y_all, val_basic_a)
    thresholds["basic_a"] = basic_threshold
    validation_model_metrics["basic_a"] = basic_best

    del x_train_raw, x_val_raw, x_train, x_val, val_raw_scores, val_gated_scores
    gc.collect()

    print("[5/8] Evaluating the full chronological XID 43 test period...", flush=True)
    test_labels, test_eligible = label_state_for_code(splits["test"], target, gpu_ids)
    score_matrix = {
        name: np.full(test_labels.shape, np.nan, dtype=np.float32)
        for name in (*models.keys(), "basic_a")
    }
    test_keys = base.half_day_keys(splits["test"])
    unique_keys = np.unique(test_keys)
    for complete, key in enumerate(unique_keys, start=1):
        time_rows = np.flatnonzero(test_keys == key)
        local_t, gpu_rows = np.where(test_eligible[time_rows])
        sample_times = splits["test"][time_rows[local_t]]
        x_raw = materialize_sequences(
            data_dir, sample_times, gpu_rows, gpu_ids, offsets, f"test {key}"
        )
        x_test = nie_features(x_raw, sample_times, gpu_rows, gpu_ids, target)
        raw_scores = model_scores(models, scaler, x_test)
        gated = gate_scores(raw_scores, gpu_rows, offender_indices)
        for name, values in gated.items():
            score_matrix[name][time_rows[local_t], gpu_rows] = values
        score_matrix["basic_a"][time_rows[local_t], gpu_rows] = np.isin(
            gpu_rows, offender_indices
        ).astype(np.float32)
        print(f"  test evaluation: {complete}/{len(unique_keys)} blocks", flush=True)
        del x_raw, x_test, raw_scores, gated
        gc.collect()

    print("[6/8] Computing PR-AUC, paper-style threshold metrics, and ranking...", flush=True)
    thresholds_for_test = {name: thresholds[name] for name in score_matrix}
    metrics = evaluate(score_matrix, test_labels, test_eligible, thresholds_for_test)

    result = {
        "experiment": {
            "paper": "Nie et al., Machine Learning Models for GPU Error Prediction in a Large Scale HPC System, DSN 2018",
            "target_xid_code": TARGET_CODE,
            "horizon": "(t, t+24h]",
            "negative_ratio": args.negative_ratio,
            "svm_max_samples": args.svm_max_samples,
            "feature_names": feature_names(),
            "models": list(models.keys()),
            "two_stage": True,
            "offender_gpu_count": int(len(offender_indices)),
        },
        "label_audit": {
            **audit,
            "target_episode_count": int(len(target)),
            "target_certain_episode_count": int((~target["uncertain_onset"]).sum()),
            "target_code": TARGET_CODE,
        },
        "splits": {
            name: {
                "count": int(len(values)),
                "start": base.ns_to_text(values.min()),
                "end": base.ns_to_text(values.max()),
            }
            for name, values in splits.items()
        },
        "training": {**train_summary, **train_info},
        "validation": {
            **validation_summary,
            "positive_rate": float(val_y_all.mean()),
            "two_stage_gate_rate": float(val_gate.mean()),
            "model_f1_selection": validation_model_metrics,
        },
        "test_summary": {
            "decision_epochs": int(len(splits["test"])),
            "eligible_samples": int(test_eligible.sum()),
            "censored_samples": int((~test_eligible).sum()),
            "positive_samples": int(test_labels[test_eligible].sum()),
            "positive_rate": float(test_labels[test_eligible].mean()),
            "test_start": base.ns_to_text(splits["test"].min()),
            "test_end": base.ns_to_text(splits["test"].max()),
        },
        "metrics": metrics,
    }
    print("[7/8] Writing compact outputs...", flush=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=base.as_json), "utf-8"
    )
    (args.output_dir / "report.md").write_text(render_report(result), "utf-8")
    print("[8/8] Done. Wrote metrics.json and report.md.", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "eda_xid_observability"
OUTPUT = ROOT / "outputs" / "risk_tape_5m"
TELEMETRY = {
    "gpu_util": ROOT / "GPU_UTIL.csv",
    "gpu_temp": ROOT / "GPU_TEMP.csv",
    "power_usage": ROOT / "POWER_USAGE.csv",
}
BIN_NS = 5 * 60 * 1_000_000_000
MINUTE_NS = 60 * 1_000_000_000
DAY_NS = 24 * 60 * MINUTE_NS
FEATURE_GAP_BINS = 2
LAST_FEATURE_LAG = FEATURE_GAP_BINS + 1
WINDOW_LAGS = np.arange(LAST_FEATURE_LAG, LAST_FEATURE_LAG + 6)
NEGATIVE_RATIO = 20
TOP_K = 100
FOLDS = [
    ("fold_1", "2023-06-15", "2023-07-01"),
    ("fold_2", "2023-07-01", "2023-07-15"),
    ("fold_3", "2023-07-15", "2023-08-01"),
    ("fold_4", "2023-08-01", "2023-08-18"),
]
FEATURES = [
    *[
        f"{metric}_{suffix}"
        for metric in TELEMETRY
        for suffix in ["last_5m", "mean_30m", "delta_30m", "node_last_5m"]
    ],
    "xid_count_30d",
    "days_since_xid",
]


def load_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SOURCE / "xid_observability_5m.npz") as data:
        return (
            data["valid_counts"],
            data["expected_counts"],
            data["bin_start_ns"],
            data["gpu_ids"].astype(str),
        )


def header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def node_matrix(metric_path: Path, node_path: Path, rows: int, nodes: int) -> None:
    if node_path.exists():
        return
    metric = np.load(metric_path, mmap_mode="r")
    output = np.lib.format.open_memmap(node_path, mode="w+", dtype=np.float32, shape=(rows, nodes))
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        values = metric[start:stop].reshape(stop - start, nodes, 8)
        valid = np.isfinite(values)
        count = valid.sum(axis=2)
        total = np.where(valid, values, 0).sum(axis=2)
        output[start:stop] = np.divide(
            total,
            count,
            out=np.full(total.shape, np.nan, dtype=np.float32),
            where=count > 0,
        )
    output.flush()


def scan_metric(
    name: str,
    path: Path,
    first_ns: int,
    shape: tuple[int, int],
    gpu_ids: np.ndarray,
    expected: np.ndarray,
) -> None:
    metric_path = OUTPUT / f"{name}_5m.npy"
    coverage_path = OUTPUT / f"{name}_coverage.npy"
    node_path = OUTPUT / f"{name}_node_5m.npy"
    if metric_path.exists() and coverage_path.exists():
        node_matrix(metric_path, node_path, shape[0], shape[1] // 8)
        print(f"[{name}] cached", flush=True)
        return
    names = header(path)
    if names[1:] != gpu_ids.tolist():
        raise ValueError(f"{name}: GPU header order differs from XID grid")
    types = {"Time": pa.string(), **{gpu: pa.float32() for gpu in gpu_ids}}
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True),
        parse_options=pacsv.ParseOptions(delimiter=","),
        convert_options=pacsv.ConvertOptions(
            column_types=types,
            null_values=[""],
            strings_can_be_null=True,
        ),
    )
    sums = np.zeros(shape, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.uint8)
    for batch_number, batch in enumerate(reader, start=1):
        times = pd.to_datetime(batch.column(0).to_pylist(), utc=True).asi8
        bins = ((times - first_ns) // BIN_NS).astype(np.int32)
        keep = (bins >= 0) & (bins < shape[0])
        if not keep.any():
            continue
        bins = bins[keep]
        frame = batch.to_pandas(split_blocks=True)
        values = frame.iloc[keep, 1:].to_numpy(dtype=np.float32, copy=False)
        starts = np.r_[0, np.flatnonzero(bins[1:] != bins[:-1]) + 1]
        grouped_bins = bins[starts]
        valid = np.isfinite(values)
        values = np.where(valid, values, 0)
        sums[grouped_bins] += np.add.reduceat(values, starts, axis=0)
        counts[grouped_bins] += np.add.reduceat(valid.astype(np.uint8), starts, axis=0)
        if batch_number % 8 == 0:
            print(f"[{name}] batches={batch_number}", flush=True)
    if not (counts <= expected[:, None]).all():
        raise AssertionError(f"{name}: coverage exceeds expected observations")
    np.divide(sums, counts, out=sums, where=counts > 0)
    sums[counts == 0] = np.nan
    np.save(metric_path, sums)
    np.save(coverage_path, counts)
    del sums, counts
    node_matrix(metric_path, node_path, shape[0], shape[1] // 8)


def build_telemetry(
    first_ns: int,
    shape: tuple[int, int],
    gpu_ids: np.ndarray,
    expected: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    for name, path in TELEMETRY.items():
        scan_metric(name, path, first_ns, shape, gpu_ids, expected)
    matrices = {
        name: np.load(OUTPUT / f"{name}_5m.npy", mmap_mode="r") for name in TELEMETRY
    }
    nodes = {
        name: np.load(OUTPUT / f"{name}_node_5m.npy", mmap_mode="r") for name in TELEMETRY
    }
    coverage_path = OUTPUT / "telemetry_min_coverage.npy"
    if not coverage_path.exists():
        minimum = np.full(shape, 255, dtype=np.uint8)
        for name in TELEMETRY:
            minimum = np.minimum(
                minimum,
                np.load(OUTPUT / f"{name}_coverage.npy", mmap_mode="r"),
            )
        np.save(coverage_path, minimum)
    return matrices, nodes, np.load(coverage_path, mmap_mode="r")


def fault_inputs(gpu_to_index: dict[str, int], first_ns: int) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    fault = pd.read_csv(SOURCE / "fault_tape_v2.csv")
    fault = fault[fault["xid_code"].isin([31, 43])].copy()
    for column in ["xid_time_raw", "fault_time_mixed"]:
        fault[column] = pd.to_datetime(fault[column], utc=True)
    fault["gpu_index"] = fault["gpu_id"].map(gpu_to_index)
    fault = fault[fault["gpu_index"].notna()].copy()
    fault["label_bin"] = ((fault["fault_time_mixed"].astype("int64") - first_ns) // BIN_NS).astype(int)
    history = {
        gpu: np.sort(group["xid_time_raw"].astype("int64").to_numpy())
        for gpu, group in fault.groupby("gpu_index")
    }
    return fault, {int(key): value for key, value in history.items()}


def row_nanmean(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    total = np.where(valid, values, 0).sum(axis=1)
    return np.divide(
        total,
        count,
        out=np.full(len(values), np.nan, dtype=np.float32),
        where=count > 0,
    )


def make_features(
    bins: np.ndarray,
    gpus: np.ndarray,
    bin_ns: np.ndarray,
    matrices: dict[str, np.ndarray],
    node_matrices: dict[str, np.ndarray],
    history: dict[int, np.ndarray],
) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for name, matrix in matrices.items():
        window = np.column_stack([matrix[bins - lag, gpus] for lag in WINDOW_LAGS])
        last = window[:, 0].astype(np.float32)
        mean = row_nanmean(window)
        data[f"{name}_last_5m"] = last
        data[f"{name}_mean_30m"] = mean
        data[f"{name}_delta_30m"] = last - mean
        data[f"{name}_node_last_5m"] = node_matrices[name][
            bins - LAST_FEATURE_LAG, gpus // 8
        ].astype(np.float32)
    cutoff = bin_ns[bins] - 10 * MINUTE_NS
    count_30d = np.zeros(len(bins), dtype=np.float32)
    days_since = np.full(len(bins), 95, dtype=np.float32)
    for gpu in np.unique(gpus):
        positions = np.flatnonzero(gpus == gpu)
        events = history.get(int(gpu))
        if events is None or not len(events):
            continue
        right = np.searchsorted(events, cutoff[positions], side="right")
        left = np.searchsorted(events, cutoff[positions] - 30 * DAY_NS, side="left")
        count_30d[positions] = right - left
        has_previous = right > 0
        previous = np.zeros(len(positions), dtype=np.int64)
        previous[has_previous] = events[right[has_previous] - 1]
        days_since[positions[has_previous]] = np.minimum(
            (cutoff[positions[has_previous]] - previous[has_previous]) / DAY_NS,
            95,
        )
    data["xid_count_30d"] = count_30d
    data["days_since_xid"] = days_since
    return pd.DataFrame(data, columns=FEATURES)


def fold_for_bins(bin_ns: np.ndarray) -> np.ndarray:
    result = np.full(len(bin_ns), "warmup", dtype="<U8")
    for name, start, end in FOLDS:
        start_ns = pd.Timestamp(start, tz="UTC").value
        end_ns = pd.Timestamp(end, tz="UTC").value
        result[(bin_ns >= start_ns) & (bin_ns < end_ns)] = name
    return result


def build_sample(
    fault: pd.DataFrame,
    valid: np.ndarray,
    expected: np.ndarray,
    bin_ns: np.ndarray,
    gpu_ids: np.ndarray,
    matrices: dict[str, np.ndarray],
    node_matrices: dict[str, np.ndarray],
    history: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    gpu_count = len(gpu_ids)
    fault = fault[(fault["label_bin"] >= int(WINDOW_LAGS.max())) & (fault["label_bin"] < len(bin_ns))]
    fault["flat"] = fault["label_bin"] * gpu_count + fault["gpu_index"].astype(int)
    positive = (
        fault.groupby("flat")
        .agg(
            positive_event_count=("event_id", "size"),
            positive_xid_codes=("xid_code", lambda x: "+".join(map(str, sorted(set(x))))),
            label_shifted_from_raw=("fault_time_mixed", lambda x: bool((x != fault.loc[x.index, "xid_time_raw"]).any())),
        )
        .reset_index()
    )
    positive_flat = positive["flat"].to_numpy(dtype=np.int64)
    positive_set = set(positive_flat.tolist())
    rng = np.random.default_rng(20230823)
    boundaries = [
        int(WINDOW_LAGS.max()),
        *[
            int(np.searchsorted(bin_ns, pd.Timestamp(start, tz="UTC").value))
            for _, start, _ in FOLDS
        ],
        int(np.searchsorted(bin_ns, pd.Timestamp(FOLDS[-1][2], tz="UTC").value)),
    ]
    negatives: set[int] = set()
    positive_bins = positive_flat // gpu_count
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        positives_in_segment = int(((positive_bins >= left) & (positive_bins < right)).sum())
        target = NEGATIVE_RATIO * positives_in_segment
        segment: set[int] = set()
        while len(segment) < target:
            size = max(2_000, 2 * (target - len(segment)))
            bins = rng.integers(left, right, size=size, dtype=np.int32)
            gpus = rng.integers(0, gpu_count, size=size, dtype=np.int32)
            observed = (expected[bins] == 20) & (valid[bins, gpus] == expected[bins])
            for flat in (bins[observed].astype(np.int64) * gpu_count + gpus[observed]).tolist():
                if flat not in positive_set:
                    segment.add(flat)
                if len(segment) >= target:
                    break
        negatives.update(segment)
    all_flat = np.r_[positive_flat, np.fromiter(negatives, dtype=np.int64)]
    labels = np.r_[np.ones(len(positive_flat), dtype=np.uint8), np.zeros(len(negatives), dtype=np.uint8)]
    bins = (all_flat // gpu_count).astype(np.int32)
    gpus = (all_flat % gpu_count).astype(np.int32)
    order = np.argsort(all_flat)
    all_flat, labels, bins, gpus = all_flat[order], labels[order], bins[order], gpus[order]
    features = make_features(bins, gpus, bin_ns, matrices, node_matrices, history)
    sample = pd.DataFrame(
        {
            "sample_id": all_flat,
            "decision_time": pd.to_datetime(bin_ns[bins], utc=True),
            "feature_cutoff_time": pd.to_datetime(bin_ns[bins] - 10 * MINUTE_NS, utc=True),
            "gpu_id": gpu_ids[gpus],
            "gpu_index": gpus,
            "label_xid31_or_43_within_5m": labels,
            "model_fold": fold_for_bins(bin_ns)[bins],
        }
    )
    sample = pd.concat([sample, features], axis=1)
    metadata = positive.set_index("flat")
    sample["positive_event_count"] = sample["sample_id"].map(metadata["positive_event_count"]).fillna(0).astype(int)
    sample["positive_xid_codes"] = sample["sample_id"].map(metadata["positive_xid_codes"]).fillna("")
    sample["label_shifted_from_raw"] = sample["sample_id"].map(metadata["label_shifted_from_raw"]).fillna(False).astype(bool)
    sample.to_parquet(OUTPUT / "decision_sample.parquet", index=False)
    return sample, positive_flat


def adjusted_probability(probability: np.ndarray, true_prior: float, sample_prior: float) -> np.ndarray:
    factor = (true_prior / (1 - true_prior)) / (sample_prior / (1 - sample_prior))
    return (probability * factor) / (1 - probability + probability * factor)


def build_observability(
    valid: np.ndarray,
    expected: np.ndarray,
    telemetry_coverage: np.ndarray,
) -> np.memmap:
    path = OUTPUT / "observability_score.npy"
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=valid.shape)
    output[:] = np.nan
    for decision in range(int(WINDOW_LAGS.max()), len(expected)):
        xid_start, xid_end = max(0, decision - 6), decision
        telemetry_start, telemetry_end = decision - int(WINDOW_LAGS.max()), decision - FEATURE_GAP_BINS
        xid_denominator = expected[xid_start:xid_end].sum()
        telemetry_denominator = expected[telemetry_start:telemetry_end].sum()
        xid_score = valid[xid_start:xid_end].sum(axis=0) / xid_denominator
        telemetry_score = telemetry_coverage[telemetry_start:telemetry_end].sum(axis=0) / telemetry_denominator
        output[decision] = np.minimum(xid_score, telemetry_score).astype(np.float16)
    output.flush()
    return output


def validation_metrics(
    name: str,
    start: int,
    end: int,
    score: np.ndarray,
    rank: np.ndarray,
    valid: np.ndarray,
    expected: np.ndarray,
    positive_flat: np.ndarray,
    gpu_count: int,
) -> dict:
    labels = np.zeros((end - start, gpu_count), dtype=bool)
    positive_bins = positive_flat // gpu_count
    keep = (positive_bins >= start) & (positive_bins < end)
    labels[positive_bins[keep] - start, positive_flat[keep] % gpu_count] = True
    eligible = valid[start:end] == expected[start:end, None]
    eligible |= labels
    y = labels[eligible]
    prediction = np.asarray(score[start:end])[eligible]
    row = {
        "model_fold": name,
        "validation_start": pd.to_datetime(int(start), unit="ns", utc=True),
        "validation_bins": end - start,
        "eligible_gpu_decisions": int(eligible.sum()),
        "positive_gpu_decisions": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier_score": float(brier_score_loss(y, prediction)),
    }
    for k in [10, 50, 100]:
        hits = int((labels & (rank[start:end] <= k) & (rank[start:end] > 0)).sum())
        row[f"recall_at_{k}"] = hits / max(int(labels.sum()), 1)
        row[f"lift_at_{k}"] = (hits / ((end - start) * k)) / max(float(y.mean()), 1e-12)
    return row


def train_oof(
    sample: pd.DataFrame,
    positive_flat: np.ndarray,
    valid: np.ndarray,
    expected: np.ndarray,
    bin_ns: np.ndarray,
    gpu_ids: np.ndarray,
    matrices: dict[str, np.ndarray],
    node_matrices: dict[str, np.ndarray],
    history: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    shape = valid.shape
    gpu_count = shape[1]
    score = np.lib.format.open_memmap(OUTPUT / "risk_score.npy", mode="w+", dtype=np.float32, shape=shape)
    rank = np.lib.format.open_memmap(OUTPUT / "risk_rank.npy", mode="w+", dtype=np.uint16, shape=shape)
    score[:] = np.nan
    rank[:] = 0
    fold_id = np.full(len(bin_ns), -1, dtype=np.int8)
    metrics = []
    models = []
    sample_times = sample["decision_time"].astype("int64").to_numpy()
    y_sample = sample["label_xid31_or_43_within_5m"].to_numpy()
    x_sample = sample[FEATURES]
    positive_bins = positive_flat // gpu_count
    for fold_number, (name, start_time, end_time) in enumerate(FOLDS, start=1):
        start_ns = pd.Timestamp(start_time, tz="UTC").value
        end_ns = pd.Timestamp(end_time, tz="UTC").value
        start = int(np.searchsorted(bin_ns, start_ns))
        end = int(np.searchsorted(bin_ns, end_ns))
        train_mask = sample_times < start_ns
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=20230823 + fold_number,
        )
        model.fit(x_sample.loc[train_mask], y_sample[train_mask])
        train_positive = int(y_sample[train_mask].sum())
        observed_train = int((valid[int(WINDOW_LAGS.max()):start] == expected[int(WINDOW_LAGS.max()):start, None]).sum())
        positive_train = int((positive_bins < start).sum())
        true_prior = positive_train / max(observed_train + positive_train, 1)
        sample_prior = train_positive / int(train_mask.sum())
        for left in range(start, end, 128):
            right = min(left + 128, end)
            bins = np.repeat(np.arange(left, right, dtype=np.int32), gpu_count)
            gpus = np.tile(np.arange(gpu_count, dtype=np.int32), right - left)
            features = make_features(bins, gpus, bin_ns, matrices, node_matrices, history)
            raw = model.predict_proba(features)[:, 1]
            calibrated = adjusted_probability(raw, true_prior, sample_prior).reshape(right - left, gpu_count)
            score[left:right] = calibrated.astype(np.float32)
            order = np.argsort(-calibrated, axis=1, kind="stable")
            ranks = np.empty_like(order, dtype=np.uint16)
            np.put_along_axis(
                ranks,
                order,
                np.arange(1, gpu_count + 1, dtype=np.uint16)[None, :],
                axis=1,
            )
            rank[left:right] = ranks
        fold_id[start:end] = fold_number
        metric = validation_metrics(name, start, end, score, rank, valid, expected, positive_flat, gpu_count)
        metric.update(
            {
                "validation_start": pd.to_datetime(bin_ns[start], utc=True),
                "validation_end": pd.to_datetime(bin_ns[end - 1] + BIN_NS, utc=True),
                "train_sample_rows": int(train_mask.sum()),
                "train_sample_positives": train_positive,
                "true_train_prevalence": true_prior,
            }
        )
        metrics.append(metric)
        models.append({"model_fold": name, "model": model, "true_prior": true_prior, "sample_prior": sample_prior})
        print(f"[{name}] AP={metric['average_precision']:.6f}, AUC={metric['roc_auc']:.4f}, R@100={metric['recall_at_100']:.3f}", flush=True)
    score.flush()
    rank.flush()
    joblib.dump({"features": FEATURES, "folds": models}, OUTPUT / "oof_models.joblib")
    result = pd.DataFrame(metrics)
    result.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    return score, rank, fold_id, result


def write_topk(
    score: np.ndarray,
    rank: np.ndarray,
    observability: np.ndarray,
    fold_id: np.ndarray,
    bin_ns: np.ndarray,
    gpu_ids: np.ndarray,
) -> int:
    path = OUTPUT / f"risk_tape_top{TOP_K}.parquet"
    writer = None
    total = 0
    fold_names = np.array(["warmup", *[x[0] for x in FOLDS]])
    for left in range(0, len(bin_ns), 256):
        right = min(left + 256, len(bin_ns))
        row_index, gpu_index = np.where((rank[left:right] > 0) & (rank[left:right] <= TOP_K))
        if not len(row_index):
            continue
        absolute_row = row_index + left
        order = np.lexsort((rank[absolute_row, gpu_index], absolute_row))
        absolute_row, gpu_index = absolute_row[order], gpu_index[order]
        table = pa.table(
            {
                "decision_time": pa.array(bin_ns[absolute_row], type=pa.timestamp("ns", tz="UTC")),
                "gpu_id": pa.array(gpu_ids[gpu_index]),
                "risk_score": pa.array(np.asarray(score[absolute_row, gpu_index]), type=pa.float32()),
                "risk_rank": pa.array(np.asarray(rank[absolute_row, gpu_index]), type=pa.uint16()),
                "observability_score": pa.array(np.asarray(observability[absolute_row, gpu_index], dtype=np.float32), type=pa.float32()),
                "model_fold": pa.array(fold_names[fold_id[absolute_row]]),
                "feature_cutoff_time": pa.array(bin_ns[absolute_row] - 10 * MINUTE_NS, type=pa.timestamp("ns", tz="UTC")),
                "target": pa.array(["XID31_or_43_within_5m"] * len(absolute_row)),
            }
        )
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        total += len(table)
    if writer is not None:
        writer.close()
    return total


def write_report(sample: pd.DataFrame, metrics: pd.DataFrame, meta: dict) -> None:
    lines = [
        "# 5분 OOF Risk Tape 결과",
        "",
        "## 설계",
        "",
        "- target: fault_time_mixed 기준 향후 5분 내 XID 31 또는 43",
        "- feature cutoff: decision time -10분; 이후 텔레메트리는 사용하지 않음",
        "- feature: GPU_UTIL·GPU_TEMP·POWER_USAGE의 GPU/노드 과거 상태와 과거 XID 이력",
        "- label precedence: observed positive > unknown > negative; XID 관측 공백은 negative에서 제외",
        "- validation: 초기 약 1개월 warm-up 이후 expanding-window 4-fold OOF",
        "",
        "## 표본과 출력",
        "",
        f"- decision sample {len(sample):,}행, positive {sample['label_xid31_or_43_within_5m'].sum():,}행",
        f"- 전체 5분 grid {meta['grid_bins']:,} × GPU {meta['gpus']:,}",
        f"- OOF 평가기간 {meta['oof_start']} ~ {meta['oof_end']}",
        f"- top-{TOP_K} Blox adapter {meta['topk_rows']:,}행",
        "",
        "## Fold 성능",
        "",
    ]
    for row in metrics.itertuples():
        lines.append(
            f"- {row.model_fold}: AP {row.average_precision:.6f}, ROC-AUC {row.roc_auc:.3f}, R@10 {row.recall_at_10:.1%}, R@50 {row.recall_at_50:.1%}, R@100 {row.recall_at_100:.1%}"
        )
    lines += [
        "",
        "## 해석 제한",
        "",
        "- risk_score는 negative downsampling 후 사전확률을 보정한 상대위험 점수이며 임상적 확률처럼 해석하지 않는다.",
        "- terminal-low onset이 확인된 156건은 mixed 시각으로 이동했지만 나머지 사건의 기록지연은 식별되지 않아 잔여 label leakage 가능성이 있다.",
        "- warm-up 구간에는 OOF 점수가 없으므로 Blox 정책 비교는 OOF 평가기간으로 제한한다.",
        "- observability_score가 낮은 GPU는 저위험이 아니라 불확실한 GPU다. drain 정책에서 별도 취급한다.",
    ]
    (OUTPUT / "RISK_TAPE_5M_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def make_plot(metrics: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(metrics))
    axes[0].bar(x - 0.18, metrics["roc_auc"], 0.36, label="ROC-AUC")
    axes[0].bar(x + 0.18, metrics["average_precision"], 0.36, label="AP")
    axes[0].set_xticks(x, metrics["model_fold"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("OOF discrimination")
    axes[0].legend()
    for key, label in [("recall_at_10", "K=10"), ("recall_at_50", "K=50"), ("recall_at_100", "K=100")]:
        axes[1].plot(x, metrics[key], marker="o", label=label)
    axes[1].set_xticks(x, metrics["model_fold"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Positive recall among top-K GPUs")
    axes[1].legend()
    axes[2].plot(x, metrics["lift_at_10"], marker="o", label="K=10")
    axes[2].plot(x, metrics["lift_at_50"], marker="o", label="K=50")
    axes[2].plot(x, metrics["lift_at_100"], marker="o", label="K=100")
    axes[2].set_xticks(x, metrics["model_fold"])
    axes[2].set_yscale("log")
    axes[2].set_title("Top-K lift over prevalence")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "risk_oof_performance.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    valid, expected, bin_ns, gpu_ids = load_grid()
    if len(gpu_ids) % 8:
        raise AssertionError("GPU columns are not divisible into 8-GPU nodes")
    nodes = [gpu.rsplit("-", 1)[0] for gpu in gpu_ids]
    if not all(nodes[index:index + 8] == [nodes[index]] * 8 for index in range(0, len(nodes), 8)):
        raise AssertionError("GPU columns are not grouped by node")
    matrices, node_matrices, telemetry_coverage = build_telemetry(
        int(bin_ns[0]), valid.shape, gpu_ids, expected
    )
    gpu_to_index = {gpu: index for index, gpu in enumerate(gpu_ids)}
    fault, history = fault_inputs(gpu_to_index, int(bin_ns[0]))
    sample, positive_flat = build_sample(
        fault, valid, expected, bin_ns, gpu_ids, matrices, node_matrices, history
    )
    observability = build_observability(valid, expected, telemetry_coverage)
    score, rank, fold_id, metrics = train_oof(
        sample, positive_flat, valid, expected, bin_ns, gpu_ids, matrices, node_matrices, history
    )
    topk_rows = write_topk(score, rank, observability, fold_id, bin_ns, gpu_ids)
    np.savez_compressed(
        OUTPUT / "risk_tape_5m.npz",
        risk_score=np.asarray(score),
        risk_rank=np.asarray(rank),
        observability_score=np.asarray(observability),
        decision_time_ns=bin_ns,
        gpu_ids=gpu_ids,
        model_fold_id=fold_id,
    )
    meta = {
        "target": "XID31_or_43_within_5m_at_fault_time_mixed",
        "feature_gap_minutes": 10,
        "features": FEATURES,
        "negative_sampling_ratio": NEGATIVE_RATIO,
        "grid_bins": len(bin_ns),
        "gpus": len(gpu_ids),
        "oof_start": FOLDS[0][1],
        "oof_end": FOLDS[-1][2],
        "top_k": TOP_K,
        "topk_rows": topk_rows,
        "positive_unique_gpu_bins": len(positive_flat),
        "shifted_event_rows": int((fault["fault_time_mixed"] < fault["xid_time_raw"]).sum()),
    }
    (OUTPUT / "risk_tape_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(sample, metrics, meta)
    make_plot(metrics)
    assert sample["sample_id"].is_unique
    assert np.isfinite(score[fold_id > 0]).all()
    assert ((rank[fold_id > 0] >= 1) & (rank[fold_id > 0] <= len(gpu_ids))).all()
    assert metrics["positive_gpu_decisions"].gt(0).all()
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

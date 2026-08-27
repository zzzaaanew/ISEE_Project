from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "eda_xid_observability"
XID_FILE = ROOT / "XID_ERRORS.csv"
FAULT_V1 = ROOT / "outputs" / "eda_blox_inputs" / "fault_tape_candidates.csv"
INVENTORY = ROOT / "outputs" / "eda_phase1" / "xid_inventory.json"
STEP_SECONDS = 15
BIN_SECONDS = 300
ROWS_PER_BIN = BIN_SECONDS // STEP_SECONDS
PRE_EVENT_ROWS = 30 * 60 // STEP_SECONDS
HORIZONS_MIN = [5, 12 * 60, 24 * 60, 72 * 60]


def max_false_run(values: np.ndarray) -> int:
    missing = ~values
    if not missing.any():
        return 0
    padded = np.r_[False, missing, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return int((changes[1::2] - changes[::2]).max())


def future_unknown_fraction(bad: np.ndarray, horizon_bins: int) -> float:
    rows, columns = bad.shape
    starts = np.arange(rows)
    ends = np.minimum(starts + horizon_bins, rows)
    total = 0
    for column in range(columns):
        cumulative = np.r_[0, np.cumsum(bad[:, column], dtype=np.int32)]
        unknown = cumulative[ends] - cumulative[starts] > 0
        unknown |= starts + horizon_bins > rows
        total += int(unknown.sum())
    return total / (rows * columns)


def read_fault() -> pd.DataFrame:
    fault = pd.read_csv(FAULT_V1)
    for column in ["xid_time_raw", "fault_time_onset", "fault_time_mixed", "end_time"]:
        fault[column] = pd.to_datetime(fault[column], errors="coerce", utc=True)
    return fault


def sorted_events(fault: pd.DataFrame, time_column: str) -> list[tuple[int, int, int]]:
    gpu_ids = XID_FILE.open("r", encoding="utf-8").readline().rstrip("\r\n").split(",")[1:]
    gpu_to_column = {gpu_id: index for index, gpu_id in enumerate(gpu_ids)}
    rows = []
    for item in fault[["event_id", "gpu_id", time_column]].itertuples(index=False):
        rows.append((int(getattr(item, time_column).value), int(item.event_id), gpu_to_column[item.gpu_id]))
    return sorted(rows)


def event_history(
    event_ns: int,
    column: int,
    tail_times: np.ndarray,
    tail_values: np.ndarray,
    batch_times: np.ndarray,
    batch_values: np.ndarray,
    position: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_start = max(0, position - PRE_EVENT_ROWS)
    times = np.r_[tail_times, batch_times[current_start:position]]
    values = np.r_[tail_values[:, column], batch_values[current_start:position, column]]
    keep = (times >= event_ns - 30 * 60 * 1_000_000_000) & (times < event_ns)
    return times[keep], values[keep]


def scan() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    fault = read_fault()
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    first = pd.to_datetime(inventory["first_time"], utc=True)
    last = pd.to_datetime(inventory["last_time"], utc=True)
    first_ns, last_ns = int(first.value), int(last.value)
    total_steps = int((last_ns - first_ns) // (STEP_SECONDS * 1_000_000_000) + 1)
    n_bins = int(np.ceil(total_steps / ROWS_PER_BIN))

    header = XID_FILE.open("r", encoding="utf-8").readline().rstrip("\r\n").split(",")
    gpu_ids = header[1:]
    n_gpu = len(gpu_ids)
    types = {header[0]: pa.string(), **{gpu_id: pa.float32() for gpu_id in gpu_ids}}
    reader = pacsv.open_csv(
        XID_FILE,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True),
        convert_options=pacsv.ConvertOptions(
            column_types=types,
            null_values=[""],
            strings_can_be_null=True,
        ),
    )

    valid_counts = np.zeros((n_bins, n_gpu), dtype=np.uint8)
    expected_counts = np.full(n_bins, ROWS_PER_BIN, dtype=np.uint8)
    expected_counts[-1] = total_steps - (n_bins - 1) * ROWS_PER_BIN
    bin_start_ns = first_ns + np.arange(n_bins, dtype=np.int64) * BIN_SECONDS * 1_000_000_000

    start_events = sorted_events(fault, "xid_time_raw")
    end_events = sorted_events(fault, "end_time")
    start_pointer = end_pointer = 0
    pending_end: list[tuple[int, int]] = []
    event_meta: dict[int, dict] = {int(event_id): {} for event_id in fault["event_id"]}
    code_by_event = fault.set_index("event_id")["xid_code"].astype(int).to_dict()

    tail_times = np.array([], dtype=np.int64)
    tail_values = np.empty((0, n_gpu), dtype=np.float32)
    previous_time_ns = -1
    global_gaps: list[tuple[int, int, int]] = []
    rows_read = 0

    for batch_number, batch in enumerate(reader, start=1):
        frame = batch.to_pandas(split_blocks=True)
        times = pd.to_datetime(frame.iloc[:, 0], utc=True)
        time_ns = times.astype("int64").to_numpy()
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
        valid = np.isfinite(values)

        bin_index = ((time_ns - first_ns) // (BIN_SECONDS * 1_000_000_000)).astype(np.int64)
        boundaries = np.r_[0, np.flatnonzero(bin_index[1:] != bin_index[:-1]) + 1, len(bin_index)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            valid_counts[bin_index[start]] += valid[start:end].sum(axis=0).astype(np.uint8)

        if previous_time_ns >= 0 and time_ns[0] - previous_time_ns > STEP_SECONDS * 1_000_000_000:
            missing = int((time_ns[0] - previous_time_ns) // (STEP_SECONDS * 1_000_000_000) - 1)
            global_gaps.append((previous_time_ns + STEP_SECONDS * 1_000_000_000, int(time_ns[0] - STEP_SECONDS * 1_000_000_000), missing))
        diffs = np.diff(time_ns)
        for position in np.flatnonzero(diffs > STEP_SECONDS * 1_000_000_000):
            missing = int(diffs[position] // (STEP_SECONDS * 1_000_000_000) - 1)
            global_gaps.append((int(time_ns[position] + STEP_SECONDS * 1_000_000_000), int(time_ns[position + 1] - STEP_SECONDS * 1_000_000_000), missing))

        for event_id, column in pending_end:
            gap = previous_time_ns < 0 or time_ns[0] - previous_time_ns > STEP_SECONDS * 1_000_000_000
            next_value = values[0, column]
            event_meta[event_id].update(
                {
                    "next_cell_time": pd.Timestamp(time_ns[0], tz="UTC"),
                    "next_cell_is_null": bool(not np.isfinite(next_value)),
                    "next_observed_xid": float(next_value) if np.isfinite(next_value) else np.nan,
                    "end_right_censored": bool(gap or not np.isfinite(next_value)),
                }
            )
        pending_end = []

        while start_pointer < len(start_events) and start_events[start_pointer][0] <= time_ns[-1]:
            event_ns, event_id, column = start_events[start_pointer]
            start_pointer += 1
            if event_ns < time_ns[0]:
                continue
            position = int(np.searchsorted(time_ns, event_ns))
            if position >= len(time_ns) or time_ns[position] != event_ns:
                continue
            if position > 0:
                previous_cell_time = int(time_ns[position - 1])
                previous_cell_value = values[position - 1, column]
            elif len(tail_times):
                previous_cell_time = int(tail_times[-1])
                previous_cell_value = tail_values[-1, column]
            else:
                previous_cell_time = -1
                previous_cell_value = np.nan

            history_times, history_values = event_history(
                event_ns, column, tail_times, tail_values, time_ns, values, position
            )
            expected = np.zeros(PRE_EVENT_ROWS, dtype=bool)
            if len(history_times):
                grid_position = ((history_times - (event_ns - 30 * 60 * 1_000_000_000)) // (STEP_SECONDS * 1_000_000_000)).astype(int)
                inside = (grid_position >= 0) & (grid_position < PRE_EVENT_ROWS)
                expected[grid_position[inside]] = np.isfinite(history_values[inside])
            observed_positions = np.flatnonzero(np.isfinite(history_values))
            zero_positions = np.flatnonzero(np.isfinite(history_values) & (history_values == 0))
            previous_observation_time = history_times[observed_positions[-1]] if len(observed_positions) else -1
            previous_observed_xid = history_values[observed_positions[-1]] if len(observed_positions) else np.nan
            last_zero_time = history_times[zero_positions[-1]] if len(zero_positions) else -1
            gap = previous_cell_time < 0 or event_ns - previous_cell_time > STEP_SECONDS * 1_000_000_000
            event_meta[event_id].update(
                {
                    "previous_cell_time": pd.Timestamp(previous_cell_time, tz="UTC") if previous_cell_time >= 0 else pd.NaT,
                    "previous_cell_is_null": bool(not np.isfinite(previous_cell_value)),
                    "previous_observation_time": pd.Timestamp(previous_observation_time, tz="UTC") if previous_observation_time >= 0 else pd.NaT,
                    "previous_observed_xid": float(previous_observed_xid) if np.isfinite(previous_observed_xid) else np.nan,
                    "last_observed_zero_time": pd.Timestamp(last_zero_time, tz="UTC") if last_zero_time >= 0 else pd.NaT,
                    "onset_left_censored": bool(gap or not np.isfinite(previous_cell_value)),
                    "early_bound_available": bool(last_zero_time >= 0),
                    "start_value_matches_xid_code": bool(
                        np.isfinite(values[position, column])
                        and int(values[position, column]) == code_by_event[event_id]
                    ),
                    "xid_valid_fraction_pre30m": float(expected.mean()),
                    "max_missing_run_pre30m_sec": int(max_false_run(expected) * STEP_SECONDS),
                }
            )

        while end_pointer < len(end_events) and end_events[end_pointer][0] <= time_ns[-1]:
            event_ns, event_id, column = end_events[end_pointer]
            end_pointer += 1
            if event_ns < time_ns[0]:
                continue
            position = int(np.searchsorted(time_ns, event_ns))
            if position >= len(time_ns) or time_ns[position] != event_ns:
                continue
            if position + 1 < len(time_ns):
                next_value = values[position + 1, column]
                gap = time_ns[position + 1] - time_ns[position] > STEP_SECONDS * 1_000_000_000
                event_meta[event_id].update(
                    {
                        "next_cell_time": pd.Timestamp(time_ns[position + 1], tz="UTC"),
                        "next_cell_is_null": bool(not np.isfinite(next_value)),
                        "next_observed_xid": float(next_value) if np.isfinite(next_value) else np.nan,
                        "end_right_censored": bool(gap or not np.isfinite(next_value)),
                    }
                )
            else:
                pending_end.append((event_id, column))

        history_times = np.r_[tail_times, time_ns]
        history_values = np.vstack([tail_values, values])
        tail_times = history_times[-PRE_EVENT_ROWS:].copy()
        tail_values = history_values[-PRE_EVENT_ROWS:].copy()
        previous_time_ns = int(time_ns[-1])
        rows_read += len(time_ns)
        if batch_number % 4 == 0:
            print(f"[xid-compact] batches={batch_number}, rows={rows_read:,}", flush=True)

    for event_id, _ in pending_end:
        event_meta[event_id].update(
            {
                "next_cell_time": pd.NaT,
                "next_cell_is_null": True,
                "next_observed_xid": np.nan,
                "end_right_censored": True,
            }
        )

    metadata = pd.DataFrame.from_dict(event_meta, orient="index").rename_axis("event_id").reset_index()
    global_frame = pd.DataFrame(global_gaps, columns=["missing_start_ns", "missing_end_ns", "missing_rows"])
    global_frame["missing_start"] = pd.to_datetime(global_frame["missing_start_ns"], utc=True)
    global_frame["missing_end"] = pd.to_datetime(global_frame["missing_end_ns"], utc=True)
    global_frame["missing_duration_sec"] = global_frame["missing_rows"] * STEP_SECONDS
    global_frame.drop(columns=["missing_start_ns", "missing_end_ns"]).to_csv(
        OUTPUT / "xid_global_time_gaps.csv", index=False
    )
    np.savez_compressed(
        OUTPUT / "xid_observability_5m.npz",
        valid_counts=valid_counts,
        expected_counts=expected_counts,
        bin_start_ns=bin_start_ns,
        gpu_ids=np.asarray(gpu_ids),
    )
    scan_meta = {
        "first_time": first,
        "last_time": last,
        "rows": rows_read,
        "gpu_columns": n_gpu,
        "n_bins": n_bins,
    }
    return valid_counts, expected_counts, bin_start_ns, np.asarray(gpu_ids), global_frame, metadata, scan_meta


def build_observability_summary(
    valid_counts: np.ndarray, expected_counts: np.ndarray, gpu_ids: np.ndarray
) -> pd.DataFrame:
    expected_total = int(expected_counts.sum())
    coverage = valid_counts / expected_counts[:, None]
    rows = []
    for column, gpu_id in enumerate(gpu_ids):
        any_missing = valid_counts[:, column] < expected_counts
        zero_coverage = valid_counts[:, column] == 0
        rows.append(
            {
                "gpu_id": gpu_id,
                "node_id": str(gpu_id).rsplit("-", 1)[0],
                "valid_observations": int(valid_counts[:, column].sum()),
                "expected_observations": expected_total,
                "xid_missing_rate": 1 - valid_counts[:, column].sum() / expected_total,
                "bins_with_any_missing_fraction": any_missing.mean(),
                "max_any_missing_run_min": max_false_run(~any_missing) * 5,
                "max_zero_coverage_run_min": max_false_run(~zero_coverage) * 5,
                "median_5m_coverage": float(np.median(coverage[:, column])),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "xid_observability_gpu_summary.csv", index=False)
    return result


def build_fault_v2(fault: pd.DataFrame, event_meta: pd.DataFrame) -> pd.DataFrame:
    result = fault.merge(event_meta, on="event_id", how="left")
    result["fault_time_early"] = result["xid_time_raw"]
    use_early = result["onset_left_censored"].fillna(True) & result["early_bound_available"].fillna(False)
    result.loc[use_early, "fault_time_early"] = (
        result.loc[use_early, "last_observed_zero_time"] + pd.Timedelta(seconds=STEP_SECONDS)
    )
    result["onset_uncertainty_sec"] = (
        result["xid_time_raw"] - result["fault_time_early"]
    ).dt.total_seconds().clip(lower=0)
    result["xid_observability_status"] = np.select(
        [
            result["xid_valid_fraction_pre30m"].ge(0.99)
            & result["max_missing_run_pre30m_sec"].le(60),
            result["xid_valid_fraction_pre30m"].ge(0.95),
        ],
        ["high", "moderate"],
        default="low",
    )
    result.to_csv(OUTPUT / "fault_tape_v2.csv", index=False)
    return result


def build_risk_labels(
    fault: pd.DataFrame,
    valid_counts: np.ndarray,
    expected_counts: np.ndarray,
    first: pd.Timestamp,
    last: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for item in fault[fault["xid_code"].isin([31, 43])].itertuples():
        for horizon_min in HORIZONS_MIN:
            horizon = pd.Timedelta(minutes=horizon_min)
            rows.append(
                {
                    "event_id": item.event_id,
                    "gpu_id": item.gpu_id,
                    "node_id": item.node_id,
                    "xid_code": item.xid_code,
                    "horizon_min": horizon_min,
                    "label_time_raw": item.xid_time_raw,
                    "positive_start_raw": max(first, item.xid_time_raw - horizon),
                    "label_time_mixed": item.fault_time_mixed,
                    "positive_start_mixed": max(first, item.fault_time_mixed - horizon),
                    "label_time_early": item.fault_time_early,
                    "positive_start_early": max(first, item.fault_time_early - horizon),
                    "onset_left_censored": item.onset_left_censored,
                    "early_bound_available": item.early_bound_available,
                    "onset_uncertainty_sec": item.onset_uncertainty_sec,
                    "xid_valid_fraction_pre30m": item.xid_valid_fraction_pre30m,
                    "xid_observability_status": item.xid_observability_status,
                }
            )
    positive = pd.DataFrame(rows)
    positive.to_csv(OUTPUT / "risk_positive_intervals.csv", index=False)

    strict_bad = valid_counts < expected_counts[:, None]
    relaxed_bad = valid_counts / expected_counts[:, None] < 0.95
    summary_rows = []
    for horizon_min in HORIZONS_MIN:
        horizon_bins = int(np.ceil(horizon_min / 5))
        strict_unknown = future_unknown_fraction(strict_bad, horizon_bins)
        relaxed_unknown = future_unknown_fraction(relaxed_bad, horizon_bins)
        for code in [31, 43]:
            group = positive[(positive["horizon_min"] == horizon_min) & (positive["xid_code"] == code)]
            summary_rows.append(
                {
                    "horizon_min": horizon_min,
                    "xid_code": code,
                    "positive_episode_intervals": len(group),
                    "left_censored_fraction": group["onset_left_censored"].mean(),
                    "early_bound_available_fraction": group["early_bound_available"].mean(),
                    "high_observability_fraction": group["xid_observability_status"].eq("high").mean(),
                    "strict_unknown_gpu_time_fraction": strict_unknown,
                    "relaxed_unknown_gpu_time_fraction": relaxed_unknown,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "risk_label_readiness_summary.csv", index=False)
    rules = {
        "grid_minutes": 5,
        "horizons_minutes": HORIZONS_MIN,
        "label_precedence": ["observed_positive", "unknown", "negative"],
        "strict_unknown": "any missing XID observation in the future horizon",
        "relaxed_unknown": "any 5-minute bin below 95% XID coverage in the future horizon",
        "negative_definition": "no positive interval and no unknown interval",
        "dense_decision_grid_materialized": False,
    }
    (OUTPUT / "risk_label_rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
    return positive, summary


def write_report(
    observability: pd.DataFrame,
    gaps: pd.DataFrame,
    fault: pd.DataFrame,
    positive: pd.DataFrame,
    label_summary: pd.DataFrame,
    meta: dict,
) -> None:
    lines = [
        "# XID Observability·Fault Tape v2·Risk Label 준비 결과",
        "",
        "## 판정",
        "",
        "- XID Observability 5분 Tape: **READY**",
        "- Fault Tape v2: **READY WITH SCENARIOS**",
        "- Compact Risk positive interval: **READY**",
        "- Dense Risk Tape: **미생성** — 다음 단계에서 시간순 out-of-fold 모델로 생성",
        "",
        "## XID 관측성",
        "",
        f"- 원본 행 {meta['rows']:,}, GPU {meta['gpu_columns']:,}, 5분 bin {meta['n_bins']:,}",
        f"- 전체 GPU 공통 timestamp gap {len(gaps):,}개",
        f"- GPU별 XID 결측률 중앙값 {observability['xid_missing_rate'].median():.2%}, p95 {observability['xid_missing_rate'].quantile(.95):.2%}, 최대 {observability['xid_missing_rate'].max():.2%}",
        f"- 결측률 5% 초과 GPU {observability['xid_missing_rate'].gt(.05).sum():,}/{len(observability):,}",
        "",
        "XID null은 0으로 채우지 않았다. 5분 bin마다 20개 예상 관측 중 실제 valid count를 저장했다.",
        "",
        "## Fault Tape v2",
        "",
        f"- episode {len(fault):,}개",
        f"- onset left-censored {fault['onset_left_censored'].fillna(True).sum():,}개",
        f"- end right-censored {fault['end_right_censored'].fillna(True).sum():,}개",
        f"- left-censored 중 30분 내 early bound 확보 {((fault['onset_left_censored'].fillna(True)) & fault['early_bound_available'].fillna(False)).sum():,}개",
        f"- 사건 전 30분 high observability {fault['xid_observability_status'].eq('high').mean():.1%}",
        "",
        "fault_time_raw·fault_time_early·fault_time_mixed를 모두 보존했다. early bound가 없는 left-censored 사건은 raw 시각만 사용하고 별도 flag로 제한한다.",
        "",
        "## Risk label",
        "",
        f"- positive interval {len(positive):,}행: XID 31/43 × 5분·12/24/72시간",
        "- dense decision×GPU 테이블은 만들지 않았다. Negative는 positive와 unknown 구간의 complement에서 샘플링한다.",
        "- precedence: observed positive > unknown > negative",
        "",
    ]
    for row in label_summary.itertuples():
        lines.append(
            f"- {int(row.horizon_min)}분 / XID {int(row.xid_code)}: positive {int(row.positive_episode_intervals):,}, left-censored {row.left_censored_fraction:.1%}, high-observability {row.high_observability_fraction:.1%}, strict unknown GPU-time {row.strict_unknown_gpu_time_fraction:.1%}, relaxed unknown {row.relaxed_unknown_gpu_time_fraction:.1%}"
        )
    lines += [
        "",
        "## 다음 단계",
        "",
        "1. 5분 observability matrix와 positive interval에서 decision samples 생성",
        "2. XID 31/43·horizon별 시간순 fold 모델 학습",
        "3. out-of-fold risk_score·risk_rank·observability_score 생성",
        "4. Risk Tape audit 후 Blox no-fault baseline 실행",
    ]
    (OUTPUT / "XID_OBSERVABILITY_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def make_plot(observability: pd.DataFrame, label_summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(observability["xid_missing_rate"], bins=40)
    axes[0].set_title("GPU-level XID missing rate")
    axes[0].set_xlabel("Missing fraction")
    axes[1].hist(np.log10(observability["max_zero_coverage_run_min"].clip(lower=5)), bins=40)
    axes[1].set_title("Maximum zero-coverage run")
    axes[1].set_xlabel("log10(minutes)")
    horizon = label_summary.drop_duplicates("horizon_min")
    axes[2].plot(horizon["horizon_min"] / 60, horizon["strict_unknown_gpu_time_fraction"], marker="o", label="strict")
    axes[2].plot(horizon["horizon_min"] / 60, horizon["relaxed_unknown_gpu_time_fraction"], marker="o", label="relaxed")
    axes[2].set_xscale("log")
    axes[2].set_title("Unknown exposure by horizon")
    axes[2].set_xlabel("Horizon (hours)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "xid_observability_profile.png", dpi=160)
    plt.close(fig)


def self_check(
    valid_counts: np.ndarray,
    expected_counts: np.ndarray,
    observability: pd.DataFrame,
    fault: pd.DataFrame,
    positive: pd.DataFrame,
) -> None:
    assert valid_counts.shape[1] == 1_992
    assert valid_counts.shape[0] == len(expected_counts)
    assert (valid_counts <= expected_counts[:, None]).all()
    assert observability["xid_missing_rate"].between(0, 1).all()
    assert fault["event_id"].is_unique and len(fault) == 2_433
    assert fault["fault_time_early"].le(fault["xid_time_raw"]).all()
    assert set(positive["xid_code"]) == {31, 43}
    assert set(positive["horizon_min"]) == set(HORIZONS_MIN)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    valid, expected, bin_start, gpu_ids, gaps, event_meta, meta = scan()
    observability = build_observability_summary(valid, expected, gpu_ids)
    fault = build_fault_v2(read_fault(), event_meta)
    positive, label_summary = build_risk_labels(
        fault, valid, expected, meta["first_time"], meta["last_time"]
    )
    write_report(observability, gaps, fault, positive, label_summary, meta)
    make_plot(observability, label_summary)
    self_check(valid, expected, observability, fault, positive)
    output_meta = {
        **{key: value.isoformat() if isinstance(value, pd.Timestamp) else value for key, value in meta.items()},
        "global_time_gaps": len(gaps),
        "fault_v2_rows": len(fault),
        "risk_positive_interval_rows": len(positive),
        "observability_matrix_shape": list(valid.shape),
    }
    (OUTPUT / "analysis_metadata.json").write_text(json.dumps(output_meta, indent=2), encoding="utf-8")
    print(json.dumps(output_meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

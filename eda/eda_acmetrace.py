from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "eda_phase1"
WIDE_FILES = {
    "dram_active": ROOT / "DRAM_ACTIVE.csv",
    "fb_used": ROOT / "FB_USED.csv",
    "gpu_temp": ROOT / "GPU_TEMP.csv",
    "gpu_util": ROOT / "GPU_UTIL.csv",
    "power_usage": ROOT / "POWER_USAGE.csv",
}
XID_FILE = ROOT / "XID_ERRORS.csv"
TRACE_FILE = ROOT / "trace_seren.csv"
EXPECTED_STEP_SECONDS = 15


def gpu_parts(gpu_id: str) -> tuple[str, int]:
    node, index = gpu_id.rsplit("-", 1)
    return node, int(index)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.readline().rstrip("\r\n").split(",")


def parse_sample(payload: bytes, expected: int) -> np.ndarray | None:
    fields = payload.rstrip(b"\r\n").split(b",")
    if len(fields) != expected:
        return None
    return np.fromiter(
        (float(value) if value else np.nan for value in fields),
        dtype=np.float32,
        count=expected,
    )


def scan_wide(name: str, path: Path, stride: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    headers = read_header(path)
    gpu_ids = headers[1:]
    expected_commas = len(headers) - 1
    sampled_values: list[np.ndarray] = []
    sampled_times: list[datetime] = []
    rows = malformed = duplicates = out_of_order = irregular = 0
    max_gap = 0.0
    first_time = last_time = previous = None

    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        handle.readline()
        for row_index, line in enumerate(handle):
            rows += 1
            raw_time, separator, payload = line.partition(b",")
            if not separator or line.count(b",") != expected_commas:
                malformed += 1
                continue
            timestamp = datetime.fromisoformat(raw_time.decode("ascii"))
            first_time = first_time or timestamp
            last_time = timestamp
            if previous is not None:
                gap = (timestamp - previous).total_seconds()
                duplicates += gap == 0
                out_of_order += gap < 0
                irregular += gap != EXPECTED_STEP_SECONDS
                max_gap = max(max_gap, gap)
            previous = timestamp
            if row_index % stride == 0:
                values = parse_sample(payload, len(gpu_ids))
                if values is None:
                    malformed += 1
                else:
                    sampled_times.append(timestamp)
                    sampled_values.append(values)

    if not sampled_values or first_time is None or last_time is None:
        raise ValueError(f"No usable rows: {path}")

    matrix = np.vstack(sampled_values)
    valid = np.isfinite(matrix)
    valid_count = valid.sum(axis=0)
    zero_count = np.sum((matrix == 0) & valid, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        quantiles = np.nanpercentile(matrix, [50, 95, 99], axis=0)
        means = np.nanmean(matrix, axis=0)
        stds = np.nanstd(matrix, axis=0)
        minimums = np.nanmin(matrix, axis=0)
        maximums = np.nanmax(matrix, axis=0)

    gpu_rows = []
    for index, gpu_id in enumerate(gpu_ids):
        node, gpu_index = gpu_parts(gpu_id)
        gpu_rows.append(
            {
                "metric": name,
                "gpu_id": gpu_id,
                "node_id": node,
                "gpu_index": gpu_index,
                "sample_count": len(sampled_values),
                "valid_count": int(valid_count[index]),
                "missing_rate": float(1 - valid_count[index] / len(sampled_values)),
                "zero_rate_among_valid": float(zero_count[index] / valid_count[index])
                if valid_count[index]
                else np.nan,
                "mean": float(means[index]),
                "std": float(stds[index]),
                "min": float(minimums[index]),
                "p50": float(quantiles[0, index]),
                "p95": float(quantiles[1, index]),
                "p99": float(quantiles[2, index]),
                "max": float(maximums[index]),
            }
        )

    time_rows = []
    for timestamp, values in zip(sampled_times, matrix, strict=True):
        observed = values[np.isfinite(values)]
        time_rows.append(
            {
                "metric": name,
                "time": timestamp.isoformat(),
                "observed_gpu_count": int(observed.size),
                "mean": float(np.mean(observed)) if observed.size else np.nan,
                "p50": float(np.percentile(observed, 50)) if observed.size else np.nan,
                "p95": float(np.percentile(observed, 95)) if observed.size else np.nan,
                "p99": float(np.percentile(observed, 99)) if observed.size else np.nan,
            }
        )

    expected_rows = int((last_time - first_time).total_seconds() / EXPECTED_STEP_SECONDS) + 1
    inventory = {
        "dataset": name,
        "file": path.name,
        "size_gib": round(path.stat().st_size / 1024**3, 3),
        "columns": len(headers),
        "gpu_columns": len(gpu_ids),
        "rows": rows,
        "expected_rows_at_15s": expected_rows,
        "first_time": first_time.isoformat(),
        "last_time": last_time.isoformat(),
        "duplicate_steps": duplicates,
        "out_of_order_steps": out_of_order,
        "irregular_steps": irregular,
        "max_gap_seconds": max_gap,
        "malformed_rows": malformed,
        "sample_stride_rows": stride,
        "sample_rows": len(sampled_values),
        "sample_missing_rate": float(1 - valid.sum() / matrix.size),
        "sample_zero_rate_among_valid": float(np.sum((matrix == 0) & valid) / valid.sum()),
    }
    return inventory, pd.DataFrame(gpu_rows), pd.DataFrame(time_rows)


def telemetry_stage(stride: int, force: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_rows = []
    gpu_frames = []
    time_frames = []
    for name, path in WIDE_FILES.items():
        stem = OUTPUT / f"telemetry_{name}"
        paths = [stem.with_suffix(".json"), Path(f"{stem}_gpu.csv"), Path(f"{stem}_time.csv")]
        if not force and all(item.exists() for item in paths):
            print(f"[telemetry] cached: {name}", flush=True)
            inventory_rows.append(json.loads(paths[0].read_text(encoding="utf-8")))
            gpu_frames.append(pd.read_csv(paths[1]))
            time_frames.append(pd.read_csv(paths[2]))
            continue
        print(f"[telemetry] scanning: {path.name}", flush=True)
        inventory, gpu_summary, time_summary = scan_wide(name, path, stride)
        paths[0].write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        gpu_summary.to_csv(paths[1], index=False)
        time_summary.to_csv(paths[2], index=False)
        inventory_rows.append(inventory)
        gpu_frames.append(gpu_summary)
        time_frames.append(time_summary)

    inventory = pd.DataFrame(inventory_rows)
    gpu_summary = pd.concat(gpu_frames, ignore_index=True)
    time_summary = pd.concat(time_frames, ignore_index=True)
    inventory.to_csv(OUTPUT / "telemetry_inventory.csv", index=False)
    gpu_summary.to_csv(OUTPUT / "telemetry_gpu_summary.csv", index=False)
    time_summary.to_csv(OUTPUT / "telemetry_timeseries_sample.csv", index=False)
    return inventory, gpu_summary, time_summary


def xid_scan() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    headers = read_header(XID_FILE)
    types = {headers[0]: pa.string(), **{name: pa.float32() for name in headers[1:]}}
    reader = pacsv.open_csv(
        XID_FILE,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True),
        parse_options=pacsv.ParseOptions(delimiter=","),
        convert_options=pacsv.ConvertOptions(
            column_types=types,
            null_values=[""],
            strings_can_be_null=True,
        ),
    )
    events: list[tuple[str, str, float]] = []
    missing = np.zeros(len(headers) - 1, dtype=np.int64)
    rows = batches = 0
    first_time = last_time = None

    for batch in reader:
        batches += 1
        batch_rows = batch.num_rows
        rows += batch_rows
        times = batch.column(0).to_pylist()
        first_time = first_time or times[0]
        last_time = times[-1]
        for column_index, gpu_id in enumerate(headers[1:], start=1):
            column = batch.column(column_index)
            missing[column_index - 1] += column.null_count
            mask = pc.fill_null(pc.not_equal(column, 0), False)
            positions = pc.indices_nonzero(mask).to_numpy(zero_copy_only=False)
            if positions.size:
                values = pc.take(column, pa.array(positions)).to_numpy(zero_copy_only=False)
                events.extend((times[int(pos)], gpu_id, float(value)) for pos, value in zip(positions, values))
        if batches % 10 == 0:
            print(f"[xid] batches={batches}, rows={rows:,}, observations={len(events):,}", flush=True)

    frame = pd.DataFrame(events, columns=["time", "gpu_id", "xid_code"])
    if not frame.empty:
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frame["xid_code"] = frame["xid_code"].astype("Int64")
        frame[["node_id", "gpu_index"]] = frame["gpu_id"].apply(
            lambda value: pd.Series(gpu_parts(value))
        )
        frame = frame.sort_values(["gpu_id", "xid_code", "time"], ignore_index=True)
        gap = frame.groupby(["gpu_id", "xid_code"])["time"].diff().dt.total_seconds()
        frame["new_episode"] = gap.isna() | (gap > 30)
        frame["episode_id"] = frame.groupby(["gpu_id", "xid_code"])["new_episode"].cumsum()
        episodes = (
            frame.groupby(["gpu_id", "node_id", "gpu_index", "xid_code", "episode_id"], as_index=False)
            .agg(start_time=("time", "min"), end_time=("time", "max"), observations=("time", "size"))
        )
        episodes["duration_seconds"] = (
            episodes["end_time"] - episodes["start_time"]
        ).dt.total_seconds()
        frame = frame.drop(columns=["new_episode", "episode_id"])
    else:
        episodes = pd.DataFrame(
            columns=[
                "gpu_id",
                "node_id",
                "gpu_index",
                "xid_code",
                "episode_id",
                "start_time",
                "end_time",
                "observations",
                "duration_seconds",
            ]
        )

    missing_frame = pd.DataFrame(
        {
            "gpu_id": headers[1:],
            "missing_count": missing,
            "missing_rate": missing / rows,
        }
    )
    inventory = {
        "dataset": "xid_errors",
        "file": XID_FILE.name,
        "size_gib": round(XID_FILE.stat().st_size / 1024**3, 3),
        "columns": len(headers),
        "gpu_columns": len(headers) - 1,
        "rows": rows,
        "first_time": first_time,
        "last_time": last_time,
        "nonzero_observations": len(frame),
        "episodes": len(episodes),
        "affected_gpus": int(frame["gpu_id"].nunique()) if not frame.empty else 0,
        "affected_nodes": int(frame["node_id"].nunique()) if not frame.empty else 0,
    }
    return frame, episodes, missing_frame, inventory


def merge_interval_bins(intervals: list[tuple[pd.Timestamp, pd.Timestamp]], step_seconds: int = 15) -> int:
    if not intervals:
        return 0
    intervals.sort()
    start, end = intervals[0]
    total_seconds = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total_seconds += (end - start).total_seconds()
            start, end = next_start, next_end
    total_seconds += (end - start).total_seconds()
    return int(round(total_seconds / step_seconds))


def horizon_balance(
    episodes: pd.DataFrame,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    rows: int,
    gpu_count: int,
) -> pd.DataFrame:
    results = []
    for label, horizon in {
        "5m": timedelta(minutes=5),
        "12h": timedelta(hours=12),
        "24h": timedelta(hours=24),
        "72h": timedelta(hours=72),
    }.items():
        positive = 0
        for _, group in episodes.groupby("gpu_id"):
            intervals = []
            for event_time in group["start_time"]:
                start = max(data_start, event_time - horizon)
                end = min(data_end + timedelta(seconds=EXPECTED_STEP_SECONDS), event_time)
                if start < end:
                    intervals.append((start, end))
            positive += merge_interval_bins(intervals)
        total = rows * gpu_count
        results.append(
            {
                "horizon": label,
                "positive_gpu_time_rows": positive,
                "total_gpu_time_rows": total,
                "positive_rate": positive / total if total else np.nan,
                "negative_to_positive_ratio": (total - positive) / positive if positive else np.inf,
            }
        )
    return pd.DataFrame(results)


def xid_stage(force: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    event_path = OUTPUT / "xid_event_observations.csv"
    episode_path = OUTPUT / "xid_event_episodes.csv"
    missing_path = OUTPUT / "xid_missingness.csv"
    inventory_path = OUTPUT / "xid_inventory.json"
    if not force and all(path.exists() for path in [event_path, episode_path, missing_path, inventory_path]):
        print("[xid] cached", flush=True)
        events = pd.read_csv(event_path, parse_dates=["time"])
        episodes = pd.read_csv(episode_path, parse_dates=["start_time", "end_time"])
        missing = pd.read_csv(missing_path)
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        print(f"[xid] scanning: {XID_FILE.name}", flush=True)
        events, episodes, missing, inventory = xid_scan()
        events.to_csv(event_path, index=False)
        episodes.to_csv(episode_path, index=False)
        missing.to_csv(missing_path, index=False)
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    if events.empty:
        code_summary = pd.DataFrame(
            columns=["xid_code", "observations", "affected_gpus", "affected_nodes", "first_time", "last_time"]
        )
    else:
        code_summary = (
            events.groupby("xid_code", as_index=False)
            .agg(
                observations=("time", "size"),
                affected_gpus=("gpu_id", "nunique"),
                affected_nodes=("node_id", "nunique"),
                first_time=("time", "min"),
                last_time=("time", "max"),
            )
            .sort_values("observations", ascending=False)
        )
    code_summary.to_csv(OUTPUT / "xid_code_summary.csv", index=False)
    balance = horizon_balance(
        episodes,
        pd.Timestamp(inventory["first_time"]),
        pd.Timestamp(inventory["last_time"]),
        int(inventory["rows"]),
        int(inventory["gpu_columns"]),
    )
    balance.to_csv(OUTPUT / "risk_horizon_balance.csv", index=False)
    return events, episodes, code_summary, inventory, balance


def trace_stage() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    print(f"[trace] reading: {TRACE_FILE.name}", flush=True)
    trace = pd.read_csv(TRACE_FILE)
    for column in ["submit_time", "start_time", "end_time"]:
        trace[column] = pd.to_datetime(trace[column], errors="coerce", utc=True)
    trace["wait_seconds"] = (trace["start_time"] - trace["submit_time"]).dt.total_seconds()
    trace["runtime_seconds"] = (trace["end_time"] - trace["start_time"]).dt.total_seconds()
    trace["duration_difference_seconds"] = trace["duration"] - trace["runtime_seconds"]
    trace["gpu_time_expected"] = trace["duration"] * trace["gpu_num"]
    trace["gpu_time_difference"] = trace["gpu_time"] - trace["gpu_time_expected"]
    trace["submit_date"] = trace["submit_time"].dt.floor("D")

    quality = {
        "dataset": "trace_seren",
        "file": TRACE_FILE.name,
        "size_gib": round(TRACE_FILE.stat().st_size / 1024**3, 3),
        "rows": len(trace),
        "unique_jobs": int(trace["job_id"].nunique()),
        "duplicate_job_ids": int(trace["job_id"].duplicated().sum()),
        "first_submit_time": trace["submit_time"].min().isoformat(),
        "last_end_time": trace["end_time"].max().isoformat(),
        "missing_start_time": int(trace["start_time"].isna().sum()),
        "missing_end_time": int(trace["end_time"].isna().sum()),
        "negative_wait": int((trace["wait_seconds"] < 0).sum()),
        "negative_runtime": int((trace["runtime_seconds"] < 0).sum()),
        "duration_mismatch_over_1s": int((trace["duration_difference_seconds"].abs() > 1).sum()),
        "gpu_time_mismatch_over_1s": int((trace["gpu_time_difference"].abs() > 1).sum()),
        "states": int(trace["state"].nunique()),
        "types": int(trace["type"].nunique()),
        "queues": int(trace["queue"].nunique()),
    }
    daily = (
        trace.groupby("submit_date", as_index=False)
        .agg(
            submitted_jobs=("job_id", "size"),
            submitted_gpus=("gpu_num", "sum"),
            median_wait_seconds=("wait_seconds", "median"),
            gpu_hours=("gpu_time", lambda values: values.sum() / 3600),
            failed_jobs=("state", lambda values: (values == "FAILED").sum()),
        )
    )
    daily["failed_rate"] = daily["failed_jobs"] / daily["submitted_jobs"]
    state_counts = trace["state"].value_counts(dropna=False).rename_axis("state").reset_index(name="jobs")
    type_counts = trace["type"].value_counts(dropna=False).rename_axis("type").reset_index(name="jobs")
    queue_counts = trace["queue"].value_counts(dropna=False).rename_axis("queue").reset_index(name="jobs")
    numeric = trace[
        ["node_num", "gpu_num", "cpu_num", "duration", "gpu_time", "wait_seconds", "runtime_seconds"]
    ].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).T

    Path(OUTPUT / "trace_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    daily.to_csv(OUTPUT / "trace_daily.csv", index=False)
    state_counts.to_csv(OUTPUT / "trace_state_counts.csv", index=False)
    type_counts.to_csv(OUTPUT / "trace_type_counts.csv", index=False)
    queue_counts.to_csv(OUTPUT / "trace_queue_counts.csv", index=False)
    numeric.to_csv(OUTPUT / "trace_numeric_summary.csv")
    return trace, quality, daily


def header_stage() -> pd.DataFrame:
    reference = set(read_header(WIDE_FILES["gpu_util"])[1:])
    rows = []
    for name, path in {**WIDE_FILES, "xid_errors": XID_FILE}.items():
        current = set(read_header(path)[1:])
        for gpu_id in sorted(reference - current):
            node, gpu_index = gpu_parts(gpu_id)
            rows.append(
                {"dataset": name, "difference": "missing_vs_gpu_util", "gpu_id": gpu_id, "node_id": node, "gpu_index": gpu_index}
            )
        for gpu_id in sorted(current - reference):
            node, gpu_index = gpu_parts(gpu_id)
            rows.append(
                {"dataset": name, "difference": "extra_vs_gpu_util", "gpu_id": gpu_id, "node_id": node, "gpu_index": gpu_index}
            )
    frame = pd.DataFrame(rows, columns=["dataset", "difference", "gpu_id", "node_id", "gpu_index"])
    frame.to_csv(OUTPUT / "header_differences.csv", index=False)
    return frame


def plots(
    telemetry_time: pd.DataFrame,
    trace_daily: pd.DataFrame,
    episodes: pd.DataFrame,
    code_summary: pd.DataFrame,
    balance: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(len(WIDE_FILES), 1, figsize=(12, 14), sharex=True)
    for axis, (metric, group) in zip(axes, telemetry_time.groupby("metric", sort=False), strict=True):
        times = pd.to_datetime(group["time"], utc=True)
        axis.plot(times, group["p50"], label="GPU median", linewidth=0.8)
        axis.plot(times, group["p95"], label="GPU p95", linewidth=0.8)
        axis.set_ylabel(metric)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time (UTC)")
    figure.suptitle("AcmeTrace telemetry — hourly samples")
    figure.tight_layout()
    figure.savefig(OUTPUT / "telemetry_overview.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    dates = pd.to_datetime(trace_daily["submit_date"], utc=True)
    axes[0].plot(dates, trace_daily["submitted_jobs"], label="Submitted jobs")
    axes[0].plot(dates, trace_daily["submitted_gpus"], label="Requested GPUs")
    axes[0].legend()
    axes[1].plot(dates, trace_daily["gpu_hours"], label="GPU hours", color="tab:green")
    axes[1].plot(dates, trace_daily["failed_rate"], label="Failed rate", color="tab:red")
    axes[1].legend()
    axes[1].set_xlabel("Submit date (UTC)")
    figure.suptitle("Trace workload by day")
    figure.tight_layout()
    figure.savefig(OUTPUT / "trace_workload.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    if not code_summary.empty:
        top = code_summary.head(15).sort_values("observations")
        axes[0].barh(top["xid_code"].astype(str), top["observations"])
        axes[0].set_title("Top XID codes")
        axes[0].set_xlabel("Non-zero observations")
    if not episodes.empty:
        daily = episodes.assign(day=episodes["start_time"].dt.floor("D")).groupby("day").size()
        axes[1].plot(daily.index, daily.values)
        axes[1].set_title("XID episodes per day")
        axes[1].set_xlabel("Day (UTC)")
    figure.tight_layout()
    figure.savefig(OUTPUT / "xid_overview.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(balance["horizon"], balance["positive_rate"] * 100)
    axis.set_ylabel("Positive GPU-time rows (%)")
    axis.set_title("Risk-label prevalence by horizon")
    figure.tight_layout()
    figure.savefig(OUTPUT / "risk_horizon_balance.png", dpi=160)
    plt.close(figure)


def report(
    telemetry_inventory: pd.DataFrame,
    header_differences: pd.DataFrame,
    xid_inventory: dict,
    code_summary: pd.DataFrame,
    balance: pd.DataFrame,
    trace_quality: dict,
) -> None:
    inventory_table = telemetry_inventory[
        [
            "dataset",
            "gpu_columns",
            "rows",
            "irregular_steps",
            "sample_missing_rate",
            "sample_zero_rate_among_valid",
        ]
    ].copy()
    inventory_table["sample_missing_rate"] = inventory_table["sample_missing_rate"].map(lambda x: f"{x:.2%}")
    inventory_table["sample_zero_rate_among_valid"] = inventory_table["sample_zero_rate_among_valid"].map(
        lambda x: f"{x:.2%}"
    )
    balance_table = balance.copy()
    balance_table["positive_rate"] = balance_table["positive_rate"].map(lambda x: f"{x:.6%}")
    balance_table["negative_to_positive_ratio"] = balance_table["negative_to_positive_ratio"].map(
        lambda x: f"{x:,.1f}"
    )
    top_codes = code_summary.head(10).copy()
    lines = [
        "# AcmeTrace 1차 EDA",
        "",
        "## 방법",
        "",
        "- 원본 CSV는 수정하지 않았다.",
        "- 텔레메트리 시간축과 행 구조는 전수 확인하고, 값 분포는 매 240행(1시간) 표본으로 계산했다.",
        "- XID 비제로 관측은 전수 추출했으며, 동일 GPU·코드가 30초 이내 반복되면 하나의 episode로 묶었다.",
        "- 위험 라벨 비율은 episode 이전 5분·12시간·24시간·72시간 구간의 합집합으로 계산했다.",
        "",
        "## 텔레메트리 품질",
        "",
        inventory_table.to_markdown(index=False),
        "",
        f"GPU_UTIL 기준 헤더 차이는 {len(header_differences):,}개다.",
        "",
        "## XID 오류",
        "",
        f"- 비제로 관측: {xid_inventory['nonzero_observations']:,}",
        f"- episode: {xid_inventory['episodes']:,}",
        f"- 영향 GPU / 노드: {xid_inventory['affected_gpus']:,} / {xid_inventory['affected_nodes']:,}",
        "",
        top_codes.to_markdown(index=False) if not top_codes.empty else "비제로 XID 관측이 없다.",
        "",
        "## 위험 라벨 불균형",
        "",
        balance_table[["horizon", "positive_gpu_time_rows", "positive_rate", "negative_to_positive_ratio"]].to_markdown(index=False),
        "",
        "## 작업 trace 품질",
        "",
        f"- 작업 수 / 고유 job_id: {trace_quality['rows']:,} / {trace_quality['unique_jobs']:,}",
        f"- 중복 job_id: {trace_quality['duplicate_job_ids']:,}",
        f"- 음수 대기시간 / 실행시간: {trace_quality['negative_wait']:,} / {trace_quality['negative_runtime']:,}",
        f"- duration 불일치(1초 초과): {trace_quality['duration_mismatch_over_1s']:,}",
        f"- gpu_time 불일치(1초 초과): {trace_quality['gpu_time_mismatch_over_1s']:,}",
        "",
        "## 해석 시 주의",
        "",
        "- trace_seren에는 node/GPU 할당 ID가 없어 텔레메트리와 GPU 단위 직접 결합할 수 없다.",
        "- 텔레메트리 분포는 1시간 간격 표본이며, XID 이벤트와 시간축 검사는 전수 결과다.",
        "- 모델링 전 XID episode 정의와 위험 horizon은 최종 파이프라인에 맞춰 확정해야 한다.",
    ]
    (OUTPUT / "EDA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check() -> None:
    base = pd.Timestamp("2023-01-01T00:00:00Z")
    assert merge_interval_bins([(base, base + timedelta(seconds=30))]) == 2
    assert merge_interval_bins(
        [(base, base + timedelta(seconds=30)), (base + timedelta(seconds=15), base + timedelta(seconds=45))]
    ) == 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal, restartable AcmeTrace phase-1 EDA")
    parser.add_argument("--stride", type=int, default=240, help="telemetry sampling stride in 15-second rows")
    parser.add_argument("--force", action="store_true", help="ignore cached summaries")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    missing = [str(path) for path in [*WIDE_FILES.values(), XID_FILE, TRACE_FILE] if not path.exists()]
    if missing:
        parser.error("missing input files: " + ", ".join(missing))

    self_check()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    header_differences = header_stage()
    telemetry_inventory, _, telemetry_time = telemetry_stage(args.stride, args.force)
    events, episodes, code_summary, xid_inventory, balance = xid_stage(args.force)
    _, trace_quality, trace_daily = trace_stage()
    plots(telemetry_time, trace_daily, episodes, code_summary, balance)
    report(telemetry_inventory, header_differences, xid_inventory, code_summary, balance, trace_quality)
    print(f"[done] {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

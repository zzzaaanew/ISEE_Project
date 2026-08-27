from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "eda_phase1" / "xid_event_episodes.csv"
OUTPUT = ROOT / "outputs" / "eda_event_windows"
METRICS = {
    "dram_active": ROOT / "DRAM_ACTIVE.csv",
    "fb_used": ROOT / "FB_USED.csv",
    "gpu_temp": ROOT / "GPU_TEMP.csv",
    "gpu_util": ROOT / "GPU_UTIL.csv",
    "power_usage": ROOT / "POWER_USAGE.csv",
}
SHORT_START, SHORT_END = -60, 60
LONG_START, LONG_END = -72, 24
MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.readline().rstrip("\r\n").split(",")


def event_index() -> pd.DataFrame:
    events = pd.read_csv(SOURCE, parse_dates=["start_time", "end_time"])
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
    events["end_time"] = pd.to_datetime(events["end_time"], utc=True)
    events = events.sort_values(["gpu_id", "start_time", "xid_code"], ignore_index=True)
    events.insert(0, "event_id", np.arange(len(events), dtype=np.int32))
    events["previous_end"] = events.groupby("gpu_id")["end_time"].transform(lambda values: values.cummax().shift())
    events["next_start"] = events.groupby("gpu_id")["start_time"].shift(-1)
    events["hours_since_previous_end"] = (
        events["start_time"] - events["previous_end"]
    ).dt.total_seconds() / 3600
    events["hours_to_next_start"] = (
        events["next_start"] - events["start_time"]
    ).dt.total_seconds() / 3600
    events["isolated_2h"] = (
        events["hours_since_previous_end"].isna() | events["hours_since_previous_end"].gt(2)
    ) & (events["hours_to_next_start"].isna() | events["hours_to_next_start"].gt(2))
    events["clean_pre_72h"] = events["hours_since_previous_end"].isna() | events[
        "hours_since_previous_end"
    ].gt(72)
    cluster_key = pd.MultiIndex.from_frame(events[["node_id", "xid_code", "start_time"]])
    events["cluster_id"] = pd.factorize(cluster_key)[0].astype(np.int32)
    events["cluster_gpu_count"] = events.groupby("cluster_id")["event_id"].transform("size")
    return events


def add_bins(
    times: np.ndarray,
    values: np.ndarray,
    event_time: int,
    unit: int,
    start: int,
    end: int,
    sums: np.ndarray,
    counts: np.ndarray,
    event_id: int,
) -> None:
    left = np.searchsorted(times, event_time + start * unit, side="left")
    right = np.searchsorted(times, event_time + end * unit, side="left")
    if left == right:
        return
    offsets = ((times[left:right] - event_time) // unit).astype(np.int16)
    selected = values[left:right]
    valid = np.isfinite(selected) & (offsets >= start) & (offsets < end)
    if not valid.any():
        return
    bins = offsets[valid] - start
    size = end - start
    sums[event_id] += np.bincount(bins, weights=selected[valid], minlength=size)
    counts[event_id] += np.bincount(bins, minlength=size).astype(np.uint16)


def scan_metric(metric: str, path: Path, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    headers = read_header(path)
    available = set(headers[1:])
    gpu_ids = [gpu for gpu in events["gpu_id"].drop_duplicates() if gpu in available]
    columns = ["Time", *gpu_ids]
    types = {"Time": pa.string(), **{gpu: pa.float32() for gpu in gpu_ids}}
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True),
        parse_options=pacsv.ParseOptions(delimiter=","),
        convert_options=pacsv.ConvertOptions(
            column_types=types,
            include_columns=columns,
            null_values=[""],
            strings_can_be_null=True,
        ),
    )
    column_index = {name: index for index, name in enumerate(reader.schema.names)}
    event_times = events["start_time"].astype("int64").to_numpy()
    event_ids_by_gpu = {
        gpu: group["event_id"].to_numpy(dtype=np.int32)
        for gpu, group in events.groupby("gpu_id", sort=False)
        if gpu in column_index
    }
    shape_short = (len(events), SHORT_END - SHORT_START)
    shape_long = (len(events), LONG_END - LONG_START)
    short_sum = np.zeros(shape_short, dtype=np.float64)
    short_count = np.zeros(shape_short, dtype=np.uint16)
    long_sum = np.zeros(shape_long, dtype=np.float64)
    long_count = np.zeros(shape_long, dtype=np.uint16)

    for batch_number, batch in enumerate(reader, start=1):
        times = pd.to_datetime(batch.column(0).to_pylist(), utc=True).asi8
        batch_start, batch_end = int(times[0]), int(times[-1])
        for gpu_id, candidate_ids in event_ids_by_gpu.items():
            starts = event_times[candidate_ids]
            left = np.searchsorted(starts, batch_start - LONG_END * HOUR_NS, side="left")
            right = np.searchsorted(starts, batch_end - LONG_START * HOUR_NS, side="right")
            if left == right:
                continue
            values = batch.column(column_index[gpu_id]).to_numpy(zero_copy_only=False)
            for event_id in candidate_ids[left:right]:
                event_time = int(event_times[event_id])
                add_bins(
                    times,
                    values,
                    event_time,
                    HOUR_NS,
                    LONG_START,
                    LONG_END,
                    long_sum,
                    long_count,
                    int(event_id),
                )
                if event_time + SHORT_END * MINUTE_NS >= batch_start and event_time + SHORT_START * MINUTE_NS <= batch_end:
                    add_bins(
                        times,
                        values,
                        event_time,
                        MINUTE_NS,
                        SHORT_START,
                        SHORT_END,
                        short_sum,
                        short_count,
                        int(event_id),
                    )
        if batch_number % 10 == 0:
            print(f"[{metric}] batches={batch_number}", flush=True)

    short = np.full(shape_short, np.nan, dtype=np.float32)
    long = np.full(shape_long, np.nan, dtype=np.float32)
    np.divide(short_sum, short_count, out=short, where=short_count > 0)
    np.divide(long_sum, long_count, out=long, where=long_count > 0)
    return short, short_count, long, long_count


def metric_curves(metric: str, path: Path, events: pd.DataFrame, force: bool):
    cache = OUTPUT / f"{metric}_event_curves.npz"
    start_ns = events["start_time"].astype("int64").to_numpy()
    if cache.exists() and not force:
        saved = np.load(cache)
        if np.array_equal(saved["start_ns"], start_ns):
            print(f"[{metric}] cached", flush=True)
            return saved["short"], saved["short_count"], saved["long"], saved["long_count"]
    print(f"[{metric}] scanning {path.name}", flush=True)
    short, short_count, long, long_count = scan_metric(metric, path, events)
    np.savez_compressed(
        cache,
        start_ns=start_ns,
        short=short,
        short_count=short_count,
        long=long,
        long_count=long_count,
    )
    return short, short_count, long, long_count


def row_window(values: np.ndarray, start: int, end: int, origin: int) -> tuple[np.ndarray, np.ndarray]:
    selected = values[:, start - origin : end - origin]
    coverage = np.isfinite(selected).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(selected, axis=1)
    return medians, coverage


def feature_table(events: pd.DataFrame, curves: dict[str, tuple[np.ndarray, ...]]) -> pd.DataFrame:
    frames = []
    short_windows = {
        "baseline_60_30m": (-60, -30),
        "pre_30_5m": (-30, -5),
        "pre_5_0m": (-5, 0),
        "post_0_5m": (0, 5),
        "post_5_30m": (5, 30),
        "post_30_60m": (30, 60),
    }
    long_windows = {
        "baseline_72_48h": (-72, -48),
        "pre_48_24h": (-48, -24),
        "pre_24_12h": (-24, -12),
        "pre_12_1h": (-12, -1),
        "pre_1_0h": (-1, 0),
        "post_0_12h": (0, 12),
        "post_12_24h": (12, 24),
    }
    metadata = [
        "event_id",
        "cluster_id",
        "gpu_id",
        "node_id",
        "xid_code",
        "start_time",
        "duration_seconds",
        "isolated_2h",
        "clean_pre_72h",
    ]
    for metric, (short, _, long, _) in curves.items():
        frame = events[metadata].copy()
        frame.insert(2, "metric", metric)
        for name, (start, end) in short_windows.items():
            frame[name], frame[f"{name}_bins"] = row_window(short, start, end, SHORT_START)
        for name, (start, end) in long_windows.items():
            frame[name], frame[f"{name}_bins"] = row_window(long, start, end, LONG_START)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def effect_summary(features: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "short_isolated_2h": ("isolated_2h", "baseline_60_30m", ["pre_30_5m", "pre_5_0m", "post_0_5m", "post_5_30m"]),
        "long_clean_pre72h": (
            "clean_pre_72h",
            "baseline_72_48h",
            ["pre_48_24h", "pre_24_12h", "pre_12_1h", "pre_1_0h", "post_0_12h"],
        ),
    }
    rows = []
    for scope, (flag, baseline, targets) in scopes.items():
        scoped = features[features[flag]]
        for code_label, code in [("all", None), ("31", 31), ("43", 43)]:
            selected = scoped if code is None else scoped[scoped["xid_code"] == code]
            for metric, group in selected.groupby("metric"):
                for target in targets:
                    pairs = group[["cluster_id", baseline, target]].dropna()
                    if pairs.empty:
                        continue
                    pairs = pairs.assign(delta=pairs[target] - pairs[baseline])
                    clusters = pairs.groupby("cluster_id", as_index=False)[[baseline, target, "delta"]].median()
                    rows.append(
                        {
                            "scope": scope,
                            "xid_code": code_label,
                            "metric": metric,
                            "comparison": target,
                            "gpu_pairs": len(pairs),
                            "node_event_clusters": len(clusters),
                            "median_baseline": clusters[baseline].median(),
                            "median_target": clusters[target].median(),
                            "median_delta": clusters["delta"].median(),
                            "delta_q25": clusters["delta"].quantile(0.25),
                            "delta_q75": clusters["delta"].quantile(0.75),
                            "clusters_increased_fraction": (clusters["delta"] > 0).mean(),
                        }
                    )
    return pd.DataFrame(rows)


def curve_summary(
    events: pd.DataFrame,
    curves: dict[str, tuple[np.ndarray, ...]],
    short: bool,
) -> pd.DataFrame:
    start, end = (SHORT_START, SHORT_END) if short else (LONG_START, LONG_END)
    flag = "isolated_2h" if short else "clean_pre_72h"
    baseline_slice = slice(-60 - start, -30 - start) if short else slice(-72 - start, -48 - start)
    minimum_baseline_bins = 15 if short else 12
    rows = []
    for metric, arrays in curves.items():
        values = arrays[0] if short else arrays[2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            baseline = np.nanmedian(values[:, baseline_slice], axis=1)
        baseline_count = np.isfinite(values[:, baseline_slice]).sum(axis=1)
        normalized = values - baseline[:, None]
        normalized[baseline_count < minimum_baseline_bins] = np.nan
        for code_label, code in [("all", None), ("31", 31), ("43", 43)]:
            mask = events[flag].to_numpy()
            if code is not None:
                mask &= events["xid_code"].eq(code).to_numpy()
            selected = pd.DataFrame(normalized[mask])
            selected.insert(0, "cluster_id", events.loc[mask, "cluster_id"].to_numpy())
            clusters = selected.groupby("cluster_id").median(numeric_only=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(clusters, axis=0)
                q25 = np.nanpercentile(clusters, 25, axis=0)
                q75 = np.nanpercentile(clusters, 75, axis=0)
            valid = np.isfinite(clusters).sum(axis=0).to_numpy()
            for index, offset in enumerate(range(start, end)):
                rows.append(
                    {
                        "metric": metric,
                        "xid_code": code_label,
                        "offset": offset,
                        "node_event_clusters": int(valid[index]),
                        "median_delta_from_baseline": median[index],
                        "q25": q25[index],
                        "q75": q75[index],
                    }
                )
    return pd.DataFrame(rows)


def plot_curves(summary: pd.DataFrame, filename: str, short: bool) -> None:
    metric_order = list(METRICS)
    figure, axes = plt.subplots(len(metric_order), 1, figsize=(12, 14), sharex=True)
    colors = {"31": "tab:blue", "43": "tab:orange"}
    for axis, metric in zip(axes, metric_order, strict=True):
        for code in ["31", "43"]:
            data = summary[(summary["metric"] == metric) & (summary["xid_code"].astype(str) == code)]
            x = data["offset"].to_numpy(dtype=float) + 0.5
            median = data["median_delta_from_baseline"].to_numpy(dtype=float)
            q25 = data["q25"].to_numpy(dtype=float)
            q75 = data["q75"].to_numpy(dtype=float)
            axis.plot(x, median, label=f"XID {code}", color=colors[code])
            axis.fill_between(x, q25, q75, color=colors[code], alpha=0.15)
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.axhline(0, color="gray", linewidth=0.6)
        axis.set_ylabel(metric)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Minutes from episode start" if short else "Hours from episode start")
    figure.suptitle(
        "Telemetry around isolated XID episodes — baseline-subtracted median and IQR"
        if short
        else "Long-horizon telemetry before XID episodes — baseline-subtracted median and IQR"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / filename, dpi=160)
    plt.close(figure)


def plot_effects(summary: pd.DataFrame) -> None:
    comparisons = ["pre_5_0m", "post_0_5m"]
    selected = summary[
        (summary["scope"] == "short_isolated_2h")
        & summary["xid_code"].isin(["31", "43"])
        & summary["comparison"].isin(comparisons)
    ]
    figure, axes = plt.subplots(len(METRICS), 1, figsize=(11, 14))
    x = np.arange(2)
    width = 0.18
    combinations = [("31", "pre_5_0m"), ("31", "post_0_5m"), ("43", "pre_5_0m"), ("43", "post_0_5m")]
    for axis, metric in zip(axes, METRICS, strict=True):
        data = selected[selected["metric"] == metric].set_index(["xid_code", "comparison"])
        for position, (code, comparison) in enumerate(combinations):
            value = data.loc[(code, comparison), "median_delta"] if (code, comparison) in data.index else np.nan
            axis.bar(x[position // 2] + (position % 2 - 0.5) * width, value, width=width, label=f"{comparison}" if position < 2 else None)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xticks(x, ["XID 31", "XID 43"])
        axis.set_ylabel(metric)
    axes[0].legend()
    figure.suptitle("Median change from -60~-30 min baseline")
    figure.tight_layout()
    figure.savefig(OUTPUT / "event_window_deltas.png", dpi=160)
    plt.close(figure)


def write_report(events: pd.DataFrame, effects: pd.DataFrame) -> None:
    short = effects[
        (effects["scope"] == "short_isolated_2h")
        & effects["xid_code"].isin(["31", "43"])
        & effects["comparison"].isin(["pre_5_0m", "post_0_5m"])
    ][
        [
            "xid_code",
            "metric",
            "comparison",
            "node_event_clusters",
            "median_baseline",
            "median_target",
            "median_delta",
            "clusters_increased_fraction",
        ]
    ].copy()
    long = effects[
        (effects["scope"] == "long_clean_pre72h")
        & effects["xid_code"].isin(["31", "43"])
        & (effects["comparison"] == "pre_1_0h")
    ][
        [
            "xid_code",
            "metric",
            "node_event_clusters",
            "median_baseline",
            "median_target",
            "median_delta",
            "clusters_increased_fraction",
        ]
    ].copy()
    short["clusters_increased_fraction"] = short["clusters_increased_fraction"].map(lambda value: f"{value:.1%}")
    long["clusters_increased_fraction"] = long["clusters_increased_fraction"].map(lambda value: f"{value:.1%}")
    lines = [
        "# XID episode 전후 텔레메트리 EDA",
        "",
        "## 방법",
        "",
        f"- GPU episode {len(events):,}개를 node·시작시각·XID 코드 기준 {events['cluster_id'].nunique():,}개 사건 cluster로 묶었다.",
        f"- 전후 2시간 내 다른 episode가 없는 단기 격리 GPU episode는 {events['isolated_2h'].sum():,}개다.",
        f"- 이전 episode 종료 후 72시간 이상 지난 장기 clean-history GPU episode는 {events['clean_pre_72h'].sum():,}개다.",
        "- 단기 곡선은 -60~+60분을 1분 단위, 장기 곡선은 -72~+24시간을 1시간 단위로 집계했다.",
        "- 각 GPU의 사건 전 기준선 중앙값을 뺀 뒤, 동일 node-time cluster 안의 GPU 중앙값을 먼저 계산했다.",
        "- p-value는 계산하지 않았다. 시간 자기상관과 동일 노드 다중 GPU 때문에 독립 표본 가정이 성립하지 않는다.",
        "",
        "## 사건 직전 5분과 직후 5분",
        "",
        short.to_markdown(index=False),
        "",
        "## 사건 직전 1시간",
        "",
        long.to_markdown(index=False),
        "",
        "## 해석 제한",
        "",
        "- 전후 비교는 연관성 EDA이며 인과효과가 아니다.",
        "- 직후 변화는 XID 상태 지속이나 운영 대응의 결과일 수 있어 예측 feature로 사용할 수 없다.",
        "- 최종 모델링 전 같은 GPU·요일·시간대의 비사건 control window와 비교해야 한다.",
    ]
    (OUTPUT / "EVENT_WINDOW_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check() -> None:
    event_time = 0
    times = np.array([-60, -45, -30, -15, 0, 15, 30, 45], dtype=np.int64) * 1_000_000_000
    values = np.arange(8, dtype=np.float32)
    sums = np.zeros((1, 2), dtype=np.float64)
    counts = np.zeros((1, 2), dtype=np.uint16)
    add_bins(times, values, event_time, MINUTE_NS, -1, 1, sums, counts, 0)
    assert counts.tolist() == [[4, 4]]
    assert sums.tolist() == [[6.0, 22.0]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-aligned telemetry EDA around XID episode starts")
    parser.add_argument("--force", action="store_true", help="ignore cached metric curves")
    args = parser.parse_args()
    required = [SOURCE, *METRICS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("missing input files: " + ", ".join(missing))
    self_check()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = event_index()
    events.to_csv(OUTPUT / "event_index.csv", index=False)
    curves = {metric: metric_curves(metric, path, events, args.force) for metric, path in METRICS.items()}
    features = feature_table(events, curves)
    effects = effect_summary(features)
    short_summary = curve_summary(events, curves, short=True)
    long_summary = curve_summary(events, curves, short=False)
    features.to_csv(OUTPUT / "event_window_features.csv", index=False)
    effects.to_csv(OUTPUT / "event_window_effect_summary.csv", index=False)
    short_summary.to_csv(OUTPUT / "event_short_curve_summary.csv", index=False)
    long_summary.to_csv(OUTPUT / "event_long_curve_summary.csv", index=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_curves(short_summary, "event_short_curves.png", short=True)
    plot_curves(long_summary, "event_long_curves.png", short=False)
    plot_effects(effects)
    write_report(events, effects)
    metadata = {
        "gpu_episodes": len(events),
        "node_event_clusters": int(events["cluster_id"].nunique()),
        "isolated_2h_gpu_episodes": int(events["isolated_2h"].sum()),
        "clean_pre72h_gpu_episodes": int(events["clean_pre_72h"].sum()),
        "short_window_minutes": [SHORT_START, SHORT_END],
        "long_window_hours": [LONG_START, LONG_END],
    }
    (OUTPUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

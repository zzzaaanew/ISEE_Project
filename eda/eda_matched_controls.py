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

import eda_event_windows as base


ROOT = Path(__file__).resolve().parent.parent
EVENT_DIR = ROOT / "outputs" / "eda_event_windows"
OUTPUT = ROOT / "outputs" / "eda_matched_controls"
METRICS = base.METRICS
MINUTE_NS = base.MINUTE_NS
WINDOW_START, WINDOW_END = -60, 60
CALIPER = 10.0


def merge_ranges(events: pd.DataFrame, before_hours: int, after_hours: int):
    before = before_hours * 3600 * 1_000_000_000
    after = after_hours * 3600 * 1_000_000_000
    ranges = sorted(
        (int(row.start_time.value) - before, int(row.end_time.value) + after)
        for row in events.itertuples()
    )
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return np.array(merged, dtype=np.int64)


def blocked(timestamp: int, ranges: np.ndarray) -> bool:
    if not len(ranges):
        return False
    index = np.searchsorted(ranges[:, 0], timestamp, side="right") - 1
    return index >= 0 and timestamp <= ranges[index, 1]


def candidate_controls(events: pd.DataFrame, data_start: pd.Timestamp, data_end: pd.Timestamp):
    gpu_blocks = {gpu: merge_ranges(group, 72, 72) for gpu, group in events.groupby("gpu_id")}
    node_blocks = {node: merge_ranges(group, 2, 2) for node, group in events.groupby("node_id")}
    point_ids = {}
    points = []
    pairs = []
    for event in events.itertuples():
        for week_shift in range(-13, 14):
            if week_shift == 0:
                continue
            control_time = event.start_time + pd.Timedelta(days=7 * week_shift)
            if control_time < data_start + pd.Timedelta(hours=1) or control_time > data_end - pd.Timedelta(hours=1):
                continue
            timestamp = int(control_time.value)
            if blocked(timestamp, gpu_blocks[event.gpu_id]) or blocked(timestamp, node_blocks[event.node_id]):
                continue
            key = (event.gpu_id, timestamp)
            if key not in point_ids:
                point_id = len(points)
                point_ids[key] = point_id
                points.append(
                    {
                        "point_id": point_id,
                        "gpu_id": event.gpu_id,
                        "node_id": event.node_id,
                        "control_time": control_time,
                    }
                )
            pairs.append(
                {
                    "event_id": event.event_id,
                    "point_id": point_ids[key],
                    "week_shift": week_shift,
                }
            )
    return pd.DataFrame(points), pd.DataFrame(pairs)


def scan_points(metric: str, path: Path, points: pd.DataFrame):
    headers = base.read_header(path)
    available = set(headers[1:])
    gpu_ids = [gpu for gpu in points["gpu_id"].drop_duplicates() if gpu in available]
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
    point_times = points["control_time"].astype("int64").to_numpy()
    point_ids_by_gpu = {
        gpu: group.sort_values("control_time")["point_id"].to_numpy(dtype=np.int32)
        for gpu, group in points.groupby("gpu_id", sort=False)
        if gpu in column_index
    }
    shape = (len(points), WINDOW_END - WINDOW_START)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint16)
    for batch_number, batch in enumerate(reader, start=1):
        times = pd.to_datetime(batch.column(0).to_pylist(), utc=True).asi8
        batch_start, batch_end = int(times[0]), int(times[-1])
        for gpu_id, candidate_ids in point_ids_by_gpu.items():
            starts = point_times[candidate_ids]
            left = np.searchsorted(starts, batch_start - WINDOW_END * MINUTE_NS, side="left")
            right = np.searchsorted(starts, batch_end - WINDOW_START * MINUTE_NS, side="right")
            if left == right:
                continue
            values = batch.column(column_index[gpu_id]).to_numpy(zero_copy_only=False)
            for point_id in candidate_ids[left:right]:
                base.add_bins(
                    times,
                    values,
                    int(point_times[point_id]),
                    MINUTE_NS,
                    WINDOW_START,
                    WINDOW_END,
                    sums,
                    counts,
                    int(point_id),
                )
        if batch_number % 10 == 0:
            print(f"[{metric}] batches={batch_number}", flush=True)
    values = np.full(shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=values, where=counts > 0)
    return values, counts


def cached_points(metric: str, path: Path, points: pd.DataFrame, force: bool):
    cache = OUTPUT / f"candidate_{metric}_curves.npz"
    timestamps = points["control_time"].astype("int64").to_numpy()
    if cache.exists() and not force:
        saved = np.load(cache)
        if np.array_equal(saved["timestamps"], timestamps):
            print(f"[{metric}] cached", flush=True)
            return saved["values"], saved["counts"]
    print(f"[{metric}] scanning {path.name}", flush=True)
    values, counts = scan_points(metric, path, points)
    np.savez_compressed(cache, timestamps=timestamps, values=values, counts=counts)
    return values, counts


def activity_state(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-np.inf, 10, 50, np.inf], right=False, labels=["idle", "low", "high"])


def match_controls(events, points, pairs, candidate_util, candidate_counts):
    event_features = pd.read_csv(EVENT_DIR / "event_window_features.csv")
    event_util = event_features[event_features["metric"] == "gpu_util"][
        ["event_id", "baseline_60_30m", "baseline_60_30m_bins"]
    ].rename(columns={"baseline_60_30m": "event_baseline_util", "baseline_60_30m_bins": "event_baseline_bins"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        control_baseline = np.nanmedian(candidate_util[:, :30], axis=1)
    control_baseline_bins = np.isfinite(candidate_util[:, :30]).sum(axis=1)
    point_quality = points[["point_id"]].copy()
    point_quality["control_baseline_util"] = control_baseline
    point_quality["control_baseline_bins"] = control_baseline_bins
    candidates = (
        pairs.merge(point_quality, on="point_id")
        .merge(event_util, on="event_id")
        .merge(events[["event_id", "gpu_id", "node_id", "cluster_id", "xid_code", "start_time", "isolated_2h"]], on="event_id")
        .merge(points[["point_id", "control_time"]], on="point_id")
    )
    candidates = candidates[
        (candidates["event_baseline_bins"] >= 15)
        & (candidates["control_baseline_bins"] >= 15)
        & candidates["event_baseline_util"].notna()
        & candidates["control_baseline_util"].notna()
    ].copy()
    candidates["event_activity"] = activity_state(candidates["event_baseline_util"])
    candidates["control_activity"] = activity_state(candidates["control_baseline_util"])
    candidates["baseline_util_abs_diff"] = (
        candidates["event_baseline_util"] - candidates["control_baseline_util"]
    ).abs()
    candidates = candidates[
        (candidates["event_activity"] == candidates["control_activity"])
        & (candidates["baseline_util_abs_diff"] <= CALIPER)
    ].copy()
    counts = candidates.groupby("event_id").size().rename("eligible_candidates")
    candidates = candidates.merge(counts, on="event_id")

    # ponytail: greedy unique matching; replace with optimal assignment only if balance materially worsens.
    used_points = set()
    rows = []
    order = counts.sort_values().index
    for event_id in order:
        options = candidates[candidates["event_id"] == event_id].sort_values(
            ["baseline_util_abs_diff", "week_shift"], key=lambda values: values.abs()
        )
        chosen = next((row for row in options.itertuples() if row.point_id not in used_points), None)
        if chosen is not None:
            used_points.add(chosen.point_id)
            rows.append(chosen._asdict())
    matched = pd.DataFrame(rows).sort_values("event_id", ignore_index=True)
    matched.insert(0, "match_id", np.arange(len(matched), dtype=np.int32))
    return matched, candidates


def window_median(values: np.ndarray, start: int, end: int):
    selected = values[:, start - WINDOW_START : end - WINDOW_START]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(selected, axis=1), np.isfinite(selected).sum(axis=1)


def paired_features(events, matched, candidate_curves):
    rows = []
    event_ids = matched["event_id"].to_numpy(dtype=np.int32)
    point_ids = matched["point_id"].to_numpy(dtype=np.int32)
    comparisons = {"pre_10_5m": (-10, -5), "pre_5_0m": (-5, 0), "post_0_5m": (0, 5)}
    for metric in METRICS:
        event_values = np.load(EVENT_DIR / f"{metric}_event_curves.npz")["short"][event_ids]
        control_values = candidate_curves[metric][0][point_ids]
        event_baseline, event_baseline_bins = window_median(event_values, -60, -30)
        control_baseline, control_baseline_bins = window_median(control_values, -60, -30)
        for comparison, (start, end) in comparisons.items():
            event_target, event_target_bins = window_median(event_values, start, end)
            control_target, control_target_bins = window_median(control_values, start, end)
            for index, match in enumerate(matched.itertuples()):
                rows.append(
                    {
                        "match_id": match.match_id,
                        "event_id": match.event_id,
                        "point_id": match.point_id,
                        "cluster_id": match.cluster_id,
                        "gpu_id": match.gpu_id,
                        "node_id": match.node_id,
                        "xid_code": match.xid_code,
                        "event_activity": match.event_activity,
                        "metric": metric,
                        "comparison": comparison,
                        "event_baseline": event_baseline[index],
                        "control_baseline": control_baseline[index],
                        "event_target": event_target[index],
                        "control_target": control_target[index],
                        "event_change": event_target[index] - event_baseline[index],
                        "control_change": control_target[index] - control_baseline[index],
                        "difference_in_change": (event_target[index] - event_baseline[index])
                        - (control_target[index] - control_baseline[index]),
                        "event_baseline_bins": event_baseline_bins[index],
                        "control_baseline_bins": control_baseline_bins[index],
                        "event_target_bins": event_target_bins[index],
                        "control_target_bins": control_target_bins[index],
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_median(values: np.ndarray, repeats: int = 1000):
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    medians = np.median(rng.choice(values, size=(repeats, len(values)), replace=True), axis=1)
    return np.quantile(medians, [0.025, 0.975])


def effect_summary(paired):
    rows = []
    scopes = {"all_matched": paired.index == paired.index, "active": paired["event_activity"] != "idle"}
    for scope, mask in scopes.items():
        scoped = paired[mask]
        for (code, metric, comparison), group in scoped[scoped["xid_code"].isin([31, 43])].groupby(
            ["xid_code", "metric", "comparison"]
        ):
            valid = group.dropna(subset=["event_change", "control_change", "difference_in_change"])
            clusters = valid.groupby("cluster_id", as_index=False)[
                ["event_change", "control_change", "difference_in_change"]
            ].median()
            low, high = bootstrap_median(clusters["difference_in_change"].to_numpy())
            rows.append(
                {
                    "scope": scope,
                    "xid_code": code,
                    "metric": metric,
                    "comparison": comparison,
                    "gpu_pairs": len(valid),
                    "node_event_clusters": len(clusters),
                    "median_event_change": clusters["event_change"].median(),
                    "median_control_change": clusters["control_change"].median(),
                    "median_difference_in_change": clusters["difference_in_change"].median(),
                    "difference_q25": clusters["difference_in_change"].quantile(0.25),
                    "difference_q75": clusters["difference_in_change"].quantile(0.75),
                    "bootstrap_median_ci_low": low,
                    "bootstrap_median_ci_high": high,
                    "clusters_positive_fraction": (clusters["difference_in_change"] > 0).mean(),
                }
            )
    return pd.DataFrame(rows)


def curve_summary(events, matched, candidate_curves):
    event_ids = matched["event_id"].to_numpy(dtype=np.int32)
    point_ids = matched["point_id"].to_numpy(dtype=np.int32)
    rows = []
    for metric in METRICS:
        event_values = np.load(EVENT_DIR / f"{metric}_event_curves.npz")["short"][event_ids]
        control_values = candidate_curves[metric][0][point_ids]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            event_base = np.nanmedian(event_values[:, :30], axis=1)
            control_base = np.nanmedian(control_values[:, :30], axis=1)
        event_norm = event_values - event_base[:, None]
        control_norm = control_values - control_base[:, None]
        invalid = (np.isfinite(event_values[:, :30]).sum(axis=1) < 15) | (
            np.isfinite(control_values[:, :30]).sum(axis=1) < 15
        )
        event_norm[invalid] = np.nan
        control_norm[invalid] = np.nan
        for scope, scope_mask in [
            ("all_matched", np.ones(len(matched), dtype=bool)),
            ("active", matched["event_activity"].ne("idle").to_numpy()),
        ]:
            for code in [31, 43]:
                mask = scope_mask & matched["xid_code"].eq(code).to_numpy()
                cluster_ids = matched.loc[mask, "cluster_id"].to_numpy()
                event_frame = pd.DataFrame(event_norm[mask]).assign(cluster_id=cluster_ids)
                control_frame = pd.DataFrame(control_norm[mask]).assign(cluster_id=cluster_ids)
                event_cluster = event_frame.groupby("cluster_id").median(numeric_only=True)
                control_cluster = control_frame.groupby("cluster_id").median(numeric_only=True)
                difference = event_cluster - control_cluster
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    event_median = np.nanmedian(event_cluster, axis=0)
                    control_median = np.nanmedian(control_cluster, axis=0)
                    diff_median = np.nanmedian(difference, axis=0)
                    q25 = np.nanpercentile(difference, 25, axis=0)
                    q75 = np.nanpercentile(difference, 75, axis=0)
                valid = np.isfinite(difference).sum(axis=0).to_numpy()
                for index, offset in enumerate(range(WINDOW_START, WINDOW_END)):
                    rows.append(
                        {
                            "scope": scope,
                            "xid_code": code,
                            "metric": metric,
                            "offset_min": offset,
                            "node_event_clusters": int(valid[index]),
                            "event_median_change": event_median[index],
                            "control_median_change": control_median[index],
                            "median_difference": diff_median[index],
                            "difference_q25": q25[index],
                            "difference_q75": q75[index],
                        }
                    )
    return pd.DataFrame(rows)


def plots(matched, curve):
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(len(METRICS), 1, figsize=(12, 14), sharex=True)
    colors = {31: "tab:blue", 43: "tab:orange"}
    for axis, metric in zip(axes, METRICS, strict=True):
        for code in [31, 43]:
            data = curve[(curve["scope"] == "active") & (curve["metric"] == metric) & (curve["xid_code"] == code)]
            x = data["offset_min"].to_numpy(dtype=float) + 0.5
            median = data["median_difference"].to_numpy(dtype=float)
            q25 = data["difference_q25"].to_numpy(dtype=float)
            q75 = data["difference_q75"].to_numpy(dtype=float)
            axis.plot(x, median, color=colors[code], label=f"XID {code}")
            axis.fill_between(x, q25, q75, color=colors[code], alpha=0.15)
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0, color="gray", linewidth=0.6)
        axis.set_ylabel(metric)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Minutes from episode start")
    figure.suptitle("Active event minus matched non-event control — median and IQR")
    figure.tight_layout()
    figure.savefig(OUTPUT / "matched_active_difference_curves.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for row, metric in enumerate(["gpu_util", "power_usage", "gpu_temp"]):
        for column, code in enumerate([31, 43]):
            axis = axes[row, column]
            data = curve[(curve["scope"] == "active") & (curve["metric"] == metric) & (curve["xid_code"] == code)]
            x = data["offset_min"].to_numpy(dtype=float) + 0.5
            axis.plot(x, data["event_median_change"], label="Event")
            axis.plot(x, data["control_median_change"], label="Control")
            axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
            axis.axhline(0, color="gray", linewidth=0.6)
            axis.set_title(f"{metric} — XID {code}")
            axis.legend()
    axes[-1, 0].set_xlabel("Minutes")
    axes[-1, 1].set_xlabel("Minutes")
    figure.suptitle("Active event and matched control changes from baseline")
    figure.tight_layout()
    figure.savefig(OUTPUT / "matched_active_event_vs_control.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = matched["event_activity"].map({"idle": "tab:gray", "low": "tab:orange", "high": "tab:blue"})
    axes[0].scatter(matched["event_baseline_util"], matched["control_baseline_util"], c=colors, s=8, alpha=0.5)
    axes[0].plot([0, 100], [0, 100], color="black", linestyle="--")
    axes[0].set_xlabel("Event baseline GPU_UTIL")
    axes[0].set_ylabel("Control baseline GPU_UTIL")
    axes[1].hist(matched["baseline_util_abs_diff"], bins=20)
    axes[1].set_xlabel("Absolute baseline GPU_UTIL difference")
    axes[1].set_ylabel("Matched pairs")
    figure.suptitle("Matching balance")
    figure.tight_layout()
    figure.savefig(OUTPUT / "matching_balance.png", dpi=160)
    plt.close(figure)


def write_report(events, points, candidate_pairs, matched, effects):
    active = effects[
        (effects["scope"] == "active")
        & effects["xid_code"].isin([31, 43])
        & effects["comparison"].isin(["pre_10_5m", "pre_5_0m"])
    ][
        [
            "xid_code",
            "metric",
            "comparison",
            "node_event_clusters",
            "median_event_change",
            "median_control_change",
            "median_difference_in_change",
            "bootstrap_median_ci_low",
            "bootstrap_median_ci_high",
        ]
    ].copy()
    lines = [
        "# Matched non-event control EDA",
        "",
        "## 매칭",
        "",
        "- 동일 GPU만 매칭했다.",
        "- control은 사건 시각에서 정확히 7일 단위로 이동해 요일과 시각을 동일하게 했다.",
        "- 동일 GPU XID episode 구간과 전후 72시간, 동일 노드 episode 구간과 전후 2시간을 제외했다.",
        "- activity 상태(idle <10, low 10~50, high >=50)를 동일하게 하고 GPU_UTIL 기준선 차이 10%p 이내 최근접 control을 선택했다.",
        "- control은 중복 사용하지 않았다.",
        f"- 후보 시점 {len(points):,}개, event-control 후보쌍 {len(candidate_pairs):,}개, 최종 매칭 {len(matched):,}/{len(events):,}개다.",
        f"- 기준선 GPU_UTIL 절대 차이 중앙값 {matched['baseline_util_abs_diff'].median():.3f}%p, p95 {matched['baseline_util_abs_diff'].quantile(.95):.3f}%p다.",
        "",
        "## Active 사건의 사전 변화 차이",
        "",
        active.to_markdown(index=False),
        "",
        "`median_difference_in_change`는 (event 변화) - (control 변화)다.",
        "",
        "## 제한",
        "",
        "- 매칭은 관측 가능한 GPU_UTIL·요일·시각만 통제하며 작업 종류와 GPU 배정 정보는 통제하지 못한다.",
        "- bootstrap 구간은 node-time 사건 cluster 재표집에 기반한 탐색적 구간이다.",
        "- 이번 단계는 -60~+60분 단기 창에 집중했다. 장기 matched control은 단기 신호가 확인된 후 추가한다.",
    ]
    (OUTPUT / "MATCHED_CONTROL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check():
    ranges = np.array([[10, 20], [30, 40]], dtype=np.int64)
    assert blocked(10, ranges) and blocked(35, ranges) and not blocked(25, ranges)
    states = activity_state(pd.Series([0, 9.9, 10, 49.9, 50, 100])).astype(str).tolist()
    assert states == ["idle", "idle", "low", "low", "high", "high"]


def main():
    parser = argparse.ArgumentParser(description="Same-GPU matched non-event control EDA")
    parser.add_argument("--force", action="store_true", help="ignore cached control curves")
    args = parser.parse_args()
    self_check()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENT_DIR / "event_index.csv", parse_dates=["start_time", "end_time"])
    data_start = pd.Timestamp("2023-05-16 17:43:15+08:00").tz_convert("UTC")
    data_end = pd.Timestamp("2023-08-18 08:00:00+08:00").tz_convert("UTC")
    points, candidate_pairs = candidate_controls(events, data_start, data_end)
    points.to_csv(OUTPUT / "control_candidate_points.csv", index=False)
    candidate_pairs.to_csv(OUTPUT / "control_candidate_pairs.csv", index=False)
    candidate_util = cached_points("gpu_util", METRICS["gpu_util"], points, args.force)
    matched, eligible = match_controls(events, points, candidate_pairs, *candidate_util)
    matched.to_csv(OUTPUT / "matched_pairs.csv", index=False)
    eligible.to_csv(OUTPUT / "eligible_pairs.csv", index=False)
    candidate_curves = {"gpu_util": candidate_util}
    for metric, path in METRICS.items():
        if metric != "gpu_util":
            candidate_curves[metric] = cached_points(metric, path, points, args.force)
    paired = paired_features(events, matched, candidate_curves)
    effects = effect_summary(paired)
    curves = curve_summary(events, matched, candidate_curves)
    paired.to_csv(OUTPUT / "paired_window_features.csv", index=False)
    effects.to_csv(OUTPUT / "matched_effect_summary.csv", index=False)
    curves.to_csv(OUTPUT / "matched_curve_summary.csv", index=False)
    plots(matched, curves)
    write_report(events, points, candidate_pairs, matched, effects)
    metadata = {
        "events": len(events),
        "candidate_points": len(points),
        "candidate_pairs": len(candidate_pairs),
        "eligible_pairs_after_activity_caliper": len(eligible),
        "matched_events": len(matched),
        "matched_fraction": len(matched) / len(events),
        "caliper_gpu_util_percentage_points": CALIPER,
        "control_reuse_count": int(matched["point_id"].duplicated().sum()),
    }
    (OUTPUT / "matching_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

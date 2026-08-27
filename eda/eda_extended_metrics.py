from __future__ import annotations

import re
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
MATCH_DIR = ROOT / "outputs" / "eda_matched_controls"
LAG_DIR = ROOT / "outputs" / "eda_xid43_timestamp_lag"
OUTPUT = ROOT / "outputs" / "eda_extended_metrics"
WIDE = {"memory_temp": ROOT / "MEMORY_TEMP.csv", "mem_clock": ROOT / "MEM_CLOCK.csv"}
IPMI = [ROOT / "GPU_AB_Power.csv", ROOT / "GPU_C_Power.csv"]
SHORT_START, SHORT_END = -60, 60
IPMI_START, IPMI_END = -12, 12  # five-minute bins
FIVE_MIN_NS = 5 * base.MINUTE_NS


def selected_pairs() -> pd.DataFrame:
    matched = pd.read_csv(MATCH_DIR / "matched_pairs.csv", parse_dates=["start_time", "control_time"])
    selected = matched[(matched["event_activity"] == "high") & matched["xid_code"].isin([31, 43])].copy()
    selected["start_time"] = pd.to_datetime(selected["start_time"], utc=True)
    selected["control_time"] = pd.to_datetime(selected["control_time"], utc=True)
    return selected.sort_values("match_id").reset_index(drop=True)


def targets(pairs: pd.DataFrame, key: str) -> pd.DataFrame:
    event = pairs[["match_id", key, "start_time"]].rename(columns={key: "entity_id", "start_time": "timestamp"})
    event["source"] = "event"
    control = pairs[["match_id", key, "control_time"]].rename(columns={key: "entity_id", "control_time": "timestamp"})
    control["source"] = "control"
    result = pd.concat([event, control], ignore_index=True)
    result.insert(0, "target_id", np.arange(len(result), dtype=np.int32))
    return result


def scan_wide(path: Path, points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    headers = base.read_header(path)
    available = set(headers[1:])
    entity_ids = [value for value in points["entity_id"].drop_duplicates() if value in available]
    columns = ["Time", *entity_ids]
    types = {"Time": pa.string(), **{value: pa.float32() for value in entity_ids}}
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True),
        convert_options=pacsv.ConvertOptions(
            include_columns=columns,
            column_types=types,
            null_values=[""],
            strings_can_be_null=True,
        ),
    )
    column_index = {name: index for index, name in enumerate(reader.schema.names)}
    point_times = points["timestamp"].astype("int64").to_numpy()
    ids_by_entity = {
        entity: group.sort_values("timestamp")["target_id"].to_numpy(dtype=np.int32)
        for entity, group in points.groupby("entity_id", sort=False)
        if entity in column_index
    }
    shape = (len(points), SHORT_END - SHORT_START)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint16)
    for batch_number, batch in enumerate(reader, start=1):
        times = pd.to_datetime(batch.column(0).to_pylist(), utc=True).asi8
        batch_start, batch_end = int(times[0]), int(times[-1])
        for entity, candidate_ids in ids_by_entity.items():
            starts = point_times[candidate_ids]
            left = np.searchsorted(starts, batch_start - SHORT_END * base.MINUTE_NS, side="left")
            right = np.searchsorted(starts, batch_end - SHORT_START * base.MINUTE_NS, side="right")
            if left == right:
                continue
            values = batch.column(column_index[entity]).to_numpy(zero_copy_only=False)
            for target_id in candidate_ids[left:right]:
                base.add_bins(
                    times,
                    values,
                    int(point_times[target_id]),
                    base.MINUTE_NS,
                    SHORT_START,
                    SHORT_END,
                    sums,
                    counts,
                    int(target_id),
                )
        if batch_number % 10 == 0:
            print(f"[{path.stem}] batches={batch_number}", flush=True)
    values = np.full(shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=values, where=counts > 0)
    return values, counts


def cached_wide(metric: str, path: Path, points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cache = OUTPUT / f"{metric}_paired_curves.npz"
    timestamps = points["timestamp"].astype("int64").to_numpy()
    if cache.exists():
        saved = np.load(cache)
        if np.array_equal(saved["timestamps"], timestamps):
            print(f"[{metric}] cached", flush=True)
            return saved["values"], saved["counts"]
    values, counts = scan_wide(path, points)
    np.savez_compressed(cache, timestamps=timestamps, values=values, counts=counts)
    return values, counts


def normalize_node(value: str) -> str | None:
    match = re.search(r"(10-140-\d+-\d+)$", value or "")
    return match.group(1).replace("-", ".") if match else None


def scan_ipmi(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    point_times = points["timestamp"].astype("int64").to_numpy()
    ids_by_node = {
        node: group.sort_values("timestamp")["target_id"].to_numpy(dtype=np.int32)
        for node, group in points.groupby("entity_id", sort=False)
    }
    shape = (len(points), IPMI_END - IPMI_START)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint16)
    for path in IPMI:
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True),
            convert_options=pacsv.ConvertOptions(
                include_columns=["Time", "Sys_Total_Power", "node"],
                column_types={"Time": pa.string(), "Sys_Total_Power": pa.float32(), "node": pa.string()},
                null_values=[""],
                strings_can_be_null=True,
            ),
        )
        for batch_number, batch in enumerate(reader, start=1):
            times = pd.to_datetime(batch.column(0).to_pylist(), utc=True).asi8
            values = batch.column(1).to_numpy(zero_copy_only=False)
            raw_nodes = batch.column(2).to_pylist()
            nodes = np.array([normalize_node(value) for value in raw_nodes], dtype=object)
            for node in set(nodes).intersection(ids_by_node):
                mask = nodes == node
                node_times = times[mask]
                node_values = values[mask]
                candidate_ids = ids_by_node[node]
                starts = point_times[candidate_ids]
                left = np.searchsorted(starts, int(node_times[0]) - IPMI_END * FIVE_MIN_NS, side="left")
                right = np.searchsorted(starts, int(node_times[-1]) - IPMI_START * FIVE_MIN_NS, side="right")
                for target_id in candidate_ids[left:right]:
                    base.add_bins(
                        node_times,
                        node_values,
                        int(point_times[target_id]),
                        FIVE_MIN_NS,
                        IPMI_START,
                        IPMI_END,
                        sums,
                        counts,
                        int(target_id),
                    )
            if batch_number % 10 == 0:
                print(f"[{path.stem}] batches={batch_number}", flush=True)
    values = np.full(shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=values, where=counts > 0)
    values[values <= 0] = np.nan
    return values, counts


def cached_ipmi(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cache = OUTPUT / "ipmi_total_power_paired_curves.npz"
    timestamps = points["timestamp"].astype("int64").to_numpy()
    if cache.exists():
        saved = np.load(cache)
        if np.array_equal(saved["timestamps"], timestamps):
            print("[ipmi] cached", flush=True)
            return saved["values"], saved["counts"]
    values, counts = scan_ipmi(points)
    np.savez_compressed(cache, timestamps=timestamps, values=values, counts=counts)
    return values, counts


def row_median(values: np.ndarray, start: int, end: int, origin: int) -> tuple[np.ndarray, np.ndarray]:
    selected = values[:, start - origin : end - origin]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(selected, axis=1)
    return result, np.isfinite(selected).sum(axis=1)


def bootstrap_median(values: np.ndarray, repeats: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(44)
    draws = np.median(rng.choice(values, size=(repeats, len(values)), replace=True), axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def bootstrap_mean(values: np.ndarray, repeats: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(44)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def paired_features(
    pairs: pd.DataFrame,
    metric: str,
    values: np.ndarray,
    origin: int,
    windows: dict[str, tuple[int, int]],
    baseline_min_bins: int,
) -> pd.DataFrame:
    size = len(pairs)
    event, control = values[:size], values[size:]
    event_baseline, event_baseline_bins = row_median(event, *windows["baseline"], origin)
    control_baseline, control_baseline_bins = row_median(control, *windows["baseline"], origin)
    rows = []
    for comparison, window in windows.items():
        if comparison == "baseline":
            continue
        event_target, event_target_bins = row_median(event, *window, origin)
        control_target, control_target_bins = row_median(control, *window, origin)
        for index, pair in enumerate(pairs.itertuples()):
            valid = (
                event_baseline_bins[index] >= baseline_min_bins
                and control_baseline_bins[index] >= baseline_min_bins
                and event_target_bins[index] > 0
                and control_target_bins[index] > 0
            )
            rows.append(
                {
                    "match_id": pair.match_id,
                    "cluster_id": pair.cluster_id,
                    "gpu_id": pair.gpu_id,
                    "node_id": pair.node_id,
                    "xid_code": pair.xid_code,
                    "metric": metric,
                    "comparison": comparison,
                    "event_baseline": event_baseline[index] if valid else np.nan,
                    "control_baseline": control_baseline[index] if valid else np.nan,
                    "event_target": event_target[index] if valid else np.nan,
                    "control_target": control_target[index] if valid else np.nan,
                    "event_change": event_target[index] - event_baseline[index] if valid else np.nan,
                    "control_change": control_target[index] - control_baseline[index] if valid else np.nan,
                    "difference_in_change": (
                        event_target[index] - event_baseline[index] - control_target[index] + control_baseline[index]
                    )
                    if valid
                    else np.nan,
                    "event_baseline_bins": event_baseline_bins[index],
                    "control_baseline_bins": control_baseline_bins[index],
                    "event_target_bins": event_target_bins[index],
                    "control_target_bins": control_target_bins[index],
                }
            )
    return pd.DataFrame(rows)


def effect_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (code, metric, comparison), group in features.groupby(["xid_code", "metric", "comparison"]):
        valid = group.dropna(subset=["difference_in_change"])
        clusters = valid.groupby("cluster_id", as_index=False)[
            ["event_change", "control_change", "difference_in_change"]
        ].median()
        if clusters.empty:
            continue
        low, high = bootstrap_median(clusters["difference_in_change"].to_numpy())
        rows.append(
            {
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
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def curve_summary(pairs: pd.DataFrame, metric: str, values: np.ndarray, origin: int, step_min: int) -> pd.DataFrame:
    size = len(pairs)
    event, control = values[:size], values[size:]
    baseline_end = -30 // step_min
    event_base, event_bins = row_median(event, origin, baseline_end, origin)
    control_base, control_bins = row_median(control, origin, baseline_end, origin)
    event_norm = event - event_base[:, None]
    control_norm = control - control_base[:, None]
    minimum = 15 if step_min == 1 else 4
    invalid = (event_bins < minimum) | (control_bins < minimum)
    event_norm[invalid] = np.nan
    control_norm[invalid] = np.nan
    rows = []
    for code in [31, 43]:
        mask = pairs["xid_code"].eq(code).to_numpy()
        cluster_ids = pairs.loc[mask, "cluster_id"].to_numpy()
        event_cluster = pd.DataFrame(event_norm[mask]).assign(cluster_id=cluster_ids).groupby("cluster_id").median()
        control_cluster = pd.DataFrame(control_norm[mask]).assign(cluster_id=cluster_ids).groupby("cluster_id").median()
        difference = event_cluster - control_cluster
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            event_median = np.nanmedian(event_cluster, axis=0)
            control_median = np.nanmedian(control_cluster, axis=0)
            diff_median = np.nanmedian(difference, axis=0)
            q25 = np.nanpercentile(difference, 25, axis=0)
            q75 = np.nanpercentile(difference, 75, axis=0)
        valid = np.isfinite(difference).sum(axis=0).to_numpy()
        for index, offset in enumerate(range(origin, -origin)):
            rows.append(
                {
                    "xid_code": code,
                    "metric": metric,
                    "offset_min": offset * step_min,
                    "node_event_clusters": int(valid[index]),
                    "event_median_change": event_median[index],
                    "control_median_change": control_median[index],
                    "median_difference": diff_median[index],
                    "difference_q25": q25[index],
                    "difference_q75": q75[index],
                }
            )
    return pd.DataFrame(rows)


def mechanism_summary(pairs: pd.DataFrame, wide: dict[str, np.ndarray], ipmi: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    size = len(pairs)
    rows = []

    memory_event, memory_control = wide["memory_temp"][:size], wide["memory_temp"][size:]
    for code in [31, 43]:
        mask = pairs["xid_code"].eq(code).to_numpy()
        cluster_ids = pairs.loc[mask, "cluster_id"].to_numpy()
        for source, values in [("event", memory_event), ("control", memory_control)]:
            baseline, bins = row_median(values, -60, -30, -60)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                peak = np.nanmax(values[:, 50:60], axis=1)
            rise = peak - baseline
            rise[(bins < 15) | ~np.isfinite(peak)] = np.nan
            cluster = pd.DataFrame({"cluster_id": pairs["cluster_id"], "rise": rise})[mask].groupby("cluster_id").median().dropna()
            rows.append(
                {
                    "xid_code": code,
                    "mechanism": "memory_temp_peak_rise_pre10m",
                    "source": source,
                    "node_event_clusters": len(cluster),
                    "median_value": cluster["rise"].median(),
                    "fraction_crossing_threshold": cluster["rise"].ge(5).mean(),
                    "threshold": ">=5 C",
                }
            )

    clock_event, clock_control = wide["mem_clock"][:size], wide["mem_clock"][size:]
    ipmi_event, ipmi_control = ipmi[:size], ipmi[size:]
    pair_mechanism = pairs[["match_id", "cluster_id", "gpu_id", "node_id", "xid_code"]].copy()
    for prefix, values, origin, base_window, final_window, min_bins in [
        ("event_clock", clock_event, -60, (-60, -30), (-3, 0), 15),
        ("control_clock", clock_control, -60, (-60, -30), (-3, 0), 15),
        ("event_node_power", ipmi_event, -12, (-12, -6), (-2, 0), 4),
        ("control_node_power", ipmi_control, -12, (-12, -6), (-2, 0), 4),
    ]:
        baseline, bins = row_median(values, *base_window, origin)
        final, final_bins = row_median(values, *final_window, origin)
        ratio = final / baseline
        ratio[(bins < min_bins) | (final_bins == 0) | (baseline <= 0)] = np.nan
        pair_mechanism[f"{prefix}_ratio"] = ratio

    lag = pd.read_csv(LAG_DIR / "xid43_lag_per_pair.csv")[
        ["match_id", "event_terminal_low_minutes_u10", "control_terminal_low_minutes_u10"]
    ]
    pair_mechanism = pair_mechanism.merge(lag, on="match_id", how="left")
    for code in [31, 43]:
        code_data = pair_mechanism[pair_mechanism["xid_code"] == code]
        cluster = code_data.groupby("cluster_id").median(numeric_only=True)
        definitions = [
            ("mem_clock_collapse", cluster["event_clock_ratio"].le(0.5), cluster["control_clock_ratio"].le(0.5), "<=50% baseline"),
            ("node_power_drop", cluster["event_node_power_ratio"].le(0.8), cluster["control_node_power_ratio"].le(0.8), "<=80% baseline"),
            (
                "gpu_low_and_node_power_drop",
                cluster["event_terminal_low_minutes_u10"].ge(3) & cluster["event_node_power_ratio"].le(0.8),
                cluster["control_terminal_low_minutes_u10"].ge(3) & cluster["control_node_power_ratio"].le(0.8),
                "GPU_UTIL<=10% for 3m and node power<=80%",
            ),
        ]
        for mechanism, event_flag, control_flag, threshold in definitions:
            if mechanism == "mem_clock_collapse":
                valid = cluster[["event_clock_ratio", "control_clock_ratio"]].notna().all(axis=1)
            else:
                valid = cluster[["event_node_power_ratio", "control_node_power_ratio"]].notna().all(axis=1)
            difference = event_flag[valid].astype(float) - control_flag[valid].astype(float)
            low, high = bootstrap_mean(difference.to_numpy()) if len(difference) else (np.nan, np.nan)
            rows.append(
                {
                    "xid_code": code,
                    "mechanism": mechanism,
                    "source": "paired_risk",
                    "node_event_clusters": int(valid.sum()),
                    "median_value": difference.mean(),
                    "fraction_crossing_threshold": event_flag[valid].mean(),
                    "control_fraction": control_flag[valid].mean(),
                    "threshold": threshold,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
    return pd.DataFrame(rows), pair_mechanism


def classification(pair_mechanism: pd.DataFrame) -> pd.DataFrame:
    data = pair_mechanism[pair_mechanism["xid_code"] == 43]
    clusters = data.groupby("cluster_id", as_index=False).agg(
        node_id=("node_id", "first"),
        gpu_pairs=("gpu_id", "size"),
        terminal_low_minutes=("event_terminal_low_minutes_u10", "median"),
        node_power_ratio=("event_node_power_ratio", "median"),
        memory_clock_ratio=("event_clock_ratio", "median"),
    )
    gpu_low = clusters["terminal_low_minutes"].ge(3)
    power_known = clusters["node_power_ratio"].notna()
    clusters["event_type"] = np.select(
        [~gpu_low, gpu_low & ~power_known, gpu_low & clusters["node_power_ratio"].le(0.8), gpu_low & clusters["node_power_ratio"].gt(0.8)],
        ["no_terminal_gpu_inactivity", "gpu_inactive_ipmi_missing", "node_or_workload_wide_shutdown_like", "gpu_specific_shutdown_like"],
        default="unclassified",
    )
    return clusters


def plots(curves: pd.DataFrame, classes: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 1, figsize=(11, 11))
    for axis, metric in zip(axes, ["memory_temp", "mem_clock", "node_total_power"], strict=True):
        data = curves[(curves["xid_code"] == 43) & (curves["metric"] == metric)]
        axis.plot(data["offset_min"], data["event_median_change"], label="XID 43 event", color="tab:orange")
        axis.plot(data["offset_min"], data["control_median_change"], label="Matched control", color="tab:gray")
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0, color="gray", linewidth=0.6)
        axis.set_ylabel(metric)
        axis.legend()
    axes[-1].set_xlabel("Minutes from XID timestamp")
    figure.suptitle("Extended telemetry around high-activity XID 43 events")
    figure.tight_layout()
    figure.savefig(OUTPUT / "xid43_extended_event_vs_control.png", dpi=170)
    plt.close(figure)

    counts = classes["event_type"].value_counts()
    figure, axis = plt.subplots(figsize=(10, 5))
    counts.sort_values().plot.barh(ax=axis, color="tab:orange")
    axis.set_xlabel("Node-event clusters")
    axis.set_ylabel("")
    axis.set_title("XID 43 event classification from GPU inactivity and node power")
    figure.tight_layout()
    figure.savefig(OUTPUT / "xid43_event_classification.png", dpi=170)
    plt.close(figure)


def self_check() -> None:
    assert normalize_node("SH-IDC1-10-140-0-241") == "10.140.0.241"
    values = np.arange(24, dtype=float).reshape(2, 12)
    medians, bins = row_median(values, -12, -6, -12)
    assert medians.tolist() == [2.5, 14.5] and bins.tolist() == [6, 6]


def main() -> None:
    self_check()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs = selected_pairs()
    gpu_targets = targets(pairs, "gpu_id")
    node_targets = targets(pairs, "node_id")
    wide_curves = {metric: cached_wide(metric, path, gpu_targets)[0] for metric, path in WIDE.items()}
    ipmi_curves = cached_ipmi(node_targets)[0]
    features = [
        paired_features(
            pairs,
            metric,
            values,
            -60,
            {"baseline": (-60, -30), "pre_30_10m": (-30, -10), "pre_10_5m": (-10, -5), "pre_5_0m": (-5, 0), "post_0_5m": (0, 5)},
            15,
        )
        for metric, values in wide_curves.items()
    ]
    features.append(
        paired_features(
            pairs,
            "node_total_power",
            ipmi_curves,
            -12,
            {"baseline": (-12, -6), "pre_30_10m": (-6, -2), "pre_10_0m": (-2, 0), "post_0_10m": (0, 2)},
            4,
        )
    )
    features = pd.concat(features, ignore_index=True)
    effects = effect_summary(features)
    curves = pd.concat(
        [
            curve_summary(pairs, "memory_temp", wide_curves["memory_temp"], -60, 1),
            curve_summary(pairs, "mem_clock", wide_curves["mem_clock"], -60, 1),
            curve_summary(pairs, "node_total_power", ipmi_curves, -12, 5),
        ],
        ignore_index=True,
    )
    mechanisms, pair_mechanism = mechanism_summary(pairs, wide_curves, ipmi_curves)
    classes = classification(pair_mechanism)
    features.to_csv(OUTPUT / "extended_pair_features.csv", index=False)
    effects.to_csv(OUTPUT / "extended_effect_summary.csv", index=False)
    curves.to_csv(OUTPUT / "extended_curve_summary.csv", index=False)
    mechanisms.to_csv(OUTPUT / "mechanism_summary.csv", index=False)
    pair_mechanism.to_csv(OUTPUT / "extended_pair_mechanisms.csv", index=False)
    classes.to_csv(OUTPUT / "xid43_event_classification.csv", index=False)
    plots(curves, classes)
    print(f"[done] {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

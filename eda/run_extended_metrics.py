import numpy as np
import pandas as pd

import eda_extended_metrics as eda


def scan_ipmi(points):
    point_times = points["timestamp"].astype("int64").to_numpy()
    ids_by_node = {
        node: group.sort_values("timestamp")["target_id"].to_numpy(dtype=np.int32)
        for node, group in points.groupby("entity_id", sort=False)
    }
    shape = (len(points), eda.IPMI_END - eda.IPMI_START)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint16)
    for path in eda.IPMI:
        reader = eda.pacsv.open_csv(
            path,
            read_options=eda.pacsv.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True),
            convert_options=eda.pacsv.ConvertOptions(
                include_columns=["Time", "Sys_Total_Power", "node"],
                column_types={"Time": eda.pa.string(), "Sys_Total_Power": eda.pa.float32(), "node": eda.pa.string()},
                null_values=[""],
                strings_can_be_null=True,
            ),
        )
        for batch_number, batch in enumerate(reader, start=1):
            parsed = pd.to_datetime(batch.column(0).to_pylist(), utc=True, errors="coerce", format="mixed")
            valid_time = ~parsed.isna()
            times = parsed[valid_time].asi8
            values = batch.column(1).to_numpy(zero_copy_only=False)[valid_time]
            raw_nodes = np.asarray(batch.column(2).to_pylist(), dtype=object)[valid_time]
            nodes = np.array([eda.normalize_node(value) for value in raw_nodes], dtype=object)
            for node in set(nodes).intersection(ids_by_node):
                mask = nodes == node
                order = np.argsort(times[mask])
                node_times = times[mask][order]
                node_values = values[mask][order]
                candidate_ids = ids_by_node[node]
                starts = point_times[candidate_ids]
                left = np.searchsorted(starts, int(node_times[0]) - eda.IPMI_END * eda.FIVE_MIN_NS, side="left")
                right = np.searchsorted(starts, int(node_times[-1]) - eda.IPMI_START * eda.FIVE_MIN_NS, side="right")
                for target_id in candidate_ids[left:right]:
                    eda.base.add_bins(
                        node_times,
                        node_values,
                        int(point_times[target_id]),
                        eda.FIVE_MIN_NS,
                        eda.IPMI_START,
                        eda.IPMI_END,
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


def classification(pair_mechanism):
    data = pair_mechanism[pair_mechanism["xid_code"] == 43]
    clusters = data.groupby("cluster_id", as_index=False).agg(
        node_id=("node_id", "first"),
        gpu_pairs=("gpu_id", "size"),
        terminal_low_minutes=("event_terminal_low_minutes_u10", "median"),
        node_power_ratio=("event_node_power_ratio", "median"),
        memory_clock_ratio=("event_clock_ratio", "median"),
    )
    lag_known = clusters["terminal_low_minutes"].notna()
    gpu_low = clusters["terminal_low_minutes"].ge(3)
    power_known = clusters["node_power_ratio"].notna()
    clusters["event_type"] = np.select(
        [
            ~lag_known,
            lag_known & ~gpu_low,
            gpu_low & ~power_known,
            gpu_low & clusters["node_power_ratio"].le(0.8),
            gpu_low & clusters["node_power_ratio"].gt(0.8),
        ],
        [
            "gpu_telemetry_missing",
            "no_terminal_gpu_inactivity",
            "gpu_inactive_ipmi_missing",
            "node_or_workload_wide_shutdown_like",
            "gpu_specific_shutdown_like",
        ],
        default="unclassified",
    )
    return clusters


eda.scan_ipmi = scan_ipmi
eda.classification = classification

if __name__ == "__main__":
    eda.main()

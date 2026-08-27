"""Memory-safe runner for AcmeTrace EDA on wide XID data."""

import gc
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

import eda_acmetrace as eda


def xid_scan_fast():
    headers = eda.read_header(eda.XID_FILE)
    types = {headers[0]: pa.string(), **{name: pa.float32() for name in headers[1:]}}
    reader = pacsv.open_csv(
        eda.XID_FILE,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024, use_threads=True),
        parse_options=pacsv.ParseOptions(delimiter=","),
        convert_options=pacsv.ConvertOptions(column_types=types, null_values=[""], strings_can_be_null=True),
    )
    time_chunks, gpu_chunks, code_chunks = [], [], []
    missing = np.zeros(len(headers) - 1, dtype=np.int64)
    rows = batches = observations = 0
    first_time = last_time = None

    for batch in reader:
        batches += 1
        rows += batch.num_rows
        times = batch.column(0).to_pylist()
        time_ns = pd.to_datetime(times, utc=True).asi8
        first_time = first_time or times[0]
        last_time = times[-1]
        for column_index in range(1, len(headers)):
            column = batch.column(column_index)
            missing[column_index - 1] += column.null_count
            positions = pc.indices_nonzero(pc.fill_null(pc.not_equal(column, 0), False)).to_numpy(
                zero_copy_only=False
            )
            if not positions.size:
                continue
            values = pc.take(column, pa.array(positions)).to_numpy(zero_copy_only=False)
            if not np.all(values == np.round(values)):
                raise ValueError("XID code contains a non-integer value")
            time_chunks.append(time_ns[positions])
            gpu_chunks.append(np.full(positions.size, column_index - 1, dtype=np.int16))
            code_chunks.append(values.astype(np.int16))
            observations += positions.size
        if batches % 10 == 0:
            print(f"[xid] batches={batches}, rows={rows:,}, observations={observations:,}", flush=True)

    times = np.concatenate(time_chunks) if time_chunks else np.array([], dtype=np.int64)
    gpu_codes = np.concatenate(gpu_chunks) if gpu_chunks else np.array([], dtype=np.int16)
    xid_codes = np.concatenate(code_chunks) if code_chunks else np.array([], dtype=np.int16)
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(times, unit="ns", utc=True),
            "gpu_id": pd.Categorical.from_codes(gpu_codes, categories=headers[1:]),
            "xid_code": xid_codes,
        }
    )
    del time_chunks, gpu_chunks, code_chunks, times, gpu_codes, xid_codes
    gc.collect()

    node_map = {gpu_id: eda.gpu_parts(gpu_id)[0] for gpu_id in headers[1:]}
    index_map = {gpu_id: eda.gpu_parts(gpu_id)[1] for gpu_id in headers[1:]}
    frame["node_id"] = frame["gpu_id"].map(node_map)
    frame["gpu_index"] = frame["gpu_id"].map(index_map).astype("int8")
    frame = frame.sort_values(["gpu_id", "xid_code", "time"], ignore_index=True)
    gap = frame.groupby(["gpu_id", "xid_code"], observed=True)["time"].diff().dt.total_seconds()
    frame["episode_id"] = (
        (gap.isna() | (gap > 30)).groupby([frame["gpu_id"], frame["xid_code"]], observed=True).cumsum().astype("int32")
    )
    episodes = (
        frame.groupby(
            ["gpu_id", "node_id", "gpu_index", "xid_code", "episode_id"], as_index=False, observed=True
        )
        .agg(start_time=("time", "min"), end_time=("time", "max"), observations=("time", "size"))
    )
    episodes["duration_seconds"] = (episodes["end_time"] - episodes["start_time"]).dt.total_seconds()
    frame = frame.drop(columns="episode_id")
    missing_frame = pd.DataFrame(
        {"gpu_id": headers[1:], "missing_count": missing, "missing_rate": missing / rows}
    )
    inventory = {
        "dataset": "xid_errors",
        "file": eda.XID_FILE.name,
        "size_gib": round(eda.XID_FILE.stat().st_size / 1024**3, 3),
        "columns": len(headers),
        "gpu_columns": len(headers) - 1,
        "rows": rows,
        "first_time": first_time,
        "last_time": last_time,
        "nonzero_observations": len(frame),
        "episodes": len(episodes),
        "affected_gpus": int(frame["gpu_id"].nunique()),
        "affected_nodes": int(frame["node_id"].nunique()),
    }
    return frame, episodes, missing_frame, inventory


def xid_stage_fast(force):
    event_path = eda.OUTPUT / "xid_event_observations.parquet"
    episode_path = eda.OUTPUT / "xid_event_episodes.csv"
    missing_path = eda.OUTPUT / "xid_missingness.csv"
    inventory_path = eda.OUTPUT / "xid_inventory.json"
    cache = [event_path, episode_path, missing_path, inventory_path]
    if not force and all(path.exists() for path in cache):
        print("[xid] cached", flush=True)
        events = pd.read_parquet(event_path)
        episodes = pd.read_csv(episode_path, parse_dates=["start_time", "end_time"])
        missing = pd.read_csv(missing_path)
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        print(f"[xid] scanning: {eda.XID_FILE.name}", flush=True)
        events, episodes, missing, inventory = xid_scan_fast()
        events.to_parquet(event_path, index=False, compression="zstd")
        episodes.to_csv(episode_path, index=False)
        missing.to_csv(missing_path, index=False)
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    code_summary = (
        events.groupby("xid_code", as_index=False, observed=True)
        .agg(
            observations=("time", "size"),
            affected_gpus=("gpu_id", "nunique"),
            affected_nodes=("node_id", "nunique"),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .sort_values("observations", ascending=False)
    )
    code_summary.to_csv(eda.OUTPUT / "xid_code_summary.csv", index=False)
    balance = eda.horizon_balance(
        episodes,
        pd.Timestamp(inventory["first_time"]),
        pd.Timestamp(inventory["last_time"]),
        int(inventory["rows"]),
        int(inventory["gpu_columns"]),
    )
    balance.to_csv(eda.OUTPUT / "risk_horizon_balance.csv", index=False)
    return events, episodes, code_summary, inventory, balance


eda.xid_stage = xid_stage_fast
eda.main()

"""Bounded-memory adapter around frozen October 26 feature calculations."""
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from source_bridge import DAY_NS, STEP_NS, METRICS, FastGridFeatureStore, basic30, feature_names
from supervise import persist

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
DATA = WORKSPACE / "DATA"
CACHE = ROOT / "cache"


def write_json(path, value):
    persist(Path(path),value)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def timestamp_ns(array):
    return array.cast(pa.timestamp("ns", tz="UTC")).cast(pa.int64()).to_numpy(zero_copy_only=False)


def audit_inputs():
    paths = [DATA/f"telemetry_5m_{m}.parquet" for m in METRICS] + [DATA/"xid_onsets_metadata.parquet"]
    records = []
    for path in paths:
        pf = pq.ParquetFile(path)
        records.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path),
                            rows=pf.metadata.num_rows, schema=str(pf.schema_arrow)))
    sources = {p.name:digest(p) for p in (ROOT/"sources").glob("*.py")}
    raw_path = DATA/"AcmeTrace/data/utilization/seren/XID_ERRORS.csv"
    first = last = None
    rows = gaps = bad = duplicates = 0
    intervals = []
    # Original log includes zero observations. Last positive onset is not coverage end.
    for part in pd.read_csv(raw_path, usecols=["Time"], chunksize=20000):
        parsed = pd.to_datetime(part.Time, utc=True, errors="coerce").dt.as_unit("ns")
        bad += int(parsed.isna().sum())
        times = parsed.dropna().astype("int64").to_numpy()
        if not len(times):
            continue
        if first is None:
            first = int(times[0])
        sequence = np.r_[last, times] if last is not None else times
        diffs = np.diff(sequence)
        duplicates += int((diffs<=0).sum())
        for i in np.flatnonzero(diffs>15_000_000_000):
            intervals.append([int(sequence[i]),int(sequence[i+1])])
        gaps += int((diffs>15_000_000_000).sum())
        rows += len(times)
        last = int(times[-1])
    if bad or duplicates or first is None:
        raise ValueError(f"Invalid XID recording timeline: {bad=}, {duplicates=}")
    audit = dict(inputs=records, sources=sources, frozen_remote_commit="1877e09f81d47753fccbb55ba9acb4fb8372b2ed",
                 raw_xid_recording=dict(path=str(raw_path),bytes=raw_path.stat().st_size,rows=rows,
                                        first_ns=first,last_ns=last,gaps=gaps,gap_intervals=intervals,
                                        coverage_basis="recording_timeline_not_last_positive",
                                        per_gpu_nonmissing_coverage="not_certified_by_time_only_audit"))
    write_json(ROOT/"input_manifest.json",audit)
    return audit


def build_grid():
    CACHE.mkdir(exist_ok=True)
    gpu_set, times_set = set(), set()
    for batch in pq.ParquetFile(DATA/"telemetry_5m_util.parquet").iter_batches(
            batch_size=200000,columns=["Time_5m","gpu_id"]):
        gpu_set.update(batch.column("gpu_id").unique().to_pylist())
        times_set.update(np.unique(timestamp_ns(batch.column("Time_5m"))).tolist())
    gpu_ids = np.array(sorted(gpu_set))
    times = np.arange(min(times_set),max(times_set)+STEP_NS,STEP_NS,dtype=np.int64)
    if len(gpu_ids)%8:
        raise ValueError("GPU count is not divisible by 8")
    for group in gpu_ids.reshape(-1,8):
        hosts = [str(x).rsplit("-",1)[0] for x in group]
        slots = {str(x).rsplit("-",1)[1] for x in group}
        if len(set(hosts)) != 1 or slots != {str(x) for x in range(8)}:
            raise ValueError("Legacy 8-GPU node grouping does not match IDs")
    mapping = {g:i for i,g in enumerate(gpu_ids)}
    shape = (len(times),len(gpu_ids))
    for m in METRICS:
        target = CACHE/f"raw_{m}.npy"
        mat = np.lib.format.open_memmap(target,mode="w+",dtype=np.float32,shape=shape)
        mat[:] = np.nan
        seen = np.zeros(shape,dtype=bool)
        duplicate_keys=0
        for batch in pq.ParquetFile(DATA/f"telemetry_5m_{m}.parquet").iter_batches(
                batch_size=150000,columns=["Time_5m","gpu_id",f"{m}_mean"]):
            b = ((timestamp_ns(batch.column("Time_5m"))-times[0])//STEP_NS).astype(np.int32)
            g = batch.column("gpu_id").to_pandas().map(mapping).to_numpy()
            if pd.isna(g).any() or np.any(b<0) or np.any(b>=len(times)):
                raise ValueError("Telemetry keys outside canonical grid")
            g = g.astype(np.int32)
            flat = b.astype(np.int64)*len(gpu_ids)+g
            duplicate_keys += int(seen[b,g].sum()) + len(flat)-len(np.unique(flat))
            seen[b,g]=True
            mat[b,g] = batch.column(f"{m}_mean").to_numpy().astype(np.float32)
        if duplicate_keys:
            raise ValueError(f"Duplicate telemetry keys: {m}, {duplicate_keys}")
        mat.flush()
        del mat,seen
        gc.collect()
        print(f"[grid] {m} complete",flush=True)
    np.savez(CACHE/"grid.npz",times=times,gpu_ids=gpu_ids)
    return dict(bins=len(times),gpus=len(gpu_ids),nodes=len(gpu_ids)//8,
                observed_time_bins=len(times_set),missing_time_bins=len(times)-len(times_set))


def load_grid():
    with np.load(CACHE/"grid.npz") as meta:
        times,gpu_ids = meta["times"],meta["gpu_ids"]
    matrices = {m:np.load(CACHE/f"raw_{m}.npy",mmap_mode="r") for m in METRICS}
    return times,gpu_ids,matrices


def ledger(times,gpu_ids):
    frame = pq.read_table(DATA/"xid_onsets_metadata.parquet").to_pandas()
    ns = pd.to_datetime(frame.onset_time,utc=True).dt.as_unit("ns").astype("int64")
    frame = frame.assign(onset_ns=ns)
    before = len(frame)
    frame = frame.drop_duplicates(["gpu_id","onset_ns","xid_code"])
    mapping = {g:i for i,g in enumerate(gpu_ids)}
    frame["gpu_idx"] = frame.gpu_id.map(mapping)
    if frame.gpu_idx.isna().any():
        raise ValueError("Onset GPU outside input population")
    history = {int(g):np.sort(np.unique(s.onset_ns.to_numpy())) for g,s in frame.groupby("gpu_idx")}
    labels = np.zeros((len(times),len(gpu_ids)),dtype=np.uint8)
    for row in frame.itertuples():
        # Strictly future: t < onset <= t+24h. Legacy code also marked t=onset positive.
        start = int(np.searchsorted(times,int(row.onset_ns)-DAY_NS,side="left"))
        end = int(np.searchsorted(times,int(row.onset_ns),side="left"))
        labels[start:end,int(row.gpu_idx)] = 1
    meta = dict(input_events=before,events_after_exact_dedup=len(frame),history_unique_gpu_onset_times=True,
                label_interval="(t,t+24h]",same_bin_onset_excluded=True)
    return labels,history,frame,meta


def feature_cache(config,history):
    times,gpu_ids,matrices = load_grid()
    B,G=len(times),len(gpu_ids)
    names = feature_names()
    extras = [n for n in names["basic30"] if n not in names["falcon45"]]
    combined = names["falcon45"]+extras
    assert len(combined)==53
    x = np.lib.format.open_memmap(CACHE/"features53.npy",mode="w+",dtype=np.float32,shape=(B,G,53))
    x2 = np.lib.format.open_memmap(CACHE/"history5.npy",mode="w+",dtype=np.float32,shape=(B,G,5))
    W=config["observation_hours"]*12
    first=W+3
    x[:first]=0
    x2[:first]=0
    for begin in range(first,B,config["feature_chunk_bins"]):
        end=min(B,begin+config["feature_chunk_bins"])
        lo=max(0,begin-W-3)
        chunk={m:np.asarray(v[lo:end]) for m,v in matrices.items()}
        nodes={}
        for m in ["util","temp","power"]:
            v=chunk[m].reshape(end-lo,G//8,8)
            count=np.isfinite(v).sum(axis=2)
            nodes[m]=np.divide(np.nansum(v,axis=2),count,
                               out=np.full(count.shape,np.nan,dtype=np.float32),where=count>0)
        engine=SimpleNamespace(num_bins=end-lo,num_gpus=G,num_nodes=G//8,matrices=chunk,
                               node_matrices=nodes,bin_start_ns=times[lo:end])
        store=FastGridFeatureStore(engine,history)
        bins=np.repeat(np.arange(begin-lo,end-lo,dtype=np.int32),G)
        gpus=np.tile(np.arange(G,dtype=np.int32),end-begin)
        f=store.extract_b1(bins,gpus,config["observation_hours"])
        b=basic30(store,bins,gpus,config["observation_hours"])
        extra_idx=[names["basic30"].index(n) for n in extras]
        x[begin:end]=np.column_stack([f,b[:,extra_idx]]).reshape(end-begin,G,53)
        x2[begin:end]=store.extract_b2(bins,gpus).reshape(end-begin,G,5)
        del store,engine,chunk,nodes,b,f
        gc.collect()
        if begin%2880<config["feature_chunk_bins"]:
            print(f"[features] {end}/{B} bins",flush=True)
    x.flush(); x2.flush()
    write_json(ROOT/"feature_manifest.json",dict(storage=combined,**names,
               basic_indices=[combined.index(n) for n in names["basic30"]],
               falcon_indices=list(range(45)),first_valid_bin=first,
               comparison="30_vs_45_feature_package_not_nested_ablation",
               peer_definition="verified_same_host_8_GPU_group_not_same_task",
               missing_value_policy="legacy_nan_to_zero_and_fixed_7_peer_denominator_preserved"))
    (CACHE/"READY").write_text("complete",encoding="ascii")


def prepare(config):
    if (CACHE/"READY").exists():
        raise FileExistsError("Cache already prepared; use existing manifest explicitly")
    manifest=ROOT/"input_manifest.json"
    audit=json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else audit_inputs()
    grid=build_grid()
    times,gpus,_=load_grid()
    labels,history,_,meta=ledger(times,gpus)
    recording=audit["raw_xid_recording"]
    valid=(times>=recording["first_ns"]) & (times+DAY_NS<=recording["last_ns"])
    for left,right in recording["gap_intervals"]:
        valid &= ~((times<right) & (times+DAY_NS>left))
    np.save(CACHE/"label_eligible.npy",valid)
    np.save(CACHE/"labels.npy",labels)
    write_json(ROOT/"grid_manifest.json",dict(**grid,**meta,label_eligible_bins=int(valid.sum()),
               label_gap_policy="exclude_all_GPU_labels_when_forward_24h_intersects_raw_recording_gap",
               raw_gap_count=recording["gaps"]))
    feature_cache(config,history)
    return audit

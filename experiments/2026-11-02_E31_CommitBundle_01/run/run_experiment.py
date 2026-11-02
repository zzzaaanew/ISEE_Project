"""E31: frozen legacy feature/model math, purged branch tapes, post-hoc fusion."""
from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data_adapter import ROOT, CACHE, DAY_NS, STEP_NS, digest, write_json, load_grid, ledger, prepare
from source_bridge import new_controller, update_adst, sample_indices, adjusted_probability


def available_memory_gib():
    class Memory(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in
                  ["total","available","page_total","page_available","virtual_total","virtual_available","extended"]]
    if os.name!="nt":
        return float("inf")
    value=Memory(); value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("GlobalMemoryStatusEx failed")
    return value.available/2**30


def resource_check(config):
    free=shutil.disk_usage(ROOT).free/2**30
    available=available_memory_gib()
    if free<config["minimum_free_disk_gib"] or available<config["minimum_available_memory_gib"]:
        raise RuntimeError(f"Resource stop: disk={free:.2f}GiB, available RAM={available:.2f}GiB")
    return dict(disk_free_gib=free,available_memory_gib=available)


def schedule(config,times,audit):
    rec=audit["raw_xid_recording"]
    end=min(len(times),int(np.searchsorted(times,int(rec["last_ns"])-DAY_NS,side="right")))
    start=config["warmup_days"]*288
    terminal=end-config["terminal_days"]*288
    assert start<terminal<end
    return dict(start_bin=start,end_bin_exclusive=end,terminal_start_bin=terminal,
                start_utc=str(pd.Timestamp(int(times[start]),tz="UTC")),
                last_score_utc=str(pd.Timestamp(int(times[end-1]),tz="UTC")),
                terminal_start_utc=str(pd.Timestamp(int(times[terminal]),tz="UTC")),
                label_recording_end_utc=str(pd.Timestamp(int(rec["last_ns"]),tz="UTC")),
                origins=list(range(start,end,288)),retrospective_terminal=True,
                label_gap_policy="shared_future_24h_eligibility_mask",
                label_coverage_limitation="global_recording_interval; per-GPU missing observations not certified")


def train_interval(origin,days,config):
    end=origin-config["purge_hours"]*12
    start=end-days*288
    assert start>=config["observation_hours"]*12+3
    assert end+config["horizon_hours"]*12<=origin
    return start,end


def flush_feedback(controller,pending,origin,config):
    remain=[]
    for row in pending:
        if row["latest_score_bin"]<=origin-config["purge_hours"]*12:
            assert row["latest_score_bin"]+config["horizon_hours"]*12<=origin
            out=update_adst(controller,row["ap"],row["prevalence"],
                    dict(origin=row["origin"],selected=dict(selected_train_days=row["train_days"])))
            out.update(update_origin=origin,feedback_latest_score_bin=row["latest_score_bin"],
                       feedback_cutoff_bin=origin-config["purge_hours"]*12,
                       feedback_label_maturity_bin=row["latest_score_bin"]+config["horizon_hours"]*12)
        else:
            remain.append(row)
    return remain


def adst_feedback(scores,labels,origin,end,eligible):
    # E14 averages full-population AP blocks at the original 30-minute cadence.
    feedback_bins=np.arange(origin,end,6,dtype=np.int32)
    feedback_bins=feedback_bins[eligible[feedback_bins]]
    if not len(feedback_bins):
        return None
    aps=[];prevalences=[]
    for b in feedback_bins:
        y=np.asarray(labels[b])
        aps.append(float(average_precision_score(y,scores[b-origin])) if y.any() else 0.)
        prevalences.append(float(y.mean()))
    return float(np.mean(aps)),float(np.mean(prevalences))


def model_b1():
    return HistGradientBoostingClassifier(loss="log_loss",learning_rate=.06,max_iter=150,
             max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=1.5,class_weight="balanced",random_state=42)


def models_b2():
    return [make_pipeline(StandardScaler(),LogisticRegression(max_iter=250,class_weight="balanced",random_state=42)),
            HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=31,l2_regularization=1.,
                                           class_weight="balanced",random_state=42)]


def sampled(labels,G,origin,days,config,eligible=None):
    start,end=train_interval(origin,days,config)
    if eligible is None:
        eligible=np.ones(len(labels),dtype=bool)
    working=np.asarray(labels[start:end]).copy()
    working[~eligible[start:end]]=0
    local=SimpleNamespace(gt_matrix=working,engine=SimpleNamespace(num_gpus=G),negative_ratio=config["negative_ratio"])
    if int(working.sum())==0:
        raise ValueError("No positive training labels; no undocumented fallback permitted")
    rng=np.random.default_rng(config["seed"]+origin*31+days)
    bins,gpus,y=sample_indices(local,0,end-start,rng)
    bins+=start
    keep=eligible[bins]
    bins,gpus,y=bins[keep],gpus[keep],y[keep]
    if len(np.unique(y))!=2:
        raise ValueError("Training sample does not contain both classes")
    prior=float(working[eligible[start:end]].mean())
    return bins,gpus,y,prior,dict(train_start_bin=start,train_end_bin_exclusive=end,
                  latest_train_bin=int(bins.max()),samples=len(y),sample_positives=int(y.sum()),true_prior=prior,
                  effective_balanced_fit_prior=.5,training_days=days,
                  label_eligible_calendar_bins=int(eligible[start:end].sum()),
                  label_maturity_latest_bin=int(bins.max())+config["horizon_hours"]*12)


def score_models(models,features,begin,end,indices,prior,config):
    G=features.shape[1]
    output=np.empty((end-begin,G),dtype=np.float32)
    for b in range(begin,end,config["predict_chunk_bins"]):
        e=min(end,b+config["predict_chunk_bins"])
        x=np.asarray(features[b:e]).reshape(-1,features.shape[-1])
        if indices is not None:
            x=x[:,indices]
        p=sum(m.predict_proba(x)[:,1] for m in models)/len(models)
        # class_weight='balanced' trains at effective prior .5, not the unweighted sampled prior.
        output[b-begin:e-begin]=adjusted_probability(p,prior,.5).reshape(e-b,G)
    return output


def close_feature_maps(*mappings):
    for mapping in mappings:
        mapping._mmap.close()
    gc.collect()


def branches(config,max_origins=None,max_new_origins=None):
    times,gpu_ids,_=load_grid()
    labels=np.load(CACHE/"labels.npy",mmap_mode="r")
    eligible=np.load(CACHE/"label_eligible.npy")
    _,history,_,_=ledger(times,gpu_ids)
    features=json.loads((ROOT/"feature_manifest.json").read_text(encoding="utf-8"))
    audit=json.loads((ROOT/"input_manifest.json").read_text(encoding="utf-8"))
    plan=schedule(config,times,audit)
    plan["development_eligible_bins"]=int(eligible[plan["start_bin"]:plan["terminal_start_bin"]].sum())
    plan["terminal_eligible_bins"]=int(eligible[plan["terminal_start_bin"]:plan["end_bin_exclusive"]].sum())
    write_json(ROOT/"split_manifest.json",plan)
    out=ROOT/"branch_scores"; out.mkdir(exist_ok=True)
    checkpoint=ROOT/"branch_checkpoint.json"
    controllers={a:new_controller() for a in ["C","D"]}
    pending={a:[] for a in controllers}
    completed=[]; fit_trace=[]
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved["config_sha256"]!=digest(ROOT/"config.json"):
            raise ValueError("Resume config differs from frozen checkpoint")
        completed=saved["completed_origins"]
        fit_trace=saved["fit_trace"]
        pending=saved["pending"]
        for a,state in saved["controllers"].items():
            controllers[a]=SimpleNamespace(**state)
        print(f"[resume] {len(completed)} committed origins",flush=True)
    selected=plan["origins"][:max_origins] if max_origins else plan["origins"]
    newly_completed=0
    for index,origin in enumerate(selected):
        if origin in completed:
            continue
        resource_check(config)
        x=np.load(CACHE/"features53.npy",mmap_mode="r")
        x2=np.load(CACHE/"history5.npy",mmap_mode="r")
        tic=time.perf_counter()
        end=min(origin+288,plan["end_bin_exclusive"])
        G=len(gpu_ids)
        for a in controllers:
            pending[a]=flush_feedback(controllers[a],pending[a],origin,config)
        bins,gpus,y,prior,tr=sampled(labels,G,origin,21,config,eligible)
        b2=models_b2()
        train_x=np.asarray(x2[bins,gpus])
        for model in b2:
            model.fit(train_x,y)
        p2=score_models(b2,x2,origin,end,None,prior,config)
        fit_trace.append(dict(origin=origin,branch="B2",arm="shared",fit_count=2,**tr))
        del b2,train_x
        preds={}
        train_days={}
        for arm,settings in config["arms"].items():
            days=21 if settings["training"]=="ST21" else controllers[arm].current_train_days
            train_days[arm]=days
            bins,gpus,y,prior,tr=sampled(labels,G,origin,days,config,eligible)
            col=features["basic_indices"] if settings["features"]=="basic30" else features["falcon_indices"]
            train_x=np.asarray(x[bins,gpus])[:,col]
            model=model_b1()
            before=time.perf_counter()
            model.fit(train_x,y)
            fit_seconds=time.perf_counter()-before
            preds[arm]=score_models([model],x,origin,end,col,prior,config)
            fit_trace.append(dict(origin=origin,branch="B1",arm=arm,fit_count=1,fit_seconds=fit_seconds,
                                  feature_count=len(col),**tr))
            if arm in controllers:
                feedback=adst_feedback(preds[arm],labels,origin,end,eligible)
                if feedback is not None:
                    ap,prevalence=feedback
                    pending[arm].append(dict(origin=origin,latest_score_bin=end-1,train_days=days,
                                             ap=ap,prevalence=prevalence,
                                             feedback_metric="mean_30minute_full_population_AP"))
            del model,train_x
            gc.collect()
        # Release file-backed resident pages before the post-commit resource check.
        # Fitted models and predictions own ordinary arrays, not mmap views.
        close_feature_maps(x,x2)
        del x,x2
        gc.collect()
        cutoff=int(times[origin])-2*STEP_NS
        counts=np.zeros(G,dtype=np.int32)
        for g,events in history.items():
            counts[g]=np.searchsorted(events,cutoff,side="left")-np.searchsorted(events,cutoff-30*DAY_NS,side="left")
        n=end-origin
        data=dict(score_bin=np.repeat(np.arange(origin,end,dtype=np.int32),G),
                  timestamp_ns=np.repeat(times[origin:end],G),gpu_idx=np.tile(np.arange(G,dtype=np.int16),n),
                  origin_bin=np.full(n*G,origin,dtype=np.int32),p_b2_raw=p2.ravel(),
                  xid_count_30d_at_origin=np.tile(counts,n))
        for a,p in preds.items():
            data[f"p_b1_{a}_raw"]=p.ravel()
        part=out/f"origin_{origin:06d}.parquet"
        if part.exists():
            raise FileExistsError(f"Uncommitted score file exists: {part}; inspect before recovery")
        pd.DataFrame(data).to_parquet(part,index=False)
        completed.append(origin)
        saved=dict(config_sha256=digest(ROOT/"config.json"),completed_origins=completed,fit_trace=fit_trace,
                   pending=pending,controllers={a:vars(v) for a,v in controllers.items()})
        write_json(checkpoint,saved)
        pd.DataFrame(fit_trace).to_csv(ROOT/"fit_trace.csv",index=False)
        for a,controller in controllers.items():
            pd.DataFrame(controller.adst_history).to_csv(ROOT/f"adst_{a}_trace.csv",index=False)
        elapsed=time.perf_counter()-tic
        write_json(ROOT/"progress.json",dict(stage="branch_scoring",completed=len(completed),total=len(plan["origins"]),
                     origin=origin,last_origin_seconds=elapsed,train_days=train_days,
                     rows_total=sum(min(b+288,plan["end_bin_exclusive"])-b for b in completed)*G,
                     last_part_sha256=digest(part),resources=resource_check(config)))
        print(f"[origin {index+1}/{len(plan['origins'])}] {origin} complete, {elapsed:.1f}s; {train_days}",flush=True)
        del preds,data,p2
        gc.collect()
        newly_completed+=1
        if max_new_origins is not None and newly_completed>=max_new_origins:
            break
    if len(completed)==len(plan["origins"]):
        (ROOT/"BRANCH_SCORES_COMPLETE").write_text("complete",encoding="ascii")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","pilot","branches","next-origin","evaluate","all"])
    args=parser.parse_args()
    config=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    os.environ.setdefault("PYTHONHASHSEED",str(config["seed"]))
    begin=time.time()
    write_json(ROOT/f"runtime_environment_{args.stage}.json",dict(python=sys.version,executable=sys.executable,
        pid=os.getpid(),created_unix=time.time(),
        argv=sys.argv,versions={p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","pyarrow","threadpoolctl"]},
        config_sha256=digest(ROOT/"config.json"),resources=resource_check(config)))
    try:
        with threadpool_limits(limits=config["threads"]):
            if args.stage in ["prepare","all"] and not (CACHE/"READY").exists():
                prepare(config)
            if args.stage=="pilot":
                branches(config,max_origins=1)
            if args.stage in ["branches","all"]:
                branches(config)
            if args.stage=="next-origin":
                branches(config,max_new_origins=1)
            if args.stage in ["evaluate","all"]:
                from evaluate_scores import evaluate
                evaluate(config)
        write_json(ROOT/f"stage_{args.stage}_status.json",dict(status="COMPLETED",seconds=time.time()-begin))
    except BaseException as exc:
        write_json(ROOT/f"stage_{args.stage}_status.json",dict(status="FAILED",error=repr(exc),traceback=traceback.format_exc(),
                                                            seconds=time.time()-begin))
        raise


if __name__=="__main__":
    main()

"""Evaluate stored branch tapes only; never fits branch models or tunes on terminal labels."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score,roc_auc_score

from data_adapter import ROOT,CACHE,DAY_NS,STEP_NS,load_grid,ledger,write_json


def inertia_stepwise(p,step_minutes,gamma,half_life,state=None):
    if gamma<=0:
        return p.copy(),state
    raw=logit(np.clip(p,1e-6,1-1e-6)).astype(np.float64)
    rho=2.**(-(step_minutes/60.)/half_life)
    result=np.empty_like(raw)
    start=0
    if state is None:
        state=raw[0].copy()
        result[0]=raw[0]
        start=1
    for i in range(start,len(raw)):
        result[i]=(raw[i]+gamma*state)/(1+gamma)
        state=rho*state+(1-rho)*raw[i]
    return np.clip(expit(result),1e-6,1-1e-6).astype(np.float32),state


def summarize(scores,labels,bins,previous_top,previous_bin,step):
    top=np.argsort(-scores,axis=1,kind="stable")[:,:min(100,scores.shape[1])]
    mask=np.zeros(scores.shape,dtype=bool)
    np.put_along_axis(mask,top,True,axis=1)
    positive=int(labels.sum())
    hits=int(np.take_along_axis(labels,top,axis=1).sum())
    p=labels.ravel();s=scores.ravel()
    ap=float(average_precision_score(p,s)) if positive else 0.
    auc=float(roc_auc_score(p,s)) if np.unique(p).size==2 else None
    entries=transitions=0
    for i in range(len(mask)):
        if previous_top is not None and previous_bin+step//5==int(bins[i]):
            entries+=int((mask[i]&~previous_top).sum());transitions+=1
        previous_top=mask[i]; previous_bin=int(bins[i])
    return dict(rows=p.size,positive=positive,hits=hits,ap=ap,auc=auc,
                entries=entries,transitions=transitions),mask,previous_top,previous_bin


def onset_denominators(events,times,eligible,plan,stride):
    result={}
    intervals={"development":(plan["start_bin"],plan["terminal_start_bin"]),
               "terminal":(plan["terminal_start_bin"],plan["end_bin_exclusive"]),
               "full":(plan["start_bin"],plan["end_bin_exclusive"])}
    full_grid=np.arange(plan["start_bin"],plan["end_bin_exclusive"],stride)
    event_t=events.onset_ns.to_numpy()
    for split,(lo,hi) in intervals.items():
        grid=full_grid[(full_grid>=lo)&(full_grid<hi)]
        ids=set()
        for i,t in enumerate(event_t):
            if not len(grid) or t-DAY_NS<times[grid[0]] or t-2*STEP_NS>times[grid[-1]]:
                continue
            sub=grid[(times[grid]>=t-DAY_NS)&(times[grid]<=t-2*STEP_NS)]
            if len(sub) and eligible[sub].all():
                ids.add(i)
        result[split]=ids
    return result


def evaluate(config):
    if not (ROOT/"BRANCH_SCORES_COMPLETE").exists():
        raise RuntimeError("Full branch scoring must finish before final evaluation")
    times,gpus,_=load_grid()
    labels=np.load(CACHE/"labels.npy",mmap_mode="r")
    eligible=np.load(CACHE/"label_eligible.npy")
    _,_,events,_=ledger(times,gpus)
    events=events.drop_duplicates(["gpu_idx","onset_ns"]).reset_index(drop=True)
    event_ns=events.onset_ns.to_numpy(dtype=np.int64)
    event_gpus=events.gpu_idx.to_numpy(dtype=np.int32)
    plan=json.loads((ROOT/"split_manifest.json").read_text(encoding="utf-8"))
    aggregates=defaultdict(list); capture=defaultdict(set); traces=[]
    states={}; previous={}
    denoms={step:onset_denominators(events,times,eligible,plan,step//5) for step in [5,30]}
    for file in sorted((ROOT/"branch_scores").glob("origin_*.parquet")):
        df=pd.read_parquet(file)
        block_bins=np.sort(df.score_bin.unique())
        G=len(gpus)
        assert len(df)==len(block_bins)*G
        assert np.array_equal(df.gpu_idx.to_numpy(),np.tile(np.arange(G),len(block_bins)))
        p2=df.p_b2_raw.to_numpy().reshape(-1,G)
        counts=df.xid_count_30d_at_origin.to_numpy().reshape(-1,G)
        for step in [5,30]:
            stride=step//5
            take=(block_bins-plan["start_bin"])%stride==0
            bins=block_bins[take]
            b2=p2[take];k=counts[take].astype(np.float32)
            candidates={"shared/B2":(b2,False)}
            for arm in config["arms"]:
                b1=df[f"p_b1_{arm}_raw"].to_numpy().reshape(-1,G)[take]
                state_key=(step,arm,1.,2.)
                q,state=inertia_stepwise(b1,step,1.,2.,states.get(state_key))
                states[state_key]=state
                lam=np.minimum(.6,1./(k+1.))
                candidates.update({f"{arm}/B1_raw":(b1,False),f"{arm}/B1_inertia":(q,False),
                                   f"{arm}/fusion_no_inertia":(lam*b1+(1-lam)*b2,False),
                                   f"{arm}/fusion_main":(lam*q+(1-lam)*b2,False)})
                if arm=="D":
                    for cap in config["development_caps"]:
                        if cap==.6:
                            continue
                        weight=np.minimum(cap,1./(k+1.))
                        candidates[f"D/dev_cap_{cap:.2f}"]=(weight*q+(1-weight)*b2,True)
                    for gamma,h in config["inertia_pairs"]:
                        if gamma in [0.,1.]:
                            continue
                        sk=(step,arm,gamma,h)
                        qx,states[sk]=inertia_stepwise(b1,step,gamma,h,states.get(sk))
                        candidates[f"D/dev_inertia_{gamma}_{h}"]=(lam*qx+(1-lam)*b2,True)
            split_masks={"development":bins<plan["terminal_start_bin"],
                         "terminal":bins>=plan["terminal_start_bin"],"full":np.ones(len(bins),dtype=bool)}
            for model,(score,dev_only) in candidates.items():
                for split,select in split_masks.items():
                    if dev_only and split!="development":
                        continue
                    select=select&eligible[bins]
                    if not select.any():
                        continue
                    current=bins[select]
                    key=(step,split,model)
                    old,old_bin=previous.get(key,(None,None))
                    metrics,top,old,old_bin=summarize(score[select],np.asarray(labels[current]),current,old,old_bin,step)
                    previous[key]=(old,old_bin)
                    aggregates[key].append(metrics)
                    traces.append(dict(cadence_minutes=step,split=split,model=model,origin_bin=int(block_bins[0]),**metrics))
                    candidates_in_time=np.flatnonzero((event_ns>=times[current[0]]+2*STEP_NS)&
                                                       (event_ns<=times[current[-1]]+DAY_NS))
                    for event_id in candidates_in_time:
                        if event_id not in denoms[step][split]:
                            continue
                        onset=int(event_ns[event_id])
                        warn=(times[current]>=onset-DAY_NS)&(times[current]<=onset-2*STEP_NS)
                        if warn.any() and top[warn,int(event_gpus[event_id])].any():
                            capture[key].add(event_id)
            fusion_dir=ROOT/"fusion_scores"/f"{step}m"
            fusion_dir.mkdir(parents=True,exist_ok=True)
            fused={"score_bin":np.repeat(bins,G),"timestamp_ns":np.repeat(times[bins],G),
                   "gpu_idx":np.tile(np.arange(G,dtype=np.int16),len(bins))}
            for arm in config["arms"]:
                for suffix in ["B1_inertia","fusion_no_inertia","fusion_main"]:
                    fused[f"{arm}_{suffix}"]=candidates[f"{arm}/{suffix}"][0].astype(np.float32).ravel()
            output=pd.DataFrame(fused)
            target=fusion_dir/file.name
            if target.exists():
                old=pd.read_parquet(target)
                if not old.equals(output):
                    raise ValueError(f"Existing fusion artifact differs: {target}")
            else:
                output.to_parquet(target,index=False)
        print(f"[evaluation] {file.name}",flush=True)
    results=[]
    for (step,split,model),values in aggregates.items():
        sums={k:sum(v[k] for v in values) for k in ["rows","positive","hits","entries","transitions"]}
        aucs=[v["auc"] for v in values if v["auc"] is not None]
        denominator=len(denoms[step][split])
        results.append(dict(cadence_minutes=step,split=split,model=model,**sums,
                            prevalence=sums["positive"]/sums["rows"],
                            mean_origin_block_AP=float(np.mean([v["ap"] for v in values])),
                            mean_origin_block_ROC_AUC=float(np.mean(aucs)) if aucs else None,
                            pooled_recall_at_100=sums["hits"]/max(sums["positive"],1),
                            churn_entries_per_hour=(sums["entries"]/sums["transitions"])*(60/step) if sums["transitions"] else None,
                            unique_onset_eligible=denominator,unique_onset_captured=len(capture[(step,split,model)]),
                            onset_capture_rate=len(capture[(step,split,model)])/denominator if denominator else None,
                            evaluated_origin_blocks=len(values),partially_observed_blocks_included=True))
    pd.DataFrame(results).to_csv(ROOT/"metrics_summary.csv",index=False)
    pd.DataFrame(traces).to_csv(ROOT/"metrics_by_origin.csv",index=False)
    write_json(ROOT/"evaluation_manifest.json",dict(status="COMPLETED",branch_refits=0,
        main_config=dict(cap=.6,gamma=1.,half_life_hours=2.),
        sensitivity="D_only_development_not_terminal_selection",selected_from_terminal=False,
        primary_metric="mean average_precision_score across eligible origin blocks; not trapezoidal PR-AUC",
        roc_metric="mean origin-block ROC-AUC, not pooled ROC-AUC",
        terminal="retrospective terminal prequential; frozen algorithms may update on mature labels",
        full_period="includes terminal; not independent second test",
        onset="unique GPU/onset-time, full 24h-10min warning window eligible, not incident sessions",
        uncertainty="single historical trace/seed, no statistical generalization proof",
        input_missingness="legacy feature missing-value assumptions preserved; per-GPU label completeness unverified"))

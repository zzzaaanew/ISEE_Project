"""Analyze frozen five-minute score artifacts; no model fitting or score edits."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from record_run import score_metrics
from analyze_comparisons import comparison


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def aggregate(ms):
    positive=sum(x['positives'] for x in ms);hits=sum(x['hits_at_100'] for x in ms)
    queries=sum(x['decision_times'] for x in ms)
    return dict(mean_ap=float(np.mean([x['pr_auc'] for x in ms])),hits=hits,positive_rows=positive,
        recall=hits/positive,precision=hits/(queries*100),decision_times=queries,score_rows=queries*1992,
        mean_ndcg=float(np.mean([x['ndcg_at_100'] for x in ms])))


def run(p5,p30,data,output):
    if output==p5 or output==p30 or p5 in output.parents or p30 in output.parents:
        raise ValueError('Analysis output must be separate from the immutable source runs')
    audit5=read(p5/'independent_audit.json');audit30=read(p30/'independent_audit.json')
    if audit5['status']!='FIVE_MINUTE_SCORE_ARTIFACTS_VERIFIED' or audit5['origins']!=64:
        raise ValueError('Five-minute full audit missing')
    if audit30['status']!='SCORE_ARTIFACTS_VERIFIED' or audit30['origins']!=64:
        raise ValueError('Thirty-minute full audit missing')
    chosen=audit5['selected']
    if (chosen['candidate_type'],chosen['policy'],chosen['inertia'],chosen['fusion'])!=('B1','S21',True,None):
        raise ValueError('This analysis contract is the selected B1 ST21 + Inertia replay')
    keys=['30m_native','5m_full','5m_common30','5m_extra','30m_held_on5','5m_no_inertia']
    metrics={k:[] for k in keys}
    tops={k:[] for k in keys if k!='5m_extra'};times={k:[] for k in tops}
    cal={k:dict(brier=0.,logloss=0.,mean_probability=0.,rows=0) for k in ['5m_full','30m_held_on5','5m_no_inertia']}
    gpu_ids=None;max_raw=0.;score_abs=0.;common_rows=0;overlap=0;exact_order=0
    t0=time.perf_counter();input_hashes={}
    for i in range(64):
        name=f'origin_{i:04d}.parquet';f5path=p5/'scores'/name;f30path=p30/'scores'/name
        h5=digest(f5path);h30=digest(f30path)
        if h5!=audit5['score_sha256'][name] or h30!=audit30['score_sha256'][name]:
            raise ValueError('Source scores changed after their independent audit')
        input_hashes[name]=dict(five_minute=h5,thirty_minute=h30)
        f5=pq.read_table(f5path,columns=['timestamp','gpu_uid','label','risk_score','b1_probability']).to_pandas()
        f30=pq.read_table(f30path,columns=['timestamp','gpu_uid','label','q_S21_I','b1_S21']).to_pandas()
        n=1992;nt=len(f5)//n;nt30=len(f30)//n
        ids=f5.gpu_uid.astype(str).to_numpy().reshape(nt,n)
        if gpu_ids is None:gpu_ids=ids[0].copy()
        if not np.array_equal(ids,np.broadcast_to(gpu_ids,ids.shape)) or not np.array_equal(f30.gpu_uid.astype(str).to_numpy().reshape(nt30,n),np.broadcast_to(gpu_ids,(nt30,n))):
            raise ValueError('Evaluation GPU population/order differs')
        ts5=f5.timestamp.astype('int64').to_numpy().reshape(nt,n)[:,0]
        ts30=f30.timestamp.astype('int64').to_numpy().reshape(nt30,n)[:,0]
        selected=np.isin(ts5,ts30)
        y=f5.label.to_numpy().reshape(nt,n);y30=f30.label.to_numpy().reshape(nt30,n)
        if not np.array_equal(ts5[selected],ts30) or not np.array_equal(y[selected],y30):
            raise ValueError('Common-time evaluation keys/labels differ')
        q5=f5.risk_score.to_numpy().reshape(nt,n);q30=f30.q_S21_I.to_numpy().reshape(nt30,n)
        raw=f5.b1_probability.to_numpy().reshape(nt,n)
        max_raw=max(max_raw,float(abs(raw[selected]-f30.b1_S21.to_numpy().reshape(nt30,n)).max()))
        hold=np.repeat(q30,6,axis=0)[:nt]
        common_rows+=int(selected.sum())*n;score_abs+=float(abs(q5[selected]-q30).sum())
        datasets={'30m_native':(q30,y30,ts30),'5m_full':(q5,y,ts5),'5m_common30':(q5[selected],y30,ts30),
            '5m_extra':(q5[~selected],y[~selected],ts5[~selected]),'30m_held_on5':(hold,y,ts5),'5m_no_inertia':(raw,y,ts5)}
        current_tops={}
        for key,(score,yy,tt) in datasets.items():
            row=score_metrics(score.ravel(),yy.ravel());row.update(origin=i,arm=key);metrics[key].append(row)
            if key in tops:
                top=np.argsort(-score,axis=1,kind='stable')[:,:100].astype('int16')
                tops[key].append(top);times[key].append(tt);current_tops[key]=top
            if key in cal:
                pp=np.clip(score,1e-6,1-1e-6)
                cal[key]['brier']+=float(((pp-yy)**2).sum())
                cal[key]['logloss']+=float(-(yy*np.log(pp)+(1-yy)*np.log1p(-pp)).sum())
                cal[key]['mean_probability']+=float(pp.sum());cal[key]['rows']+=pp.size
        for a,b in zip(current_tops['30m_native'],current_tops['5m_common30']):
            overlap+=len(np.intersect1d(a,b));exact_order+=int(np.array_equal(a,b))
        if i%8==7:print(json.dumps(dict(stage='analysis_origins',completed=i+1,total=64)),flush=True)
    xidpath=data/'xid_onsets_metadata.parquet';xidsha=digest(xidpath)
    if xidsha!=read(p5/'feature_manifest.json')['input_sha256']['xid_onsets_metadata.parquet']:
        raise ValueError('Raw onset input identity differs')
    xid=pq.read_table(xidpath).to_pandas();xid['gpu_id']=xid.gpu_id.astype(str)
    xid['onset_time']=pd.to_datetime(xid.onset_time,utc=True);xid=xid.drop_duplicates(['gpu_id','onset_time'])
    mapping={g:i for i,g in enumerate(gpu_ids)};tall=np.concatenate(times['5m_full'])
    event_ns=xid.onset_time.astype('int64').to_numpy()
    eligible=xid.gpu_id.isin(mapping).to_numpy() & (event_ns-24*3600*10**9>=tall[0]) & (event_ns<=tall[-1])
    events=xid[eligible].copy();event_ns=events.onset_time.astype('int64').to_numpy();event_gpu=events.gpu_id.map(mapping).to_numpy()
    operations={}
    for key in tops:
        top=np.concatenate(tops[key]);tt=np.concatenate(times[key]);flat=top.ravel()
        order=np.argsort(flat,kind='stable');ordered=np.repeat(tt,100)[order]
        counts=np.bincount(flat,minlength=1992);offset=np.r_[0,np.cumsum(counts)]
        first=np.full(len(events),np.iinfo(np.int64).min,dtype='int64')
        for e,(g,stop) in enumerate(zip(event_gpu,event_ns)):
            a=ordered[offset[g]:offset[g+1]];j=np.searchsorted(a,stop-24*3600*10**9)
            if j<len(a) and a[j]<stop:first[e]=a[j]
        found=first!=np.iinfo(np.int64).min;lead=(event_ns[found]-first[found])/(3600*10**9)
        turnover=sum((100-len(np.intersect1d(a,b)))/100 for a,b in zip(top[:-1],top[1:]))
        hours=(tt[-1]-tt[0])/(3600*10**9)
        operations[key]=dict(eligible_onsets=len(events),captured_onsets=int(found.sum()),capture_rate=float(found.mean()),
            median_lead_hours=float(np.median(lead)),mean_lead_hours=float(np.mean(lead)),mean_turnover=turnover/(len(tt)-1),
            replacements_per_hour=turnover*100/hours,unique_selected_gpus=int((counts>0).sum()))
    frame=pd.DataFrame([x for rows in metrics.values() for x in rows]);pairs=[]
    for a,b in [('5m_common30','30m_native'),('5m_full','30m_held_on5'),('5m_full','5m_no_inertia')]:
        c=comparison(frame,a,b);c.pop('origins');pairs.append(c)
    for v in cal.values():
        for key in ['brier','logloss','mean_probability']:v[key]/=v['rows']
    recorded=pd.read_csv(p5/'metrics_by_origin.csv').seconds
    trace=[json.loads(s) for s in (p5/'fit_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    inference=[r for r in trace if r['stage']=='frozen_model_inference'];status=read(p5/'watcher_status.json')
    chain=(dt.datetime.fromisoformat(status['completed']['utc'])-dt.datetime.fromisoformat(status['created']['utc'])).total_seconds()
    aggregate_metrics={k:aggregate(v) for k,v in metrics.items()}
    for key,field in [('origin_mean_pr_auc','mean_ap'),('pooled_recall_at_100','recall'),('hits_at_100','hits')]:
        if not np.isclose(audit5[key],aggregate_metrics['5m_full'][field],rtol=1e-12,atol=1e-12):
            raise ValueError('Recomputed five-minute metrics differ from the independent audit')
    result=dict(status='FIVE_MINUTE_COMPARISON_ANALYZED',generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        selected='B1 ST21 + Inertia; frozen daily GBDT/LR; final ranking uses B1 only',metrics=aggregate_metrics,
        common_time=dict(raw_B1_max_abs_difference=max_raw,inertia_score_mean_abs_difference=score_abs/common_rows,
            mean_top100_overlap=overlap/3053,identical_top100_order_queries=exact_order,queries=3053),
        operations=operations,paired=pairs,calibration=cal,
        runtime=dict(origin_seconds_sum=float(recorded.sum()),origin_seconds_median=float(recorded.median()),
            origin_seconds_max=float(recorded.max()),cuda_inference_origins=sum(str(r['actual_backend']).startswith('cuda') for r in inference),
            chain_seconds=chain),
        provenance=dict(source_5m=str(p5),source_30m=str(p30),source_5m_audit_sha256=digest(p5/'independent_audit.json'),
            source_30m_audit_sha256=digest(p30/'independent_audit.json'),onset_sha256=xidsha,
            script_sha256=digest(Path(__file__)),score_parts_sha256=input_hashes),
        definitions=dict(hold='Use the latest available native 30min score until the next scheduled update; never a future score.',
            event='Unique GPU/onset with full preceding 24h inside the evaluation; selected at least once in [onset-24h,onset). Not verified hardware failure episodes.',
            turnover='Fraction of Top100 replaced per adjacent update; replacements/hour normalizes differing cadences.',
            lead='Conditional on capture; captured-event sets differ between methods.',
            uncertainty='Descriptive non-circular 3/7-origin moving-block bootstrap, 5000 replicates, seed 20260905. Development selection; no multiplicity-adjusted superiority claim.'),
        training_executed=False,analysis_seconds=time.perf_counter()-t0)
    output.mkdir(parents=True,exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    frame.to_csv(output/'comparisons_by_origin.csv',index=False)
    print(json.dumps(dict(status=result['status'],output=str(output),queries=aggregate_metrics['5m_full']['decision_times'],analysis_seconds=result['analysis_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-5m',type=Path,required=True);p.add_argument('--source-30m',type=Path,required=True)
    p.add_argument('--data-directory',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'results')
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a.source_5m.resolve(),a.source_30m.resolve(),a.data_directory.resolve(),a.output.resolve())

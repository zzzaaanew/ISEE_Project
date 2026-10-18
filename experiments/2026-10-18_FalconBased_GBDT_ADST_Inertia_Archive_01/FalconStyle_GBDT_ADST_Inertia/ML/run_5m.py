"""Conditional 5-minute inference replay of a frozen, positively evaluated setting.

GBDT/LR models and daily ADST routing come from the audited 30-minute run.
KL inertia preserves the original half-life at the new 5-minute cadence.
Selected E19 fusion uses the original causal daily model. No new fitting.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from threadpoolctl import threadpool_limits
import xgboost as xgb

import run_experiment as r

ROOT=r.ROOT

# Preserve the numerical implementation while tolerating transient Windows
# file sharing locks on status/manifest replacement in this new replay only.
_original_write_json=r.write_json
def durable_json(path,value):
    deadline=time.monotonic()+30
    while True:
        try:
            return _original_write_json(path,value)
        except PermissionError:
            if time.monotonic()>=deadline:
                raise
            time.sleep(.2)
r.write_json=durable_json


def expand_schedule(schedule):
    """Keep exactly the original first and last evaluation instants."""
    result=[]
    for i,sch in enumerate(schedule):
        end=schedule[i+1]['origin'] if i+1<len(schedule) else int(sch['bins'][-1])+1
        result.append(dict(origin=sch['origin'],bins=np.arange(sch['origin'],end,dtype='int32')))
    return result


def strict_selection(source,selection_path):
    selection=json.loads(Path(selection_path).read_text(encoding='utf-8'))
    if selection.get('decision')!='PROMOTE_5M':
        raise ValueError('No positive-result promotion was authorized by the frozen gate')
    supervisor=json.loads((source/'supervisor_status.json').read_text(encoding='utf-8'))
    audit=json.loads((source/'independent_audit.json').read_text(encoding='utf-8'))
    if supervisor['status']!='COMPLETED_AND_AUDITED' or audit['origins']!=64 or audit['status']!='SCORE_ARTIFACTS_VERIFIED':
        raise ValueError('The complete 30-minute source must pass its independent audit')
    causal=audit['source_fidelity_audit']
    if causal['purge_violations']!=0 or causal['label_maturity_violations']!=0:
        raise ValueError('Source causal audit rejected')
    contract=json.loads((source/'execution_contract.json').read_text(encoding='utf-8'))
    if contract['source_sha256']!=r.digest(ROOT/'ML/run_experiment.py'):
        raise ValueError('Frozen 30-minute implementation changed')
    for name,sha in contract['vendor_sha256'].items():
        if r.digest(ROOT/'vendor'/name)!=sha:
            raise ValueError('Frozen feature/fusion source changed')
    selected=selection['selected']
    if selected['policy'] not in r.POLICIES or not isinstance(selected['inertia'],bool):
        raise ValueError('Invalid promoted branch policy')
    if selected['candidate_type'] not in ('B1','Fusion'):
        raise ValueError('Invalid promoted predictor type')
    if selected['candidate_type']=='Fusion' and selected['fusion'] not in ('P','E'):
        raise ValueError('Invalid promoted fusion')
    if Path(selection['source']).resolve()!=source.resolve():
        raise ValueError('Promoted model source differs')
    if selection['source_contract_sha256']!=r.digest(source/'execution_contract.json'):
        raise ValueError('Promoted source execution contract differs')
    for relative,expected in selection['source_model_sha256'].items():
        if r.digest(source/relative)!=expected:
            raise ValueError('Audited daily trained model changed')
    return selection,contract['config'],audit


def model_predictions(engine,events,source,index,sch,policy,config,trace,device='cuda'):
    meta=json.loads((source/'branches'/f'origin_{index:04d}.json').read_text(encoding='utf-8'))
    days=meta['adst_days'] if policy=='A' else int(policy[1:])
    info=next(x for x in meta['fit_info'] if x['days']==days)
    origin=sch['origin']
    if info['training_end_bin_exclusive']!=origin-r.PURGE or info['origin_bin']!=origin:
        raise ValueError('Frozen GBDT training origin/purge differs')
    b=np.repeat(sch['bins'],engine.num_gpus)
    g=np.tile(np.arange(engine.num_gpus,dtype='int32'),len(sch['bins']))
    x=r.telemetry(engine,b,g,config['feature_chunk'],trace,f'five_minute_origin_{index}')
    model_path=source/'models'/f'b1_origin_{index:04d}_days_{days}.json'
    if info['rows']:
        model=xgb.XGBClassifier()
        model.load_model(model_path)
        model.set_params(device=device,n_jobs=config['cpu_threads'])
        actual=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        if device=='cuda' and not actual.startswith('cuda'):
            raise ValueError('Five-minute inference silently left CUDA')
        raw=model.predict_proba(x)[:,1].astype('float64')
        b1=r.natural_probability(raw,info['natural_prior'])
        model_sha=r.digest(model_path)
        del model
    else:
        b1=np.full(len(b),np.clip(info['natural_prior'],r.EPS,1-r.EPS));raw=b1.copy()
        actual='constant_fallback';model_sha=None
    del x
    source_b2=pq.read_table(source/'branches'/f'origin_{index:04d}.parquet',
        columns=['b2_prior','pareto_lambda']).to_pandas()
    prior2=float(source_b2.b2_prior.iloc[0])
    lam=source_b2.pareto_lambda.to_numpy()[:engine.num_gpus]
    b2_path=source/'models'/f'b2_origin_{index:04d}.joblib'
    if b2_path.exists():
        model2=joblib.load(b2_path)
        raw2=model2.predict_proba(r.history_features(engine,events,b,g))[:,1]
        b2=r.natural_probability(raw2,prior2)
        b2_sha=r.digest(b2_path)
        del model2
    elif prior2 in (0.,1.):
        b2=np.full(len(b),np.clip(prior2,r.EPS,1-r.EPS));raw2=b2.copy();b2_sha=None
    else:
        raise ValueError('Frozen B2 model missing')
    trace.emit('frozen_model_inference',origin=index,policy=policy,days=days,
        actual_backend=actual,model_sha256=model_sha,b2_model_sha256=b2_sha,
        model_training_end_bin_exclusive=info['training_end_bin_exclusive'],
        model_origin_bin=origin,rows=len(b),fresh_gbdt_fits=0,fresh_lr_fits=0)
    return dict(b=b,g=g,raw_b1=raw,b1=b1,b1_prior=info['natural_prior'],raw_b2=raw2,
        b2=b2,b2_prior=prior2,lam=lam,days=days,b1_model_sha256=model_sha,b2_model_sha256=b2_sha)


def compare_shared_times(source,index,policy,new_frame):
    original=pq.read_table(source/'branches'/f'origin_{index:04d}.parquet',
        columns=['timestamp','gpu_uid','label',f'b1_{policy}','b2']).to_pandas()
    common=new_frame[new_frame.timestamp.isin(original.timestamp.unique())]
    if not common[['timestamp','gpu_uid','label']].reset_index(drop=True).equals(original[['timestamp','gpu_uid','label']]):
        raise ValueError('Shared 30-minute query keys/ground truth changed')
    max1=float(np.max(np.abs(common.b1_probability.to_numpy()-original[f'b1_{policy}'].to_numpy())))
    max2=float(np.max(np.abs(common.b2_probability.to_numpy()-original.b2.to_numpy())))
    if max1>1e-6 or max2>1e-9:
        raise ValueError(f'Frozen predictor changed at common times: B1={max1}, B2={max2}')
    return dict(common_rows=len(common),b1_max_score_difference=max1,b2_max_score_difference=max2)


def run(source,output,selection_path,limit,resume):
    selection,config,source_audit=strict_selection(source,selection_path)
    chosen=selection['selected'];policy=chosen['policy']
    use_e19=chosen['candidate_type']=='Fusion' and chosen['fusion']=='E'
    use_inertia=chosen['inertia']
    output.mkdir(parents=True,exist_ok=True)
    trace=r.Trace(output)
    signature=dict(source_sha256=r.digest(__file__),selection_sha256=r.digest(selection_path),
        source_execution_contract_sha256=r.digest(source/'execution_contract.json'),
        frozen_main_sha256=r.digest(ROOT/'ML/run_experiment.py'),selected=chosen,
        source_directory=str(source),config=config,scoring_minutes=5,retraining_hours=24,
        b1_b2='frozen per-origin models; no new fitting',
        adst='frozen causal daily window routing from the 30-minute source',
        inertia='same half-life; rho=2**(-(5/60)/half_life_hours)',
        fusion='Frozen causal daily E19 model if selected; Pareto unchanged otherwise',
        statistical_scope='Development-period model selection and denser replay, not independent confirmation')
    contract=output/'execution_contract.json'
    if contract.exists():
        if not resume or json.loads(contract.read_text(encoding='utf-8'))!=signature:
            raise ValueError('Five-minute resume identity differs')
    else:
        r.write_json(contract,signature)
    engine,events,gt,math,manifest=r.load_engine(config,output,trace)
    original_schedule=r.schedule_from_reference(engine,config)
    schedule=expand_schedule(original_schedule)[:limit]
    sdir=output/'scores';sdir.mkdir(exist_ok=True)
    mdir=output/'fusion_models';mdir.mkdir(exist_ok=True)
    topdir=output/'top100';topdir.mkdir(exist_ok=True)
    cp=output/'state/checkpoint.joblib';cp.parent.mkdir(exist_ok=True)
    if cp.exists():
        state=joblib.load(cp);memory=state['memory'];last_ns=state['last_ns']
        metrics=state['metrics'];start=state['last_origin']+1
    else:
        memory=None;last_ns=None;metrics=[];start=0
    for i in range(start,len(schedule)):
        sch=schedule[i];origin_ns=int(engine.bin_start_ns[sch['origin']]);t0=time.perf_counter()
        trace.emit('five_minute_origin_start',origin=i,total=len(schedule),queries=len(sch['bins']))
        if use_e19:
            base=policy+('_I' if use_inertia else '_0')
            fusion=json.loads((source/'fusion_models'/f'origin_{i:04d}.json').read_text(encoding='utf-8'))[base]
            if fusion['cutoff_ns']!=origin_ns-36*12*r.STEP:
                raise ValueError('Frozen E19 cutoff differs')
            if fusion.get('latest_fit_score_ns',0)+r.HORIZON*r.STEP>origin_ns:
                raise ValueError('Frozen E19 used immature labels')
        else:
            fusion=dict(status='not_selected')
        r.write_json(mdir/f'origin_{i:04d}.json',fusion)
        p=model_predictions(engine,events,source,i,sch,policy,config,trace)
        n=engine.num_gpus;nt=len(sch['bins'])
        raw1=p['b1'].reshape(nt,n);b2=p['b2'].reshape(nt,n)
        y=gt[p['b'],p['g']].astype('uint8').reshape(nt,n)
        q=np.empty_like(raw1);risk=np.empty_like(raw1);mem_rows=np.empty_like(raw1)
        previous_times=[];rhos=[]
        for j,binid in enumerate(sch['bins']):
            ts=int(engine.bin_start_ns[binid]);z=math.clipped_logit(raw1[j])
            reset=last_ns is None or ts-last_ns>=r.DAY*r.STEP
            prior=z.copy() if reset else memory.copy()
            if not reset and last_ns>=ts:
                raise ValueError('Future/current inertia state')
            eta=(z+config['inertia_gamma']*prior)/(1+config['inertia_gamma']) if use_inertia else z
            q[j]=np.clip(expit(eta),r.EPS,1-r.EPS)
            if chosen['candidate_type']=='B1':
                risk[j]=q[j]
            elif chosen['fusion']=='P':
                risk[j]=p['lam']*q[j]+(1-p['lam'])*b2[j]
            else:
                risk[j]=r.e19_predict(eta,math.clipped_logit(b2[j]),b2[j],fusion,math)
            mem_rows[j]=prior
            previous_times.append(np.iinfo(np.int64).min if reset else last_ns)
            delta=(ts-last_ns)/(12*r.STEP) if last_ns is not None else 1/12
            rho=2**(-delta/config['inertia_half_life_hours']);rhos.append(rho)
            memory=z.copy() if reset else rho*memory+(1-rho)*z
            last_ns=ts
        order=np.argsort(-risk,axis=1,kind='stable')
        ranks=np.empty_like(order,dtype='int16')
        np.put_along_axis(ranks,order,np.arange(1,n+1,dtype='int16')[None,:],axis=1)
        frame=pd.DataFrame(dict(timestamp=pd.to_datetime(engine.bin_start_ns[p['b']],utc=True),
            gpu_uid=engine.gpu_ids[p['g']].astype(str),label=y.ravel(),b1_model_raw=p['raw_b1'],
            b1_prior=p['b1_prior'],b1_probability=p['b1'],b2_model_raw=p['raw_b2'],b2_prior=p['b2_prior'],
            b2_probability=p['b2'],b1_inertia_probability=q.ravel(),memory_logit=mem_rows.ravel(),
            memory_previous_timestamp=pd.to_datetime(np.repeat(np.asarray(previous_times,dtype='int64'),n),utc=True),
            inertia_rho=np.repeat(rhos,n),pareto_lambda=np.tile(p['lam'],nt),
            risk_score=risk.ravel(),rank=ranks.ravel(),b1_train_days=p['days'],model_origin_bin=sch['origin']))
        frame['feature_cutoff_time']=frame.timestamp-pd.Timedelta(minutes=10)
        frame['latest_telemetry_time']=frame.timestamp-pd.Timedelta(minutes=15)
        common=compare_shared_times(source,i,policy,frame)
        part=sdir/f'origin_{i:04d}.parquet'
        frame.to_parquet(part,index=False,compression='zstd')
        frame[frame['rank']<=100].to_parquet(topdir/part.name,index=False,compression='zstd')
        entry=dict(origin=i,policy=policy,inertia=use_inertia,fusion=chosen['fusion'],
            fusion_status=fusion['status'],model_training_end_bin_exclusive=sch['origin']-r.PURGE,
            model_origin_bin=sch['origin'],b1_model_sha256=p['b1_model_sha256'],b2_model_sha256=p['b2_model_sha256'],
            score_sha256=r.digest(part),seconds=time.perf_counter()-t0,common_30m=common,**r.metric(risk,y))
        r.write_json(sdir/f'origin_{i:04d}.identity.json',entry)
        metrics.append(entry)
        pd.DataFrame([{k:v for k,v in m.items() if not isinstance(v,dict)} for m in metrics]).to_csv(output/'metrics_by_origin.csv',index=False)
        state=dict(memory=memory,last_ns=last_ns,metrics=metrics,last_origin=i)
        tmp=cp.with_suffix('.tmp');joblib.dump(state,tmp);tmp.replace(cp)
        trace.emit('five_minute_origin_done',origin=i,completed=i+1,total=len(schedule),
            rows=len(frame),queries=nt,fusion_status=fusion['status'],seconds=entry['seconds'])
        del p,frame,raw1,b2,y,q,risk,mem_rows
        gc.collect()
    positive=sum(m['positives'] for m in metrics);hits=sum(m['hits_at_100'] for m in metrics)
    result=dict(status='FULL_5M_FINISHED_PENDING_AUDIT' if len(metrics)==64 else 'PARTIAL_5M',
        origins_completed=len(metrics),decision_times=sum(m['decision_times'] for m in metrics),
        score_rows=sum(m['decision_times'] for m in metrics)*1992,positive_rows=positive,
        hits_at_100=hits,pooled_recall_at_100=hits/positive if positive else 0.,
        origin_mean_pr_auc=float(np.mean([m['pr_auc'] for m in metrics])),selected=chosen,
        scoring_minutes=5,purge_hours=36,inertia_half_life_hours=config['inertia_half_life_hours'],
        regular_5m_rho=2**(-(1/12)/config['inertia_half_life_hours']),
        fresh_gbdt_fits=0,fresh_lr_fits=0,fresh_fusion_fits=0,e19_frozen_trained_origins=sum(m['fusion_status']=='trained' for m in metrics),
        score_provenance='Fresh 5-minute features and scores with frozen daily GBDT/LR models',
        evaluation_scope='Denser replay on the selected development period; not a new test set')
    r.write_json(output/'summary.json',result);r.write_json(output/'status.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--limit-origins',type=int,default=64);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 1<=a.limit_origins<=64:
        raise ValueError('Invalid origin limit')
    with threadpool_limits(limits=2):
        run(a.source.resolve(),a.output.resolve(),a.selection.resolve(),a.limit_origins,a.resume)

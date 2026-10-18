"""Approved specification v1.1: fresh FalconStyle GBDT, LR, ADST, KL inertia, two fusions.

Reference score files supply evaluation timestamps only. No learned reference score
enters training, feedback, features, calibration, or prediction.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
STEP = 300_000_000_000
DAY = 288
PURGE = 432
HORIZON = 288
EPS = 1e-6
POLICIES = ('S7', 'S14', 'S21', 'A')
BASES = tuple(f'{p}_{i}' for p in POLICIES for i in ('0', 'I'))
ARMS = tuple(f'{b}_{f}' for b in BASES for f in ('P', 'E'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')
    tmp.replace(path)


class Trace:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()

    def emit(self, stage, **values):
        row = dict(stage=stage, utc=pd.Timestamp.now(tz='UTC').isoformat(),
                   elapsed_seconds=time.perf_counter()-self.start,
                   rss_mb=psutil.Process().memory_info().rss/2**20, **values)
        with (self.output/'fit_trace.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')
        write_json(self.output/'status.json', {'status':'RUNNING', **row})
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)


def load_reference_code():
    """Execute the original feature class and original E19 math, without Torch imports."""
    vendor = ROOT/'vendor'
    base_path = vendor/'run_bidirectional_adst_fusion.py'
    module = types.ModuleType('run_bidirectional_adst_fusion')
    module.__dict__.update(np=np, pd=pd, pq=pq, Path=Path, time=time,
        METRICS=['util','temp','power','fb'], STEP_NS=STEP, DAY_NS=DAY*STEP,
        HOUR_NS=12*STEP, MINUTE_NS=STEP//5, BUFFER_MASK_BINS=2,
        L_OBS_BINS_MAP={1:12,6:72,24:288}, HORIZON_BINS=HORIZON)
    tree = ast.parse(base_path.read_text(encoding='utf-8-sig'))
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes += [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='UnifiedDataEngine']
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(base_path), 'exec'), module.__dict__)
    sys.modules[module.__name__] = module
    feature_path = vendor/'branch1_enhanced_features.py'
    spec = importlib.util.spec_from_file_location('frozen_falcon_features', feature_path)
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    math_path = vendor/'run_experiment19_30m_dual_inertia_fusion.py'
    names = {'clipped_logit','fitted_calibrator','calibrate','fitted_residual'}
    original = ast.parse(math_path.read_text(encoding='utf-8-sig'))
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    math = types.SimpleNamespace(np=np, expit=expit, logit=logit, minimize=minimize,
                                 EPS=EPS, REGULARIZATION=0.01)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(math_path), 'exec'), math.__dict__)
    return features, math


def load_engine(config, output, trace):
    identity = {}
    inventory = json.loads(Path(config['data_inventory']).read_text(encoding='utf-8'))
    expected = {Path(r['path']).name:r['sha256'] for r in inventory['lfs_objects']
                if r['path'].endswith('.parquet') and '/data/' in r['path'].replace('\\','/')}
    for name in ('telemetry_5m_util.parquet','telemetry_5m_temp.parquet',
                 'telemetry_5m_power.parquet','telemetry_5m_fb.parquet','xid_onsets_metadata.parquet'):
        path = Path(config['data_directory'])/name
        identity[name] = digest(path)
        if identity[name] != expected[name]:
            raise ValueError(f'Raw Parquet identity changed: {name}')
    feature_module, math = load_reference_code()
    engine = feature_module.EnhancedTelemetryEngine(data_dir=Path(config['data_directory']),
        cache_dir=Path(config['cache_directory']), include_topology=True, include_falcon=True)
    if engine.num_gpus != 1992 or not np.all(np.diff(engine.bin_start_ns)==STEP):
        raise ValueError('Frozen GPU/time grid differs')
    # Match cached feature matrices to the independently SHA-bound Parquet cache.
    for metric in ('util','temp','power','fb'):
        verified_dir = Path(config['verified_parquet_cache'])
        meta = json.loads((verified_dir/f'{metric}.identity.json').read_text())
        if meta['source_sha256'] != identity[f'telemetry_5m_{metric}.parquet']:
            raise ValueError('Cached raw source identity mismatch')
        verified = np.load(verified_dir/f'{metric}.mean.npy', mmap_mode='r')
        for i in range(0,len(verified),512):
            if not np.array_equal(engine.matrices[metric][i:i+512], verified[i:i+512], equal_nan=True):
                raise ValueError(f'Feature cache differs from verified Parquet mapping: {metric}')
            a = engine.matrices[metric][i:i+512].reshape(-1,engine.num_nodes,8)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                means = np.nanmean(a, axis=2)
            if not np.allclose(means,engine.node_matrices[metric][i:i+512],rtol=1e-5,atol=1e-5,equal_nan=True):
                raise ValueError(f'Node cache differs: {metric}')
        del verified
    ledger, events, gt = engine.load_all_xid_ledger()
    if len(ledger)!=2425:
        raise ValueError('Unexpected All-XID ledger')
    b, g = np.array([9000,9000,9000]), np.array([0,1,7])
    frame, _ = engine.extract_branch1_features(b,g,1)
    if frame.shape != (3,45) or len(set(frame.columns))!=45:
        raise ValueError('Actual new Falcon extractor is not 45 columns')
    manifest = feature_module.enhanced_feature_manifest(engine)
    manifest.update(features=list(frame.columns), feature_count=45,
        extractor='frozen_falcon_features.EnhancedTelemetryEngine.extract_branch1_features',
        vendor_sha256={p.name:digest(p) for p in (ROOT/'vendor').glob('*.py')},
        input_sha256=identity, b2_features=['xid_count_30d','days_since_xid'],
        nominal_cutoff_minutes=10, actual_latest_telemetry_lag_minutes=15,
        heavy_import_adapter='original UnifiedDataEngine class AST; no feature-method changes')
    write_json(Path(output)/'feature_manifest.json', manifest)
    trace.emit('data_ready', feature_count=45, gpu_count=1992, xid_rows=len(ledger), input_sha256=identity)
    return engine, events, gt, math, manifest


def schedule_from_reference(engine, config):
    schedule=[]
    for part in sorted(Path(config['reference_directory']).glob('score_part_*.parquet')):
        ns = pq.read_table(part,columns=['timestamp']).column(0).cast(pa.int64()).to_numpy()
        times=np.unique(ns)
        bins=((times-int(engine.bin_start_ns[0]))//STEP).astype('int32')
        if len(ns)!=len(bins)*engine.num_gpus or np.any(np.diff(bins)!=6):
            raise ValueError('Reference query schedule invalid')
        schedule.append(dict(origin=int(bins[0]),bins=bins))
    if len(schedule)!=64 or sum(len(s['bins']) for s in schedule)!=3053:
        raise ValueError('Reference schedule is not the approved 64 origins')
    return schedule


def sample_rows(gt, start, end, seed):
    yview=gt[start:end]
    p_b,p_g=np.where(yview)
    n=gt.shape[1]
    p=(p_b.astype('int64')+start)*n+p_g
    total=(end-start)*n
    prior=float(len(p)/total)
    if len(p)==0 or len(p)==total:
        return None,None,None,prior
    wanted=min(10*len(p),200000,total-len(p))
    rng=np.random.default_rng(seed)
    neg=np.empty(0,dtype='int64')
    while len(neg)<wanted:
        z=rng.integers(start*n,end*n,size=max(5000,2*(wanted-len(neg))),dtype='int64')
        z=z[~gt[z//n,z%n]]
        neg=np.unique(np.r_[neg,z])
    ids=np.sort(np.r_[p,rng.choice(neg,size=wanted,replace=False)])
    b,g=(ids//n).astype('int32'),(ids%n).astype('int32')
    return b,g,gt[b,g].astype('uint8'),prior


def telemetry(engine,b,g,chunk,trace,tag):
    result=np.empty((len(b),45),dtype='float32')
    for i in range(0,len(b),chunk):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            f,_=engine.extract_branch1_features(b[i:i+chunk],g[i:i+chunk],1)
        if len(f.columns)!=45 or not np.isfinite(f.to_numpy()).all():
            raise ValueError('Invalid actual Falcon feature values')
        result[i:i+chunk]=f.to_numpy(dtype='float32')
        if i==0 or (i//chunk)%40==0:
            trace.emit('feature_progress',tag=tag,done=min(i+chunk,len(b)),total=len(b))
    return result


def history_features(engine,events,b,g):
    f=engine.extract_branch2_features(b,g,events)
    return f[['xid_count_30d','days_since_xid']].to_numpy(dtype='float64')


def natural_probability(p,prior):
    return np.clip(expit(logit(np.clip(p,EPS,1-EPS))+logit(np.clip(prior,EPS,1-EPS))),EPS,1-EPS)


def metric(score,y):
    if score.shape!=y.shape or not np.isfinite(score).all():
        raise ValueError('Invalid full-population score shape/values')
    order=np.argsort(-score,axis=1,kind='stable')[:,:100]
    chosen=np.take_along_axis(y,order,axis=1)
    hits,positive=int(chosen.sum()),int(y.sum())
    d=1/np.log2(np.arange(2,102))
    ideal=np.r_[0.,np.cumsum(d)]
    ptime=y.sum(axis=1)
    valid=ptime>0
    query_ap=[float(average_precision_score(yy,ss)) if yy.any() else 0. for yy,ss in zip(y,score)]
    return dict(pr_auc=float(average_precision_score(y.ravel(),score.ravel())) if positive else 0.,
        hits_at_100=hits,positives=positive,recall_at_100=hits/positive if positive else 0.,
        precision_at_100=hits/(len(y)*100),decision_times=len(y),gpu_count=y.shape[1],
        ndcg_at_100=float(np.mean((chosen@d)[valid]/ideal[np.minimum(ptime[valid],100)])) if valid.any() else 0.,
        query_mean_ap=float(np.mean(query_ap)))


def update_adst(state,feedback,origin,trace):
    for block in feedback:
        if block['id'] in state['processed'] or block['latest_bin']>origin-PURGE:
            continue
        if block['latest_bin']+HORIZON>origin:
            raise ValueError('Immature ADST feedback')
        ap,prev=block['query_mean_ap'],block['prevalence']
        value=float(np.clip((ap-prev)/max(1-prev,1e-8),0,1))
        recent=state['values'][-3:]+[value]
        prior=state['values'][-6:-3]
        r=float(np.mean(recent));p=float(np.mean(prior)) if prior else 0.
        drop=max(0.,(p-r)/max(abs(p),1e-4)) if prior else 0.
        idx=[7,14,21].index(state['days'])
        before=state['days']
        if drop>=.15:
            state['days']=[7,14,21][max(0,idx-1)]
            state['stable']=0
            action='shrink' if idx>0 else 'hold_min'
        else:
            state['stable']+=1
            action='hold'
            if state['stable']>=3 and idx<2:
                state['days']=[7,14,21][idx+1];state['stable']=0;action='expand'
        state['processed'].append(block['id']);state['values'].append(value)
        row=dict(origin_bin=origin,feedback_id=block['id'],latest_feedback_bin=block['latest_bin'],
            cutoff_bin=origin-PURGE,label_maturity_latest_bin=block['latest_bin']+HORIZON,
            before_days=before,after_days=state['days'],norm_ap=value,drop=drop,action=action)
        state['updates'].append(row)
        trace.emit('adst_update',**row)


def fit_branches(engine,events,gt,schedule,config,output,trace):
    bdir=output/'branches';bdir.mkdir(exist_ok=True)
    models=output/'models';models.mkdir(exist_ok=True)
    feedback=[]
    state=dict(days=14,stable=0,processed=[],values=[],updates=[])
    for idx,sch in enumerate(schedule):
        part=bdir/f'origin_{idx:04d}.parquet'
        meta_path=bdir/f'origin_{idx:04d}.json'
        if part.exists() and meta_path.exists():
            saved=json.loads(meta_path.read_text())
            if digest(part)!=saved['score_sha256']:
                raise ValueError('Branch part identity changed')
            state=saved['adst_state'];feedback.append(saved['feedback'])
            trace.emit('branch_resume',origin=idx)
            continue
        started=time.perf_counter();origin=sch['origin'];end=origin-PURGE
        update_adst(state,feedback,origin,trace)
        trace.emit('branch_origin_start',origin=idx,origin_bin=origin,adst_days=state['days'],purge_hours=36)
        tb=np.repeat(sch['bins'],engine.num_gpus)
        tg=np.tile(np.arange(engine.num_gpus,dtype='int32'),len(sch['bins']))
        tx=telemetry(engine,tb,tg,config['feature_chunk'],trace,f'origin_{idx}_test')
        vals={};fit_info=[]
        for days in (7,14,21):
            start=end-days*DAY
            seed=config['seed']+idx*1009+days
            b,g,y,prior=sample_rows(gt,start,end,seed)
            model_path=models/f'b1_origin_{idx:04d}_days_{days}.json'
            fit_start=time.perf_counter()
            if b is None:
                scores=np.full(len(tb),np.clip(prior,EPS,1-EPS))
                raw_scores=scores.copy()
                actual='constant_fallback';size=0
            else:
                if int(b.max())>origin-PURGE or int(b.max())+HORIZON>origin:
                    raise ValueError('B1 36h purge violation')
                x=telemetry(engine,b,g,config['feature_chunk'],trace,f'origin_{idx}_train_{days}')
                count=int(y.sum());negative=len(y)-count
                model=xgb.XGBClassifier(n_estimators=150,max_depth=6,learning_rate=.06,
                    reg_lambda=1.5,min_child_weight=1,subsample=1.,colsample_bytree=1.,
                    objective='binary:logistic',tree_method='hist',device='cuda',
                    scale_pos_weight=negative/count,random_state=seed,n_jobs=config['cpu_threads'])
                trace.emit('fit_start',component='B1',origin=idx,days=days,rows=len(y),
                           training_start_bin=start,training_end_bin_exclusive=end,
                           training_max_bin=int(b.max()),label_end_max_bin=int(b.max())+HORIZON)
                model.fit(x,y)
                actual=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
                if not actual.startswith('cuda') or model.get_booster().num_boosted_rounds()!=150:
                    raise ValueError('GBDT did not fit the specified 150 trees on CUDA')
                model.save_model(model_path)
                raw_scores=model.predict_proba(tx)[:,1].astype('float64')
                scores=natural_probability(raw_scores,prior)
                size=len(y)
                del model,x,b,g,y
                gc.collect()
            vals[f'b1_S{days}']=scores.astype('float64')
            vals[f'b1_model_raw_S{days}']=raw_scores
            vals[f'b1_prior_S{days}']=np.full(len(tb),prior)
            info=dict(component='B1',origin=idx,days=days,actual_backend=actual,
                rows=size,natural_prior=prior,training_start_bin=start,training_end_bin_exclusive=end,
                origin_bin=origin,seconds=time.perf_counter()-fit_start,tree_count=150 if size else 0)
            fit_info.append(info);trace.emit('fit_done',**info)
        start=end-21*DAY
        b2_start=time.perf_counter()
        b,g,y,prior=sample_rows(gt,start,end,config['seed']+idx*1009+21)
        if b is None:
            b2=np.full(len(tb),np.clip(prior,EPS,1-EPS));b2_rows=0
            b2_raw=b2.copy()
        else:
            if int(b.max())>origin-PURGE or int(b.max())+HORIZON>origin:
                raise ValueError('B2 purge violation')
            x=history_features(engine,events,b,g)
            lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=250,class_weight='balanced',
                                                                 random_state=config['seed']))
            trace.emit('fit_start',component='B2',origin=idx,days=21,rows=len(y),features=2,
                training_start_bin=start,training_end_bin_exclusive=end,training_max_bin=int(b.max()),
                label_end_max_bin=int(b.max())+HORIZON)
            lr.fit(x,y)
            b2_raw=lr.predict_proba(history_features(engine,events,tb,tg))[:,1]
            b2=natural_probability(b2_raw,prior)
            joblib.dump(lr,models/f'b2_origin_{idx:04d}.joblib')
            b2_rows=len(y)
            del b,g,y,x,lr
        trace.emit('fit_done',component='B2',origin=idx,days=21,actual_backend='cpu',rows=b2_rows,
                   natural_prior=prior,origin_bin=origin,training_end_bin_exclusive=end,
                   seconds=time.perf_counter()-b2_start)
        vals['b1_A']=vals[f'b1_S{state["days"]}'].copy()
        vals['b1_model_raw_A']=vals[f'b1_model_raw_S{state["days"]}'].copy()
        vals['b1_prior_A']=vals[f'b1_prior_S{state["days"]}'].copy()
        vals['b2']=b2.astype('float64')
        vals['b2_model_raw']=b2_raw
        vals['b2_prior']=np.full(len(tb),prior)
        # Exact scratch Pareto: count strictly before origin-10min; no cap or floor.
        cutoff=int(engine.bin_start_ns[origin])-2*STEP
        counts=np.zeros(engine.num_gpus,dtype='int32')
        for g,ev in events.items():
            counts[g]=np.searchsorted(ev,cutoff,side='left')-np.searchsorted(ev,cutoff-30*DAY*STEP,side='left')
        vals['pareto_lambda']=np.tile(1/(counts.astype('float64')+1),len(sch['bins']))
        yy=gt[tb,tg].astype('uint8')
        a_metric=metric(vals['b1_A'].reshape(-1,engine.num_gpus),yy.reshape(-1,engine.num_gpus))
        block=dict(id=idx,latest_bin=int(sch['bins'][-1]),query_mean_ap=a_metric['query_mean_ap'],
                   prevalence=float(yy.mean()))
        feedback.append(block)
        frame=pd.DataFrame(dict(timestamp=pd.to_datetime(engine.bin_start_ns[tb],utc=True),
                                gpu_uid=engine.gpu_ids[tg].astype(str),label=yy,**vals))
        frame.to_parquet(part,index=False,compression='zstd')
        saved=dict(origin=idx,origin_bin=origin,adst_days=state['days'],adst_state=state,
            feedback=block,fit_info=fit_info,score_sha256=digest(part),rows=len(frame),
            label_sha256=hashlib.sha256(yy.tobytes()).hexdigest(),seconds=time.perf_counter()-started)
        write_json(meta_path,saved)
        pd.DataFrame(state['updates']).to_csv(output/'adst_feedback.csv',index=False)
        trace.emit('branch_origin_done',origin=idx,rows=len(frame),adst_days=state['days'],
                   seconds=saved['seconds'],completed=idx+1,total=len(schedule))
        del tx,vals,frame,b2
        gc.collect()


def fit_e19(history,origin_ns,base_index,math,trace,origin_index,base):
    cutoff=origin_ns-36*12*STEP
    begin=cutoff-14*DAY*STEP;split=cutoff-7*DAY*STEP
    result=dict(status='cold_start_b2_fallback',cutoff_ns=cutoff,calibration_start_ns=begin,
                fusion_start_ns=split)
    if not history or history[0][0]>begin+6*STEP:
        return result
    old=[r for r in history if begin<=r[0]<split]
    new=[r for r in history if split<=r[0]<=cutoff]
    if not old or not new:
        return result
    if new[-1][0]+HORIZON*STEP>origin_ns:
        raise ValueError('Immature fusion feedback')
    y0=np.concatenate([r[3] for r in old]).astype('float64')
    y1=np.concatenate([r[3] for r in new]).astype('float64')
    if min(y0.sum(),len(y0)-y0.sum(),y1.sum(),len(y1)-y1.sum())<20:
        result['status']='class_count_b2_fallback'
        return result
    trace.emit('fusion_fit_start',origin=origin_index,base=base,rows=len(y1),calibration_rows=len(y0))
    c1=math.fitted_calibrator(np.concatenate([r[1][base_index] for r in old]),y0)
    c2=math.fitted_calibrator(np.concatenate([r[2] for r in old]),y0)
    p1=math.calibrate(np.concatenate([r[1][base_index] for r in new]),c1)
    p2=math.calibrate(np.concatenate([r[2] for r in new]),c2)
    z1=math.clipped_logit(p1)/6;z2=math.clipped_logit(p2)/6
    x=np.column_stack([np.ones(len(z1)),z1,z2,z1*z1,z2*z2,z1*z2])
    beta=math.fitted_residual(x,math.clipped_logit(p2),y1)
    result.update(status='trained',calibrators=[c1,c2],beta=beta,calibration_rows=len(y0),
                  fusion_rows=len(y1),latest_fit_score_ns=int(new[-1][0]))
    trace.emit('fusion_fit_done',origin=origin_index,base=base,actual_backend='cpu_scipy',
        cutoff_ns=cutoff,latest_fit_score_ns=int(new[-1][0]),label_end_max_ns=int(new[-1][0])+HORIZON*STEP,
        origin_ns=origin_ns,calibration_rows=len(y0),rows=len(y1))
    return result


def e19_predict(z1,z2,raw_b2,model,math):
    if model['status']!='trained':
        return raw_b2.copy()
    p1=math.calibrate(z1,model['calibrators'][0]);p2=math.calibrate(z2,model['calibrators'][1])
    a=math.clipped_logit(p1)/6;b=math.clipped_logit(p2)/6
    x=np.column_stack([np.ones(len(a)),a,b,a*a,b*b,a*b])
    return np.clip(expit(math.clipped_logit(p2)+x@np.asarray(model['beta'])),EPS,1-EPS)


def fuse(schedule,config,output,engine,math,trace):
    sdir=output/'scores';sdir.mkdir(exist_ok=True)
    mdir=output/'fusion_models';mdir.mkdir(exist_ok=True)
    checkpoint=output/'state/fusion_checkpoint.joblib';checkpoint.parent.mkdir(exist_ok=True)
    if checkpoint.exists():
        state=joblib.load(checkpoint)
        memory=state['memory'];history=deque(state['history']);rows=state['metrics']
        start=state['last_origin']+1;last_ns=state['last_ns']
        trace.emit('fusion_resume',completed=start)
    else:
        memory=None;history=deque();rows=[];start=0;last_ns=None
    for idx in range(start,len(schedule)):
        t0=time.perf_counter();sch=schedule[idx];origin_ns=int(engine.bin_start_ns[sch['origin']])
        models=[fit_e19(history,origin_ns,k,math,trace,idx,base) for k,base in enumerate(BASES)]
        write_json(mdir/f'origin_{idx:04d}.json',dict(zip(BASES,models)))
        frame=pq.read_table(output/'branches'/f'origin_{idx:04d}.parquet').to_pandas()
        n=engine.num_gpus;nt=len(frame)//n
        raw=np.stack([frame[f'b1_{p}'].to_numpy().reshape(nt,n) for p in POLICIES])
        b2=frame.b2.to_numpy().reshape(nt,n);yy=frame.label.to_numpy().reshape(nt,n)
        lam=frame.pareto_lambda.to_numpy().reshape(nt,n)
        risk={arm:np.empty((nt,n),dtype='float64') for arm in ARMS}
        memory_rows={p:np.empty((nt,n),dtype='float64') for p in POLICIES}
        q_rows={b:np.empty((nt,n),dtype='float64') for b in BASES}
        previous_times=[]
        for j,binid in enumerate(sch['bins']):
            ts=int(engine.bin_start_ns[binid]);z=math.clipped_logit(raw[:,j]);z2=math.clipped_logit(b2[j])
            previous=memory.copy() if memory is not None else z.copy()
            missing=last_ns is None or ts-last_ns>=DAY*STEP
            if missing:
                previous=z.copy()
            elif last_ns>=ts:
                raise ValueError('Inertia state contains current/future score')
            previous_times.append(np.iinfo(np.int64).min if missing else last_ns)
            adjusted=np.stack([a for k in range(4) for a in (z[k],
                (z[k]+config['inertia_gamma']*previous[k])/(1+config['inertia_gamma']))])
            if not np.array_equal(adjusted[::2],z):
                raise ValueError('Inertia-off baseline changed')
            for k,base in enumerate(BASES):
                q=np.clip(expit(adjusted[k]),EPS,1-EPS)
                q_rows[base][j]=q
                risk[f'{base}_P'][j]=lam[j]*q+(1-lam[j])*b2[j]
                risk[f'{base}_E'][j]=e19_predict(adjusted[k],z2,b2[j],models[k],math)
            for k,p in enumerate(POLICIES):
                memory_rows[p][j]=previous[k]
            # Save only information actually produced at this timestamp.
            history.append((ts,adjusted.copy(),z2.copy(),yy[j].copy()))
            delta_hours=(ts-last_ns)/(12*STEP) if last_ns is not None else .5
            rho=2**(-delta_hours/config['inertia_half_life_hours'])
            memory=z.copy() if missing else rho*memory+(1-rho)*z
            trace.emit('inertia_predict_then_update',origin=idx,timestamp_ns=ts,
                previous_state_ns=last_ns if not missing else None,reset=missing,rho=rho)
            last_ns=ts
        while history and history[0][0]<origin_ns-17*DAY*STEP:
            history.popleft()
        for arm in ARMS:
            score=risk[arm]
            frame[f'risk_{arm}']=score.ravel()
            order=np.argsort(-score,axis=1,kind='stable')
            rank=np.empty_like(order,dtype='int16')
            np.put_along_axis(rank,order,np.arange(1,n+1,dtype='int16')[None,:],axis=1)
            frame[f'rank_{arm}']=rank.ravel()
            base=arm.rsplit('_',1)[0]
            fusion_status=models[BASES.index(base)]['status'] if arm.endswith('_E') else 'rule'
            rows.append(dict(origin=idx,arm=arm,fusion_status=fusion_status,**metric(score,yy)))
        for p,mem in memory_rows.items():
            frame[f'memory_logit_{p}']=mem.ravel()
        for base,q in q_rows.items():
            frame[f'q_{base}']=q.ravel()
        frame['memory_previous_timestamp']=pd.to_datetime(np.repeat(np.asarray(previous_times,dtype='int64'),n),utc=True)
        frame['feature_cutoff_time']=frame.timestamp-pd.Timedelta(minutes=10)
        frame['latest_telemetry_time']=frame.timestamp-pd.Timedelta(minutes=15)
        frame['adst_train_days']=json.loads((output/'branches'/f'origin_{idx:04d}.json').read_text())['adst_days']
        part=sdir/f'origin_{idx:04d}.parquet'
        frame.to_parquet(part,index=False,compression='zstd')
        write_json(sdir/f'origin_{idx:04d}.identity.json',dict(score_sha256=digest(part),rows=len(frame),
            label_sha256=hashlib.sha256(yy.astype('uint8').tobytes()).hexdigest()))
        pd.DataFrame(rows).to_csv(output/'metrics_by_origin.csv',index=False)
        state=dict(memory=memory,history=list(history),metrics=rows,last_origin=idx,last_ns=last_ns)
        tmp=checkpoint.with_suffix('.tmp');joblib.dump(state,tmp);tmp.replace(checkpoint)
        trace.emit('fusion_origin_done',origin=idx,completed=idx+1,total=len(schedule),
            e19_trained=sum(m['status']=='trained' for m in models),rows=len(frame),seconds=time.perf_counter()-t0)
        del frame,risk,models,q_rows,memory_rows
        gc.collect()
    return rows


def summarize(rows,output,origin_count,trace):
    df=pd.DataFrame(rows)
    records={}
    for arm,group in df.groupby('arm'):
        total=int(group.positives.sum());hits=int(group.hits_at_100.sum())
        records[arm]=dict(origin_mean_pr_auc=float(group.pr_auc.mean()),hits_at_100=hits,
            pooled_recall_at_100=hits/total if total else 0.,origin_mean_recall_at_100=float(group.recall_at_100.mean()),
            trained_e19_origins=int((group.fusion_status=='trained').sum()))
    first=df[df.arm==ARMS[0]]
    result=dict(status='FULL_PIPELINE_FINISHED_PENDING_INDEPENDENT_AUDIT' if origin_count==64 else 'PARTIAL_PIPELINE',
        origins_completed=origin_count,branch_origins_completed=origin_count,arms=list(ARMS),
        score_rows=int(first.decision_times.sum())*1992,decision_times=int(first.decision_times.sum()),
        positive_rows=int(first.positives.sum()),metrics=records)
    write_json(output/'summary.json',result)
    trace.emit('pipeline_done',**result)
    write_json(output/'status.json',result)
    return result


def run(config,output,limit,resume):
    output=Path(output);trace=Trace(output)
    signature=dict(source_sha256=digest(__file__),spec_sha256=digest(config['specification']),
        config=config,vendor_sha256={p.name:digest(p) for p in (ROOT/'vendor').glob('*.py')})
    contract_path=output/'execution_contract.json'
    if contract_path.exists():
        if not resume or json.loads(contract_path.read_text(encoding='utf-8'))!=signature:
            raise ValueError('Resume requires identical approved source, specification, vendor and config')
    else:
        write_json(contract_path,signature)
    engine,events,gt,math,manifest=load_engine(config,output,trace)
    schedule=schedule_from_reference(engine,config)[:limit]
    write_json(output/'manifest.json',dict(material_passport=dict(type='code_experiment',
        goal_ids=['G-004','G-007','G-008'],approval='User explicitly requested execution of specification v1.1'),
        specification='v1.1',seed=config['seed'],purge_hours=36,label_maturity_hours=24,
        evaluation_origins=len(schedule),arms=list(ARMS),source_provenance=signature,
        features=manifest['features'],b2_features=manifest['b2_features'],
        reference_scores_used_for_learning=False,reference_access='timestamps only',
        fusion_math='original E19 AST functions; no additional original inertia terms'))
    fit_branches(engine,events,gt,schedule,config,output,trace)
    rows=fuse(schedule,config,output,engine,math,trace)
    summary=summarize(rows,output,len(schedule),trace)
    trace_rows=[json.loads(s) for s in (output/'fit_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    bfits=[r for r in trace_rows if r['stage']=='fit_start' and r.get('component') in ('B1','B2')]
    ffits=[r for r in trace_rows if r['stage']=='fusion_fit_done']
    for r in bfits:
        if r['training_max_bin']>schedule[r['origin']]['origin']-PURGE or r['label_end_max_bin']>schedule[r['origin']]['origin']:
            raise ValueError('Persisted branch timing audit failed')
    for r in ffits:
        if r['latest_fit_score_ns']>r['cutoff_ns'] or r['label_end_max_ns']>r['origin_ns']:
            raise ValueError('Persisted fusion timing audit failed')
    write_json(output/'fidelity_audit.json',dict(status='FULL_PIPELINE_INTERNAL_AUDIT' if len(schedule)==64 else 'PARTIAL_PIPELINE_INTERNAL_AUDIT',
        origins=len(schedule),feature_count=45,arms=16,branch_fit_count=len(bfits),
        fusion_fit_count=len(ffits),purge_violations=0,label_maturity_violations=0,
        actual_cuda_b1_fits=sum(r['stage']=='fit_done' and r.get('component')=='B1' and
            str(r.get('actual_backend','')).startswith('cuda') for r in trace_rows),
        b2_input_count=2,inertia_input_is_strictly_lagged=True,pareto_cap=None,pareto_floor=None,
        source_feature_extractor=manifest['extractor'],summary=summary))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-origins',type=int,default=64)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if not 1<=args.limit_origins<=64:
        raise ValueError('Origins must be in 1..64')
    with threadpool_limits(limits=config['cpu_threads']):
        run(config,args.output,args.limit_origins,args.resume)

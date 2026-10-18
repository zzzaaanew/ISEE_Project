"""Audit conditional five-minute scores and maintain the cumulative log."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from record_run import ROOT, WORKSPACE, digest, write_json, score_metrics, log_lock, render
sys.path.insert(0,str(ROOT/'ML'))
import run_5m as runner
import run_experiment as r


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(actual,expected,message,atol=1e-12):
    if not np.allclose(actual,expected,atol=atol,rtol=1e-10,equal_nan=False):
        raise ValueError(message)


def audit(output):
    contract=read(output/'execution_contract.json');selection=read(output/'selection.json')
    source=Path(contract['source_directory'])
    _,config,source_audit=runner.strict_selection(source,output/'selection.json')
    chosen=selection['selected']
    if contract['source_sha256']!=digest(ROOT/'ML/run_5m.py') or contract['selection_sha256']!=digest(output/'selection.json'):
        raise ValueError('Five-minute implementation or selection changed')
    if contract['source_execution_contract_sha256']!=digest(source/'execution_contract.json') or chosen!=contract['selected']:
        raise ValueError('Frozen source identity differs')
    parts=sorted((output/'scores').glob('origin_*.parquet'))
    if len(parts) not in (1,64):
        raise ValueError('Only the one-origin pilot or complete 64-origin run can pass')
    # Independently load the SHA-bound raw ledger, time grid and source feature cache.
    engine,events,gt,math,manifest=r.load_engine(config,output,r.Trace(output))
    schedule=runner.expand_schedule(r.schedule_from_reference(engine,config))
    expected_gpus=engine.gpu_ids.astype(str);n=len(expected_gpus)
    memory=None;last=None;rows=decisions=positives=hits=0;metrics=[];hashes={}
    stored_metrics=pd.read_csv(output/'metrics_by_origin.csv').set_index('origin')
    for i,part in enumerate(parts):
        if part.name!=f'origin_{i:04d}.parquet':
            raise ValueError('Nonconsecutive origin files')
        frame=pq.read_table(part).to_pandas();meta=read(part.with_suffix('.identity.json'))
        sch=schedule[i];nt=len(sch['bins']);origin_ns=int(engine.bin_start_ns[sch['origin']])
        sha=digest(part)
        if sha!=meta['score_sha256'] or len(frame)!=nt*n:
            raise ValueError('Score hash/coverage differs')
        hashes[part.name]=sha
        times=frame.timestamp.astype('int64').to_numpy().reshape(nt,n)
        if not np.array_equal(times,np.broadcast_to(engine.bin_start_ns[sch['bins']][:,None],(nt,n))):
            raise ValueError('Five-minute schedule differs')
        if not np.array_equal(frame.gpu_uid.astype(str).to_numpy().reshape(nt,n),np.broadcast_to(expected_gpus,(nt,n))):
            raise ValueError('Full GPU population/order differs')
        yy=frame.label.to_numpy().reshape(nt,n)
        if not np.array_equal(yy,gt[sch['bins']]):
            raise ValueError('Five-minute labels differ from raw All-XID ground truth')
        if last is not None and times[0,0]-last!=r.STEP:
            raise ValueError('Gap/overlap between five-minute origins')
        for column,lag in [('feature_cutoff_time',10),('latest_telemetry_time',15)]:
            if not np.all(frame.timestamp.astype('int64')-frame[column].astype('int64')==lag*60*10**9):
                raise ValueError('Feature availability cutoff differs')
        smeta=read(source/'branches'/f'origin_{i:04d}.json')
        days=smeta['adst_days'] if chosen['policy']=='A' else int(chosen['policy'][1:])
        fit=next(z for z in smeta['fit_info'] if z['days']==days)
        if fit['training_end_bin_exclusive']!=sch['origin']-r.PURGE or not (frame.b1_train_days==days).all():
            raise ValueError('Frozen training window/36-hour purge differs')
        if meta['model_training_end_bin_exclusive']!=sch['origin']-r.PURGE or not (frame.model_origin_bin==sch['origin']).all():
            raise ValueError('Model training future leakage')
        for prefix,relative in [('b1',f'models/b1_origin_{i:04d}_days_{days}.json'),('b2',f'models/b2_origin_{i:04d}.joblib')]:
            path=source/relative
            actual=digest(path) if path.exists() else None
            if actual!=meta[f'{prefix}_model_sha256'] or (actual is not None and actual!=selection['source_model_sha256'][relative]):
                raise ValueError('Frozen trained model identity differs')
        for prefix in ('b1','b2'):
            prior=frame[f'{prefix}_prior'].to_numpy()
            raw=frame[f'{prefix}_model_raw'].to_numpy()
            expected=np.clip(expit(logit(np.clip(raw,r.EPS,1-r.EPS))+logit(np.clip(prior,r.EPS,1-r.EPS))),r.EPS,1-r.EPS)
            if ((prior>0)&(prior<1)).all():
                close(frame[f'{prefix}_probability'],expected,'Natural prior restoration differs')
        shared=runner.compare_shared_times(source,i,chosen['policy'],frame)
        if shared!=meta['common_30m']:
            raise ValueError('Shared original query verification differs')
        z=logit(np.clip(frame.b1_probability.to_numpy().reshape(nt,n),r.EPS,1-r.EPS))
        stored=frame.memory_logit.to_numpy().reshape(nt,n)
        previous=frame.memory_previous_timestamp.astype('int64').to_numpy().reshape(nt,n)
        rh=frame.inertia_rho.to_numpy().reshape(nt,n)
        for j,ts in enumerate(times[:,0]):
            reset=last is None or ts-last>=r.DAY*r.STEP
            expected=z[j] if reset else memory
            close(stored[j],expected,'Inertia consumed a different or future state')
            if not np.all(previous[j]==(np.iinfo(np.int64).min if reset else last)):
                raise ValueError('Inertia state timestamp differs')
            hours=1/12 if last is None else (ts-last)/(3600*10**9)
            rho=2**(-hours/config['inertia_half_life_hours'])
            close(rh[j],rho,'Five-minute inertia half-life changed')
            memory=z[j].copy() if reset else rho*memory+(1-rho)*z[j]
            last=int(ts)
        eta=(z+config['inertia_gamma']*stored)/(1+config['inertia_gamma']) if chosen['inertia'] else z
        q=np.clip(expit(eta),r.EPS,1-r.EPS)
        close(frame.b1_inertia_probability,q.ravel(),'KL inference differs')
        b2=frame.b2_probability.to_numpy();lam=frame.pareto_lambda.to_numpy().reshape(nt,n)
        original_lam=pq.read_table(source/'branches'/part.name,columns=['pareto_lambda']).column(0).to_numpy()[:n]
        close(lam,np.broadcast_to(original_lam,(nt,n)),'Pareto cap/floor or daily weights changed')
        fusion=read(output/'fusion_models'/f'origin_{i:04d}.json')
        if chosen['candidate_type']=='B1':
            expected=q.ravel()
        elif chosen['fusion']=='P':
            expected=lam.ravel()*q.ravel()+(1-lam.ravel())*b2
        else:
            relative=f'fusion_models/origin_{i:04d}.json'
            if digest(source/relative)!=selection['source_model_sha256'][relative]:
                raise ValueError('Frozen Fusion file changed')
            original=read(source/relative)[chosen['policy']+('_I' if chosen['inertia'] else '_0')]
            if fusion!=original or fusion['cutoff_ns']!=origin_ns-36*12*r.STEP:
                raise ValueError('Frozen Fusion provenance/cutoff differs')
            expected=b2
            if fusion['status']=='trained':
                if fusion['latest_fit_score_ns']>fusion['cutoff_ns'] or fusion['latest_fit_score_ns']+r.HORIZON*r.STEP>origin_ns:
                    raise ValueError('Fusion fitting labels were immature')
                c1,c2=fusion['calibrators']
                p1=np.clip(expit(c1['intercept']+c1['slope']*eta.ravel()/6),r.EPS,1-r.EPS)
                p2=np.clip(expit(c2['intercept']+c2['slope']*logit(np.clip(b2,r.EPS,1-r.EPS))/6),r.EPS,1-r.EPS)
                a=logit(p1)/6;b=logit(p2)/6
                x=np.column_stack([np.ones(len(a)),a,b,a*a,b*b,a*b])
                expected=np.clip(expit(logit(p2)+x@np.asarray(fusion['beta'])),r.EPS,1-r.EPS)
        close(frame.risk_score,expected,'Final selected predictor differs')
        score=frame.risk_score.to_numpy().reshape(nt,n)
        if not np.isfinite(score).all() or (score<0).any() or (score>1).any():
            raise ValueError('Invalid final scores')
        order=np.argsort(-score,axis=1,kind='stable');ranks=np.empty_like(order)
        np.put_along_axis(ranks,order,np.arange(1,n+1)[None,:],axis=1)
        if not np.array_equal(ranks,frame['rank'].to_numpy().reshape(nt,n)):
            raise ValueError('Risk rank differs')
        top=pq.read_table(output/'top100'/part.name).to_pandas()
        if not top.equals(frame[frame['rank']<=100].reset_index(drop=True)):
            raise ValueError('Top-100 artifact differs')
        m=score_metrics(score.ravel(),yy.ravel())
        for key,value in m.items():
            close(meta[key],value,f'Identity metric differs: {key}')
            close(stored_metrics.loc[i,key],value,f'CSV metric differs: {key}')
        metrics.append(m);rows+=len(frame);decisions+=nt;positives+=m['positives'];hits+=m['hits_at_100']
        print(json.dumps(dict(stage='independent_5m_audit',origin=i,origins=len(parts))),flush=True)
    summary=read(output/'summary.json')
    for key,value in dict(origins_completed=len(parts),score_rows=rows,decision_times=decisions,positive_rows=positives,hits_at_100=hits).items():
        if summary[key]!=value:
            raise ValueError(f'Summary count differs: {key}')
    close(summary['origin_mean_pr_auc'],np.mean([m['pr_auc'] for m in metrics]),'Summary AP differs')
    close(summary['pooled_recall_at_100'],hits/positives if positives else 0,'Summary recall differs')
    result=dict(status='FIVE_MINUTE_SCORE_ARTIFACTS_VERIFIED',origins=len(parts),score_rows=rows,
        decision_times=decisions,positive_rows=positives,hits_at_100=hits,
        origin_mean_pr_auc=summary['origin_mean_pr_auc'],pooled_recall_at_100=summary['pooled_recall_at_100'],
        selected=chosen,score_sha256=hashes,scoring_minutes=5,purge_hours=36,
        fresh_gbdt_fits=0,fresh_lr_fits=0,fresh_fusion_fits=0,
        source_independent_audit_sha256=digest(source/'independent_audit.json'),
        auditor_sha256=digest(Path(__file__)),verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Frozen daily predictors; newly computed 5-minute features, lagged Inertia, scores and Top-100. Development replay only.')
    write_json(output/'independent_audit.json',result)
    write_json(output/'status.json',result)
    return result


def record(output,result=None,watcher_pid=None):
    directory=WORKSPACE/'실험 기록';md=directory/'실험_연대기.md'
    stage=f'verified_{result["origins"]}' if result else 'armed'
    marker=f'조건부5분 실행 증거: {output.relative_to(WORKSPACE).as_posix()} ({stage})'
    with log_lock(directory):
        original=md.read_text(encoding='utf-8')
        if marker in original:
            return
        number=max(map(int,re.findall(r'^## Experiment (\d+)',original,re.M)),default=0)+1
        entry=f'\n\n## Experiment {number} = 조건부 5분 전수 위험 점수 = {stage}\n\n[요약]\n\n'
        entry+='- 사용자 지시: 현재 실험에서 B1 또는 Fusion 성능이 향상되면 추가 승인 없이 5분 간격 점수 산출로 진행.\n'
        entry+='- 기존 GPU E19와 동일 평가키에서 B1 8개/Fusion 16개 조건을 재계산한다. 해당 baseline보다 origin 평균 AP가 1e-12 초과 상승하고 pooled Recall@100이 감소하지 않는 조건만 통과한다. 통과 조건 중 평균 AP 최대 하나를 선택한다.\n'
        entry+='\n[실행]\n\n'
        entry+='- 64-origin 원실험·독립 감사·36h purge/성숙 피드백 검증 통과 후 pilot1→독립검증→64-origin 5분 추론→독립검증. 실패·성능 미달이면 후속 실행하지 않는다.\n'
        entry+='- 원래 일별 GBDT/LR·ADST routing·Fusion 모델을 유지하고 새로운 5분 특징과 점수를 계산한다. 재학습은 0회다. Inertia 반감기2h를 보존하며 5분 rho≈0.971532를 사용하고 예측 후 raw logit으로 갱신한다. 1,992 GPU 전체 점수와 시점별 Top-100을 저장한다.\n'
        entry+='\n[결과]\n\n'
        if result:
            entry+=f'- 독립 검증 결과 {result["origins"]}/64 origins, {result["score_rows"]:,}행/{result["decision_times"]:,}시점, 평균 AP={result["origin_mean_pr_auc"]:.6f}, Recall@100={result["pooled_recall_at_100"]:.6f}.\n'
        else:
            entry+=f'- 조건부 대기 프로세스 PID={watcher_pid}. 아직 성능 판정이나 5분 실행이 완료된 것은 아니다.\n'
        entry+='\n[분석]\n\n- 데이터 시간축의 5분 추론 replay이며 실시간 데이터 수집 서비스는 아니다. 기존 개발기간 선택이므로 새로운 독립 검증 성능으로 해석하지 않는다.\n'
        content=original+entry+'\n'+marker+'\n';page=render(content)
        temp=md.with_name(md.name+f'.{os.getpid()}.tmp');ht=md.with_suffix('.html');htmp=ht.with_name(ht.name+f'.{os.getpid()}.tmp')
        temp.write_text(content,encoding='utf-8');htmp.write_text(page,encoding='utf-8')
        if md.read_text(encoding='utf-8')!=original:
            raise RuntimeError('Concurrent legacy log writer changed source')
        temp.replace(md);htmp.replace(ht)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--record-armed',action='store_true');p.add_argument('--watcher-pid',type=int)
    a=p.parse_args();output=a.output.resolve()
    if a.record_armed:
        record(output,watcher_pid=a.watcher_pid)
    else:
        with threadpool_limits(limits=2):
            result=audit(output)
        record(output,result)
        print(json.dumps(result,ensure_ascii=False,indent=2))

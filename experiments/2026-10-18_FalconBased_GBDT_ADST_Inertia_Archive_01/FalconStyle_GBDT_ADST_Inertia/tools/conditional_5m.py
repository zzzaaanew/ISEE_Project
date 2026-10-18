"""Bounded, identity-checked continuation of the approved experiment chain."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from record_run import ROOT, WORKSPACE, digest, score_metrics, write_json

POLICY = ROOT/'policies/conditional_5m_policy_v1.json'
KST = dt.timezone(dt.timedelta(hours=9))


def clock():
    now=dt.datetime.now(dt.timezone.utc)
    return {'utc':now.isoformat(),'kst':now.astimezone(KST).isoformat()}


def command_digest(command):
    return hashlib.sha256(json.dumps(command,ensure_ascii=True,separators=(',',':')).encode('ascii')).hexdigest()


def describe_candidate(column):
    kind='B1' if column.startswith('q_') else 'Fusion'
    if kind=='B1':
        base=column[2:];fusion=None
    else:
        base,fusion=column[5:].rsplit('_',1)
    policy,flag=base.rsplit('_',1)
    return dict(candidate_type=kind,policy=policy,inertia=flag=='I',fusion=fusion,score_column=column)


def select_candidate_records(candidates, baseline, policy):
    """Pure predeclared gate, also suitable for synthetic checks before completion."""
    qualified=[];rows=[]
    ap_threshold=policy['gate']['mean_ap_delta_strictly_greater_than']
    recall_tolerance=policy['gate']['pooled_recall_degradation_tolerance']
    for candidate in candidates:
        row=dict(candidate)
        base=baseline[row['candidate_type']]
        row['mean_ap_delta']=row['metrics']['origin_mean_pr_auc']-base['origin_mean_pr_auc']
        row['pooled_recall_delta']=row['metrics']['pooled_recall_at_100']-base['pooled_recall_at_100']
        row['passes_ap']=row['mean_ap_delta']>ap_threshold
        row['passes_recall_noninferiority']=row['pooled_recall_delta']>=-recall_tolerance
        row['eligible']=row['passes_ap'] and row['passes_recall_noninferiority']
        rows.append(row)
        if row['eligible']:
            qualified.append(row)
    if not qualified:
        return None,rows
    highest=max(row['metrics']['origin_mean_pr_auc'] for row in qualified)
    ties=[row for row in qualified if abs(row['metrics']['origin_mean_pr_auc']-highest)<=policy['selection']['ap_tie_tolerance']]
    ties.sort(key=lambda row:(-row['metrics']['pooled_recall_at_100'],row['score_column']))
    return ties[0],rows


def validate_source(source,policy):
    if source != (WORKSPACE/policy['source_relative_to_workspace']).resolve():
        raise ValueError('Source differs from frozen conditional policy')
    supervisor=json.loads((source/'supervisor_status.json').read_text(encoding='utf-8'))
    audit=json.loads((source/'independent_audit.json').read_text(encoding='utf-8'))
    internal=json.loads((source/'fidelity_audit.json').read_text(encoding='utf-8'))
    if supervisor.get('status')!=policy['required_source_supervisor_status']:
        raise ValueError('Current 30-minute experiment is not completed and audited')
    if supervisor.get('supervisor_pid')!=policy['supervisor_identity']['pid']:
        raise ValueError('Source completion came from a different supervisor')
    if audit.get('status')!=policy['required_independent_audit_status']:
        raise ValueError('Source independent audit did not pass')
    for field,expected in [('origins',64),('score_rows',6081576),('decision_times',3053),('positive_rows',52945)]:
        if audit.get(field)!=expected:
            raise ValueError(f'Source full coverage failed: {field}')
    if internal.get('status')!='FULL_PIPELINE_INTERNAL_AUDIT' or internal.get('purge_violations')!=0 or internal.get('label_maturity_violations')!=0:
        raise ValueError('Source purge/maturity audit failed')
    if digest(source/'execution_contract.json')!=policy['source_contract_sha256']:
        raise ValueError('Source execution contract changed')
    contract=json.loads((source/'execution_contract.json').read_text(encoding='utf-8'))
    if contract['source_sha256']!=policy['source_main_sha256'] or digest(ROOT/'ML/run_experiment.py')!=policy['source_main_sha256']:
        raise ValueError('Source main hash differs from frozen contract')
    for name,value in contract['vendor_sha256'].items():
        if digest(ROOT/'vendor'/name)!=value:
            raise ValueError('Source vendor changed')
    return audit,contract


def aggregate(values):
    positive=sum(row['positives'] for row in values)
    hits=sum(row['hits_at_100'] for row in values)
    decisions=sum(row['decision_times'] for row in values)
    return dict(origin_mean_pr_auc=float(np.mean([row['pr_auc'] for row in values])),
        pooled_recall_at_100=hits/positive if positive else 0.,hits_at_100=hits,
        positive_rows=positive,decision_times=decisions,origins=len(values),
        origin_ap=[row['pr_auc'] for row in values],origin_hits=[row['hits_at_100'] for row in values])


def gate(source,target,policy):
    audit,contract=validate_source(source,policy)
    baseline_dir=(WORKSPACE/policy['baseline_relative_to_workspace']).resolve(strict=True)
    parts=sorted((source/'scores').glob('origin_*.parquet'))
    if len(parts)!=64 or len(list(baseline_dir.glob('score_part_*.parquet')))!=64:
        raise ValueError('Gate requires exactly 64 source and baseline score parts')
    columns=policy['b1_candidates']+policy['fusion_candidates']
    values={column:[] for column in columns}
    baseline_values={kind:[] for kind in policy['baseline_columns']}
    source_hashes={};baseline_hashes={}
    for index,part in enumerate(parts):
        if part.name!=f'origin_{index:04d}.parquet':
            raise ValueError('Nonconsecutive source origins')
        sha=digest(part)
        if sha!=audit['score_sha256'][part.name]:
            raise ValueError('Source score changed after independent audit')
        frame=pq.read_table(part,columns=['timestamp','gpu_uid','label']+columns).to_pandas()
        frame['gpu_uid']=frame.gpu_uid.astype(str)
        key=pd.MultiIndex.from_frame(frame[['timestamp','gpu_uid']])
        if not key.is_unique:
            raise ValueError('Source gate keys duplicated')
        reference=baseline_dir/f'score_part_{index:04d}.parquet'
        base=pq.read_table(reference,columns=['timestamp','gpu_uid']+list(policy['baseline_columns'].values())).to_pandas()
        base['gpu_uid']=base.gpu_uid.astype(str)
        base_key=pd.MultiIndex.from_frame(base[['timestamp','gpu_uid']])
        if len(base)!=len(frame) or not base_key.is_unique or len(key.difference(base_key)):
            raise ValueError('Baseline/source GPU-time evaluation populations differ')
        base=base.set_index(['timestamp','gpu_uid']).reindex(key)
        y=frame.label.to_numpy(dtype='uint8')
        for column in columns:
            values[column].append(score_metrics(frame[column].to_numpy(),y))
        for kind,column in policy['baseline_columns'].items():
            probability=base[column].to_numpy()
            if not np.isfinite(probability).all():
                raise ValueError('Invalid aligned baseline probability')
            baseline_values[kind].append(score_metrics(probability,y))
        source_hashes[part.name]=sha;baseline_hashes[reference.name]=digest(reference)
        print(json.dumps({'stage':'gate_origin','origin':index,'origins':64,**clock()}),flush=True)
    baseline={kind:aggregate(items) for kind,items in baseline_values.items()}
    candidates=[dict(**describe_candidate(column),metrics=aggregate(values[column])) for column in columns]
    selected,candidates=select_candidate_records(candidates,baseline,policy)
    model_hashes={path.relative_to(source).as_posix():digest(path)
        for folder in ['models','fusion_models'] for path in sorted((source/folder).glob('*')) if path.is_file()}
    result=dict(decision='PROMOTE_5M' if selected else 'NO_POSITIVE_IMPROVEMENT',selected=selected,
        baseline=baseline,all_candidates=candidates,source=str(source),
        baseline_directory=str(baseline_dir),policy=policy,policy_sha256=digest(POLICY),
        source_contract_sha256=digest(source/'execution_contract.json'),source_main_sha256=contract['source_sha256'],
        source_vendor_sha256=contract['vendor_sha256'],source_model_sha256=model_hashes,
        source_score_sha256=source_hashes,baseline_score_sha256=baseline_hashes,
        source_independent_audit_sha256=digest(source/'independent_audit.json'),
        selected_on='Already examined development period; no independent test or causally isolated architecture improvement',
        candidate_count=len(candidates),**clock())
    write_json(target/'selection.json',result)
    return result


def own_process(child,identity,deadline,status):
    while child.poll() is None:
        if time.monotonic()>=deadline:
            process=psutil.Process(child.pid)
            if abs(process.create_time()-identity['create_time'])>.01 or process.ppid()!=os.getpid() or process.cmdline()!=identity['command']:
                raise RuntimeError('Refusing timeout kill outside owned child identity')
            descendants=[(p,p.create_time()) for p in process.children(recursive=True)]
            for descendant,created in reversed(descendants):
                try:
                    if abs(descendant.create_time()-created)<.01:
                        descendant.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill();child.wait(timeout=30)
            status(status='FOLLOWUP_TIMEOUT',stopped_owned_child=identity)
            return 2
        time.sleep(min(5,max(.1,deadline-time.monotonic())))
    return child.returncode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--wait-supervisor-pid',type=int,default=22224)
    p.add_argument('--max-wait-seconds',type=float,default=43200)
    p.add_argument('--max-run-seconds',type=float,default=43200)
    args=p.parse_args()
    if args.max_wait_seconds<=0 or args.max_run_seconds<=0:
        raise ValueError('Time bounds must be positive')
    source=args.source.resolve(strict=True);target=args.target.resolve()
    if source==target or source in target.parents:
        raise ValueError('Follow-up output must be separate from frozen source run')
    target.mkdir(parents=True,exist_ok=True)
    policy=json.loads(POLICY.read_text(encoding='utf-8'))
    expected=policy['supervisor_identity']
    if args.wait_supervisor_pid!=expected['pid']:
        raise ValueError('Watcher PID differs from frozen supervisor identity')
    state=dict(status='INITIALIZING',watcher_pid=os.getpid(),source=str(source),target=str(target),
        source_supervisor_identity=expected,policy_sha256=digest(POLICY),watcher_source_sha256=digest(Path(__file__)),
        max_wait_seconds=args.max_wait_seconds,max_run_seconds=args.max_run_seconds,retries=0,
        created=clock())

    def status(**updates):
        state.update(updates,updated=clock())
        write_json(target/'watcher_status.json',state)

    status(status='WAITING_FOR_SOURCE_COMPLETION')
    try:
        wait_deadline=time.monotonic()+args.max_wait_seconds
        while True:
            observed=json.loads((source/'supervisor_status.json').read_text(encoding='utf-8'))
            source_status=observed.get('status','UNKNOWN')
            if source_status=='COMPLETED_AND_AUDITED':
                break
            if source_status=='TIMEOUT' or 'FAILED' in source_status:
                status(status='NO_FOLLOWUP_SOURCE_FAILED',source_status=source_status)
                return 1
            if time.monotonic()>=wait_deadline:
                status(status='NO_FOLLOWUP_WAIT_TIMEOUT',source_status=source_status)
                return 2
            try:
                process=psutil.Process(args.wait_supervisor_pid)
                identity=dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline())
                if abs(identity['create_time']-expected['create_time'])>.01 or command_digest(identity['command'])!=expected['command_sha256']:
                    raise ValueError('Existing supervisor PID was reused or command changed')
                if process.status()==psutil.STATUS_ZOMBIE or not process.is_running():
                    raise psutil.NoSuchProcess(process.pid)
            except psutil.NoSuchProcess:
                # Allow a just-written terminal status to become visible; never start from RUNNING alone.
                observed=json.loads((source/'supervisor_status.json').read_text(encoding='utf-8'))
                if observed.get('status')=='COMPLETED_AND_AUDITED':
                    break
                status(status='NO_FOLLOWUP_SOURCE_PROCESS_EXITED',source_status=observed.get('status'))
                return 1
            status(status='WAITING_FOR_SOURCE_COMPLETION',source_status=source_status,observed_supervisor=identity)
            time.sleep(min(10,max(.1,wait_deadline-time.monotonic())))
        status(status='EVALUATING_FROZEN_GATE')
        selected=gate(source,target,policy)
        if selected['decision']!='PROMOTE_5M':
            status(status='NO_POSITIVE_IMPROVEMENT',decision=selected['decision'],selected=None)
            return 0
        status(status='PROMOTED_5M',selected=selected['selected'])
        python=WORKSPACE/'Falcon_AcmeTrace_ST_Top100/.venv/Scripts/python.exe'
        if not python.exists():
            raise FileNotFoundError('Approved experiment Python environment missing')
        runner=ROOT/'ML/run_5m.py';auditor=ROOT/'tools/audit_5m.py'
        for path in [runner,auditor]:
            if not path.exists():
                raise FileNotFoundError(path)
        state['followup_source_sha256']={path.name:digest(path) for path in [runner,auditor]}
        selection=target/'selection.json'
        def command(limit):
            return [str(python),'-u',str(runner),'--source',str(source),'--output',str(target),
                    '--selection',str(selection),'--limit-origins',str(limit),'--resume']
        audit=[str(python),'-u',str(auditor),'--output',str(target)]
        phases=[('FIVE_MINUTE_PILOT',command(1),''),('PILOT_AUDIT',audit,'pilot_audit_'),
                ('FIVE_MINUTE_FULL',command(64),''),('FULL_AUDIT',audit,'full_audit_')]
        deadline=time.monotonic()+args.max_run_seconds
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
        env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
        for phase,cmd,prefix in phases:
            status(status=phase,phase=phase,command=cmd)
            with (target/f'{prefix}stdout.log').open('a',encoding='utf-8') as out, \
                    (target/f'{prefix}stderr.log').open('a',encoding='utf-8') as err:
                child=subprocess.Popen(cmd,cwd=WORKSPACE,stdout=out,stderr=err,env=env,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                owned=psutil.Process(child.pid)
                identity=dict(pid=owned.pid,create_time=owned.create_time(),command=owned.cmdline())
                status(owned_child=identity)
                code=own_process(child,identity,deadline,status)
            if code:
                if state.get('status')!='FOLLOWUP_TIMEOUT':
                    status(status=f'{phase}_FAILED',exit_code=code)
                return code
            if phase in ('PILOT_AUDIT','FULL_AUDIT'):
                verified=json.loads((target/'independent_audit.json').read_text(encoding='utf-8'))
                expected_origins=1 if phase=='PILOT_AUDIT' else 64
                if verified.get('status')!='FIVE_MINUTE_SCORE_ARTIFACTS_VERIFIED' or verified.get('origins')!=expected_origins:
                    raise ValueError(f'{phase}: independent artifact audit did not verify {expected_origins} origins')
            status(last_completed_phase=phase,last_exit_code=0)
        status(status='FIVE_MINUTE_COMPLETED_AND_AUDITED',completed=clock())
        return 0
    except Exception as error:
        status(status='FOLLOWUP_BLOCKED_OR_FAILED',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':
    raise SystemExit(main())

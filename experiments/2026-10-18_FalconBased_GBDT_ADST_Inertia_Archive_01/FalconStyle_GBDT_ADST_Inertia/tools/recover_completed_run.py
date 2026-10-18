"""Recover postprocessing after supervisor failure, without rerunning training."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import psutil

from record_run import ROOT, WORKSPACE, digest, write_json, log_lock, render

POLICY=ROOT/'policies/conditional_5m_policy_v1.json'
KST=dt.timezone(dt.timedelta(hours=9))


def clock():
    now=dt.datetime.now(dt.timezone.utc)
    return dict(utc=now.isoformat(),kst=now.astimezone(KST).isoformat())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def command_sha(command):
    return hashlib.sha256(json.dumps(command,ensure_ascii=True,separators=(',',':')).encode('ascii')).hexdigest()


def identity(process):
    command=process.cmdline()
    return dict(pid=process.pid,create_time=process.create_time(),command=command,
                command_sha256=command_sha(command))


def assert_no_existing_pipeline():
    blocked={str(ROOT/'ML/run_experiment.py').casefold(),str(ROOT/'tools/supervise.py').casefold(),
             str(ROOT/'tools/conditional_5m.py').casefold(),str(ROOT/'ML/run_5m.py').casefold()}
    active=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            name=str(process.info['name'] or '').casefold()
            command=process.info['cmdline']
            if command is None and name.startswith(('python','pythonw')):
                raise RuntimeError(f'Cannot verify Python process command line: PID {process.pid}')
            if command and any(str(argument).casefold() in blocked for argument in command):
                active.append(identity(process))
        except psutil.NoSuchProcess:
            continue
    if active:
        raise RuntimeError('Existing experiment/watcher is still active: '+json.dumps(active,ensure_ascii=True))


def validate_finished_training(source,policy):
    if source!=(WORKSPACE/policy['source_relative_to_workspace']).resolve():
        raise ValueError('Source differs from frozen conditional policy')
    contract_path=source/'execution_contract.json';contract=read(contract_path)
    if digest(contract_path)!=policy['source_contract_sha256']:
        raise ValueError('Original execution contract hash changed')
    if contract['source_sha256']!=policy['source_main_sha256'] or digest(ROOT/'ML/run_experiment.py')!=policy['source_main_sha256']:
        raise ValueError('Original main implementation changed')
    if read(ROOT/'config.json')!=contract['config']:
        raise ValueError('Original config differs from completed training contract')
    if digest(Path(contract['config']['specification']))!=contract['spec_sha256']:
        raise ValueError('Original specification changed')
    for name,expected in contract['vendor_sha256'].items():
        if digest(ROOT/'vendor'/name)!=expected:
            raise ValueError('Original vendor code changed')
    summary=read(source/'summary.json');fidelity=read(source/'fidelity_audit.json')
    if summary.get('status')!='FULL_PIPELINE_FINISHED_PENDING_INDEPENDENT_AUDIT':
        raise ValueError('Only the completed pending-audit pipeline is eligible for this recovery')
    for field,value in [('origins_completed',64),('branch_origins_completed',64),
                        ('score_rows',6081576),('decision_times',3053),('positive_rows',52945)]:
        if summary.get(field)!=value:
            raise ValueError(f'Incomplete training/scoring summary: {field}')
    if len(summary.get('arms',[]))!=16:
        raise ValueError('Completed source is not the approved 16-arm experiment')
    for folder in ['branches','scores']:
        actual=sorted(path.name for path in (source/folder).glob('origin_*.parquet'))
        if actual!=[f'origin_{i:04d}.parquet' for i in range(64)]:
            raise ValueError(f'Missing completed source parts: {folder}')
    if fidelity.get('status')!='FULL_PIPELINE_INTERNAL_AUDIT' or fidelity.get('origins')!=64:
        raise ValueError('Full internal fidelity audit unavailable')
    if fidelity.get('purge_violations')!=0 or fidelity.get('label_maturity_violations')!=0:
        raise ValueError('Original training violated temporal controls')
    if fidelity.get('actual_cuda_b1_fits')!=192 or fidelity.get('branch_fit_count')!=256:
        raise ValueError('Expected completed CUDA/B2 fit counts differ')
    supervisor=read(source/'supervisor_status.json')
    if supervisor.get('supervisor_pid')!=policy['supervisor_identity']['pid']:
        raise ValueError('Original supervisor provenance differs')
    return contract


def preserve_evidence(source,stopped,evidence):
    if (evidence/'snapshot_manifest.json').exists():
        raise FileExistsError('This recovery evidence folder already belongs to an attempt; use a fresh recovery directory')
    evidence.mkdir(parents=True,exist_ok=True)
    paths=[]
    names=['supervisor_status.json','supervisor_status.json.22224.tmp','supervisor_stderr.log',
           'supervisor_stdout.log','status.json','stdout.log','stderr.log','summary.json',
           'fidelity_audit.json','independent_audit.json','execution_contract.json','feature_manifest.json']
    paths += [(source/name,evidence/'source_before_recovery'/name) for name in names if (source/name).is_file()]
    if stopped.exists():
        paths += [(path,evidence/'stopped_watcher'/path.name) for path in stopped.iterdir() if path.is_file()]
    manifest=[]
    for original,copy in paths:
        copy.parent.mkdir(parents=True,exist_ok=True)
        if copy.exists():
            raise FileExistsError(f'Refusing to overwrite recovery evidence: {copy}')
        shutil.copy2(original,copy)
        if digest(original)!=digest(copy):
            raise ValueError('Evidence copy hash differs')
        manifest.append(dict(original=str(original),snapshot=copy.relative_to(evidence).as_posix(),
            sha256=digest(copy),bytes=copy.stat().st_size,
            source_mtime_utc=dt.datetime.fromtimestamp(original.stat().st_mtime,dt.timezone.utc).isoformat()))
    write_json(evidence/'snapshot_manifest.json',dict(created=clock(),files=manifest,
        reason='Original supervisor terminated on PermissionError during status temp.replace; completed training persisted independently.',
        root_cause='PermissionError is observed. External locking or the responsible application is not confirmed.'))


def immutable_identity(source):
    models={path.relative_to(source).as_posix():digest(path)
            for folder in ['models','fusion_models'] for path in sorted((source/folder).glob('*')) if path.is_file()}
    return dict(model_sha256=models,fit_trace_sha256=digest(source/'fit_trace.jsonl'),
                main_sha256=digest(ROOT/'ML/run_experiment.py'),config_sha256=digest(ROOT/'config.json'),
                vendor_sha256={p.name:digest(p) for p in (ROOT/'vendor').glob('*.py')})


def replace_bounded(temporary,target):
    deadline=time.monotonic()+30
    while True:
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if time.monotonic()>=deadline:
                raise
            time.sleep(.2)


def record_recovery_start(source,stopped,evidence,own):
    """Append a correction; the earlier armed entry remains historical evidence."""
    directory=WORKSPACE/'실험 기록';md=directory/'실험_연대기.md'
    marker=f'감독 복구 증거: {evidence.relative_to(WORKSPACE).as_posix()} (recovery_pid={own["pid"]})'
    with log_lock(directory):
        original=md.read_text(encoding='utf-8')
        if marker in original:
            return
        number=max(map(int,re.findall(r'^## Experiment (\d+)',original,re.M)),default=0)+1
        entry=f'\n\n## Experiment {number} = 감독 연결 실패 정정 및 완료 산출물 후처리 복구\n\n[요약]\n\n'
        entry+='- Experiment17의 조건부 대기 기록을 이어 정정한다. 원감독 PID22224는 2026-10-18 02:08 KST경 supervisor_status.json 임시파일 교체에서 PermissionError/WinError5로 종료했다. 오류는 확인됐으며 외부 잠금 여부와 잠금 주체는 확정하지 않는다. 기존 기록을 덮어쓰지 않는다.\n'
        entry+='- 별도 계산 프로세스는 03:22:04.796 KST에64/64 origin 학습·Fusion을 완료했다. 감독 종료와 계산 종료는 서로 다른 사건이며, 감독의 RUNNING JSON은 최종 계산 상태를 반영하지 못했다.\n\n[실행]\n\n'
        entry+='- 사용자 기존 조건부 자동 실행 승인 범위에서 복구한다. source/main/config/vendor·학습 데이터·완료 모델·전수점수·fit trace를 보존하고 재학습은0회다. source의 원감독 상태·오류·완료 요약과 중단 watcher PID25100의 상태를 recovery_20261018 증거 폴더에 복사·SHA256으로 보존한다.\n'
        entry+='- 기존 연구 pipeline/supervisor/watcher의 활성 프로세스가 없는지 확인한 뒤 독립 감사→paired·운영 분석→누적 기록을 실행한다. 성공한 경우에만 원감독 실패 사실과 recovery PID를 명시하여 완료 상태를 복구하고, 별도 새 target에서 고정 정책의 조건부5분 watcher를 재개한다. 자동 재시도는 없으며 소유 자식만12시간 상한으로 감시한다.\n\n[결과]\n\n'
        entry+='- 현재 원본 내부 감사는64 origins·36h purge/라벨 성숙 위반0·CUDA B1 192fits·B2포함 branch256fits를 기록한다. 전체 독립 감사·분석 및5분 후속은 이번 복구에서 확인할 단계이며 아직 완료했다고 주장하지 않는다.\n'
        entry+='- 이전 watcher는 NO_FOLLOWUP_SOURCE_PROCESS_EXITED 상태로 종료했다. 따라서 이전 자동 연결에서5분 성능 선정·추론은 실행되지 않았다. 중단 target은 보존한다.\n\n[분석]\n\n'
        entry+='- 계산 실패와 감독 메타데이터 쓰기 실패를 구분한다. 재학습 없이 완료 산출물을 검사해 감독 이후 후처리를 복구한다. 후속5분 실행은 기존24후보의 AP향상·Recall비열화 조건을 만족할 때에만 진행하며, 개발기간 선택이라는 범위는 유지한다.\n\n[실행할 것]\n\n1. 64-origin 독립 감사·비교 분석·기록 성공을 확인한다.\n2. 기존 정책의 단일 선정 조건만5분 pilot→감사→64-origin→감사로 진행한다.\n\n'+marker+'\n'
        content=original+entry;page=render(content)
        html_path=md.with_suffix('.html')
        md_tmp=md.with_name(md.name+f'.{os.getpid()}.tmp')
        html_tmp=html_path.with_name(html_path.name+f'.{os.getpid()}.tmp')
        md_tmp.write_text(content,encoding='utf-8');html_tmp.write_text(page,encoding='utf-8')
        if md.read_text(encoding='utf-8')!=original:
            raise RuntimeError('Concurrent writer changed research log during recovery')
        replace_bounded(md_tmp,md);replace_bounded(html_tmp,html_path)


def stop_owned(child,record):
    if child.poll() is not None:
        return
    process=psutil.Process(child.pid)
    if abs(process.create_time()-record['create_time'])>.01 or process.ppid()!=os.getpid() or process.cmdline()!=record['command']:
        raise RuntimeError('Refusing to stop a child whose identity changed')
    descendants=[(p,p.create_time()) for p in process.children(recursive=True)]
    for descendant,created in reversed(descendants):
        try:
            if abs(descendant.create_time()-created)<.01:
                descendant.kill()
        except psutil.NoSuchProcess:
            continue
    process.kill();child.wait(timeout=30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'runs/full_01')
    parser.add_argument('--target',type=Path,default=ROOT/'runs/conditional_5m_recovery_01')
    parser.add_argument('--stopped-target',type=Path,default=ROOT/'runs/conditional_5m_01')
    parser.add_argument('--recovery-directory',type=Path)
    parser.add_argument('--max-seconds',type=float,default=43200)
    args=parser.parse_args()
    if not 0<args.max_seconds<=43200:
        raise ValueError('Recovery time bound must be positive and no longer than 12 hours')
    source=args.source.resolve(strict=True);target=args.target.resolve();stopped=args.stopped_target.resolve()
    evidence=args.recovery_directory.resolve() if args.recovery_directory else source/'recovery_20261018'
    if source not in evidence.parents:
        raise ValueError('Recovery evidence must remain inside the completed source output')
    if target==stopped or target==source or source in target.parents:
        raise ValueError('Follow-up requires a separate, fresh output directory')
    if target.exists() and any(target.iterdir()):
        raise ValueError('Recovery follow-up target is not empty; preserve it and choose a fresh target')
    own=identity(psutil.Process())
    state=dict(status='INITIALIZING',recovery_identity=own,recovery_pid=os.getpid(),
        source=str(source),followup_target=str(target),stopped_target=str(stopped),evidence_directory=str(evidence),
        created=clock(),max_seconds=args.max_seconds,retries=0,training_rerun=False,
        recovery_script_sha256=digest(Path(__file__)))

    def report(**values):
        state.update(values,updated=clock())
        write_json(source/'recovery_status.json',state)

    started=time.monotonic();deadline=started+args.max_seconds
    report(status='VALIDATING_COMPLETED_SOURCE')
    try:
        policy=read(POLICY)
        assert_no_existing_pipeline()
        validate_finished_training(source,policy)
        preserve_evidence(source,stopped,evidence)
        frozen=immutable_identity(source)
        write_json(evidence/'immutable_before.json',frozen)
        original_supervisor=read(source/'supervisor_status.json')
        original_error=(source/'supervisor_stderr.log').read_text(encoding='utf-8',errors='replace')
        if 'PermissionError' not in original_error or 'WinError 5' not in original_error:
            raise ValueError('Expected supervisor PermissionError evidence is unavailable')
        record_recovery_start(source,stopped,evidence,own)
        python=WORKSPACE/'Falcon_AcmeTrace_ST_Top100/.venv/Scripts/python.exe'
        environment=os.environ.copy();environment['PYTHONIOENCODING']='utf-8'
        environment['OMP_NUM_THREADS']='2';environment['OPENBLAS_NUM_THREADS']='2'

        def run_phase(name,command):
            if time.monotonic()>=deadline:
                raise TimeoutError('Recovery deadline exhausted before next phase')
            report(status=name,phase=name,command=command)
            with (evidence/f'{name.lower()}_stdout.log').open('a',encoding='utf-8') as out, \
                    (evidence/f'{name.lower()}_stderr.log').open('a',encoding='utf-8') as err:
                child=subprocess.Popen(command,cwd=WORKSPACE,stdout=out,stderr=err,env=environment,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                record=identity(psutil.Process(child.pid))
                report(owned_child=record)
                while child.poll() is None:
                    elapsed=time.monotonic()-started
                    if time.monotonic()>=deadline:
                        stop_owned(child,record)
                        report(status='RECOVERY_TIMEOUT',elapsed_seconds=elapsed,stopped_owned_child=record)
                        raise TimeoutError('Recovery hard deadline reached; stopped owned child tree')
                    report(status=name,elapsed_seconds=elapsed)
                    time.sleep(min(5,max(.1,deadline-time.monotonic())))
            report(last_phase=name,last_exit_code=child.returncode)
            if child.returncode:
                raise RuntimeError(f'{name} exited with code {child.returncode}; no retry')

        record=[str(python),str(ROOT/'tools/record_run.py'),'--output',str(source)]
        run_phase('INDEPENDENT_AUDIT',record+['--audit','--no-record'])
        audit=read(source/'independent_audit.json')
        if audit.get('status')!='SCORE_ARTIFACTS_VERIFIED' or audit.get('origins')!=64:
            raise ValueError('Recovered source independent full audit did not pass')
        run_phase('COMPARISON_ANALYSIS',[str(python),str(ROOT/'tools/analyze_comparisons.py'),'--output',str(source)])
        comparisons=read(source/'comparison.json');operations=read(source/'operational_metrics.json')
        if comparisons.get('status')!='DESCRIPTIVE_PAIRED_ANALYSIS' or len(comparisons.get('comparisons',[]))!=28:
            raise ValueError('Recovered paired comparison analysis is incomplete')
        if operations.get('status')!='OPERATIONAL_DESCRIPTIVE_METRICS' or operations.get('decision_times')!=3053:
            raise ValueError('Recovered operational analysis is incomplete')
        run_phase('RECORD_SOURCE',record+['--record-only'])
        assert_no_existing_pipeline()
        after=immutable_identity(source)
        if after!=frozen:
            raise ValueError('Source model, fit trace, main, config or vendor changed during recovery')
        write_json(evidence/'immutable_after_source_audit.json',after)
        # Compatibility field keeps the original PID as provenance. Recovery success is explicit.
        recovered=dict(original_supervisor)
        recovered.update(status='COMPLETED_AND_AUDITED',updated_utc=clock()['utc'],
            original_supervisor_completed_successfully=False,
            original_supervisor_outcome='Terminated on PermissionError/WinError5 during status replacement',
            original_supervisor_status_before_recovery=original_supervisor.get('status'),
            original_supervisor_updated_utc=original_supervisor.get('updated_utc'),
            completion_provenance='Original pipeline completed independently; this recovery performed audit, analysis and recording only. No training rerun.',
            recovery_pid=os.getpid(),recovery_identity=own,recovery_completed=clock(),
            recovery_evidence_directory=str(evidence),audit_exit_code=0,analysis_exit_code=0,record_exit_code=0)
        write_json(source/'supervisor_status.json',recovered)
        report(status='SOURCE_COMPLETION_RECOVERED',source_audit_status=audit['status'])
        remaining=deadline-time.monotonic()
        if remaining<=30:
            raise TimeoutError('Not enough bounded time remains to start conditional follow-up')
        followup=[str(python),'-u',str(ROOT/'tools/conditional_5m.py'),'--source',str(source),
            '--target',str(target),'--wait-supervisor-pid',str(policy['supervisor_identity']['pid']),
            '--max-wait-seconds','30','--max-run-seconds',str(max(1,int(remaining)-15))]
        run_phase('CONDITIONAL_FIVE_MINUTE',followup)
        followup_state=read(target/'watcher_status.json')
        decision=read(target/'selection.json')
        if followup_state.get('status')=='NO_POSITIVE_IMPROVEMENT' and decision.get('decision')=='NO_POSITIVE_IMPROVEMENT':
            final='RECOVERED_NO_POSITIVE_IMPROVEMENT'
        elif followup_state.get('status')=='FIVE_MINUTE_COMPLETED_AND_AUDITED':
            five=read(target/'independent_audit.json')
            if five.get('status')!='FIVE_MINUTE_SCORE_ARTIFACTS_VERIFIED' or five.get('origins')!=64:
                raise ValueError('Five-minute child reported success without full independent verification')
            final='RECOVERED_AND_FIVE_MINUTE_VERIFIED'
        else:
            raise ValueError('Conditional child did not reach an expected terminal outcome')
        if immutable_identity(source)!=frozen:
            raise ValueError('Frozen source was modified by conditional replay')
        report(status=final,followup_status=followup_state['status'],selected=decision.get('selected'),
            completed=clock(),elapsed_seconds=time.monotonic()-started)
        return 0
    except Exception as error:
        report(status='RECOVERY_TIMEOUT' if isinstance(error,TimeoutError) else 'RECOVERY_FAILED',
            error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started)
        raise


if __name__=='__main__':
    raise SystemExit(main())

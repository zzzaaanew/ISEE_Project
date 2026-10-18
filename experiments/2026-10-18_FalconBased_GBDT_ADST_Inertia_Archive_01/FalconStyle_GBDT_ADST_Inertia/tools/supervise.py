"""Queue the approved experiment behind one identity-checked existing run."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    payload=json.dumps(value, ensure_ascii=False, indent=2)
    deadline=time.monotonic()+30
    while True:
        try:
            temp.write_text(payload, encoding='utf-8')
            temp.replace(path)
            return
        except PermissionError:
            if time.monotonic()>=deadline:
                raise
            time.sleep(.2)


def identity(process: psutil.Process) -> dict:
    return {'pid': process.pid, 'create_time': process.create_time(), 'command': process.cmdline()}


def same_process(record: dict) -> psutil.Process | None:
    try:
        proc = psutil.Process(record['pid'])
        if abs(proc.create_time() - record['create_time']) > .01:
            return None
        if proc.cmdline() != record['command']:
            raise RuntimeError('Process command changed after identity capture')
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return None
        return proc
    except psutil.NoSuchProcess:
        return None


def stop_own_child(child: subprocess.Popen, record: dict) -> None:
    if child.poll() is not None:
        return
    proc = same_process(record)
    if proc is None or proc.ppid() != os.getpid():
        raise RuntimeError('Refusing to stop a process outside this supervisor')
    descendants = [(p, p.create_time()) for p in proc.children(recursive=True)]
    for descendant, created in reversed(descendants):
        try:
            if abs(descendant.create_time() - created) < .01:
                descendant.kill()
        except psutil.NoSuchProcess:
            pass
    proc.kill()
    child.wait(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, default=11820)
    parser.add_argument('--max-seconds', type=float, default=43200)
    parser.add_argument('--pilot-first', action='store_true')
    args = parser.parse_args()
    if args.max_seconds <= 0:
        raise ValueError('max-seconds must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = args.config.resolve(strict=True)
    state = {'status': 'INITIALIZING', 'supervisor_pid': os.getpid(), 'created_utc': utc(),
             'origin_target': 64, 'config': str(config), 'output': str(output),
             'max_seconds': args.max_seconds, 'retries': 0, 'wait_pid': args.wait_pid}

    def report(**values):
        state.update(values, updated_utc=utc())
        write_json(output / 'supervisor_status.json', state)

    try:
        try:
            previous = psutil.Process(args.wait_pid)
            wait_identity = identity(previous)
        except psutil.NoSuchProcess:
            wait_identity = None
        if wait_identity is not None:
            expected = ROOT.parent / 'Falcon_AcmeTrace_ST_Top100' / 'ML' / 'run_falcon_acmetrace_st_top100.py'
            command_paths = [str(v).casefold() for v in wait_identity['command']]
            if str(expected).casefold() not in command_paths:
                raise RuntimeError('wait-pid does not belong to the expected existing Falcon pipeline')
            report(status='WAITING_FOR_EXISTING_PIPELINE', wait_identity=wait_identity,
                   wait_started_utc=utc())
            while same_process(wait_identity) is not None:
                time.sleep(10)
                report(status='WAITING_FOR_EXISTING_PIPELINE')
        else:
            report(wait_identity=None, wait_result='PID absent at initial identity check')
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'utf-8'
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        started = time.monotonic()
        report(started_utc=utc(), pilot_first=args.pilot_first)

        def run_owned(command, phase, log_prefix):
            report(status=phase, command=command, phase=phase)
            with (output/f'{log_prefix}stdout.log').open('a',encoding='utf-8') as stdout, \
                    (output/f'{log_prefix}stderr.log').open('a',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,cwd=ROOT.parent,stdout=stdout,stderr=stderr,
                    env=environment,creationflags=flags)
                child_identity=identity(psutil.Process(child.pid))
                report(status=phase,pipeline_identity=child_identity)
                while child.poll() is None:
                    elapsed=time.monotonic()-started
                    if elapsed>=args.max_seconds:
                        stop_own_child(child,child_identity)
                        report(status='TIMEOUT',phase=phase,elapsed_seconds=elapsed,stopped_utc=utc())
                        return 2
                    report(status=phase,elapsed_seconds=elapsed)
                    time.sleep(min(10,max(.1,args.max_seconds-elapsed)))
            report(last_phase=phase,last_exit_code=child.returncode,elapsed_seconds=time.monotonic()-started)
            return child.returncode

        def pipeline(limit):
            return [sys.executable,'-u',str(ROOT/'ML/run_experiment.py'),'--config',str(config),
                    '--output',str(output),'--limit-origins',str(limit),'--resume']

        record=[sys.executable,str(ROOT/'tools/record_run.py'),'--output',str(output)]
        phases=[]
        if args.pilot_first:
            phases += [(pipeline(1),'PILOT_RUNNING',''),
                       (record+['--audit'],'PILOT_AUDITING','pilot_audit_')]
        phases += [(pipeline(64),'RUNNING',''),
                   (record+['--audit','--no-record'],'AUDITING','audit_'),
                   ([sys.executable,str(ROOT/'tools/analyze_comparisons.py'),'--output',str(output)],'ANALYZING','analysis_'),
                   (record+['--record-only'],'RECORDING','record_')]
        for command,phase,prefix in phases:
            code=run_owned(command,phase,prefix)
            if code:
                if state.get('status')!='TIMEOUT':
                    report(status=f'{phase}_FAILED',exit_code=code,completed_utc=utc())
                return code
            if phase=='PILOT_AUDITING':
                report(pilot_verified_utc=utc(),pilot_audit_exit_code=0)
        report(status='COMPLETED_AND_AUDITED',completed_utc=utc(),audit_exit_code=0,
               analysis_exit_code=0,record_exit_code=0)
        return 0
    except Exception as error:
        report(status='SUPERVISOR_FAILED', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())

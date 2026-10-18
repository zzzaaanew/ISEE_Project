"""Attach a new deadline watcher to an existing, identity-checked pipeline."""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ML'))
from run_falcon_acmetrace_st_top100 import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    handoff = json.loads(args.handoff.read_text(encoding='utf-8'))
    pipeline = psutil.Process(handoff['pipeline_pid'])
    if abs(pipeline.create_time()-handoff['pipeline_create_time']) > .01:
        raise RuntimeError('Pipeline PID was reused')
    command = pipeline.cmdline()
    expected_script = str(ROOT/'ML/run_falcon_acmetrace_st_top100.py').casefold()
    if expected_script not in [str(x).casefold() for x in command]:
        raise RuntimeError('Pipeline command identity mismatch')
    if str(output).casefold() not in [str(x).casefold() for x in command]:
        raise RuntimeError('Pipeline output identity mismatch')
    deadline = handoff['original_start_epoch']+handoff['max_seconds']
    status = {'status': 'RUNNING', 'supervisor_pid': os.getpid(),
              'pipeline_pid': pipeline.pid, 'pipeline_create_time': pipeline.create_time(),
              'command': command, 'max_seconds': handoff['max_seconds'],
              'origin_target': 64, 'attached_existing_pipeline': True,
              'original_start_epoch': handoff['original_start_epoch'],
              'deadline_utc': dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat(),
              'previous_supervisor_pid': handoff['old_supervisor_pid']}
    write_json(output/'extension_watcher_ready.json', status)
    write_json(output/'supervisor_status.json', status)
    while True:
        try:
            exit_code = pipeline.wait(timeout=5)
            break
        except psutil.TimeoutExpired:
            if time.time() < deadline:
                continue
            status.update(status='TIMEOUT', elapsed_seconds=time.time()-handoff['original_start_epoch'])
            write_json(output/'supervisor_status.json', status)
            print('Extended hard timeout reached; stopping only the identified pipeline.', flush=True)
            if abs(pipeline.create_time()-handoff['pipeline_create_time']) > .01:
                raise RuntimeError('Pipeline identity changed at deadline')
            for child in pipeline.children(recursive=True):
                child.kill()
            pipeline.kill()
            write_json(output/'status.json', status)
            return 2
    status.update(exit_code=exit_code, elapsed_seconds=time.time()-handoff['original_start_epoch'])
    state = json.loads((output/'status.json').read_text(encoding='utf-8'))
    if state.get('status') != 'FULL_PIPELINE_VERIFIED' or (exit_code not in (0, None)):
        status['status'] = 'FAILED_OR_INCOMPLETE'
        status['pipeline_status'] = state
        write_json(output/'supervisor_status.json', status)
        return 1
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    audit = subprocess.run([sys.executable, str(ROOT/'tools/record_run.py'), '--output', str(output)],
        cwd=ROOT.parent, capture_output=True, text=True, encoding='utf-8', env=env,
        creationflags=subprocess.CREATE_NO_WINDOW)
    (output/'audit_stdout.log').write_text(audit.stdout, encoding='utf-8')
    (output/'audit_stderr.log').write_text(audit.stderr, encoding='utf-8')
    status.update(audit_exit_code=audit.returncode,
        status='COMPLETED_AND_AUDITED' if audit.returncode == 0 else 'COMPLETED_AUDIT_FAILED')
    write_json(output/'supervisor_status.json', status)
    return audit.returncode


if __name__ == '__main__':
    raise SystemExit(main())

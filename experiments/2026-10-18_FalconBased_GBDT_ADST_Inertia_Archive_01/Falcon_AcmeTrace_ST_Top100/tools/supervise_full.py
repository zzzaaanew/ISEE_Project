"""Supervise the approved continuation; no retries or unrelated process control."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ML'))
from run_falcon_acmetrace_st_top100 import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=int,default=14400)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u',str(ROOT/'ML/run_falcon_acmetrace_st_top100.py'),
             '--config',str(ROOT/'config.json'),'--output',str(output),'--limit-origins','64','--resume']
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
    start=time.monotonic()
    with (output/'stdout.log').open('w',encoding='utf-8') as stdout, (output/'stderr.log').open('w',encoding='utf-8') as stderr:
        proc=subprocess.Popen(command,cwd=ROOT.parent,stdout=stdout,stderr=stderr,env=env,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        launch={'status':'RUNNING','supervisor_pid':os.getpid(),'pipeline_pid':proc.pid,
                'command':command,'max_seconds':args.max_seconds,'origin_target':64}
        write_json(output/'supervisor_status.json',launch)
        while proc.poll() is None:
            if time.monotonic()-start > args.max_seconds:
                launch.update(status='TIMEOUT',elapsed_seconds=time.monotonic()-start)
                write_json(output/'supervisor_status.json',launch)
                print('Hard timeout reached; stopping only this owned pipeline.',flush=True)
                owned=psutil.Process(proc.pid)
                for child in owned.children(recursive=True):
                    child.kill()
                proc.kill();proc.wait()
                write_json(output/'status.json',launch)
                return 2
            time.sleep(5)
        launch.update(exit_code=proc.returncode,elapsed_seconds=time.monotonic()-start)
        if proc.returncode:
            launch['status']='FAILED'
            write_json(output/'supervisor_status.json',launch)
            return proc.returncode
    audit=subprocess.run([sys.executable,str(ROOT/'tools/record_run.py'),'--output',str(output)],
                         cwd=ROOT.parent,capture_output=True,text=True,encoding='utf-8',env=env)
    (output/'audit_stdout.log').write_text(audit.stdout,encoding='utf-8')
    (output/'audit_stderr.log').write_text(audit.stderr,encoding='utf-8')
    launch.update(audit_exit_code=audit.returncode,
                  status='COMPLETED_AND_AUDITED' if audit.returncode==0 else 'COMPLETED_AUDIT_FAILED')
    write_json(output/'supervisor_status.json',launch)
    return audit.returncode


if __name__=='__main__':
    raise SystemExit(main())

"""Frozen, sequential prepare -> pilot -> full branch scoring -> postprocessing."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def persist(path,value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(path)


def main():
    import hashlib
    files=[*ROOT.glob("*.py"),ROOT/"config.json",*ROOT.joinpath("sources").glob("*.py")]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    freeze=ROOT/"frozen_code.json"
    if freeze.exists():
        old=json.loads(freeze.read_text(encoding="utf-8"))
        if old["sha256"]!=hashes:
            raise ValueError("Code changed after freeze; inspect previous attempt before restarting")
    else:
        persist(freeze,dict(sha256=hashes,created_unix=time.time(),local_head="42191204ff0eecfa2e0250c7da232713781203a1",
                           remote_source_commit="1877e09f81d47753fccbb55ba9acb4fb8372b2ed"))
    logdir=ROOT/"logs";logdir.mkdir(exist_ok=True)
    stages=["prepare","pilot","branches","evaluate"]
    history=[]
    monitor=ROOT/"monitor.json"
    for stage in stages:
        log=logdir/f"{stage}.log"
        if log.exists():
            raise FileExistsError(f"Existing stage log, automatic retry disabled: {log}")
        before=time.time()
        env=os.environ.copy()
        env.update(PYTHONHASHSEED="42",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
        command=[sys.executable,"-B",str(ROOT/"run_experiment.py"),stage]
        with log.open("w",encoding="utf-8") as output:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,
                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            while process.poll() is None:
                persist(monitor,dict(status="RUNNING",stage=stage,worker_pid=process.pid,supervisor_pid=os.getpid(),
                        stage_elapsed_seconds=time.time()-before,completed_stages=history,log=str(log)))
                time.sleep(15)
        code=process.returncode
        history.append(dict(stage=stage,exit_code=code,seconds=time.time()-before))
        if code:
            persist(monitor,dict(status="FAILED",stage=stage,completed_stages=history,log=str(log)))
            raise SystemExit(code)
        if stage=="pilot":
            validate=[sys.executable,"-B",str(ROOT/"verify_run.py"),"pilot"]
            with (logdir/"pilot_validation.log").open("w",encoding="utf-8") as output:
                result=subprocess.run(validate,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
            if result.returncode:
                persist(monitor,dict(status="FAILED",stage="pilot_validation",completed_stages=history))
                raise SystemExit(result.returncode)
    with (logdir/"final_validation.log").open("w",encoding="utf-8") as output:
        result=subprocess.run([sys.executable,"-B",str(ROOT/"verify_run.py"),"full"],cwd=ROOT,env=env,
                              stdout=output,stderr=subprocess.STDOUT)
    persist(monitor,dict(status="COMPLETED" if result.returncode==0 else "FAILED",stage="final_validation",completed_stages=history))
    raise SystemExit(result.returncode)


if __name__=="__main__":
    main()

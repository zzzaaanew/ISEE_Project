"""Explicit checkpoint resume, one origin per process, resilient Windows status IO."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT.parents[2]/"migrate/migration/e19-reproduction/.venv/Scripts/python.exe"


def persist(path,value,critical=True):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    payload=json.dumps(value,ensure_ascii=False,indent=2,default=str)
    for attempt in range(14):
        try:
            tmp.write_text(payload,encoding="utf-8")
            tmp.replace(path)
            return True
        except PermissionError:
            if attempt==13:
                if critical:
                    raise
                return False
            time.sleep(min(.02*2**attempt,.25))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def checkpoint():
    return json.loads((ROOT/"branch_checkpoint.json").read_text(encoding="utf-8"))


def assert_preserved(folder,expected_count=None):
    old=json.loads((folder/"before/branch_checkpoint.json").read_text(encoding="utf-8"))
    current=checkpoint()
    assert current["completed_origins"][:len(old["completed_origins"])]==old["completed_origins"]
    assert current["fit_trace"][:len(old["fit_trace"])]==old["fit_trace"]
    assert digest(ROOT/"config.json")==digest(folder/"before/config.json")==current["config_sha256"]
    for record in json.loads((folder/"preserved_scores.json").read_text(encoding="utf-8-sig")):
        path=ROOT/"branch_scores"/record["name"]
        assert path.stat().st_size==record["bytes"] and digest(path)==record["sha256"],path
    if expected_count is not None:
        assert len(current["completed_origins"])==expected_count
    return current


def verify_first_resume(folder):
    import pyarrow.parquet as pq
    old=json.loads((folder/"before/branch_checkpoint.json").read_text(encoding="utf-8"))
    current=assert_preserved(folder,len(old["completed_origins"])+1)
    origin=current["completed_origins"][-1]
    path=ROOT/"branch_scores"/f"origin_{origin:06d}.parquet"
    assert pq.ParquetFile(path).metadata.num_rows==288*1992
    assert sum(x["fit_count"] for x in current["fit_trace"])==6*len(current["completed_origins"])
    for row in current["fit_trace"][-5:]:
        assert row["origin"]==origin and row["label_maturity_latest_bin"]<=origin
        assert row["latest_train_bin"]<=origin-432
    persist(folder/"first_resumed_origin_validation.json",dict(status="PASS",origin=origin,
            completed_origins=len(current["completed_origins"]),preserved_origins=len(old["completed_origins"]),
            old_scores_unchanged=True,old_fit_trace_unchanged=True,new_classifier_fits=6,
            new_score_rows=288*1992,config_unchanged=True))


def execute(attempt_id):
    folder=ROOT/"attempts"/attempt_id
    if not (folder/"before/branch_checkpoint.json").exists():
        raise FileNotFoundError("Resume baseline archive missing")
    if (folder/"STARTED.json").exists():
        raise FileExistsError("Attempt already started; automatic retry disabled")
    frozen=json.loads((ROOT/"frozen_code.json").read_text(encoding="utf-8"))
    for name,expected in frozen["sha256"].items():
        assert digest(ROOT/name)==expected,name
    old=assert_preserved(folder)
    started_count=len(old["completed_origins"])
    persist(folder/"STARTED.json",dict(supervisor_pid=os.getpid(),created_unix=time.time(),
             from_completed_origins=started_count,attempt_id=attempt_id))
    history=[]
    stage="initializing"
    env=os.environ.copy()
    env.update(PYTHONHASHSEED="42",PYTHONUNBUFFERED="1",PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")

    def update(value):
        value.update(attempt_id=attempt_id,supervisor_pid=os.getpid(),updated_unix=time.time(),
                     completed_stages=list(history),authoritative_attempt_monitor=str(folder/"monitor.json"))
        for path in [folder/"monitor.json",ROOT/"monitor.json"]:
            if not persist(path,value,critical=False):
                with (folder/"status_io_warnings.log").open("a",encoding="utf-8") as f:
                    f.write(f"{time.time()} PermissionError writing {path}; worker supervision continues\n")

    def worker(label,command):
        nonlocal stage
        stage=label
        before=time.time()
        log=folder/"logs"/(label+".log")
        if log.exists():
            raise FileExistsError(log)
        with log.open("w",encoding="utf-8") as output:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            while p.poll() is None:
                update(dict(status="RUNNING",stage=label,worker_pid=p.pid,
                            stage_elapsed_seconds=time.time()-before,
                            completed_origins=len(checkpoint()["completed_origins"]),total_origins=63))
                time.sleep(5)
        history.append(dict(stage=label,exit_code=p.returncode,seconds=time.time()-before,log=str(log)))
        if p.returncode:
            raise RuntimeError(f"{label} exited {p.returncode}; see {log}")

    success=False
    try:
        persist(ROOT/"stage_branches_status.json",dict(status="RESUMING",attempt_id=attempt_id,
                from_completed_origins=started_count,remaining_origins=63-started_count))
        persist(ROOT/"completion_handoff_status.json",dict(status="WAITING_FOR_EXPERIMENT",attempt_id=attempt_id,
                supervisor_pid=os.getpid(),started_unix=time.time()))
        while len(checkpoint()["completed_origins"])<63:
            count=len(checkpoint()["completed_origins"])
            worker(f"origin_{count+1:03d}",[str(PYTHON),"-B",str(ROOT/"run_experiment.py"),"next-origin"])
            assert len(checkpoint()["completed_origins"])==count+1
            for name in ["stage_next-origin_status.json","runtime_environment_next-origin.json"]:
                source=ROOT/name
                if source.exists():
                    (folder/"logs"/f"origin_{count+1:03d}_{name}").write_bytes(source.read_bytes())
            if count==started_count:
                worker("first_resume_validation",[str(PYTHON),"-B",str(Path(__file__)),"--attempt",attempt_id,"--verify-first"])
        assert (ROOT/"BRANCH_SCORES_COMPLETE").exists()
        persist(ROOT/"stage_branches_status.json",dict(status="COMPLETED",attempt_id=attempt_id,origins=63))
        worker("evaluate",[str(PYTHON),"-B",str(ROOT/"run_experiment.py"),"evaluate"])
        worker("final_validation",[str(PYTHON),"-B",str(ROOT/"verify_run.py"),"full"])
        assert_preserved(folder,63)
        success=True
        terminal=dict(status="COMPLETED",stage="final_validation",completed_origins=63)
    except BaseException as exc:
        terminal=dict(status="FAILED",stage=stage,error=repr(exc),traceback=traceback.format_exc(),
                      completed_origins=len(checkpoint()["completed_origins"]))
        if stage.startswith("origin_"):
            persist(ROOT/"stage_branches_status.json",terminal,critical=False)
    terminal.update(attempt_id=attempt_id,completed_stages=list(history),updated_unix=time.time())
    # New terminal file has no previous readers; it remains authoritative if root monitor is locked.
    persist(folder/"terminal_status.json",terminal)
    update(terminal.copy())
    log=folder/"logs/completion_handoff.log"
    with log.open("w",encoding="utf-8") as output:
        handoff=subprocess.run([str(PYTHON),"-B",str(ROOT/"finish_handoff.py"),"--attempt",attempt_id,"--once"],
                              cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
    persist(folder/"handoff_exit.json",dict(exit_code=handoff.returncode,experiment_success=success))
    return 0 if success and handoff.returncode==0 else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt",required=True)
    parser.add_argument("--verify-first",action="store_true")
    args=parser.parse_args()
    if not args.attempt.startswith("resume_") or not args.attempt[7:].isdigit():
        raise ValueError("Invalid resume attempt identifier")
    if args.verify_first:
        verify_first_resume(ROOT/"attempts"/args.attempt)
        return
    raise SystemExit(execute(args.attempt))


if __name__=="__main__":
    main()

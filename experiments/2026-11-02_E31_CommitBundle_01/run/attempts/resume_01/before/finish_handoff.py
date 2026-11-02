"""Finish the approved long-running experiment's local handoff; no retraining or messages."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]
STATUS=ROOT/"completion_handoff_status.json"


def save(value):
    tmp=STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(STATUS)


def insert_text(path,anchor,text):
    old=path.read_text(encoding="utf-8-sig")
    if "E31_COMPLETION_HANDOFF" in old:
        raise ValueError(f"Completion already recorded: {path.name}")
    value=old.replace(anchor,anchor+"\n\n"+text,1) if anchor in old else old+"\n\n"+text
    temp=path.with_suffix(path.suffix+".e31tmp")
    temp.write_text(value,encoding="utf-8")
    if path.read_text(encoding="utf-8-sig")!=old:
        raise RuntimeError(f"Concurrent document change: {path.name}; no overwrite")
    temp.replace(path)


def finish(monitor):
    success=monitor["status"]=="COMPLETED"
    validation=ROOT/"final_validation.json"
    if success:
        if not validation.exists() or json.loads(validation.read_text(encoding="utf-8"))["status"]!="PASS":
            raise ValueError("Supervisor completion without final validation PASS")
        import pandas as pd
        metrics=pd.read_csv(ROOT/"metrics_summary.csv")
        checks=json.loads(validation.read_text(encoding="utf-8"))
        matrix=pd.read_csv(ROOT/"REQUIREMENTS.csv")
        matrix["status"]="confirmed"
        matrix["runtime_evidence"]=matrix["runtime_evidence"].astype(str)+"; final_validation.json; evaluation_manifest.json"
        matrix.to_csv(ROOT/"REQUIREMENTS.csv",index=False)
        rows=metrics[(metrics.cadence_minutes==5)&~metrics.model.str.contains("dev_")&
                     (metrics.model.str.endswith("fusion_main")|(metrics.model=="shared/B2"))]
        lines=["| 구간 | 모델 | 평균 block AP | pooled R@100 | churn/시간 | 고유 onset 포착/적격 |",
               "|---|---|---:|---:|---:|---:|"]
        for split in ["development","terminal","full"]:
            for row in rows[rows.split==split].itertuples():
                lines.append(f"| {split} | {row.model} | {row.mean_origin_block_AP:.6f} | "
                             f"{row.pooled_recall_at_100:.6f} | {row.churn_entries_per_hour:.3f} | "
                             f"{int(row.unique_onset_captured)}/{int(row.unique_onset_eligible)} |")
        table="\n".join(lines)
        fits=pd.read_csv(ROOT/"fit_trace.csv")
        windows={a:{str(k):int(v) for k,v in fits[fits.arm==a].training_days.value_counts().sort_index().items()}
                 for a in ["C","D"]}
        result=(f"- 최종검증PASS: {checks['origins']} origin·{checks['rows']:,}개실제5분GPU-time행·"
                f"{checks['actual_branch_estimator_fits']} estimator fit. 저장Fusion전행등식검사"
                f"{checks['full_tape_fusion_identity_checks']}개통과. 감독기전체단계exit0.\n"
                f"- ADST 실제 학습창 origin 수: C={windows['C']}, D={windows['D']}. "
                "원점수는branch_scores,주Fusion은fusion_scores/5m와30m,전체지표는metrics_summary.csv에보존했다.\n\n"+table)
        interpretation=("- 위수치는공통라벨마스크·고정설정·동일역사적trace의결과다. terminal은과거열람된"
                        "retrospective prequential이며완전새독립test가아니다. 30분·Branch단독·관성없는Fusion·개발민감도는"
                        "metrics_summary.csv를함께확인한다. 통계적일반화/최종모델채택/실제운영이득을자동판정하지않는다.")
        title="E31 전체 학습·관성/Fusion 평가·전수 검증 완료"
    else:
        failed=[]
        for p in ROOT.glob("stage_*_status.json"):
            data=json.loads(p.read_text(encoding="utf-8"))
            if data.get("status")=="FAILED":
                failed.append(f"{p.name}: {data.get('error','미기록')}")
        if not failed:
            log_path=ROOT/"logs"/f"{monitor.get('stage','final_validation')}.log"
            if log_path.exists():
                failed.append("\n".join(log_path.read_text(encoding="utf-8",errors="replace").splitlines()[-12:]))
        result=(f"- 감독기실패단계: {monitor.get('stage','미기록')}. 오류: "+"; ".join(failed)+
                ". 완료된origin/부분산출물은보존했다. 자동재학습/설정변경/재시도없음. 전체실험완료로주장하지않는다.")
        interpretation="- 원인은로그와stage별status를확인한다. 새시도는기존산출물을덮어쓰지않는복구범위와승인에따른다."
        title="E31 실행 중단·부분 산출물 보존"
    marker="<!-- E31_COMPLETION_HANDOFF -->"
    entry="### 2026-11-02 "+title+"\n\n"+marker+"\n\n"+result+"\n\n"+interpretation
    # Append dated evidence to the existing E31 section, preserving prior partial results verbatim.
    log=WORKSPACE/"실험 기록/실험_연대기.md"
    old=log.read_text(encoding="utf-8-sig")
    if marker in old:
        raise ValueError("E31 completion already in chronology")
    anchor="## Experiment 31 ="
    index=old.find(anchor)
    if index<0:
        raise ValueError("E31 chronology section missing")
    next_section=old.find("\n## Experiment ",index+len(anchor))
    point=next_section if next_section>=0 else len(old)
    value=old[:point]+"\n\n"+entry+"\n"+old[point:]
    temp=log.with_suffix(".md.e31tmp");temp.write_text(value,encoding="utf-8")
    if log.read_text(encoding="utf-8-sig")!=old:
        raise RuntimeError("Concurrent chronology edit; no overwrite")
    temp.replace(log)
    brief=(marker+"\n\n### 2026-11-02 "+title+"\n\n"+
           result.split("\n\n")[0]+"\n\n"+interpretation+
           "\n\n근거: E31 monitor.json·final_validation.json(성공시)·metrics_summary.csv 및 누적 실험 연대기. "
           "G-004·G-007·G-008의광범위연구목표완료/최종채택과구분한다. E30·설치·커밋/푸시없음.")
    insert_text(WORKSPACE/"프로젝트 목표 및 결정사항.md","## 최근 결정사항",brief)
    insert_text(WORKSPACE/"프로젝트 진행 상황.md",
                "> 이 문서는 회의에서 확정된 장기 방향과 현재 실험 상태를 계속 참고하기 위한 문서다. 회의 중 제안·질문은 확정사항과 분리해 기록한다. 다음 회의 일정은 Google Calendar에서만 관리한다.",brief)
    bundle=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    env=os.environ.copy();env["NODE_PATH"]=str(bundle/"node_modules")
    subprocess.run([str(bundle/"bin/node.exe"),str(WORKSPACE/"실험 기록/render_experiment_log.cjs")],
                   cwd=WORKSPACE,env=env,check=True)
    html=(WORKSPACE/"실험 기록/실험_연대기.html").read_text(encoding="utf-8")
    if title not in html:
        raise ValueError("Chronology HTML lacks completion section")
    save(dict(status="COMPLETED",experiment_success=success,supervisor_status=monitor["status"],
              completed_unix=time.time(),updated_project_records=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def main():
    if STATUS.exists():
        raise FileExistsError("Existing completion handoff; automatic retry disabled")
    save(dict(status="WAITING_FOR_EXPERIMENT",pid=os.getpid(),started_unix=time.time()))
    try:
        while True:
            path=ROOT/"monitor.json"
            if path.exists():
                monitor=json.loads(path.read_text(encoding="utf-8"))
                if monitor.get("status") in ["COMPLETED","FAILED"]:
                    finish(monitor)
                    return
            time.sleep(20)
    except BaseException as exc:
        save(dict(status="FAILED",error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    main()

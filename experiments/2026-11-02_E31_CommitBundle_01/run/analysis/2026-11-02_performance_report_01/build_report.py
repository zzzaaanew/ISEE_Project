"""Post-hoc report from immutable E31 metrics; never trains or rewrites experiment results."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
WORKSPACE=ROOT.parents[2]
REPORT_MD=WORKSPACE/"실험 기록/2026-11-02_E31_성능분석보고서.md"
REPORT_HTML=REPORT_MD.with_suffix(".html")


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def export(name,rows):
    target=OUT/name
    if target.exists():
        raise FileExistsError(target)
    with target.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|",
                      *["| "+" | ".join(str(x) for x in row)+" |" for row in rows]])


def link(name,path):
    return "["+name+"](<"+"/"+path.as_posix()+">)"


def main():
    if REPORT_MD.exists() or REPORT_HTML.exists() or (OUT/"analysis.json").exists():
        raise FileExistsError("Report already exists; do not overwrite the analysis")
    sources=[ROOT/name for name in ["metrics_summary.csv","metrics_by_origin.csv","fit_trace.csv",
              "config.json","split_manifest.json","evaluation_manifest.json","grid_manifest.json",
              "final_validation.json","evaluate_scores.py","SPEC.md","cache/grid.npz","cache/label_eligible.npy"]]
    before={str(path.relative_to(ROOT)):sha(path) for path in sources}
    summary=read_csv(ROOT/"metrics_summary.csv")
    daily=read_csv(ROOT/"metrics_by_origin.csv")
    fits=read_csv(ROOT/"fit_trace.csv")
    config=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    split_manifest=json.loads((ROOT/"split_manifest.json").read_text(encoding="utf-8"))
    validation=json.loads((ROOT/"final_validation.json").read_text(encoding="utf-8"))
    assert validation["status"]=="PASS"
    idx={(r["cadence_minutes"],r["split"],r["model"]):r for r in summary}
    groups=defaultdict(list)
    for row in daily:
        groups[row["cadence_minutes"],row["split"],row["model"]].append(row)
    maximum_ap_error=maximum_recall_error=0.
    for key,blocks in groups.items():
        reported=idx[key]
        for name in ["rows","positive","hits"]:
            assert sum(int(r[name]) for r in blocks)==int(reported[name]),(key,name)
        ap=statistics.mean(float(r["ap"]) for r in blocks)
        recall=sum(int(r["hits"]) for r in blocks)/max(sum(int(r["positive"]) for r in blocks),1)
        maximum_ap_error=max(maximum_ap_error,abs(ap-float(reported["mean_origin_block_AP"])))
        maximum_recall_error=max(maximum_recall_error,abs(recall-float(reported["pooled_recall_at_100"])))
        transitions=sum(int(r["transitions"]) for r in blocks)
        churn=sum(int(r["entries"]) for r in blocks)/transitions*60/int(key[0]) if transitions else None
        if churn is not None:
            assert abs(churn-float(reported["churn_entries_per_hour"]))<1e-8,key
    assert maximum_ap_error<1e-12 and maximum_recall_error<1e-12

    def row(model,split="full",step="5"):
        return idx[step,split,model]

    def val(model,key,split="full",step="5"):
        return float(row(model,split,step)[key])

    population=[]
    for split in ["development","terminal","full"]:
        base=row("shared/B2",split)
        blocks=groups["5",split,"shared/B2"]
        population.append(dict(split=split,evaluated_rows=int(base["rows"]),positive_rows=int(base["positive"]),
              pooled_prevalence=float(base["prevalence"]),eligible_origin_blocks=len(blocks),
              two_class_origin_blocks=sum(0<int(b["positive"])<int(b["rows"]) for b in blocks),
              eligible_5minute_points=int(base["rows"])//1992,
              onset_eligible=int(base["unique_onset_eligible"]),
              mean_block_prevalence=statistics.mean(int(b["positive"])/int(b["rows"]) for b in blocks)))
    export("population.csv",population)

    robust=[]
    for key,blocks in groups.items():
        if key[0]!="5":
            continue
        good=lambda minimum:[b for b in blocks if int(b["rows"])>=minimum*1992]
        average=lambda xs:statistics.mean(float(b["ap"]) for b in xs) if xs else None
        weighted=sum(float(b["ap"])*int(b["rows"]) for b in blocks)/sum(int(b["rows"]) for b in blocks)
        robust.append(dict(split=key[1],model=key[2],original_equal_block_AP=average(blocks),
               posthoc_time_weighted_block_AP=weighted,
               posthoc_exclude_under30min_AP=average(good(6)),
               posthoc_at_least12h_AP=average(good(144)),
               posthoc_complete24h_AP=average(good(288)),
               original_blocks=len(blocks),at_least12h_blocks=len(good(144)),complete24h_blocks=len(good(288)),
               status="posthoc_diagnostic_not_a_replacement_primary_metric"))
    ridx={(r["split"],r["model"]):r for r in robust}
    export("robustness_5m.csv",robust)
    rw=lambda model,split="terminal":ridx[split,model]["posthoc_time_weighted_block_AP"]

    main_rows=[]
    for split in ["development","terminal","full"]:
        for arm in ["A","B","C","D"]:
            r=row(arm+"/fusion_main",split)
            main_rows.append({name:r[name] for name in ["split","model","mean_origin_block_AP",
                 "pooled_recall_at_100","mean_origin_block_ROC_AUC","churn_entries_per_hour",
                 "unique_onset_captured","unique_onset_eligible","rows","positive"]})
    export("main_fusion_5m.csv",main_rows)
    terminal_rows=[r for r in summary if r["cadence_minutes"]=="5" and r["split"]=="terminal"]
    export("terminal_models_5m.csv",terminal_rows)

    def contrast(split,a,b):
        return dict(split=split,reference=a,comparison=b,
            AP_difference=val(b,"mean_origin_block_AP",split)-val(a,"mean_origin_block_AP",split),
            AP_relative_percent=100*(val(b,"mean_origin_block_AP",split)/val(a,"mean_origin_block_AP",split)-1),
            recall_difference_percentage_points=100*(val(b,"pooled_recall_at_100",split)-val(a,"pooled_recall_at_100",split)),
            churn_relative_percent=100*(val(b,"churn_entries_per_hour",split)/val(a,"churn_entries_per_hour",split)-1),
            onset_count_difference=int(row(b,split)["unique_onset_captured"])-int(row(a,split)["unique_onset_captured"]))
    contrasts=[]
    pairs=[("A/fusion_main","B/fusion_main"),("A/fusion_main","C/fusion_main"),
           ("B/fusion_main","D/fusion_main"),("C/fusion_main","D/fusion_main")]
    for split in ["development","terminal","full"]:
        for a,b in pairs:
            contrasts.append(contrast(split,a,b))
    export("factorial_contrasts_5m.csv",contrasts)
    inertia=[contrast(split,a+"/fusion_no_inertia",a+"/fusion_main")
             for split in ["development","terminal","full"] for a in ["A","B","C","D"]]
    export("inertia_effects_5m.csv",inertia)
    paired=[]
    for split in ["development","terminal","full"]:
        for a,b in pairs+[("D/fusion_no_inertia","D/fusion_main")]:
            aa={int(r["origin_bin"]):float(r["ap"]) for r in groups["5",split,a]}
            bb={int(r["origin_bin"]):float(r["ap"]) for r in groups["5",split,b]}
            assert aa.keys()==bb.keys()
            differences=[bb[k]-aa[k] for k in sorted(aa)]
            paired.append(dict(split=split,reference=a,comparison=b,n=len(differences),
              wins=sum(d>1e-12 for d in differences),ties=sum(abs(d)<=1e-12 for d in differences),
              losses=sum(d< -1e-12 for d in differences),mean_difference=statistics.mean(differences),
              median_difference=statistics.median(differences),inference="descriptive_no_significance_test"))
    export("paired_origin_AP.csv",paired)

    cap_rows=[]
    for step in ["5","30"]:
        for cap in [.5,.6,.7,.8,1.]:
            model="D/fusion_main" if cap==.6 else f"D/dev_cap_{cap:.2f}"
            r=row(model,"development",step)
            cap_rows.append(dict(cadence_minutes=int(step),cap=cap,split="development",
                 AP=float(r["mean_origin_block_AP"]),recall_at100=float(r["pooled_recall_at_100"]),
                 churn=float(r["churn_entries_per_hour"]),onset_captured=int(r["unique_onset_captured"]),
                 onset_eligible=int(r["unique_onset_eligible"])))
    export("cap_sensitivity_development_D.csv",cap_rows)
    cadence=[r for r in summary if r["model"] in ["A/fusion_main","B/fusion_main","C/fusion_main","D/fusion_main",
                                                "shared/B2","B/B1_inertia","C/B1_inertia"]]
    export("cadence_comparison.csv",cadence)
    training=[]
    for arm in ["A","B","C","D"]:
        these=[r for r in fits if r["arm"]==arm]
        training.append(dict(arm=arm,classifier_fits=len(these),
                 mean_samples=statistics.mean(int(r["samples"]) for r in these),
                 sum_fit_wall_seconds=sum(float(r["fit_seconds"]) for r in these),
                 median_fit_wall_seconds=statistics.median(float(r["fit_seconds"]) for r in these),
                 training_window_counts={str(d):sum(int(r["training_days"])==d for r in these) for d in [7,14,21]}))
    export("training_cost.csv",training)

    tiny=next(b for b in groups["5","terminal","C/fusion_main"] if int(b["origin_bin"])==23616)
    assert int(tiny["rows"])==1992
    with np.load(ROOT/"cache/grid.npz") as meta:
        times=meta["times"]
    eligible=np.load(ROOT/"cache/label_eligible.npy")
    good=np.flatnonzero(eligible[23616:23616+288])+23616
    assert len(good)==1
    assert (int(good[0])-split_manifest["start_bin"])%6!=0
    witness=dict(origin_bin=23616,valid_points=1,gpu_time_rows=1992,positive_gpu_time_rows=int(tiny["positive"]),
                 score_time_utc=dt.datetime.fromtimestamp(int(times[good[0]])/1e9,dt.timezone.utc).isoformat(),
                 C_fusion_AP=float(tiny["ap"]),macro_terminal_weight=1/15,
                 time_weight=1/3454,absent_from_30minute_sample=True)
    evidence=[
      dict(claim="Pipeline completed with 63 origins and 378 classifier fits",source="final_validation.json",condition="E31",status="confirmed",verification="PASS plus persisted tapes",action="retain"),
      dict(claim="Current AP macro mean equally weights a one-point origin",source="metrics_by_origin.csv; evaluate_scores.py:167",condition="5m terminal origin23616",status="confirmed",verification="1992 rows; 1/15 macro weight vs1/3454 time weight",action="report robustness; do not silently replace metric"),
      dict(claim="D is consistently best",source="metrics_summary.csv; robustness_5m.csv",condition="four arm comparison",status="unsupported",verification="rank changes by split/metric/aggregation",action="withhold claim"),
      dict(claim="Inertia decreases ranking churn",source="metrics_summary.csv",condition="D full and terminal fixed main",status="confirmed",verification="47.85% and48.48% lower churn; event capture also lower",action="state tradeoff"),
      dict(claim="Fusion has general predictive advantage over B1",source="metrics_summary.csv; robustness_5m.csv",condition="B1-inertia vs same-arm fusion",status="unsupported",verification="all four full macro AP lower after fusion",action="audit B2, calibration and fusion"),
      dict(claim="Cap0.60 is optimal",source="metrics_summary.csv",condition="D development sensitivity",status="unsupported",verification="cap1 better AP/R/churn but lower onset capture",action="specify objective before selecting cap"),
      dict(claim="ADST reduces training compute",source="fit_trace.csv",condition="same63 fits per B1 arm",status="unsupported",verification="fewer samples; recorded fit wall time higher",action="measure iterations and cost under controlled conditions"),
      dict(claim="Transient peaks suppressed by inertia explain lost onsets",source="metrics_summary.csv",condition="onset capture/churn tradeoff",status="hypothesis",verification="no per-event diagnostic completed",action="inspect event traces under matched alert budget"),
      dict(claim="New unseen generalization established",source="evaluation_manifest.json",condition="retrospective terminal",status="unsupported",verification="previously inspected trace; online mature-label updates",action="use retrospective wording"),
      dict(claim="Re-evaluate aggregation from saved scores without branch retraining",source="branch_scores; fusion_scores",condition="future protocol amendment",status="proposed",verification="existing tapes available; not executed here",action="separate approval before replacing publication metrics")]
    export("claim_evidence.csv",evidence)
    random_ref=dict(expected_random_recall_at100=100/1992,
                   independent_5min_random_onset_capture=1-(1-100/1992)**287,
                   expected_random_new_entries_per_hour=100*(1-100/1992)*12,
                   actual_random_experiment_run=False)
    tdict={r["arm"]:r for r in training}
    ip={r["split"]:r for r in inertia if r["comparison"]=="D/fusion_main"}
    n=lambda x:f"{x:,}"
    f6=lambda x:f"{float(x):.6f}"
    pct=lambda x:f"{100*float(x):.2f}%"
    on=lambda r:f"{int(r['unique_onset_captured'])}/{int(r['unique_onset_eligible'])}"
    pop_table=table(["구간","평가 GPU-time 행","양성 행","양성률","유효 5분 시점","AP block 수","적격 onset"],
       [[r["split"],n(r["evaluated_rows"]),n(r["positive_rows"]),pct(r["pooled_prevalence"]),
         n(r["eligible_5minute_points"]),r["eligible_origin_blocks"],r["onset_eligible"]] for r in population])
    arms_table=table(["조건","B1 특징","B1 학습창","B2"],
       [["A","기본 30개","고정 21일","공통 LR+GBDT / ST21"],["B","Falcon-inspired 45개","고정 21일","동일"],
        ["C","기본 30개","ADST 7·14·21일","동일"],["D","Falcon-inspired 45개","ADST 7·14·21일","동일"]])
    main_table=table(["조건 / 주 Fusion","개발 AP","마지막14일 AP","전체 AP","마지막14일 R@100","마지막14일 churn/h","onset"],
       [[a,f6(val(a+"/fusion_main","mean_origin_block_AP","development")),
         f6(val(a+"/fusion_main","mean_origin_block_AP","terminal")),f6(val(a+"/fusion_main","mean_origin_block_AP")),
         pct(val(a+"/fusion_main","pooled_recall_at_100","terminal")),
         f"{val(a+'/fusion_main','churn_entries_per_hour','terminal'):.1f}",on(row(a+"/fusion_main","terminal"))] for a in ["A","B","C","D"]])
    robust_table=table(["조건 / 주 Fusion","기존 동일가중 AP","시점 수 가중 block AP","30분 미만 block 제외","완전24h block만"],
       [[a,f6(ridx["terminal",a+"/fusion_main"]["original_equal_block_AP"]),
         f6(rw(a+"/fusion_main")),f6(ridx["terminal",a+"/fusion_main"]["posthoc_exclude_under30min_AP"]),
         f6(ridx["terminal",a+"/fusion_main"]["posthoc_complete24h_AP"])] for a in ["A","B","C","D"]])
    branch_models=["shared/B2","A/B1_raw","A/B1_inertia","B/B1_inertia","C/B1_inertia","D/B1_inertia","C/fusion_main","D/fusion_main"]
    branch_table=table(["마지막14일 / 5분","기존 AP","가중 block AP(진단)","R@100","block평균 ROC-AUC","churn/h","onset"],
       [[m,f6(val(m,"mean_origin_block_AP","terminal")),f6(rw(m)),pct(val(m,"pooled_recall_at_100","terminal")),
         f"{val(m,'mean_origin_block_ROC_AUC','terminal'):.3f}",f"{val(m,'churn_entries_per_hour','terminal'):.1f}",
         on(row(m,"terminal"))] for m in branch_models])
    effect_table=table(["D / 주기5분","관성없음 AP","관성 AP","AP 상대변화","R@100 변화","churn 변화","onset"],
       [[s,f6(val("D/fusion_no_inertia","mean_origin_block_AP",s)),f6(val("D/fusion_main","mean_origin_block_AP",s)),
         f"{ip[s]['AP_relative_percent']:+.2f}%",f"{ip[s]['recall_difference_percentage_points']:+.3f}%p",
         f"{ip[s]['churn_relative_percent']:+.2f}%",
         on(row("D/fusion_no_inertia",s))+" → "+on(row("D/fusion_main",s))] for s in ["full","terminal"]])
    cap_table=table(["개발 D / 5분 cap","AP","R@100","churn/h","onset /724"],
       [[f"{r['cap']:.2f}",f6(r["AP"]),pct(r["recall_at100"]),f"{r['churn']:.1f}",r["onset_captured"]]
        for r in cap_rows if r["cadence_minutes"]==5])
    cadence_table=table(["마지막14일 / 주 Fusion","5분 기존 AP","5분의 1시점 block 제외 AP","30분 AP"],
       [[a,f6(val(a+"/fusion_main","mean_origin_block_AP","terminal")),
         f6(ridx["terminal",a+"/fusion_main"]["posthoc_exclude_under30min_AP"]),
         f6(val(a+"/fusion_main","mean_origin_block_AP","terminal","30"))] for a in ["A","B","C","D"]])
    cost_table=table(["B1 조건","fit 수","평균 학습 표본","fit 경과시간 합(초)","fit 중앙값(초)","7/14/21일 origin 수"],
       [[r["arm"],r["classifier_fits"],f"{r['mean_samples']:,.0f}",f"{r['sum_fit_wall_seconds']:.1f}",
         f"{r['median_fit_wall_seconds']:.2f}"," / ".join(str(r["training_window_counts"][str(d)]) for d in [7,14,21])]
         for r in training])
    stamp=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")
    summary_link=link("원본 성능표",ROOT/"metrics_summary.csv")
    origin_link=link("origin별 원본 지표",ROOT/"metrics_by_origin.csv")
    code_link=link("실행된 평가 코드",ROOT/"evaluate_scores.py")
    diagnostic_link=link("후행 강건성 진단표",OUT/"robustness_5m.csv")
    evidence_link=link("주장–근거 매트릭스",OUT/"claim_evidence.csv")
    report=f"""# E31 성능 분석 보고서

작성: {stamp}  
관련 목표: G-004·G-007·G-008 / 대상: IISE 2027 초록의 ML 근거 검토  
판정: **실행·수식 검증 완료 / 성능 우위 주장 보류 / 부분 origin 집계 보완 우선**

## 1. 핵심 결론

1. 63개 origin, 실제 5분 GPU-time {n(validation['rows'])}행과 378개 분류기 fit을 완료했다. 저장 Fusion의 전행 수식 일치 검사 1,008개가 통과했다. 이것은 실행 충실도의 근거이며, 통계적 성능 우위·확률 보정·독립 일반화까지 증명한 것은 아니다.
2. **현재 동일가중 block AP는 매우 짧은 평가 구간에 민감하다.** 마지막 14일의 한 origin에는 5분 시점 1개만 남았지만 24시간 block과 같은 가중치를 받는다. 따라서 현재 AP를 최종 초록의 확정 성능이나 5분 추론의 우월성으로 바로 사용하기 어렵다.
3. 사전 고정한 전체 구성 D(Falcon-inspired+ADST+관성+Fusion)는 일관되게 가장 좋지 않았다. 개발·전체 구간의 주 Fusion AP는 B가 높고, 마지막 구간은 C가 높다. 마지막 구간을 보고 C를 새 최종 모델로 선택하면 별도의 확인 평가가 필요하다.
4. 가장 분명한 관측은 **관성의 순위 안정성–고유 onset 포착 절충**이다. D의 전체 기간 churn은 47.85% 줄었지만 고유 onset 포착은 647→420으로 줄었다. AP/R@100만으로 모든 측면이 개선됐다고 말할 수 없다.
5. B2와 현재 Fusion, cap0.60, ADST의 계산 절감은 추가 검증이 필요하다. 원점수는 그대로 보존하고 평가 집계부터 보완하는 것이 다음 우선 작업이다.

분석 근거: {summary_link}, {origin_link}, {code_link}. 원본 결과 파일과 모델 설정은 이 분석에서 수정하지 않았다. 본 보고서의 가중/구간제외 수치는 **사후 진단**이며 주 지표를 교체하거나 모델을 재선정한 결과가 아니다.

## 2. 실제 실행한 방법과 비교 조건

{arms_table}

- B1은 모든 조건에서 동일 파라미터의 HistGradientBoosting이다. B2는 XID history와 달력 특징을 사용하는 LR+GBDT 평균 ensemble로 네 조건이 공유한다.
- All-XID onset의 다음24시간 (t,t+24h]를 예측한다. 6시간 특징, 마지막10분 차단, 학습·피드백36시간 purge, 24시간 재학습, 전체1,992 GPU의 실제5분 추론을 사용했다.
- ADST는 초기14일, 후보7/14/21일, 정규화 AP 하락15%, 안정3회인 기존 E11 상태전이와 E14의 성숙 피드백 규칙을 사용한다. 피드백은 기존30분 full-GPU AP의 평균이다.
- B1에만 gamma1·반감기2시간 관성을 적용한다. Fusion의 B1 가중치는 lambda=min(0.60,1/(k+1)), k는 origin−10분 이전 최근30일 고유 GPU-onset 횟수이며 하루 동안 유지한다.
- 30분은 동일 Branch 원점수를6개중1개 추출한 다음30분 감쇠로 관성/Fusion을 별도 계산했다. 30분 결과를5분으로 유지 보간한 실험이 아니다.
- 기본30개와 Falcon-inspired45개는 포함관계가 아니다. 기존 max/min8개를 제외하고 다른23개를 추가한 특징 묶음 비교다. 원형 Falcon 재현이나 학습된 GNN의 기여로 표현하지 않는다.
- 원본 XID 기록 공백11곳과 미래24시간이 겹치는 라벨을 공통 제외했다. 개별 GPU의 nonmissing XID 관측 완전성은 미인증이며, 기존 결측 대치/peer 평균 가정도 남아 있다.

## 3. 평가 모집단과 지표의 의미

평가 점수 시각: UTC2023-06-15 09:40–08-17 00:00. 마지막14일 달력 구간은 UTC08-03 00:05–08-17 00:00이다. terminal의 유효5분 시점3,454개는 약12일분이며, 실제 달력14일 전체가 동일하게 평가 가능한 것은 아니다.

{pop_table}

원점수35,913,768행 중 유효라벨을 사용한5분 평가 행은32,294,304행이다. 나머지3,619,464행의 위험점수도 보존했다. 양성률은 약0.765%로 낮다. 3천만 행은 독립된 고장 사례 수가 아니며 시간·GPU 간 상관을 고려해야 한다.

- AP: 각 origin의 유효 GPU-time을 펼쳐 average_precision_score를 구하고 block별 동일가중 평균한다. trapezoidal PR-AUC, 전체 기간 pooled AP, 매시점 GPU ranking AP와 구별한다.
- R@100: 매시점 전체 GPU 중 상위100개에서 포착한 양성 GPU-time / 전체 양성 GPU-time이다. 고유 오류 사건 포착률이 아니다.
- ROC-AUC: 양·음성이 모두 있는 origin block의 평균이다. 개발/전체의 유효 AP block은48/62개, ROC-AUC 계산 가능 block은46/60개다.
- churn/h: 연속 유효시점 Top100의 신규 진입 수를 시간당으로 환산한다. 실제 GPU 이주·작업 재시작·비용 감소를 측정한 수치가 아니다.
- onset: 완전한24시간~10분 경고창이 평가 가능한 고유 GPU/onset-time의 포착 수다. session episode나 모든 관측 오류의 포착률이 아니다.

개발48개와 terminal15개 block을 더한 수가 full62개와 다른 것은 분할 경계가 한 origin을 나누기 때문이다. full은 terminal을 포함한다. onset의 full912개가 개발724+terminal161개와 다른 것도 full에서 분할 경계를 가로지르는 경고창27개를 추가로 평가할 수 있기 때문이다.

## 4. 가장 먼저 보완할 AP 집계 문제

평가 코드167행의 평균은 block 길이에 관계없이 각 AP를 동일하게 반영한다. 마지막 구간 origin23616에는 **{witness['score_time_utc']}의 5분 시점 하나**, 1,992행·양성57행만 남았다. 이 block의 C/Fusion AP는 {witness['C_fusion_AP']:.6f}다.

이5분 block의 terminal 평균 가중치는1/15=6.67%다. 유효시점 수 비중은1/3,454≈0.029%다. 24시간 block과 같은 가중치를 주는 정의는 명세와 일치하지만, 이를 동일한 하루의 성능처럼 해석하면 매우 작은 잔여 구간을 과대 반영한다. 제가 작성한 평가 코드에서 이 부분을 충분히 검토하지 못했다.

### 마지막14일 주 Fusion AP의 집계 민감도

{robust_table}

시점 수 가중 block AP는 sum(n_origin×AP_origin)/sum(n_origin)이다. **전체점수 pooled AP도, 매시점 AP의 평균도 아니다.** 30분 미만/완전24시간 표는 같은 문제를 점검하는 후행 민감도 분석이다. 임계값을 새 선택 규칙으로 확정하지 않았다. 완전24시간 terminal block은9개뿐이라 해당 행만 따로 일반화해서도 안 된다.

예를 들어 B 대 D의 terminal AP 차이는 원래0.020579 대0.016707이지만, 시점 수 가중값은0.011385 대0.011383으로 거의 같다. 원래 등가중 평균만으로 D의 큰 열세를 확정하는 것도 강건하지 않다. 반면 C의 주 Fusion은 가중 진단에서도 D보다 높다.

진단 근거: {diagnostic_link}. 이 결과는 학습·원점수의 실패를 의미하지 않는다. **논문에서 무엇을 평균내는지와 부분 관측 구간을 어떻게 다룰지 먼저 명시해야 한다는 뜻이다.**

## 5. 네 가지 구성의 원래 공식 결과

아래 AP는 원본 파일의 동일가중 block AP다. 앞 절의 민감도를 전제로 읽는다.

{main_table}

### 특징과 ADST

- 고정21일에서 A→B의 전체 Fusion AP는10.88% 높지만 R@100은0.097%p 낮고 churn은31.99% 높다. 특징 증가가 모든 지표의 개선으로 이어지지는 않는다. B1 원점수의 전체 AP는 A0.013622/B0.013494로 거의 같거나 B가 조금 낮다.
- A→C의 전체 Fusion AP는6.51% 낮고 R@100은1.510%p 낮다. terminal에서는 반대로 C의 AP가6.48% 높고 R@100이0.384%p 높다. ADST 이득이 기간에 따라 바뀐다.
- B→D의 개발/전체 AP는10.26%/12.87% 낮다. terminal 원래 AP 차이18.81%는 짧은 구간의 영향이 커, 가중 진단에서는 거의 사라진다. 그래도 D의 일관된 우위를 뒷받침하지 않는다.
- ADST의 실제7/14/21일 사용은 C30/24/9회, D38/20/5회다. 동적 제어가 실행된 것은 확인됐지만 성능 이득과는 별개다. C/D의 특징 비교에는 특징에 따라 달라진 ADST 창 궤적도 포함된다.
- 원래 terminal C−A AP의 origin별 승/무/패는7/1/7이다. 평균 차이만으로 안정적인 우월성이나 통계적 유의성을 주장하지 않는다.

## 6. Branch 1 단독과 Fusion 비교

{branch_table}

전체 기간에서 B1+관성 AP는 B0.015511, D0.014379, A0.014043, C0.013987이다. **개발·전체 AP 기준으로는 B1+관성의 B가 강한 비교 기준**이다. 동일 arm에 B2를 결합하면 전체 AP는 A/B/C/D에서 모두 낮아진다. B의 경우 AP0.015511→0.014452, R@100 8.91%→7.49%, onset455→407, churn176.0→158.0/h다.

terminal의 기존 AP에서는 Fusion이 B1+관성보다 아주 조금 높은 경우가 있지만, 시점 수 가중 block AP에서는 네 arm 모두 Fusion 후 낮아진다. C의 가중 AP는 B1+관성0.014061→Fusion0.013127이다. 따라서 현재 Fusion의 예측 기여를 확정하기 어렵다. 일부 조건에서 churn이나 onset 포착이 개선되는지를 별도로 보아야 한다.

B2 단독의 전체 R@100은4.61%, block평균 ROC-AUC는0.480이다. 전체1,992개중100개를 균등 무작위로 선택할 때의 R@100 기댓값은5.02%다. 이는 수학적 참고값이며 실제 랜덤 실험은 하지 않았다. 현재 B2가 신뢰할 만한 Top100 순위 기여를 보인다고 보기 어렵다.

**원인 미확정:** B2 자체 정보력, 학습창/분포 이동, LR·GBDT 평균, 점수 보정, GPU별 가중치 중 무엇이 손실을 만드는지는 이 실험으로 분리하지 않았다. origin 내 공통 prior 보정은 순위를 기본적으로 보존하므로 B2 순위 성능 문제를 보정 하나로 설명해서도 안 된다. 경험적 calibration은 인증되지 않았다.

## 7. 관성: 강한 안정성 효과와 포착 손실

{effect_table}

D의 terminal 시점 수 가중 block AP도 관성없음0.010679→관성0.011383으로6.59% 높아진다. 따라서 AP 개선 방향 자체가 전부 작은 block 하나의 결과인 것은 아니다. 다만 원래13.41%와는 효과 크기가 다르다.

반면 고유 onset 포착은 전체647→420, terminal118→83으로 감소한다. 순위가 안정됐다는 사실을 오류 포착·운영 성능 전체가 개선됐다는 뜻으로 사용하면 안 된다. 짧은 위험 상승이 평활화됐다는 설명은 가능한 가설이며, 사건별 추론 경로를 확인해야 확정할 수 있다.

같은 Top100 크기에서도 순위 갱신이 많으면24시간 안에 한 번이라도 경고하는 onset 포착 기회가 늘어난다. 예를 들어 매5분 독립적으로 무작위100개를 선택하는 가정에서는287번의 경고 기회로 onset 포착 기댓값이약99.99996%가 되지만 churn도약1,140/h로 높다. 이 역시 실제 실험이 아닌 계산 예시다. onset 포착은 경고 횟수·cooldown·GPU-days-in-warning·lead time을 맞춰 비교할 필요가 있다.

## 8. Pareto 상한0.60 민감도

{cap_table}

cap1.00은0.60보다 개발 AP11.63%, R@100 0.219%p 개선되고 churn은5.04% 낮지만, onset 포착은333→293으로 줄었다. AP·R·churn만 보면0.60을 지지하지 않으며, onset까지 포함하면 목적 간 절충이다. **0.60의 최적성은 입증되지 않았다.**

alpha1인 현재 식에서 cap≥0.50의 차이는 k=0 GPU에만 영향을 준다. k=1이면 이미B1 가중치0.5이기 때문이다. 이 표는 오류 이력이 없는 GPU에 B1을 얼마나 맡길지에 대한 민감도다. 개발 구간 D만 평가한 결과로 모든 arm/기간에 일반화하지 않는다. 이 결과를 보고 terminal 설정을 바꾸지 않았다.

## 9. 5분 대30분

{cadence_table}

5분의1시점 block은30분 부분추출에 포함되지 않는다. 따라서 AP를 평균낸 origin 집합도15개와14개로 다르다. C의 AP는5분0.021733 대30분0.016298이지만, 문제의1시점 block을 제외한5분값은0.016310이다. **원래 차이를5분 추론의 우월성으로 해석할 수 없다.**

전체 D의 R@100은5분6.617%,30분6.606%로 가깝다. onset은420/912 대345/912, churn은140.1/h 대37.5/h다. 더 촘촘한 판단은 더 많은 경고/교체 기회를 주므로 동일한 경고·개입 예산을 적용한 비교가 필요하다. 두 주기의 학습 모델은 같으며, 이번 비교는 서로 다른 모델을 재학습한 우열 실험이 아니다.

## 10. 학습 비용과 ADST 계산 절감 주장

{cost_table}

ADST는 C/D의 학습 표본을 A/B 대비평균23.63%/26.24% 줄였다. 하지만 분류기fit은 arm마다63회로 같다. 이 구현은 상태에 따라창을 조절하며 매일 재학습한다. action 이름의 skip/hold를 fit 생략으로 해석하지 않는다.

기록된 B1 fit 경과시간 합은 C가 A보다36.56%, D가 B보다42.67% 길다. 이 값은 fit 구간의 wall time이며 전처리·추론·B2공통비용을 모두 포함한 시스템 비용이 아니다. 초기/재개 실행 방식과 시스템 부하 차이도 있어 알고리즘의 본질적 효율 차이로 단정하지 않는다. 실제 반복 수/자원 조건을 통제하기 전에는 **ADST의 계산 절감 기여를 주장할 수 없다.**

## 11. 초록에 사용할 수 있는 주장과 보류할 주장

### 현재 자료로 가능한 서술

- 시간 누수를 통제한 회고적 All-XID 순위 예측 비교를 구축하고, 고정/동적 학습창과 두 특징 묶음을 대조했다.
- 관성은 순위 교체를 크게 줄였지만 고유 onset 포착 손실을 동반했으며, 이는 예측–안정성 절충을 보여준다.
- ADST와 history Fusion의 효과는 구간·특징·목적에 따라 달랐다.
- 관측 공백과 부분 origin의 AP 집계가 성능 해석에 큰 영향을 주므로 이를 별도 검토해야 한다.

### 현재 보류할 서술

- Falcon+ADST+Fusion 전체 구조가 모든 baseline을 능가한다.
- ADST가 정확도와 학습 비용을 동시에 개선한다.
- cap0.60이 최적이거나 이론적으로 보장된다.
- 5분 추론이30분보다AP를 확실히 개선한다.
- 독립된 새 미관측 데이터에서 일반화가 검증됐다.
- 낮은 churn으로 실제 작업 지연·GPU 이주·자원배분 성능이 개선됐다.

terminal은 과거 열람 이력이 있는 retrospective prequential 구간이며, 사전 고정 규칙의 성숙 후 업데이트를 허용했다. terminal 최고 모델을 지금 재선정하면 이 구간은 추가 선택에도 사용된 것이 된다. 새 확인 평가 없이 독립 test라고 부르지 않는다. E30 운영 최적화는 이번 ML 수치에서 검증하지 않았다.

## 12. 다음 작업 제안 — 실행/채택 승인과 구분

1. **최우선: 평가 집계 확정.** 원래 동일가중 AP를 보존하면서 유효시점 가중 block AP·매결정시점 AP·완전 block 민감도를 비교하고, 논문 목적에 맞는 AP 집계를 문서로 확정한다. 5분/30분은 공통 origin 집합도 비교한다. 현재 저장 점수로 가능하며 Branch 재학습이 필요하지 않다. 본 보고서는 원본 성능표를 교체하지 않았다.
2. **B2와 Fusion 감사.** 최근오류 k=0/1/2+ cohort, Branch 점수 분포/보정, B1대B2 단독 포착을 먼저 확인한다. LR/GBDT를 따로 평가하려면 개별 모델 재생성/재학습이 추가로 필요할 수 있다. B2를 유지해야 한다는 전제 없이 그 기여를 검증한다.
3. **ADST 근거 보완.** 개발 구간에서 선택한 고정7/14/21일 baseline과 비교하고, AP 잡음/창 전환/학습시간/실제fit 반복 수를 함께 남긴다. 신규 학습은 별도 계획·승인 대상이다.
4. **경고 예산과 불확실성.** cooldown과 경고량·lead time을 맞춘 onset 평가, 시간 의존성을 고려한 block 기반 불확실성 분석을 보완한다. 수천만 GPU-time 행을 독립 표본으로 한 검정은 하지 않는다. 이 보고서에서는 신뢰구간·유의성 검정을 수행하지 않았다.
5. **논문 모델 채택 보류.** B1+관성은 강한 후보 비교군이지만, 지금 terminal 최고인 C를 최종 모델로 확정하지 않는다. 집계 보완 후 개발 기준의 선택 절차와 별도 확인 자료를 정한다.

현재 초록은 **예측 성능과 순위 안정성의 절충 및 누수·관측 공백을 고려한 회고적 평가**를 중심으로 구성하는 편이 자료에 충실하다. 일괄 성능 향상을 전제한 최종안은 보완이 필요하다.

## 13. 근거와 재현

- {summary_link}: 기존 공식 출력, 수정하지 않음.
- {origin_link}: 모든 후행 집계·paired 승/패의 직접 입력.
- {code_link}: AP 동일가중 평균, Top100·onset·관성 정의.
- {link('설정',ROOT/'config.json')}, {link('평가 계약',ROOT/'SPEC.md')}, {link('분할 명세',ROOT/'split_manifest.json')}: 사전 설정과 실제 구간.
- {link('최종 실행 검증',ROOT/'final_validation.json')}: 행수·fit·전행 Fusion 일치.
- {link('분석 JSON·입력 해시',OUT/'analysis.json')}, {link('분석 재현 코드',Path(__file__))}: 원본 집계 재확인 및 보고서 생성.
- {evidence_link}: confirmed / hypothesis / proposed / unsupported 판정.

이 분석은 읽은 원본 지표의 합계·AP·R@100·churn을 다시 대조했다. 원본 지표·Branch/Fusion 점수·config·실험 코드는 수정하지 않았고, 재학습·추론 재실행·설치·커밋·푸시도 수행하지 않았다.
"""
    REPORT_MD.write_text(report,encoding="utf-8")
    after={str(path.relative_to(ROOT)):sha(path) for path in sources}
    assert before==after,"Experiment inputs were changed"
    result=dict(created_kst=stamp,goals=["G-004","G-007","G-008"],input_sha256=before,
                source_metrics_unchanged=True,branch_refits=0,new_inference=0,
                posthoc_diagnostics=True,primary_metric_replaced=False,model_selected=False,
                original_aggregation_verified=dict(groups=len(groups),max_AP_error=maximum_ap_error,max_recall_error=maximum_recall_error),
                tiny_block_witness=witness,random_reference=random_ref,
                main_findings=["macro AP sensitive to tiny partial origin","no consistent D superiority",
                               "inertia stability/capture tradeoff","Fusion gain and cap optimality unsupported"],
                statistical_significance_test_run=False,
                outputs=[p.name for p in sorted(OUT.glob("*.csv"))],
                report_markdown=str(REPORT_MD),report_html=str(REPORT_HTML))
    (OUT/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    css="""body{font-family:system-ui,'Malgun Gothic',sans-serif;max-width:1120px;margin:42px auto;padding:0 24px;color:#17212e;line-height:1.75;background:#fff}h1{font-size:2rem;line-height:1.3;margin-bottom:20px}h2{margin-top:48px;padding-top:16px;border-top:2px solid #dce3eb;font-size:1.35rem}h3{font-size:1.1rem;margin-top:26px}a{color:#185a93;overflow-wrap:anywhere}strong{font-weight:720}table{border-collapse:collapse;width:100%;font-size:.88rem;margin:22px 0;display:block;overflow-x:auto}th,td{padding:9px 11px;border:1px solid #d8dee7;white-space:nowrap;text-align:right}th{background:#edf2f7;color:#22384f}th:first-child,td:first-child{text-align:left}tr:nth-child(even){background:#fafbfd}li{margin:7px 0}blockquote{border-left:4px solid #aa7000;background:#fff9e9;padding:12px 20px;margin:24px 0}.meta{font-size:.85rem;color:#536273}.footer{margin:44px 0 12px;border-top:1px solid #dce3eb;padding-top:16px;color:#536273;font-size:.85rem}@media print{body{max-width:none;margin:0;padding:0;color:#111;font-size:10pt}h2,h3{break-after:avoid}table{display:table;font-size:8pt}th,td{white-space:normal;padding:5px}tr{break-inside:avoid}a{color:inherit;text-decoration:none}}"""
    node_code="""
const fs=require('fs');const path=require('path');const {marked}=require('marked');
const args=process.argv.slice(1);let body=marked.parse(fs.readFileSync(args[0],'utf8'),{gfm:true});
body=body.replace(/href="\\/C:([^"]+)"/g,(_m,rest)=>{const abs=decodeURIComponent('C:'+rest);
return 'href="'+encodeURI(path.relative(path.dirname(args[1]),abs).replace(/\\\\/g,'/'))+'"';});
const html='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E31 성능 분석 보고서</title><style>'+CSS+'</style></head><body>'+body+'<div class="footer">E31 성능 분석 · 원본 보존 · 후행 진단 별도 표시 · 2026-11-02</div></body></html>';
fs.writeFileSync(args[1],html,'utf8');
""".replace("CSS",json.dumps(css,ensure_ascii=False))
    bundle=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    env=os.environ.copy();env["NODE_PATH"]=str(bundle/"node_modules")
    subprocess.run([str(bundle/"bin/node.exe"),"-e",node_code,str(REPORT_MD),str(REPORT_HTML)],env=env,check=True)
    html=REPORT_HTML.read_text(encoding="utf-8")
    assert "E31 성능 분석 보고서" in html and "시점 수 가중" in html
    assert "<script" not in html and 'src="http' not in html and 'href="https:' not in html
    assert len(list(OUT.glob("*.csv")))==11
    print(json.dumps(dict(status="PASS",report_markdown=str(REPORT_MD),report_html=str(REPORT_HTML),
          tables=11,verified_groups=len(groups),max_AP_error=maximum_ap_error,inputs_unchanged=True),ensure_ascii=True))


if __name__=="__main__":
    main()

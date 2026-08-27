from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "eda_blox_inputs"
TRACE = ROOT / "trace_seren.csv"
EPISODES = ROOT / "outputs" / "eda_phase1" / "xid_event_episodes.csv"
EVENT_INDEX = ROOT / "outputs" / "eda_event_windows" / "event_index.csv"
LAG = ROOT / "outputs" / "eda_xid43_timestamp_lag" / "xid43_lag_per_pair.csv"
CLASSIFICATION = ROOT / "outputs" / "eda_extended_metrics" / "xid43_event_classification.csv"
XID_INVENTORY = ROOT / "outputs" / "eda_phase1" / "xid_inventory.json"
GPU_CAPACITY = 1_992
GPUS_PER_NODE = 8


def q(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile)) if len(values) else np.nan


def job_eda(xid_start: pd.Timestamp, xid_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    columns = [
        "job_id",
        "submit_time",
        "start_time",
        "end_time",
        "duration",
        "gpu_time",
        "gpu_num",
        "node_num",
        "cpu_num",
        "type",
        "user",
        "state",
    ]
    jobs = pd.read_csv(TRACE, usecols=columns)
    for column in ["submit_time", "start_time", "end_time"]:
        jobs[column] = pd.to_datetime(jobs[column], errors="coerce", utc=True)

    jobs["runtime_seconds"] = (jobs["end_time"] - jobs["start_time"]).dt.total_seconds()
    jobs["in_fault_window"] = jobs["submit_time"].between(xid_start, xid_end)
    jobs["gpu_job"] = jobs["gpu_num"].gt(0)
    jobs["completed"] = jobs["state"].eq("COMPLETED")
    jobs["positive_duration"] = jobs["duration"].gt(0)
    jobs["cluster_capacity_feasible"] = jobs["gpu_num"].le(GPU_CAPACITY)
    jobs["topology_feasible_8gpu"] = jobs["gpu_num"].le(jobs["node_num"] * GPUS_PER_NODE)
    jobs["main_eligible"] = (
        jobs["in_fault_window"]
        & jobs["gpu_job"]
        & jobs["completed"]
        & jobs["positive_duration"]
        & jobs["cluster_capacity_feasible"]
        & jobs["topology_feasible_8gpu"]
    )

    window_gpu = jobs[jobs["in_fault_window"] & jobs["gpu_job"]].copy()
    selected = jobs[jobs["main_eligible"]].copy().sort_values(["submit_time", "job_id"])
    selected["arrival_time_sec"] = (selected["submit_time"] - xid_start).dt.total_seconds().astype("int64")
    selected["duration_sec"] = selected["duration"].astype("int64")
    selected["historical_state"] = selected["state"]
    job_tape = selected[
        [
            "job_id",
            "submit_time",
            "arrival_time_sec",
            "duration_sec",
            "gpu_num",
            "node_num",
            "cpu_num",
            "type",
            "user",
            "historical_state",
        ]
    ]
    job_tape.to_csv(OUTPUT / "job_tape_main.csv", index=False)

    state_summary = (
        window_gpu.groupby("state", dropna=False)
        .agg(
            jobs=("job_id", "size"),
            gpu_hours=("gpu_time", lambda x: x.sum() / 3600),
            median_duration_sec=("duration", "median"),
            p95_duration_sec=("duration", lambda x: x.quantile(0.95)),
            median_gpu_num=("gpu_num", "median"),
            p95_gpu_num=("gpu_num", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    state_summary["job_fraction"] = state_summary["jobs"] / state_summary["jobs"].sum()
    state_summary.to_csv(OUTPUT / "job_state_selection_summary.csv", index=False)

    selection = window_gpu.assign(
        selection_group=np.where(window_gpu["main_eligible"], "main_completed_eligible", "excluded_gpu_job")
    )
    bias_rows = []
    for name, group in selection.groupby("selection_group"):
        bias_rows.append(
            {
                "selection_group": name,
                "jobs": len(group),
                "job_fraction": len(group) / len(selection),
                "gpu_hours": group["gpu_time"].sum() / 3600,
                "median_duration_sec": group["duration"].median(),
                "p90_duration_sec": q(group["duration"], 0.90),
                "p95_duration_sec": q(group["duration"], 0.95),
                "p99_duration_sec": q(group["duration"], 0.99),
                "median_gpu_num": group["gpu_num"].median(),
                "p95_gpu_num": q(group["gpu_num"], 0.95),
                "multi_gpu_fraction": group["gpu_num"].gt(1).mean(),
                "multi_node_fraction": group["node_num"].gt(1).mean(),
            }
        )
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(OUTPUT / "job_inclusion_bias.csv", index=False)

    size_summary = (
        selected.groupby(["gpu_num", "node_num"], as_index=False)
        .agg(
            jobs=("job_id", "size"),
            median_duration_sec=("duration", "median"),
            p90_duration_sec=("duration", lambda x: x.quantile(0.90)),
            p95_duration_sec=("duration", lambda x: x.quantile(0.95)),
            total_gpu_hours=("gpu_time", lambda x: x.sum() / 3600),
        )
        .sort_values("jobs", ascending=False)
    )
    size_summary.to_csv(OUTPUT / "job_size_duration_summary.csv", index=False)

    type_summary = (
        selected.groupby("type", dropna=False)
        .agg(
            jobs=("job_id", "size"),
            median_duration_sec=("duration", "median"),
            p95_duration_sec=("duration", lambda x: x.quantile(0.95)),
            median_gpu_num=("gpu_num", "median"),
            total_gpu_hours=("gpu_time", lambda x: x.sum() / 3600),
        )
        .reset_index()
        .sort_values("jobs", ascending=False)
    )
    type_summary["job_fraction"] = type_summary["jobs"] / type_summary["jobs"].sum()
    type_summary.to_csv(OUTPUT / "job_type_summary.csv", index=False)

    selected["submit_day"] = selected["submit_time"].dt.floor("D")
    daily = (
        selected.groupby("submit_day", as_index=False)
        .agg(
            jobs=("job_id", "size"),
            requested_gpus=("gpu_num", "sum"),
            offered_gpu_seconds=("gpu_time", "sum"),
        )
    )
    daily["offered_load_ratio"] = daily["offered_gpu_seconds"] / (GPU_CAPACITY * 86_400)
    daily.to_csv(OUTPUT / "job_daily_offered_load.csv", index=False)

    interarrival = selected["submit_time"].sort_values().diff().dt.total_seconds().dropna()
    checks = {
        "trace_rows": int(len(jobs)),
        "gpu_jobs_in_fault_window": int(len(window_gpu)),
        "main_job_tape_rows": int(len(job_tape)),
        "main_inclusion_rate_among_window_gpu_jobs": float(len(job_tape) / len(window_gpu)),
        "completed_gpu_jobs_before_validity_filters": int((window_gpu["completed"]).sum()),
        "zero_duration_completed_gpu_jobs": int((window_gpu["completed"] & ~window_gpu["positive_duration"]).sum()),
        "topology_infeasible_completed_gpu_jobs": int((window_gpu["completed"] & ~window_gpu["topology_feasible_8gpu"]).sum()),
        "capacity_infeasible_completed_gpu_jobs": int((window_gpu["completed"] & ~window_gpu["cluster_capacity_feasible"]).sum()),
        "duration_runtime_mismatch_over_1s": int((jobs["duration"].sub(jobs["runtime_seconds"]).abs() > 1).sum()),
        "job_tape_first_submit": job_tape["submit_time"].min().isoformat(),
        "job_tape_last_submit": job_tape["submit_time"].max().isoformat(),
        "median_duration_sec": float(job_tape["duration_sec"].median()),
        "p95_duration_sec": q(job_tape["duration_sec"], 0.95),
        "p99_duration_sec": q(job_tape["duration_sec"], 0.99),
        "median_gpu_num": float(job_tape["gpu_num"].median()),
        "p95_gpu_num": q(job_tape["gpu_num"], 0.95),
        "multi_gpu_fraction": float(job_tape["gpu_num"].gt(1).mean()),
        "multi_node_fraction": float(job_tape["node_num"].gt(1).mean()),
        "median_interarrival_sec": float(interarrival.median()),
        "daily_offered_load_median": float(daily["offered_load_ratio"].median()),
        "daily_offered_load_p95": q(daily["offered_load_ratio"], 0.95),
        "daily_offered_load_max": float(daily["offered_load_ratio"].max()),
    }
    return job_tape, {"checks": checks, "state": state_summary, "bias": bias, "daily": daily}


def fault_eda(xid_start: pd.Timestamp, xid_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    episodes = pd.read_csv(EPISODES, parse_dates=["start_time", "end_time"])
    events = pd.read_csv(EVENT_INDEX, parse_dates=["start_time", "end_time"])
    for frame in [episodes, events]:
        for column in ["start_time", "end_time"]:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    events = events[events["start_time"].between(xid_start, xid_end)].copy()

    lag = pd.read_csv(LAG, parse_dates=["start_time", "event_inferred_low_onset_u10"])
    lag["start_time"] = pd.to_datetime(lag["start_time"], utc=True)
    lag["event_inferred_low_onset_u10"] = pd.to_datetime(lag["event_inferred_low_onset_u10"], utc=True)
    lag = lag.sort_values("match_id").drop_duplicates("event_id")
    events = events.merge(
        lag[["event_id", "event_terminal_low_minutes_u10", "event_inferred_low_onset_u10"]],
        on="event_id",
        how="left",
    )

    classes = pd.read_csv(CLASSIFICATION)
    classes["xid_code"] = 43
    events = events.merge(
        classes[["cluster_id", "node_id", "xid_code", "node_power_ratio", "event_type"]],
        on=["cluster_id", "node_id", "xid_code"],
        how="left",
    )
    events["fault_episode_id"] = events.apply(
        lambda r: f"xid{int(r.xid_code)}_{r.gpu_id}_e{int(r.episode_id)}", axis=1
    )
    events["fault_cluster_id"] = events["cluster_id"].map(lambda x: f"fc_{int(x):06d}")
    events["xid_time_raw"] = events["start_time"]
    events["terminal_low_supported_3m"] = events["event_terminal_low_minutes_u10"].ge(3)
    events["fault_time_onset"] = events["event_inferred_low_onset_u10"].where(
        events["terminal_low_supported_3m"]
    )
    events["fault_time_mixed"] = events["fault_time_onset"].fillna(events["xid_time_raw"])
    events["timestamp_evidence"] = np.select(
        [events["terminal_low_supported_3m"], events["event_terminal_low_minutes_u10"].notna()],
        ["terminal_low_onset_supported", "audited_no_3m_terminal_low"],
        default="not_audited",
    )
    events["event_type"] = events["event_type"].fillna("unclassified")
    events["fault_scope_evidence"] = np.where(
        events["cluster_gpu_count"].gt(1), "multi_gpu_same_node_time", "single_gpu_observed"
    )
    events["interrupt_evidence"] = np.select(
        [
            events["event_type"].str.contains("shutdown_like", na=False),
            events["event_type"].eq("no_terminal_gpu_inactivity"),
            events["event_type"].str.contains("missing", na=False),
        ],
        ["shutdown_like", "no_terminal_inactivity", "observability_limited"],
        default="unclassified",
    )
    events["conservative_interrupt_candidate"] = events["interrupt_evidence"].eq("shutdown_like")
    events["severity_status"] = "not_observed_requires_scenario"

    fault_columns = [
        "fault_episode_id",
        "fault_cluster_id",
        "event_id",
        "gpu_id",
        "node_id",
        "gpu_index",
        "xid_code",
        "xid_time_raw",
        "fault_time_onset",
        "fault_time_mixed",
        "end_time",
        "duration_seconds",
        "observations",
        "cluster_gpu_count",
        "fault_scope_evidence",
        "event_terminal_low_minutes_u10",
        "timestamp_evidence",
        "event_type",
        "interrupt_evidence",
        "conservative_interrupt_candidate",
        "severity_status",
    ]
    fault_tape = events[fault_columns].sort_values(["xid_time_raw", "node_id", "gpu_id"])
    fault_tape.to_csv(OUTPUT / "fault_tape_candidates.csv", index=False)

    clusters = (
        events.groupby(["cluster_id", "fault_cluster_id", "node_id", "xid_code"], as_index=False)
        .agg(
            xid_time_raw=("xid_time_raw", "min"),
            gpu_count=("gpu_id", "nunique"),
            max_episode_duration_sec=("duration_seconds", "max"),
            timestamp_audited_gpus=("event_terminal_low_minutes_u10", lambda x: x.notna().sum()),
            terminal_low_supported_gpus=("terminal_low_supported_3m", "sum"),
            event_type=("event_type", "first"),
        )
    )
    clusters["multi_gpu_cluster"] = clusters["gpu_count"].gt(1)
    clusters.to_csv(OUTPUT / "fault_cluster_tape.csv", index=False)

    code_summary = (
        events.groupby("xid_code", as_index=False)
        .agg(
            gpu_episodes=("event_id", "size"),
            fault_clusters=("cluster_id", "nunique"),
            affected_gpus=("gpu_id", "nunique"),
            affected_nodes=("node_id", "nunique"),
            multi_gpu_episode_fraction=("cluster_gpu_count", lambda x: x.gt(1).mean()),
            median_episode_duration_sec=("duration_seconds", "median"),
            p95_episode_duration_sec=("duration_seconds", lambda x: x.quantile(0.95)),
            timestamp_audited_episodes=("event_terminal_low_minutes_u10", lambda x: x.notna().sum()),
            onset_supported_episodes=("terminal_low_supported_3m", "sum"),
        )
    )
    code_summary.to_csv(OUTPUT / "fault_code_summary.csv", index=False)

    scope_summary = (
        clusters.groupby(["xid_code", "gpu_count"], as_index=False)
        .agg(fault_clusters=("cluster_id", "size"))
        .sort_values(["xid_code", "gpu_count"])
    )
    scope_summary.to_csv(OUTPUT / "fault_scope_summary.csv", index=False)

    recurrence = events.sort_values(["gpu_id", "xid_code", "xid_time_raw"]).copy()
    recurrence["interarrival_hours"] = (
        recurrence.groupby(["gpu_id", "xid_code"])["xid_time_raw"].diff().dt.total_seconds() / 3600
    )
    recurrence_summary = []
    for code, group in recurrence.groupby("xid_code"):
        values = group["interarrival_hours"].dropna()
        recurrence_summary.append(
            {
                "xid_code": code,
                "repeat_intervals": len(values),
                "median_interarrival_hours": values.median(),
                "p25_interarrival_hours": q(values, 0.25),
                "p75_interarrival_hours": q(values, 0.75),
                "within_1h_fraction": values.le(1).mean(),
                "within_24h_fraction": values.le(24).mean(),
                "within_7d_fraction": values.le(24 * 7).mean(),
            }
        )
    recurrence_summary = pd.DataFrame(recurrence_summary)
    recurrence_summary.to_csv(OUTPUT / "fault_recurrence_summary.csv", index=False)

    clusters["day"] = clusters["xid_time_raw"].dt.floor("D")
    daily = clusters.groupby(["day", "xid_code"], as_index=False).agg(fault_clusters=("cluster_id", "size"))
    daily.to_csv(OUTPUT / "fault_daily_clusters.csv", index=False)

    checks = {
        "fault_tape_rows": int(len(fault_tape)),
        "fault_clusters": int(clusters["cluster_id"].nunique()),
        "affected_gpus": int(events["gpu_id"].nunique()),
        "affected_nodes": int(events["node_id"].nunique()),
        "multi_gpu_cluster_fraction": float(clusters["multi_gpu_cluster"].mean()),
        "timestamp_audited_episodes": int(events["event_terminal_low_minutes_u10"].notna().sum()),
        "onset_supported_3m_episodes": int(events["terminal_low_supported_3m"].sum()),
        "shutdown_like_episodes": int(events["interrupt_evidence"].eq("shutdown_like").sum()),
        "unclassified_episodes": int(events["event_type"].eq("unclassified").sum()),
        "max_cluster_gpu_count": int(clusters["gpu_count"].max()),
        "max_episode_duration_days": float(events["duration_seconds"].max() / 86_400),
    }
    return fault_tape, {"checks": checks, "clusters": clusters, "code": code_summary, "daily": daily}


def risk_audit(risk_path: Path | None, xid_start: pd.Timestamp, xid_end: pd.Timestamp) -> dict:
    contract = pd.DataFrame(
        [
            ["decision_time", "timestamp with timezone", True, "replay window 안이며 decision-time feature만 사용"],
            ["gpu_id", "string", True, "AcmeTrace 1,992개 GPU ID 집합에 포함"],
            ["risk_score", "finite numeric", True, "높을수록 위험한 방향으로 일관"],
            ["risk_rank", "positive integer", True, "동일 decision_time에서 score 내림차순 순위"],
            ["model_fold", "string", True, "시간순 out-of-fold 예측 provenance"],
            ["feature_cutoff_time", "timestamp with timezone", True, "decision_time 이하; ambiguous zone 제외 검증"],
        ],
        columns=["field", "expected_type", "required", "validation"],
    )
    contract.to_csv(OUTPUT / "risk_tape_contract.csv", index=False)

    rows = []
    if risk_path is None:
        rows = [
            ["risk_tape_supplied", "BLOCKED", "Risk Tape 파일이 제공되지 않음"],
            ["required_schema", "BLOCKED", "decision_time, gpu_id, risk_score, risk_rank 필요"],
            ["out_of_fold_provenance", "BLOCKED", "model_fold와 시간순 분할 정보 필요"],
            ["leakage_cutoff", "BLOCKED", "feature_cutoff_time과 terminal-low ambiguous zone 제외 증명 필요"],
        ]
    else:
        risk = pd.read_csv(risk_path)
        required = {"decision_time", "gpu_id", "risk_score", "risk_rank"}
        missing = sorted(required - set(risk.columns))
        rows.append(["risk_tape_supplied", "PASS", str(risk_path)])
        rows.append(["required_schema", "PASS" if not missing else "FAIL", ", ".join(missing) or "all present"])
        if not missing:
            decision = pd.to_datetime(risk["decision_time"], errors="coerce", utc=True)
            score = pd.to_numeric(risk["risk_score"], errors="coerce")
            rank = pd.to_numeric(risk["risk_rank"], errors="coerce")
            rows.extend(
                [
                    ["decision_time_parseable", "PASS" if decision.notna().all() else "FAIL", f"invalid={decision.isna().sum()}"],
                    ["decision_time_coverage", "PASS" if decision.between(xid_start, xid_end).all() else "WARN", f"outside_window={(~decision.between(xid_start, xid_end)).sum()}"],
                    ["finite_risk_score", "PASS" if np.isfinite(score).all() else "FAIL", f"invalid={(~np.isfinite(score)).sum()}"],
                    ["positive_risk_rank", "PASS" if rank.ge(1).all() else "FAIL", f"invalid={(~rank.ge(1)).sum()}"],
                    ["unique_decision_gpu", "PASS" if not risk.duplicated(["decision_time", "gpu_id"]).any() else "FAIL", f"duplicates={risk.duplicated(['decision_time', 'gpu_id']).sum()}"],
                ]
            )
            rows.append(["out_of_fold_provenance", "PASS" if "model_fold" in risk else "BLOCKED", "model_fold required"])
            rows.append(["leakage_cutoff", "PASS" if "feature_cutoff_time" in risk else "BLOCKED", "feature_cutoff_time required"])
    audit = pd.DataFrame(rows, columns=["check", "status", "detail"])
    audit.to_csv(OUTPUT / "risk_tape_audit.csv", index=False)
    return {"path": str(risk_path) if risk_path else None, "audit": audit.to_dict("records")}


def plots(job: dict, fault: dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    state = job["state"].sort_values("jobs", ascending=False)
    axes[0, 0].bar(state["state"].astype(str), state["jobs"])
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].set_title("GPU jobs in fault window by historical state")
    axes[0, 0].set_ylabel("Jobs")
    bias = job["bias"].set_index("selection_group")
    axes[0, 1].bar(bias.index, bias["median_duration_sec"])
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].set_title("Completed-only selection: median duration")
    axes[0, 1].set_ylabel("Seconds")
    daily = job["daily"]
    axes[1, 0].plot(daily["submit_day"], daily["jobs"])
    axes[1, 0].set_title("Main Job Tape arrivals")
    axes[1, 0].set_ylabel("Jobs/day")
    axes[1, 1].plot(daily["submit_day"], daily["offered_load_ratio"])
    axes[1, 1].axhline(1, color="tab:red", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Offered GPU load / 1,992-GPU capacity")
    axes[1, 1].set_ylabel("Ratio")
    fig.tight_layout()
    fig.savefig(OUTPUT / "job_tape_profile.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    daily_fault = fault["daily"].pivot(index="day", columns="xid_code", values="fault_clusters").fillna(0)
    for code in daily_fault.columns:
        axes[0].plot(daily_fault.index, daily_fault[code], label=f"XID {int(code)}")
    axes[0].set_title("Fault clusters by day")
    axes[0].legend()
    sizes = fault["clusters"]["gpu_count"].value_counts().sort_index()
    axes[1].bar(sizes.index.astype(str), sizes.values)
    axes[1].set_title("Same-node simultaneous cluster size")
    axes[1].set_xlabel("GPUs")
    code = fault["code"]
    axes[2].bar(code["xid_code"].astype(str), code["fault_clusters"])
    axes[2].set_title("Fault clusters by XID code")
    axes[2].set_xlabel("XID code")
    fig.tight_layout()
    fig.savefig(OUTPUT / "fault_tape_profile.png", dpi=160)
    plt.close(fig)


def write_report(job: dict, fault: dict, risk: dict, xid_start: pd.Timestamp, xid_end: pd.Timestamp) -> None:
    j = job["checks"]
    f = fault["checks"]
    risk_ready = any(row["check"] == "required_schema" and row["status"] == "PASS" for row in risk["audit"])
    lines = [
        "# Blox Replay 입력 적합성 EDA",
        "",
        "## 판정",
        "",
        "- Job Tape: **조건부 READY** — 공통 fault 기간의 COMPLETED GPU job으로 Blox 입력 후보를 생성했다.",
        "- Fault Tape: **조건부 READY** — XID episode와 동시 GPU cluster를 구성했지만 severity·downtime·fault time은 시나리오 파라미터다.",
        f"- Risk Tape: **{'READY' if risk_ready else 'BLOCKED'}** — {'기본 schema 검사를 통과했다.' if risk_ready else '파일과 out-of-fold/leakage provenance가 아직 없다.'}",
        "- 전체 Blox 실행: **Risk Tape와 fault scenario 확정 전에는 정책 비교를 시작하지 않는다.**",
        "",
        "## 공통 Replay 기간",
        "",
        f"- Fault 관측 시작: {xid_start.isoformat()}",
        f"- Fault 관측 종료: {xid_end.isoformat()}",
        "- Job Tape는 submit_time이 이 구간에 포함된 GPU job만 사용했다.",
        "- 초기 빈 cluster 편향과 종료부 fault 소실을 줄이기 위해 실제 Blox 실행에서는 warm-up·cool-down 구간을 별도로 둔다.",
        "",
        "## 1. Job Tape",
        "",
        f"- fault 기간 GPU job: {j['gpu_jobs_in_fault_window']:,}개",
        f"- main COMPLETED·양의 duration·8GPU/node topology 적합 job: {j['main_job_tape_rows']:,}개 ({j['main_inclusion_rate_among_window_gpu_jobs']:.1%})",
        f"- duration과 end-start 불일치 >1초: {j['duration_runtime_mismatch_over_1s']:,}개",
        f"- COMPLETED 중 duration 0: {j['zero_duration_completed_gpu_jobs']:,}개",
        f"- COMPLETED 중 gpu_num > node_num×8: {j['topology_infeasible_completed_gpu_jobs']:,}개",
        f"- duration 중앙값 {j['median_duration_sec']:,.0f}초, p95 {j['p95_duration_sec']:,.0f}초, p99 {j['p99_duration_sec']:,.0f}초",
        f"- multi-GPU job {j['multi_gpu_fraction']:.1%}, multi-node job {j['multi_node_fraction']:.1%}",
        f"- 일별 offered load 중앙값 {j['daily_offered_load_median']:.3f}, p95 {j['daily_offered_load_p95']:.3f}, 최대 {j['daily_offered_load_max']:.3f}",
        "",
        "해석: historical queue/start는 Blox에서 재계산하고, submit_time·duration·GPU/node 요구량만 재생한다. COMPLETED-only 결과는 서비스시간이 알려진 workload에 조건부인 결과이므로 제외 job과의 분포 차이를 sensitivity로 보고한다.",
        "",
        "## 2. Fault Tape",
        "",
        f"- GPU-level XID episode: {f['fault_tape_rows']:,}개",
        f"- same-node/time fault cluster: {f['fault_clusters']:,}개",
        f"- affected GPU {f['affected_gpus']:,}개, node {f['affected_nodes']:,}개",
        f"- multi-GPU cluster 비율: {f['multi_gpu_cluster_fraction']:.1%}; 최대 cluster 크기 {f['max_cluster_gpu_count']} GPU",
        f"- terminal-low timestamp 감사 episode: {f['timestamp_audited_episodes']:,}개; 3분 이상 onset 지원: {f['onset_supported_3m_episodes']:,}개",
        f"- shutdown-like로 분류된 GPU episode: {f['shutdown_like_episodes']:,}개; 미분류: {f['unclassified_episodes']:,}개",
        f"- XID episode 최대 지속시간: {f['max_episode_duration_days']:.1f}일 — downtime으로 사용하지 않는다.",
        "",
        "Fault time은 raw XID, terminal-low onset, mixed 세 시나리오를 유지한다. fault scope는 single-GPU와 same-node correlated 두 시나리오를 비교한다. severity와 repair time은 원본 관측값이 아니므로 Fault Tape에 확정값으로 쓰지 않는다.",
        "",
        "## 3. Risk Tape",
        "",
    ]
    for row in risk["audit"]:
        lines.append(f"- {row['check']}: **{row['status']}** — {row['detail']}")
    lines += [
        "",
        "Risk Tape는 decision_time 이전 정보만 사용한 시간순 out-of-fold score여야 한다. historical telemetry 기반 score를 replay에서 고정하면 결과는 '고정된 historical risk ranking이 주어진 조건'의 정책 효용으로 한정한다.",
        "",
        "## Blox 실행 전 남은 결정",
        "",
        "1. COMPLETED-only main workload와 excluded-job duration 대체 sensitivity의 두 Job Tape를 사용할지 확정",
        "2. raw/onset/mixed fault time 시나리오 확정",
        "3. single-GPU/node-correlated fault scope 확정",
        "4. interruption severity, repair time, checkpoint interval, restart overhead grid 확정",
        "5. Risk Tape 생성 후 schema·out-of-fold·leakage audit 통과",
        "6. no-fault/no-drain Blox baseline으로 arrival·utilization·JCT sanity check",
        "",
        "## 생성 파일",
        "",
        "- job_tape_main.csv: Blox 변환용 main Job Tape",
        "- fault_tape_candidates.csv: GPU episode 단위 Fault Tape 후보",
        "- fault_cluster_tape.csv: same-node/time correlated fault 후보",
        "- risk_tape_contract.csv, risk_tape_audit.csv: Risk Tape 입력 계약과 현재 readiness",
        "- 나머지 CSV·PNG: 선택편향, job size/duration, offered load, fault code/scope/recurrence 근거",
    ]
    (OUTPUT / "BLOX_INPUT_EDA_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def self_check(job_tape: pd.DataFrame, fault_tape: pd.DataFrame) -> None:
    assert not job_tape.empty and not fault_tape.empty
    assert job_tape["duration_sec"].gt(0).all()
    assert job_tape["gpu_num"].gt(0).all()
    assert job_tape["gpu_num"].le(job_tape["node_num"] * GPUS_PER_NODE).all()
    assert fault_tape["fault_episode_id"].is_unique
    assert fault_tape["xid_time_raw"].notna().all()
    onset = fault_tape["fault_time_onset"].notna()
    assert (pd.to_datetime(fault_tape.loc[onset, "fault_time_onset"], utc=True) <= pd.to_datetime(fault_tape.loc[onset, "xid_time_raw"], utc=True)).all()


def main() -> None:
    parser = argparse.ArgumentParser(description="Blox replay Job/Fault/Risk Tape pre-EDA")
    parser.add_argument("--risk-tape", type=Path, default=None)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(XID_INVENTORY.read_text(encoding="utf-8"))
    xid_start = pd.to_datetime(inventory["first_time"], utc=True)
    xid_end = pd.to_datetime(inventory["last_time"], utc=True)

    job_tape, job = job_eda(xid_start, xid_end)
    fault_tape, fault = fault_eda(xid_start, xid_end)
    risk = risk_audit(args.risk_tape, xid_start, xid_end)
    plots(job, fault)
    write_report(job, fault, risk, xid_start, xid_end)
    self_check(job_tape, fault_tape)

    metadata = {
        "replay_window_start": xid_start.isoformat(),
        "replay_window_end": xid_end.isoformat(),
        "gpu_capacity": GPU_CAPACITY,
        "gpus_per_node": GPUS_PER_NODE,
        "job": job["checks"],
        "fault": fault["checks"],
        "risk": risk,
    }
    (OUTPUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

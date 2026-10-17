"""
Generate publication-grade visualization figures for the Blox + RiskLab 17-run Optimization Benchmark.
Saves PNG charts to:
  1. C:/Users/jehan/Desktop/IISE 컨퍼런스/보고서/Blox연동/figures/
  2. Artifact directory for direct markdown embedding.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 180


def main():
    blox_dir = Path(__file__).resolve().parent
    project_root = blox_dir.parent.parent.parent
    report_fig_dir = project_root.parent / "보고서" / "Blox연동" / "figures"
    report_fig_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(r"C:\Users\jehan\.gemini\antigravity\brain\492e8e47-19a0-4e49-8667-1677687f0d46")

    csv_path = blox_dir / "runs_actual" / "benchmark_20261017" / "comparison_all.csv"
    df = pd.read_csv(csv_path).set_index("run_id")

    # -------------------------------------------------------------------------
    # Figure 1: Step-by-Step Ablation Waterfall (Track 1)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    fig.patch.set_facecolor("#ffffff")

    ablation_ids = [
        "01_baseline_blind",
        "03_junho_prefer_e19",
        "04_solA_falcon_prefer",
        "05_solA_vuln_weighted_dp",
        "06_solA_metaheuristic",
        "08_solA_full_co_opt",
    ]
    labels = [
        "01. Blind\n(위험도 미사용)",
        "03. 준호 E19\n+ 단일작업 DP",
        "04. 해법 A\n(Falcon+Pareto\n+ 단일작업 DP)",
        "05. 해법 A\n+ 취약도 가중\n다중작업 DP",
        "06. 해법 A\n+ 메타휴리스틱\n(2-Opt + SA)",
        "08. 최종 통합\n(메타휴리스틱\n+ 선점 가드)",
    ]
    colors = ["#64748b", "#94a3b8", "#3b82f6", "#10b981", "#059669", "#1d4ed8"]

    # Subplot 1: Job Failure Loss (s)
    vals_fail = [df.loc[r, "job_failure_lost_s"] for r in ablation_ids]
    bars1 = axes[0].bar(range(len(ablation_ids)), vals_fail, color=colors, width=0.62, edgecolor="#0f172a", linewidth=0.8)
    axes[0].set_title("(a) 작업 실패 손실 (JobFailLoss, 낮을수록 우수)", fontsize=11.5, fontweight="bold", pad=12)
    axes[0].set_ylabel("손실 시간 (Job Wall Seconds)", fontsize=10)
    axes[0].set_xticks(range(len(ablation_ids)))
    axes[0].set_xticklabels(labels, fontsize=8.5)
    axes[0].set_ylim(0, 1550)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for idx, b in enumerate(bars1):
        h = b.get_height()
        pct = (h - vals_fail[0]) / vals_fail[0] * 100
        txt = f"{int(h)}s\n({pct:+.1f}%)" if idx > 0 else f"{int(h)}s"
        if h == 0:
            txt = "0s (-100%)\n완전 제거!"
        axes[0].text(b.get_x() + b.get_width() / 2, h + 35, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Subplot 2: Vulnerability-Weighted Risk Exposure (Billions)
    vals_vuln = [df.loc[r, "vuln_weighted_risk_exposure"] / 1e9 for r in ablation_ids]
    bars2 = axes[1].bar(range(len(ablation_ids)), vals_vuln, color=colors, width=0.62, edgecolor="#0f172a", linewidth=0.8)
    axes[1].set_title("(b) 취약도 가중 위험 노출 (VulnRiskExp, 낮을수록 우수)", fontsize=11.5, fontweight="bold", pad=12)
    axes[1].set_ylabel("취약도 × 위험도 노출 합계 (Billion, 10^9)", fontsize=10)
    axes[1].set_xticks(range(len(ablation_ids)))
    axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylim(0, 118)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    for idx, b in enumerate(bars2):
        h = b.get_height()
        pct = (h - vals_vuln[0]) / vals_vuln[0] * 100
        txt = f"{h:.1f}B\n({pct:+.1f}%)" if idx > 0 else f"{h:.1f}B"
        axes[1].text(b.get_x() + b.get_width() / 2, h + 2.2, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Subplot 3: Preemption Overhead (Hours) & Avg JCT reduction
    vals_preempt_h = [df.loc[r, "scheduling_preemption_overhead_s"] / 3600.0 for r in ablation_ids]
    bars3 = axes[2].bar(range(len(ablation_ids)), vals_preempt_h, color=colors, width=0.62, edgecolor="#0f172a", linewidth=0.8)
    axes[2].set_title("(c) 스케줄링 선점 오버헤드 (PreemptCost, 낮을수록 우수)", fontsize=11.5, fontweight="bold", pad=12)
    axes[2].set_ylabel("누적 선점 낭비 (GPU-Hours)", fontsize=10)
    axes[2].set_xticks(range(len(ablation_ids)))
    axes[2].set_xticklabels(labels, fontsize=8.5)
    axes[2].set_ylim(0, 940)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    for idx, b in enumerate(bars3):
        h = b.get_height()
        pct = (h - vals_preempt_h[0]) / vals_preempt_h[0] * 100
        txt = f"{h:.0f}h\n({pct:+.1f}%)" if abs(pct) >= 1.0 else f"{h:.0f}h"
        axes[2].text(b.get_x() + b.get_width() / 2, h + 18, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle("Blox + RiskLab 단계별 최적화(Ablation) 핵심 성과 비교 (100-GPU, 16,560 Jobs)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    f1_path = report_fig_dir / "fig1_ablation_waterfall.png"
    fig.savefig(f1_path, bbox_inches="tight")
    shutil.copy2(f1_path, artifact_dir / "fig1_ablation_waterfall.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: 2D Pareto Frontier (Avg JCT vs Total Interruption & Risk Exposure)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    fig.patch.set_facecolor("#ffffff")

    pts = [
        ("01_baseline_blind", "01. Blind (베이스라인)", "#64748b", "o", 110),
        ("03_junho_prefer_e19", "03. 준호 E19 Greedy DP", "#ef4444", "s", 120),
        ("04_solA_falcon_prefer", "04. 해법A: Falcon Greedy DP", "#f59e0b", "^", 130),
        ("05_solA_vuln_weighted_dp", "05. 해법A: 취약도 Exact DP", "#10b981", "D", 130),
        ("06_solA_metaheuristic", "06. 해법A: 메타휴리스틱 (2-Opt+SA)", "#059669", "P", 160),
        ("08_solA_full_co_opt", "08. 최종 통합 (메타휴리스틱+선점가드)", "#1d4ed8", "*", 240),
    ]

    for rid, lbl, col, mk, sz in pts:
        jct = df.loc[rid, "avg_jct_s"]
        floss = df.loc[rid, "job_failure_lost_s"]
        axes[0].scatter(jct, floss, color=col, marker=mk, s=sz, label=lbl, edgecolor="#0f172a", linewidth=0.9, zorder=5)
        offset_y = 45 if rid != "05_solA_vuln_weighted_dp" else 95
        axes[0].annotate(lbl.split(":")[0], (jct, floss), xytext=(0, offset_y / 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    axes[0].set_title("(a) 평균 작업 완료시간(AvgJCT) vs 작업 실패 손실(JobFailLoss)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("평균 작업 완료시간 AvgJCT (초, 좌측일수록 빠름)", fontsize=10)
    axes[0].set_ylabel("작업 실패 손실 JobFailLoss (초, 하단일수록 안전)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].set_xlim(50350, 51600)
    axes[0].set_ylim(-120, 1480)

    for rid, lbl, col, mk, sz in pts:
        p95w = df.loc[rid, "p95_queue_wait_s"] / 3600.0
        vuln = df.loc[rid, "vuln_weighted_risk_exposure"] / 1e9
        axes[1].scatter(p95w, vuln, color=col, marker=mk, s=sz, label=lbl, edgecolor="#0f172a", linewidth=0.9, zorder=5)
        axes[1].annotate(lbl.split(":")[0], (p95w, vuln), xytext=(6, 4), textcoords="offset points", fontsize=8.5, fontweight="bold")

    axes[1].set_title("(b) P95 꼬리 대기시간(시간) vs 취약도 가중 위험 노출(Billion)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("P95 큐 대기시간 (시간, 좌측일수록 빠름)", fontsize=10)
    axes[1].set_ylabel("취약도 가중 위험 노출 (Billion, 하단일수록 안전)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].set_xlim(49.5, 59.0)
    axes[1].set_ylim(15, 110)
    axes[1].legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.suptitle("다목적 파레토 프론티어(Pareto Frontier): 속도(JCT/Wait)와 안전성(Failure/Risk) 동시 개선", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    f2_path = report_fig_dir / "fig2_pareto_frontier.png"
    fig.savefig(f2_path, bbox_inches="tight")
    shutil.copy2(f2_path, artifact_dir / "fig2_pareto_frontier.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Cross-Track Comparison (Track 1 Uniform vs Track 2 Sol B vs Track 3 Mixed)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fig.patch.set_facecolor("#ffffff")

    tracks = ["Track 1: 동일 하드웨어 격리\n(Uniform Hardware)", "Track 2: 해법 B\n(XID 43 소프트 리셋)", "Track 3: 혼합 시나리오\n(XID 31 8건 + XID 43 23건)"]
    x = np.arange(len(tracks))
    w = 0.20

    blind_loss = [df.loc["01_baseline_blind", "job_failure_lost_s"], df.loc["09_solB_blind_diff10", "job_failure_lost_s"], df.loc["14_mixed31_43_blind", "job_failure_lost_s"]]
    greedy_loss = [df.loc["03_junho_prefer_e19", "job_failure_lost_s"], df.loc["10_solB_junho_prefer_diff10", "job_failure_lost_s"], df.loc["15_mixed31_43_junho_prefer", "job_failure_lost_s"]]
    meta_loss = [df.loc["06_solA_metaheuristic", "job_failure_lost_s"], df.loc["11_solAB_metaheuristic_diff10", "job_failure_lost_s"], df.loc["16_mixed31_43_metaheuristic", "job_failure_lost_s"]]
    coopt_loss = [df.loc["08_solA_full_co_opt", "job_failure_lost_s"], df.loc["12_solAB_full_co_opt_diff10", "job_failure_lost_s"], df.loc["17_mixed31_43_full_co_opt", "job_failure_lost_s"]]

    b1 = axes[0].bar(x - 1.5 * w, blind_loss, w, label="위험도 미사용 (Blind)", color="#64748b")
    b2 = axes[0].bar(x - 0.5 * w, greedy_loss, w, label="단일작업 Greedy DP", color="#f59e0b")
    b3 = axes[0].bar(x + 0.5 * w, meta_loss, w, label="제안 1: 메타휴리스틱 (2-Opt+SA)", color="#059669")
    b4 = axes[0].bar(x + 1.5 * w, coopt_loss, w, label="제안 2: 최종 통합 (메타+선점가드)", color="#1d4ed8")

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 25, f"{int(h)}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[0].set_title("(a) 시나리오 트랙별 작업 실패 손실 (JobFailLoss)", fontsize=11.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tracks, fontsize=9.5)
    axes[0].set_ylabel("작업 실패 손실 (초)", fontsize=10)
    axes[0].set_ylim(0, 1850)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5)

    # Subplot 2: JCT comparison across tracks
    blind_jct = [df.loc["01_baseline_blind", "avg_jct_s"], df.loc["09_solB_blind_diff10", "avg_jct_s"], df.loc["14_mixed31_43_blind", "avg_jct_s"]]
    greedy_jct = [df.loc["03_junho_prefer_e19", "avg_jct_s"], df.loc["10_solB_junho_prefer_diff10", "avg_jct_s"], df.loc["15_mixed31_43_junho_prefer", "avg_jct_s"]]
    meta_jct = [df.loc["06_solA_metaheuristic", "avg_jct_s"], df.loc["11_solAB_metaheuristic_diff10", "avg_jct_s"], df.loc["16_mixed31_43_metaheuristic", "avg_jct_s"]]
    coopt_jct = [df.loc["08_solA_full_co_opt", "avg_jct_s"], df.loc["12_solAB_full_co_opt_diff10", "avg_jct_s"], df.loc["17_mixed31_43_full_co_opt", "avg_jct_s"]]

    axes[1].bar(x - 1.5 * w, blind_jct, w, label="위험도 미사용 (Blind)", color="#64748b")
    axes[1].bar(x - 0.5 * w, greedy_jct, w, label="단일작업 Greedy DP", color="#f59e0b")
    axes[1].bar(x + 0.5 * w, meta_jct, w, label="제안 1: 메타휴리스틱 (2-Opt+SA)", color="#059669")
    bars_c = axes[1].bar(x + 1.5 * w, coopt_jct, w, label="제안 2: 최종 통합 (메타+선점가드)", color="#1d4ed8")

    for idx_t, bar in enumerate(bars_c):
        h = bar.get_height()
        diff = h - blind_jct[idx_t]
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 40, f"{diff:.0f}s\n단축!", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1d4ed8")

    axes[1].set_title("(b) 시나리오 트랙별 평균 작업 완료시간 (AvgJCT)", fontsize=11.5, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tracks, fontsize=9.5)
    axes[1].set_ylabel("평균 작업 완료시간 (초)", fontsize=10)
    axes[1].set_ylim(49800, 52300)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("해법 A (ML+메타휴리스틱) & 해법 B (XID 31/43 이원화) 전 트랙 일관성 검증", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    f3_path = report_fig_dir / "fig3_track123_comparison.png"
    fig.savefig(f3_path, bbox_inches="tight")
    shutil.copy2(f3_path, artifact_dir / "fig3_track123_comparison.png")
    plt.close(fig)

    print(f"[Visualizations Generated] Saved to {report_fig_dir} and {artifact_dir}")


if __name__ == "__main__":
    main()

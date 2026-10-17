"""
Comprehensive Benchmark Runner for Blox + RiskLab Optimization Simulator (2026-10-17)
=====================================================================================
Evaluates:
1. Solution A: Falcon-45 + Optimal Capped Pareto Dual-Branch (b1_risk, b2_risk, fused_risk)
   vs. Junho's 09-22 Model Tape.
2. Solution B: Differentiated XID 43 vs XID 31 Impact Semantics (uniform_hardware vs differentiated_43_31
   with rho_43 = 1.0 and 0.5, plus mixed XID 31 + 43 scenario).
3. Optimization Solvers:
   - risk_blind_packed (Baseline 1: Risk-Blind Packed LAS)
   - risk_mask_packed (Baseline 2: 99% Recall Hard Mask)
   - risk_prefer_packed (Baseline 3: Junho 09-22 Single-Job Greedy DP)
   - vuln_weighted_packed (Proposed 1: Vulnerability-Weighted Batch Exact DP)
   - metaheuristic_packed (Proposed 2: Warm-Start 2-Opt + Simulated Annealing Batch Co-Optimizer)
   - topk_quarantine_packed (Proposed 3: Budgeted Top-K Quarantine + Vulnerability Batch DP)
   - full_co_optimized (Proposed 4: Metaheuristic Batch Co-Optimizer + Cost-Benefit Preemption Guard)
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BLOX_RISKLAB_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = BLOX_RISKLAB_DIR.parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent.parent

sys.path.insert(0, str(BLOX_RISKLAB_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "ML"))

from risklab.feeds import sha256
from risklab.replay_runner import atomic_json, execute


def build_falcon_pareto_inputs() -> Path:
    """
    Build inputs/falcon_pareto_30m_20261017 using the completed Falcon-45 + Capped Pareto (Cap 0.60, alpha=1.0)
    ML model on the exact 611 timestamps and 100-GPU cohort of reportfaithful_30m_20260922.
    """
    base_in = BLOX_RISKLAB_DIR / "inputs" / "reportfaithful_30m_20260922"
    out_in = BLOX_RISKLAB_DIR / "inputs" / "falcon_pareto_30m_20261017"
    out_in.mkdir(parents=True, exist_ok=True)

    manifest_path = out_in / "run_manifest.json"
    prob_path = out_in / "probability_tape.parquet"

    if manifest_path.exists() and prob_path.exists():
        print(f"[Input Ready] {out_in} already exists.")
        return manifest_path

    print("[Step 1] Generating Falcon-45 + Capped Pareto 30m probability tape for the 100-GPU cohort...")
    import run_unified_adst_fusion as runner

    data_dir = runner.base.find_data_dir()
    cache_dir = PROJECT_ROOT / "outputs" / "branch1" / "cache"
    engine = runner.base.UnifiedDataEngine(data_dir=data_dir, cache_dir=cache_dir)
    _, history_map, gt_matrix = engine.load_all_xid_ledger()

    pipeline = runner.UnifiedADSTPipeline(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=PROJECT_ROOT / "outputs" / "unified_adst",
    )

    metrics_df = pd.read_csv(PROJECT_ROOT / "outputs" / "unified_adst" / "unified_adst_metrics.csv")
    ref_prob = pd.read_parquet(base_in / "probability_tape.parquet")
    ref_times = np.sort(ref_prob["timestamp"].unique())
    ref_top = pd.read_parquet(base_in / "gpu_manifest.parquet").sort_values("cohort_index")
    cohort_uids = list(ref_top["gpu_uid"])
    uid_to_idx = {str(u): idx for idx, u in enumerate(engine.gpu_ids)}
    cohort_indices = np.array([uid_to_idx[u] for u in cohort_uids], dtype=np.int32)

    # Map ref_times to engine bin indices
    bin_ts = pd.to_datetime(engine.bin_start_ns, utc=True)
    ts_to_bin = {ts: b for b, ts in enumerate(bin_ts)}
    target_bins = np.array([ts_to_bin[pd.Timestamp(t)] for t in ref_times], dtype=np.int32)

    cadence_bins = int(24 * 60 / runner.base.STEP_MINUTES)
    warmup_bins = int(30 * 24 * 60 / runner.base.STEP_MINUTES)
    rolling_origins = np.arange(warmup_bins, engine.num_bins - cadence_bins, cadence_bins)

    rng = np.random.default_rng(42)
    rows = []

    # Group target_bins by their corresponding rolling origin
    for b in target_bins:
        # Find latest origin <= b
        valid_origins = np.where(rolling_origins <= b)[0]
        origin_idx = int(valid_origins[-1]) if len(valid_origins) > 0 else 0
        origin_idx = min(origin_idx, len(metrics_df) - 1)
        origin_bin = int(rolling_origins[origin_idx])

    # Fit per needed origin_idx
    bin_to_origin_idx = {}
    for b in target_bins:
        valid_origins = np.where(rolling_origins <= b)[0]
        oi = int(valid_origins[-1]) if len(valid_origins) > 0 else 0
        oi = min(oi, len(metrics_df) - 1)
        bin_to_origin_idx[int(b)] = oi

    unique_ois = sorted(set(bin_to_origin_idx.values()))
    print(f"  -> Fitting Falcon-45 + Capped Pareto across {len(unique_ois)} rolling origins covering 611 decision steps...")

    for oi in unique_ois:
        origin_bin = int(rolling_origins[oi])
        best_l_train = int(metrics_df["selected_L_train_days"].iloc[oi])
        best_l_obs = int(metrics_df["selected_L_obs_hours"].iloc[oi])

        final_tr_start = max(0, origin_bin - int(best_l_train * 24 * 60 / runner.base.STEP_MINUTES))
        final_tr_bins, final_tr_gpus, final_tr_labels = pipeline.sample_indices(final_tr_start, origin_bin, rng)

        b1_tr = pipeline.extract_enhanced_branch1_features(final_tr_bins, final_tr_gpus, best_l_obs)
        b2_tr = engine.extract_branch2_features(final_tr_bins, final_tr_gpus, history_map)

        b1_model = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.06, max_iter=150, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.5, class_weight="balanced", random_state=42,
        )
        b1_model.fit(b1_tr, final_tr_labels)

        b2_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=250, class_weight="balanced", random_state=42))
        b2_lr.fit(b2_tr, final_tr_labels)
        b2_gbdt = HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=31, l2_regularization=1.0, class_weight="balanced", random_state=42,
        )
        b2_gbdt.fit(b2_tr, final_tr_labels)

        true_tr_pos = int(gt_matrix[final_tr_start:origin_bin].sum())
        true_tr_tot = (origin_bin - final_tr_start) * engine.num_gpus
        true_prior = true_tr_pos / max(true_tr_tot, 1)
        sample_prior = int(final_tr_labels.sum()) / max(len(final_tr_labels), 1)

        origin_ns = int(engine.bin_start_ns[origin_bin])
        cutoff_ns = origin_ns - runner.base.BUFFER_MASK_BINS * runner.base.STEP_NS
        start_ns = cutoff_ns - 30 * runner.base.DAY_NS
        xid_counts = np.zeros(len(cohort_indices), dtype=np.float64)
        for idx_c, gpu in enumerate(cohort_indices):
            events = history_map.get(int(gpu))
            if events is not None and len(events) > 0:
                right = int(np.searchsorted(events, cutoff_ns, side="left"))
                left = int(np.searchsorted(events, start_ns, side="left"))
                xid_counts[idx_c] = max(0, right - left)

        pareto_lam = np.minimum(0.60, (1.0 / (xid_counts + 1.0)) ** 1.0)

        bins_for_oi = [b for b, mapped_oi in bin_to_origin_idx.items() if mapped_oi == oi]
        for b in sorted(bins_for_oi):
            bs = np.full(len(cohort_indices), b, dtype=np.int32)
            b1_te = pipeline.extract_enhanced_branch1_features(bs, cohort_indices, best_l_obs)
            b2_te = engine.extract_branch2_features(bs, cohort_indices, history_map)

            p1 = np.clip(runner.base.adjusted_probability(b1_model.predict_proba(b1_te)[:, 1], true_prior, sample_prior), 0.0, 1.0)
            p2_raw = 0.5 * b2_lr.predict_proba(b2_te)[:, 1] + 0.5 * b2_gbdt.predict_proba(b2_te)[:, 1]
            p2 = np.clip(runner.base.adjusted_probability(p2_raw, true_prior, sample_prior), 0.0, 1.0)
            pf = np.clip(pareto_lam * p1 + (1.0 - pareto_lam) * p2, 0.0, 1.0)

            ts_val = pd.Timestamp(engine.bin_start_ns[b], tz="UTC")
            rows.append(pd.DataFrame({
                "timestamp": [ts_val] * len(cohort_uids),
                "gpu_uid": cohort_uids,
                "probability": pf.astype(np.float32),
                "b1_probability": p1.astype(np.float32),
                "b2_probability": p2.astype(np.float32),
                "model_version": "falcon45_capped_pareto_v1",
            }))

    tape_df = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "gpu_uid"]).reset_index(drop=True)
    # Match exact ordering of reference tape
    tape_df.to_parquet(prob_path, index=False)

    # Copy shared topology, job_trace, failure_tape, threshold_manifest
    for fname in ["gpu_manifest.parquet", "job_trace.parquet", "failure_tape.parquet", "threshold_manifest.json"]:
        shutil.copy2(base_in / fname, out_in / fname)

    base_manifest = json.loads((base_in / "run_manifest.json").read_text(encoding="utf-8"))
    for key, fname in [
        ("probability_tape", "probability_tape.parquet"),
        ("failure_tape", "failure_tape.parquet"),
        ("gpu_manifest", "gpu_manifest.parquet"),
        ("job_trace", "job_trace.parquet"),
        ("threshold_manifest", "threshold_manifest.json"),
    ]:
        base_manifest["inputs"][key] = {"path": fname, "sha256": sha256(out_in / fname)}
    base_manifest["placement_policy"] = "full_co_optimized"
    atomic_json(manifest_path, base_manifest)
    print(f"[Step 1 Complete] Saved Falcon-45 + Pareto inputs to {out_in}")
    return manifest_path


def build_mixed_xid31_43_inputs(falcon_manifest_path: Path) -> Path:
    """
    Build a mixed XID-31 + XID-43 scenario directory where historical episodes on the top chronic
    hardware nodes carry XID 31 (hardware memory fault) while transient app crashes carry XID 43,
    enabling direct evaluation of Solution B when both XID 31 and XID 43 coexist.
    """
    src_dir = falcon_manifest_path.parent
    out_dir = BLOX_RISKLAB_DIR / "inputs" / "falcon_pareto_mixed_xid31_43_20261017"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        return manifest_path

    for fname in ["probability_tape.parquet", "gpu_manifest.parquet", "job_trace.parquet", "threshold_manifest.json"]:
        shutil.copy2(src_dir / fname, out_dir / fname)

    ft = pd.read_parquet(src_dir / "failure_tape.parquet")
    # In the 31 episodes, assign XID 31 to long persistent multi-observation episodes (duration > 600s:
    # the 8 GPU episodes on node 10.140.0.134 lasting ~55 minutes) and keep the 23 short episodes as XID 43
    new_codes = []
    for _, row in ft.iterrows():
        dur = (pd.Timestamp(row["episode_end"]) - pd.Timestamp(row["episode_start"])).total_seconds()
        if dur > 600:
            new_codes.append("[31]")
        else:
            new_codes.append("[43]")
    ft["xid_codes"] = new_codes
    ft.to_parquet(out_dir / "failure_tape.parquet", index=False)

    base_manifest = json.loads(falcon_manifest_path.read_text(encoding="utf-8"))
    for key, fname in [
        ("probability_tape", "probability_tape.parquet"),
        ("failure_tape", "failure_tape.parquet"),
        ("gpu_manifest", "gpu_manifest.parquet"),
        ("job_trace", "job_trace.parquet"),
        ("threshold_manifest", "threshold_manifest.json"),
    ]:
        base_manifest["inputs"][key] = {"path": fname, "sha256": sha256(out_dir / fname)}
    atomic_json(manifest_path, base_manifest)
    return manifest_path


def main():
    junho_manifest = BLOX_RISKLAB_DIR / "inputs" / "reportfaithful_30m_20260922" / "run_manifest.json"
    falcon_manifest = build_falcon_pareto_inputs()
    mixed_manifest = build_mixed_xid31_43_inputs(falcon_manifest)

    bench_root = BLOX_RISKLAB_DIR / "runs_actual" / "benchmark_20261017"
    if bench_root.exists():
        shutil.rmtree(bench_root)
    bench_root.mkdir(parents=True, exist_ok=True)

    experiments = [
        # --- Track 1: Uniform Hardware Mode (Direct Apples-to-Apples with Junho 09-22) ---
        ("01_baseline_blind", junho_manifest, "risk_blind_packed", "Las", "uniform_hardware", 1.0, "Baseline 1: 위험도 미사용 (Blox LAS)"),
        ("02_junho_mask99", junho_manifest, "risk_mask_packed", "Las", "uniform_hardware", 1.0, "Baseline 2: 준호 99% 임계값 마스킹 (Starvation)"),
        ("03_junho_prefer_e19", junho_manifest, "risk_prefer_packed", "Las", "uniform_hardware", 1.0, "Baseline 3: 준호 E19 + 단일작업 Greedy DP"),
        ("04_solA_falcon_prefer", falcon_manifest, "risk_prefer_packed", "Las", "uniform_hardware", 1.0, "해법 A (ML 교체): Falcon-45+Pareto + 단일작업 DP"),
        ("05_solA_vuln_weighted_dp", falcon_manifest, "vuln_weighted_packed", "Las", "uniform_hardware", 1.0, "해법 A + 수리모형 1: 취약도 가중 다중작업 Exact DP"),
        ("06_solA_metaheuristic", falcon_manifest, "metaheuristic_packed", "Las", "uniform_hardware", 1.0, "해법 A + 수리모형 2: Warm-Start 메타휴리스틱 (2-Opt+SA)"),
        ("07_solA_topk_quarantine", falcon_manifest, "topk_quarantine_packed", "Las", "uniform_hardware", 1.0, "해법 A + 수리모형 3: 예산 제약 Top-K 격리 + 취약도 배치"),
        ("08_solA_full_co_opt", falcon_manifest, "full_co_optimized", "Las", "uniform_hardware", 1.0, "해법 A + 최종 통합: 메타휴리스틱 배치 + 선점 가드"),

        # --- Track 2: Solution B Differentiated XID 43 Soft-Reset Mode (Actual Cohort: 31 XID-43 episodes) ---
        ("09_solB_blind_diff10", falcon_manifest, "risk_blind_packed", "Las", "differentiated_43_31", 1.0, "해법 B (XID43 소프트리셋, ρ=1.0): 위험도 미사용"),
        ("10_solB_junho_prefer_diff10", junho_manifest, "risk_prefer_packed", "Las", "differentiated_43_31", 1.0, "해법 B (XID43 소프트리셋, ρ=1.0): 준호 Greedy DP"),
        ("11_solAB_metaheuristic_diff10", falcon_manifest, "metaheuristic_packed", "Las", "differentiated_43_31", 1.0, "해법 A+B (ρ=1.0): Falcon+Pareto + 메타휴리스틱 배치"),
        ("12_solAB_full_co_opt_diff10", falcon_manifest, "full_co_optimized", "Las", "differentiated_43_31", 1.0, "해법 A+B (ρ=1.0): 메타휴리스틱 배치 + 선점 가드"),
        ("13_solAB_full_co_opt_diff05", falcon_manifest, "full_co_optimized", "Las", "differentiated_43_31", 0.5, "해법 A+B (ρ=0.5 민감도): 메타휴리스틱 배치 + 선점 가드"),

        # --- Track 3: Mixed XID-31 (Hardware) + XID-43 (Soft) Scenario ---
        ("14_mixed31_43_blind", mixed_manifest, "risk_blind_packed", "Las", "differentiated_43_31", 1.0, "혼합(XID 31+43) 시나리오: 위험도 미사용"),
        ("15_mixed31_43_junho_prefer", mixed_manifest, "risk_prefer_packed", "Las", "differentiated_43_31", 1.0, "혼합(XID 31+43) 시나리오: 준호 Greedy DP"),
        ("16_mixed31_43_metaheuristic", mixed_manifest, "metaheuristic_packed", "Las", "differentiated_43_31", 1.0, "혼합(XID 31+43) 시나리오: Falcon+Pareto + 메타휴리스틱"),
        ("17_mixed31_43_full_co_opt", mixed_manifest, "full_co_optimized", "Las", "differentiated_43_31", 1.0, "혼합(XID 31+43) 시나리오: 메타휴리스틱 + 선점 가드"),
    ]

    results = []
    for run_id, m_path, pol, sched, xid_mode, rho43, desc in experiments:
        t0 = time.perf_counter()
        out_dir = bench_root / run_id
        summary = execute(
            m_path,
            out_dir,
            policy=pol,
            scheduler_name=sched,
            xid_severity_mode=xid_mode,
            xid43_interrupt_prob=rho43,
        )
        elapsed = time.perf_counter() - t0
        row = {
            "run_id": run_id,
            "description": desc,
            "ml_tape": "Falcon45_Pareto" if "falcon" in str(m_path) else "Junho_E19",
            "policy": pol,
            "xid_mode": xid_mode,
            "rho43": rho43,
            "completed_jobs": summary["completed_jobs"],
            "unfinished_jobs": summary["unfinished_jobs"],
            "failure_checkpoint_loss_s": summary["failure_checkpoint_loss_seconds"],
            "failure_recovery_delay_s": summary["failure_recovery_delay_seconds"],
            "job_failure_lost_s": summary["job_failure_lost_seconds"],
            "failure_gpu_unavailable_s": summary["failure_gpu_unavailable_seconds"],
            "soft_xid_reset_s": summary["soft_xid_reset_seconds"],
            "scheduling_checkpoint_s": summary["scheduling_checkpoint_seconds"],
            "scheduling_relaunch_s": summary["scheduling_relaunch_seconds"],
            "scheduling_preemption_overhead_s": summary["scheduling_preemption_overhead_seconds"],
            "total_job_interruption_waste_s": summary["total_job_interruption_waste_seconds"],
            "risk_exposure_gpu_s": round(summary["risk_exposure_gpu_seconds"], 2),
            "vuln_weighted_risk_exposure": round(summary["vuln_weighted_risk_exposure"], 2),
            "avg_queue_wait_s": round(summary["avg_queue_wait_seconds"], 2),
            "p95_queue_wait_s": round(summary["p95_queue_wait_seconds"], 2),
            "avg_jct_s": round(summary["avg_jct_seconds"], 2) if summary["avg_jct_seconds"] is not None else None,
            "avoided_xid_on_idle": summary["avoided_xid_episodes_on_idle"],
            "active_xid_interruptions": summary["active_xid_job_interruptions"],
            "elapsed_sec": round(elapsed, 2),
        }
        results.append(row)
        print(
            f"[{run_id}] ({elapsed:.1f}s) | Completed={row['completed_jobs']:,} | "
            f"JobFailLoss={row['job_failure_lost_s']:.0f}s | PreemptCost={row['scheduling_preemption_overhead_s']:,.0f}s | "
            f"RiskExp={row['risk_exposure_gpu_s']:,.1f} | AvgWait={row['avg_queue_wait_s']:,.1f}s | AvgJCT={row['avg_jct_s']}",
            flush=True,
        )

    df = pd.DataFrame(results)
    df.to_csv(bench_root / "comparison_all.csv", index=False, encoding="utf-8-sig")
    atomic_json(bench_root / "comparison_summary.json", results)
    print(f"\n[All 17 Benchmark Runs Complete] Saved to {bench_root / 'comparison_all.csv'}")


if __name__ == "__main__":
    main()

"""
Computes exact Top-100 Hits, Recall@100, and Lift@100 for:
1. Branch 1 alone (45-feat GBDT)
2. Branch 2 alone (History)
3. Dual-Branch Fused (Optimal Cap 0.60 + alpha=1.0)
Across all 63 origins on the full 1,992-GPU cluster.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ML"))

import run_unified_adst_fusion as runner

data_dir = runner.base.find_data_dir()
cache_dir = Path("outputs/branch1/cache") if Path("outputs/branch1/cache").exists() else Path("../outputs/branch1/cache")
engine = runner.base.UnifiedDataEngine(data_dir=data_dir, cache_dir=cache_dir)
_, history_map, gt_matrix = engine.load_all_xid_ledger()

pipeline = runner.UnifiedADSTPipeline(
    engine=engine,
    history_map=history_map,
    gt_matrix=gt_matrix,
    output_dir=Path("outputs/unified_adst"),
)

metrics_path = Path("outputs/unified_adst/unified_adst_metrics.csv")
metrics_df = pd.read_csv(metrics_path)

cadence_bins = int(24 * 60 / runner.base.STEP_MINUTES)
warmup_bins = int(30 * 24 * 60 / runner.base.STEP_MINUTES)
rolling_origins = np.arange(warmup_bins, engine.num_bins - cadence_bins, cadence_bins)

n_origins = min(len(rolling_origins), len(metrics_df))
print(f"[Starting Evaluation] Total {n_origins} rolling origins on 1,992 GPUs...")

rng = np.random.default_rng(42)

records = []
total_positives = 0
total_hits_b1 = 0
total_hits_b2 = 0
total_hits_fused = 0

t0_all = time.time()

for origin_idx in range(n_origins):
    t_start = time.time()
    origin_bin = rolling_origins[origin_idx]
    best_l_train = int(metrics_df["selected_L_train_days"].iloc[origin_idx])
    best_l_obs = int(metrics_df["selected_L_obs_hours"].iloc[origin_idx])
    test_cycle_end = origin_bin + cadence_bins

    # Train windows
    final_tr_start = max(0, origin_bin - int(best_l_train * 24 * 60 / runner.base.STEP_MINUTES))
    final_tr_bins, final_tr_gpus, final_tr_labels = pipeline.sample_indices(final_tr_start, origin_bin, rng)

    # Extract train features
    b1_tr = pipeline.extract_enhanced_branch1_features(final_tr_bins, final_tr_gpus, best_l_obs)
    b2_tr = engine.extract_branch2_features(final_tr_bins, final_tr_gpus, history_map)

    # Fit B1
    b1_model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.06, max_iter=150, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.5, class_weight="balanced", random_state=42,
    )
    b1_model.fit(b1_tr, final_tr_labels)

    # Fit B2
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

    # Test features in batch
    test_bins_range = np.arange(origin_bin, test_cycle_end, 6, dtype=np.int32)
    cur_bins = np.repeat(test_bins_range, engine.num_gpus)
    all_g = np.tile(np.arange(engine.num_gpus, dtype=np.int32), len(test_bins_range))

    b1_te = pipeline.extract_enhanced_branch1_features(cur_bins, all_g, best_l_obs)
    b2_te = engine.extract_branch2_features(cur_bins, all_g, history_map)

    p1 = runner.base.adjusted_probability(b1_model.predict_proba(b1_te)[:, 1], true_prior, sample_prior).reshape(48, engine.num_gpus)
    p2 = runner.base.adjusted_probability(0.5 * b2_lr.predict_proba(b2_te)[:, 1] + 0.5 * b2_gbdt.predict_proba(b2_te)[:, 1], true_prior, sample_prior).reshape(48, engine.num_gpus)

    # Pareto lam
    origin_ns = int(engine.bin_start_ns[origin_bin])
    cutoff_ns = origin_ns - runner.base.BUFFER_MASK_BINS * runner.base.STEP_NS
    start_ns = cutoff_ns - 30 * runner.base.DAY_NS
    xid_counts = np.zeros(engine.num_gpus, dtype=np.float64)
    for gpu in range(engine.num_gpus):
        events = history_map.get(gpu)
        if events is not None and len(events) > 0:
            right = int(np.searchsorted(events, cutoff_ns, side="left"))
            left = int(np.searchsorted(events, start_ns, side="left"))
            xid_counts[gpu] = max(0, right - left)

    pareto_lam = np.minimum(0.60, (1.0 / (xid_counts + 1.0)) ** 1.0)
    pf = pareto_lam[None, :] * p1 + (1.0 - pareto_lam[None, :]) * p2

    gt_test = gt_matrix[test_bins_range]
    positives = int(gt_test.sum())

    def get_hits_and_pr(scores: np.ndarray) -> tuple[int, float, float]:
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(1, engine.num_gpus + 1)[None, :], axis=1)
        hits = int((gt_test & (ranks <= 100)).sum())
        rec = hits / max(positives, 1)
        ap = float(average_precision_score(gt_test.ravel(), scores.ravel())) if gt_test.any() else 0.0
        return hits, rec, ap

    h1, r1, ap1 = get_hits_and_pr(p1)
    h2, r2, ap2 = get_hits_and_pr(p2)
    hf, rf, apf = get_hits_and_pr(pf)

    total_positives += positives
    total_hits_b1 += h1
    total_hits_b2 += h2
    total_hits_fused += hf

    records.append({
        "origin_idx": origin_idx,
        "positives": positives,
        "hits_b1": h1,
        "recall_b1": r1,
        "pr_auc_b1": ap1,
        "hits_b2": h2,
        "recall_b2": r2,
        "pr_auc_b2": ap2,
        "hits_fused": hf,
        "recall_fused": rf,
        "pr_auc_fused": apf,
    })

    if (origin_idx + 1) % 5 == 0 or origin_idx == n_origins - 1:
        print(f"[{origin_idx+1:2d}/{n_origins}] Elapsed: {time.time()-t0_all:.1f}s | "
              f"Cumulative Hits: B1={total_hits_b1:,} ({total_hits_b1/total_positives:.1%}), "
              f"B2={total_hits_b2:,} ({total_hits_b2/total_positives:.1%}), "
              f"Fused={total_hits_fused:,} ({total_hits_fused/total_positives:.1%})", flush=True)

eval_df = pd.DataFrame(records)
out_csv = PROJECT_ROOT / "outputs" / "unified_adst" / "b1_b2_fused_exact_comparison.csv"
eval_df.to_csv(out_csv, index=False)
print(f"\n[Done] Saved full evaluation to {out_csv}")

print("\n" + "=" * 90)
print("=== FINAL APPLES-TO-APPLES BENCHMARK (EXACT 63-ORIGIN 1,992-GPU CLUSTER EVALUATION) ===")
print("=" * 90)
print(f"Total Cluster Positives: {total_positives:,}")
print(f"B1 Only (45-feat GBDT)   : Hits = {total_hits_b1:,} | Pooled Recall = {total_hits_b1/total_positives:.4f} ({total_hits_b1/total_positives*100:.2f}%) | Cycle Mean Recall = {eval_df['recall_b1'].mean():.4f} | PR-AUC = {eval_df['pr_auc_b1'].mean():.4f}")
print(f"B2 Only (History)        : Hits = {total_hits_b2:,} | Pooled Recall = {total_hits_b2/total_positives:.4f} ({total_hits_b2/total_positives*100:.2f}%) | Cycle Mean Recall = {eval_df['recall_b2'].mean():.4f} | PR-AUC = {eval_df['pr_auc_b2'].mean():.4f}")
print(f"Dual-Branch Fused        : Hits = {total_hits_fused:,} | Pooled Recall = {total_hits_fused/total_positives:.4f} ({total_hits_fused/total_positives*100:.2f}%) | Cycle Mean Recall = {eval_df['recall_fused'].mean():.4f} | PR-AUC = {eval_df['pr_auc_fused'].mean():.4f}")

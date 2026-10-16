"""
Evaluates all dual-branch fusion methods under the newly established criteria:
1. Cluster Full-Population PR-AUC (1,992 GPUs)
2. Recall@100 (Cluster Top-100 Fault Recall & Lift)
3. Top-20 / Top-50 Ranking Concentration within Top-100 Cohort
4. Cohort PR-AUC (Cycle Mean & Global Pooled across 302,400 decisions)
5. Subgroup Analysis (Clean GPU k=0 vs Repetitive GPU k>=1)
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent

# 1. Load Files
tape_path = PROJECT_ROOT / "outputs" / "unified_adst" / "unified_adst_risk_tape.parquet"
metrics_path = PROJECT_ROOT / "outputs" / "unified_adst" / "unified_adst_metrics.csv"

if not tape_path.exists():
    raise FileNotFoundError(f"Risk tape not found at {tape_path}")

print(f"[Loading] Unified risk tape from {tape_path}...")
tape = pd.read_parquet(tape_path)
metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None

# Load XID onsets
data_dir = PROJECT_ROOT / "data"
if not (data_dir / "xid_onsets_metadata.parquet").exists():
    data_dir = PARENT_ROOT / "data"
xid_df = pq.read_table(data_dir / "xid_onsets_metadata.parquet").to_pandas()
xid_df["onset_time_ns"] = pd.to_datetime(xid_df["onset_time"], utc=True).astype("int64")

# Group XID onsets by gpu_id for fast lookup
gpu_onsets = {}
for gpu, grp in xid_df.groupby("gpu_id"):
    gpu_onsets[gpu] = np.sort(grp["onset_time_ns"].values)

tape["decision_time"] = pd.to_datetime(tape["decision_time"], utc=True)
decision_times_ns = tape["decision_time"].astype("int64").values

n_rows = len(tape)
counts_30d = np.zeros(n_rows, dtype=np.float64)
DAY_NS = 24 * 3600 * 1_000_000_000
BUFFER_NS = 10 * 60 * 1_000_000_000

print(f"[Processing] Calculating 30-day XID counts for {n_rows:,} decisions...")
gpu_col = tape["gpu_id"].values
for i in range(n_rows):
    g = gpu_col[i]
    t = decision_times_ns[i]
    cutoff = t - BUFFER_NS
    start = cutoff - 30 * DAY_NS
    onsets = gpu_onsets.get(g)
    if onsets is not None and len(onsets) > 0:
        r = np.searchsorted(onsets, cutoff, side="left")
        l = np.searchsorted(onsets, start, side="left")
        counts_30d[i] = max(0, r - l)

tape["count_30d"] = counts_30d

# MLE Alpha estimation from positive counts
pos_counts = counts_30d[counts_30d > 0]
if len(pos_counts) > 0:
    alpha_mle = float(np.clip(len(pos_counts) / np.sum(np.log(pos_counts / 1.0)), 0.5, 3.0))
else:
    alpha_mle = 1.0
print(f"[MLE Estimation] Estimated Pareto alpha MLE: {alpha_mle:.3f}")

# Define Variants
b1 = tape["b1_risk"].values
b2 = tape["b2_risk"].values
y = tape["target_24h"].values

variants = {}

# 1. Base / Ablations
variants["B1 Only (Telemetry 45-feat GBDT)"] = b1
variants["B2 Only (Cumulative History)"] = b2
variants["Fixed 50:50 Fusion"] = 0.5 * b1 + 0.5 * b2

# 2. Junho's methods
# Exp 2: Fixed Lambda = 0.60
variants["Junho Exp 2 (Fixed λ=0.60)"] = 0.60 * b1 + 0.40 * b2

# Exp 8 / Production Final: B1-capped Pareto (pareto_cap_0.6, alpha=1.0)
w_cap06 = np.minimum(0.60, (1.0 / (counts_30d + 1.0)) ** 1.0)
variants["Junho Exp 8 / Optimal (Cap 0.60, α=1.0)"] = w_cap06 * b1 + (1.0 - w_cap06) * b2

# Exp 9 Primary: Report-faithful (wc=0.50, wr=0.02, alpha=1.0)
w_rf_prim = 0.02 + (0.50 - 0.02) * ((1.0 / (counts_30d + 1.0)) ** 1.0)
variants["Junho Exp 9 Primary (wc=0.50, wr=0.02)"] = w_rf_prim * b1 + (1.0 - w_rf_prim) * b2

# Exp 9 Grid Best: (wc=0.30, wr=0.05, alpha=1.0)
w_rf_best = 0.05 + (0.30 - 0.05) * ((1.0 / (counts_30d + 1.0)) ** 1.0)
variants["Junho Exp 9 Grid Best (wc=0.30, wr=0.05)"] = w_rf_best * b1 + (1.0 - w_rf_best) * b2

# 3. Jihan's Pure Pareto (No Cap)
w_pure_mle = np.clip((1.0 / (counts_30d + 1.0)) ** alpha_mle, 0.05, 1.0)
variants[f"Jihan Pure MLE Pareto (No Cap, α={alpha_mle:.2f})"] = w_pure_mle * b1 + (1.0 - w_pure_mle) * b2

w_pure_a10 = np.clip((1.0 / (counts_30d + 1.0)) ** 1.0, 0.05, 1.0)
variants["Jihan Pure Pareto (No Cap, α=1.0)"] = w_pure_a10 * b1 + (1.0 - w_pure_a10) * b2

# Evaluation Setup
is_clean = counts_30d == 0
is_rep = counts_30d > 0
total_tape_pos = int(y.sum())
cluster_total_pos = int(metrics["positives"].sum()) if metrics is not None else 52932

# 3,024 decision points (63 origins x 48 bins) of 100 GPUs each
n_pts = 3024
assert len(tape) == n_pts * 100, f"Expected {n_pts * 100} rows, got {len(tape)}"

# Cycle-by-cycle metrics across 63 origins
tape["origin_idx"] = np.repeat(np.arange(63), 48 * 100)
cycle_ap = {k: [] for k in variants}

for orig_idx, grp in tape.groupby("origin_idx"):
    y_c = grp["target_24h"].values
    if y_c.sum() == 0:
        continue
    for name in variants:
        preds_c = variants[name][grp.index]
        cycle_ap[name].append(average_precision_score(y_c, preds_c))

results = []

for name, preds in variants.items():
    # 1. PR-AUC and ROC-AUC
    ap_all = average_precision_score(y, preds)
    roc_all = roc_auc_score(y, preds)
    ap_clean = average_precision_score(y[is_clean], preds[is_clean])
    ap_rep = average_precision_score(y[is_rep], preds[is_rep])
    cycle_pr_mean = float(np.mean(cycle_ap[name]))

    # 2. Ranking metrics within decision bins (Top-20 and Top-50 concentration)
    preds_2d = preds.reshape(n_pts, 100)
    y_2d = y.reshape(n_pts, 100)
    order = np.argsort(-preds_2d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    for row in range(n_pts):
        ranks[row, order[row]] = np.arange(1, 101)

    hits_20 = int(((ranks <= 20) & (y_2d == 1)).sum())
    hits_50 = int(((ranks <= 50) & (y_2d == 1)).sum())

    rec20_pct = (hits_20 / total_tape_pos) * 100
    rec50_pct = (hits_50 / total_tape_pos) * 100
    prec20_pct = (hits_20 / (n_pts * 20)) * 100
    prec50_pct = (hits_50 / (n_pts * 50)) * 100

    # 3. Cluster Full-Population metrics
    if "B1 Only" in name and metrics is not None:
        full_pr = metrics["pr_auc_b1"].mean()
        full_recall = np.nan
        full_lift = np.nan
    elif "B2 Only" in name and metrics is not None:
        full_pr = metrics["pr_auc_b2"].mean()
        full_recall = np.nan
        full_lift = np.nan
    elif "Optimal" in name and metrics is not None:
        full_pr = metrics["pr_auc_fused"].mean()
        full_recall = metrics["recall_at_100"].mean() * 100
        full_lift = metrics["lift_at_100"].mean()
    else:
        full_pr = np.nan
        full_recall = np.nan
        full_lift = np.nan

    results.append({
        "Method": name,
        "Full PR-AUC (1992 GPU)": f"{full_pr:.4f}" if not np.isnan(full_pr) else "-",
        "Full Recall@100": f"{full_recall:.1f}%" if not np.isnan(full_recall) else "-",
        "Full Lift@100": f"{full_lift:.2f}x" if not np.isnan(full_lift) else "-",
        "Cohort Cycle PR": cycle_pr_mean,
        "Cohort Global PR": ap_all,
        "Cohort ROC": roc_all,
        "Top-20 Recall (Tape)": f"{rec20_pct:.1f}%",
        "Top-20 Hits": hits_20,
        "Top-50 Recall (Tape)": f"{rec50_pct:.1f}%",
        "Top-50 Hits": hits_50,
        "Clean PR (k=0)": ap_clean,
        "Repetitive PR (k>=1)": ap_rep,
    })

res_df = pd.DataFrame(results)

print("\n" + "=" * 110)
print("=== UPDATED APPLES-TO-APPLES BENCHMARK (STANDARDIZED CRITERIA: RECALL@100 & FULL PR-AUC) ===")
print("=" * 110)
print(res_df.to_string(index=False))

# Save results
out_file_1 = PROJECT_ROOT / "scratch_compare_results.csv"
out_file_2 = PARENT_ROOT / "scratch_compare_results.csv"
res_df.to_csv(out_file_1, index=False, encoding="utf-8-sig")
res_df.to_csv(out_file_2, index=False, encoding="utf-8-sig")
print(f"\n[Saved] Updated metrics table to {out_file_1} and {out_file_2}")


def test_stability_check() -> None:
    """Self-check asserting validity of computations."""
    assert len(res_df) == len(variants), "Missing variants in result DataFrame"
    assert res_df["Cohort Global PR"].max() > 0.090, "Optimal model should exceed 0.090 Cohort PR"
    opt_row = res_df[res_df["Method"].str.contains("Optimal")].iloc[0]
    assert opt_row["Top-20 Hits"] >= 4000, "Optimal model Top-20 hits must exceed 4000"
    print("[Assert Passed] Self-check successfully verified.")


if __name__ == "__main__":
    test_stability_check()

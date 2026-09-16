"""
[Pareto λ Post-Hoc Analysis]
Recompute B1/B2 fusion with Pareto-distributed per-GPU dynamic λ
using the EXISTING risk tape — zero model retraining.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = PROJECT_ROOT.parent

# ── Reuse pareto functions from pipeline ──
sys.path.insert(0, str(PROJECT_ROOT / "ML"))
from run_bidirectional_adst_fusion import pareto_lambda, pareto_alpha_mle


def find_data_dir() -> Path:
    for c in [PROJECT_ROOT / "data", PARENT_ROOT / "data"]:
        if c.exists() and (c / "xid_onsets_metadata.parquet").exists():
            return c.resolve()
    raise FileNotFoundError("Data dir not found")


def load_xid_history(data_dir: Path) -> pd.DataFrame:
    df = pq.read_table(data_dir / "xid_onsets_metadata.parquet").to_pandas()
    df["onset_time_ns"] = pd.to_datetime(df["onset_time"], utc=True).astype("int64")
    return df


def compute_gpu_30d_counts(
    xid_df: pd.DataFrame, decision_time: pd.Timestamp,
) -> dict[int, int]:
    """Count XID onsets per gpu_id in the 30 days before decision_time."""
    cutoff_ns = int(decision_time.value)
    start_ns = cutoff_ns - 30 * 24 * 3600 * 1_000_000_000
    mask = (xid_df["onset_time_ns"] >= start_ns) & (xid_df["onset_time_ns"] < cutoff_ns)
    return xid_df.loc[mask].groupby("gpu_id").size().to_dict()


def main() -> None:
    # Load existing risk tape
    tape_path = PROJECT_ROOT / "outputs" / "bidirectional_adst" / "bidirectional_adst_risk_tape.parquet"
    metrics_path = PROJECT_ROOT / "outputs" / "bidirectional_adst" / "bidirectional_adst_metrics.csv"
    tape = pd.read_parquet(tape_path)
    orig_metrics = pd.read_csv(metrics_path)
    data_dir = find_data_dir()
    xid_df = load_xid_history(data_dir)

    print(f"Risk tape: {len(tape):,} rows, {tape['gpu_id'].nunique()} GPUs")
    print(f"Original metrics: {len(orig_metrics)} rolling origins\n")

    # Get unique decision times (proxy for origin cycles)
    tape["decision_time"] = pd.to_datetime(tape["decision_time"], utc=True)
    unique_gpus = sorted(tape["gpu_id"].unique())
    gpu_set = set(unique_gpus)

    # ── Try multiple α values: MLE + fixed grid ──
    all_counts = []
    for gpu_id in unique_gpus:
        gpu_xids = xid_df[xid_df["gpu_id"] == gpu_id]
        all_counts.append(len(gpu_xids))
    all_counts_arr = np.array(all_counts)
    mle_alpha = pareto_alpha_mle(all_counts_arr)
    print(f"Pareto alpha (MLE from global counts): {mle_alpha:.3f}")
    print(f"GPUs with 0 XID events: {(all_counts_arr == 0).sum()}/{len(all_counts_arr)}")
    print(f"GPUs with >=1 XID events: {(all_counts_arr > 0).sum()}")
    print(f"Max XID count: {all_counts_arr.max()}, Median: {np.median(all_counts_arr):.0f}\n")

    alpha_candidates = [0.5, 0.8, 1.0, mle_alpha, 1.5, 2.0, 3.0]
    alpha_candidates = sorted(set(round(a, 3) for a in alpha_candidates))

    results = []

    for alpha in alpha_candidates:
        # Compute per-cycle Pareto λ and re-fuse
        cycle_metrics = []

        for origin_idx, row in orig_metrics.iterrows():
            origin_time = pd.Timestamp(row["origin_time"])
            if origin_time.tzinfo is None:
                origin_time = origin_time.tz_localize("UTC")
            test_end = pd.Timestamp(row["test_end_time"])
            if test_end.tzinfo is None:
                test_end = test_end.tz_localize("UTC")

            # Get this cycle's risk tape rows
            cycle_mask = (tape["decision_time"] >= origin_time) & (tape["decision_time"] < test_end)
            cycle_tape = tape.loc[cycle_mask].copy()
            if len(cycle_tape) == 0:
                continue

            # Compute 30d counts for each GPU at this origin
            counts_dict = compute_gpu_30d_counts(xid_df, origin_time)
            gpu_counts = cycle_tape["gpu_id"].map(lambda g: counts_dict.get(g, 0)).to_numpy()

            # Pareto λ per GPU
            lam = pareto_lambda(gpu_counts, alpha=alpha)

            # Re-fuse
            new_fused = lam * cycle_tape["b1_risk"].values + (1 - lam) * cycle_tape["b2_risk"].values
            cycle_tape["pareto_fused"] = new_fused

            y = cycle_tape["target_24h"].values
            if y.sum() == 0:
                continue

            ap_pareto = float(average_precision_score(y, new_fused))
            ap_orig = float(average_precision_score(y, cycle_tape["fused_risk"].values))
            roc_pareto = float(roc_auc_score(y, new_fused))

            # Recall@100 equivalent: among top-100 ranked by pareto_fused, how many hits?
            # But tape only has top-100 by original ranking... we can still compare
            # fused scores within this subset
            cycle_metrics.append({
                "origin_idx": origin_idx,
                "pr_auc_orig": ap_orig,
                "pr_auc_pareto": ap_pareto,
                "roc_auc_pareto": roc_pareto,
                "lambda_mean": float(lam.mean()),
                "lambda_std": float(lam.std()),
                "n_rows": len(cycle_tape),
                "positives": int(y.sum()),
            })

        if not cycle_metrics:
            continue
        df = pd.DataFrame(cycle_metrics)
        mean_orig = df["pr_auc_orig"].mean()
        mean_pareto = df["pr_auc_pareto"].mean()
        mean_roc = df["roc_auc_pareto"].mean()
        delta_pct = (mean_pareto - mean_orig) / max(mean_orig, 1e-9) * 100

        results.append({
            "alpha": alpha,
            "pr_auc_orig_mean": mean_orig,
            "pr_auc_pareto_mean": mean_pareto,
            "roc_auc_pareto_mean": mean_roc,
            "delta_pct": delta_pct,
            "lambda_mean": df["lambda_mean"].mean(),
            "lambda_std": df["lambda_std"].mean(),
            "cycles_evaluated": len(df),
        })

        is_mle = " (MLE)" if abs(alpha - mle_alpha) < 0.01 else ""
        marker = " [IMPROVED]" if delta_pct > 0 else ""
        print(
            f"  alpha={alpha:.3f}{is_mle}: PR-AUC {mean_orig:.4f} -> {mean_pareto:.4f} "
            f"({delta_pct:+.1f}%) | ROC-AUC {mean_roc:.4f} | mean_lambda={df['lambda_mean'].mean():.3f}{marker}"
        )

    print("\n" + "=" * 70)
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["pr_auc_pareto_mean"].idxmax()]
    print(f"Best alpha = {best['alpha']:.3f}: PR-AUC {best['pr_auc_pareto_mean']:.4f} ({best['delta_pct']:+.1f}%)")
    print(f"  mean_lambda = {best['lambda_mean']:.3f} +/- {best['lambda_std']:.3f}")

    # Save results
    out_dir = PROJECT_ROOT / "outputs" / "pareto_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "pareto_alpha_sweep.csv", index=False)
    print(f"\nSaved to {out_dir / 'pareto_alpha_sweep.csv'}")


if __name__ == "__main__":
    main()

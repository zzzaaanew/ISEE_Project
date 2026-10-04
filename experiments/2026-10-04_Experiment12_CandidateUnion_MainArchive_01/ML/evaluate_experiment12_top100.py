"""Validate Experiment 12's saved full-population scores at Top 100.

This is evaluation only: no fitting, threshold selection, or change to source
scores. Labels reproduce UnifiedDataEngine.load_all_xid_ledger() exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score


PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "experiments" / "2026-09-29_All-XID_HanADST_B1LightCascade_B2Sliding_TemporalMoE_01"
OUTPUT = PROJECT / "experiments" / "2026-10-01_Experiment12_Top100_Validation_01"
GRID_META = PROJECT / "outputs" / "branch1" / "cache" / "grid_meta.npz"
ONSET_LEDGER = PROJECT.parent / "data" / "xid_onsets_metadata.parquet"
STEP_NS = 5 * 60 * 1_000_000_000
DECISION_NS = 30 * 60 * 1_000_000_000
HORIZON_BINS = 24 * 60 // 5
K = 100
SCORE_COLUMNS = {
    "B1": "b1_probability",
    "B2": "b2_probability",
    "Fusion": "probability",
}
PR_COLUMNS = {"B1": "pr_auc_b1", "B2": "pr_auc_b2", "Fusion": "pr_auc_fused"}
DISCOUNT = 1.0 / np.log2(np.arange(2, K + 2, dtype=np.float64))
IDEAL_DCG = np.r_[0.0, np.cumsum(DISCOUNT)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_ground_truth() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Match the original 5-minute grid/onset-bin inclusive label construction."""
    with np.load(GRID_META, allow_pickle=True) as meta:
        gpu_ids = np.asarray(meta["gpu_ids"]).astype(str)
        bins_ns = np.asarray(meta["bin_start_ns"], dtype=np.int64)
    check(len(gpu_ids) == 1992, "Grid GPU count is not 1,992")
    check(np.all(np.diff(bins_ns) == STEP_NS), "Grid is not contiguous at 5 minutes")
    check(len(set(gpu_ids)) == len(gpu_ids), "Duplicate GPU IDs in grid")
    ledger = pq.read_table(ONSET_LEDGER, columns=["gpu_id", "onset_time", "xid_code"]).to_pandas()
    onset_ns = pd.to_datetime(ledger["onset_time"], utc=True).astype("int64").to_numpy()
    gpu_map = {gpu: idx for idx, gpu in enumerate(gpu_ids)}
    gpu_idx = ledger["gpu_id"].map(gpu_map)
    gt = np.zeros((len(bins_ns), len(gpu_ids)), dtype=np.bool_)
    matched = 0
    for ns, idx in zip(onset_ns, gpu_idx):
        if pd.isna(idx):
            continue
        matched += 1
        onset_bin = int((int(ns) - int(bins_ns[0])) // STEP_NS)
        start_bin = max(0, onset_bin - HORIZON_BINS)
        end_bin = min(len(bins_ns) - 1, onset_bin)
        gt[start_bin : end_bin + 1, int(idx)] = True
    return gpu_ids, bins_ns, gt, {
        "onset_rows": int(len(ledger)),
        "matched_onset_rows": matched,
        "label_rule": "onset_bin-288 through onset_bin inclusive on original 5-minute grid",
    }


def evaluate_part(
    part: Path,
    reference: pd.Series,
    gpu_ids: np.ndarray,
    bins_ns: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[list[dict], dict]:
    columns = ["timestamp", "gpu_uid", "probability", "b1_probability", "b2_probability", "risk_rank", "model_state_id"]
    frame = pq.read_table(part, columns=columns).to_pandas()
    state_id = str(reference["model_state_id"])
    n_gpu = len(gpu_ids)
    n_times = int(reference["decision_time_count"])
    check(len(frame) == n_times * n_gpu, f"{state_id}: row count mismatch")
    check(frame["model_state_id"].eq(state_id).all(), f"{state_id}: state IDs mismatch")
    observed_gpus = frame["gpu_uid"].to_numpy().reshape(n_times, n_gpu)
    check(np.array_equal(observed_gpus, np.broadcast_to(gpu_ids, observed_gpus.shape)), f"{state_id}: GPU order mismatch")
    times_ns = pd.to_datetime(frame["timestamp"], utc=True).astype("int64").to_numpy().reshape(n_times, n_gpu)
    check(np.all(times_ns == times_ns[:, :1]), f"{state_id}: mixed timestamps within a query")
    query_ns = times_ns[:, 0]
    check(np.all(np.diff(query_ns) == DECISION_NS), f"{state_id}: decision cadence mismatch")
    check((query_ns[0] - bins_ns[0]) % STEP_NS == 0, f"{state_id}: off-grid timestamp")
    bin_idx = (query_ns - bins_ns[0]) // STEP_NS
    check(np.all((bin_idx >= 0) & (bin_idx < len(bins_ns))), f"{state_id}: timestamp out of grid")
    check(np.array_equal(bins_ns[bin_idx], query_ns), f"{state_id}: grid timestamp mismatch")
    check(pd.Timestamp(query_ns[0], unit="ns", tz="UTC") == pd.Timestamp(reference["origin_time"]), f"{state_id}: origin mismatch")
    end_ns = int(pd.Timestamp(reference["score_end_exclusive"]).value)
    check(int(query_ns[-1]) < end_ns <= int(query_ns[-1]) + DECISION_NS, f"{state_id}: end mismatch")
    labels = ground_truth[bin_idx]
    positives = int(labels.sum())
    prevalence = positives / len(frame)
    check(positives == int(reference["positives"]), f"{state_id}: positive count mismatch: {positives} vs {reference['positives']}")
    check(np.isclose(prevalence, float(reference["prevalence"]), rtol=0, atol=1e-14), f"{state_id}: prevalence mismatch")
    positive_per_query = labels.sum(axis=1).astype(np.int32)
    positive_queries = positive_per_query > 0
    y_flat = labels.ravel().astype(np.uint8)
    rows = []
    for model, column in SCORE_COLUMNS.items():
        scores = frame[column].to_numpy(dtype=np.float64).reshape(n_times, n_gpu)
        check(np.isfinite(scores).all(), f"{state_id}/{model}: nonfinite score")
        check(((scores >= 0) & (scores <= 1)).all(), f"{state_id}/{model}: score outside [0,1]")
        order = np.argsort(-scores, axis=1, kind="stable")
        top_indices = order[:, :K]
        top_labels = np.take_along_axis(labels, top_indices, axis=1)
        hits = int(top_labels.sum())
        precision = hits / (n_times * K)
        recall = hits / max(positives, 1)  # Original runner's zero-positive convention.
        lift = precision / max(prevalence, 1e-12)
        dcg = top_labels @ DISCOUNT
        ndcg = np.full(n_times, np.nan, dtype=np.float64)
        ndcg[positive_queries] = dcg[positive_queries] / IDEAL_DCG[np.minimum(positive_per_query[positive_queries], K)]
        ap = float(average_precision_score(y_flat, scores.ravel())) if positives else 0.0
        original_ap = float(reference[PR_COLUMNS[model]])
        check(np.isclose(ap, original_ap, rtol=1e-9, atol=1e-12), f"{state_id}/{model}: PR-AUC mismatch: {ap} vs {original_ap}")
        if model == "Fusion":
            original_ranks = frame["risk_rank"].to_numpy().reshape(n_times, n_gpu)
            expected_ranks = np.empty_like(order, dtype=np.uint16)
            np.put_along_axis(expected_ranks, order, np.arange(1, n_gpu + 1, dtype=np.uint16), axis=1)
            check(np.array_equal(expected_ranks, original_ranks), f"{state_id}: stored Fusion ranks mismatch")
            check(np.isclose(recall, float(reference["recall_at_100"]), rtol=0, atol=1e-12), f"{state_id}: Fusion Recall@100 mismatch")
            check(np.isclose(lift, float(reference["lift_at_100"]), rtol=0, atol=1e-12), f"{state_id}: Fusion Lift@100 mismatch")
        rows.append({
            "origin_idx": int(reference["origin_idx"]),
            "model_state_id": state_id,
            "origin_time": str(reference["origin_time"]),
            "model": model,
            "decision_time_count": n_times,
            "gpu_count": n_gpu,
            "top_k": K,
            "positive_queries": int(positive_queries.sum()),
            "zero_positive_queries": int((~positive_queries).sum()),
            "positives": positives,
            "prevalence": prevalence,
            "hits_at_100": hits,
            "recall_at_100": recall,
            "precision_at_100": precision,
            "lift_at_100": lift,
            "ndcg_at_100_positive_queries": float(np.nanmean(ndcg)) if positive_queries.any() else None,
            "ndcg_sum_positive_queries": float(np.nansum(ndcg)),
            "pr_auc": ap,
        })
    return rows, {"part": str(part.relative_to(SOURCE)), "rows": len(frame), "timestamps": n_times}


def summarize(metrics: pd.DataFrame, part_checks: list[dict], label_info: dict) -> dict:
    report: dict = {
        "status": "VERIFIED",
        "source_experiment": SOURCE.name,
        "origin_count": int(metrics["origin_idx"].nunique()),
        "score_rows": int(sum(part["rows"] for part in part_checks)),
        "decision_time_count": int(sum(part["timestamps"] for part in part_checks)),
        "gpu_count": 1992,
        "top_k": K,
        "label_info": label_info,
        "models": {},
        "fusion_minus_b2": {},
        "fidelity": "64/64 positive counts, prevalence, PR-AUC; Fusion stored ranks, Recall@100, Lift@100 matched",
    }
    for model in SCORE_COLUMNS:
        group = metrics.loc[metrics["model"] == model]
        check(len(group) == 64, f"{model}: not 64 origins")
        total_positive_queries = int(group["positive_queries"].sum())
        report["models"][model] = {
            "macro_origin_recall_at_100": float(group["recall_at_100"].mean()),
            "micro_gpu_time_recall_at_100": float(group["hits_at_100"].sum() / group["positives"].sum()),
            "macro_origin_ndcg_at_100": float(group["ndcg_at_100_positive_queries"].mean()),
            "pooled_positive_query_ndcg_at_100": float(group["ndcg_sum_positive_queries"].sum() / total_positive_queries),
            "macro_origin_precision_at_100": float(group["precision_at_100"].mean()),
            "macro_origin_lift_at_100": float(group["lift_at_100"].mean()),
            "macro_origin_pr_auc": float(group["pr_auc"].mean()),
            "hits_at_100": int(group["hits_at_100"].sum()),
            "gpu_time_positives": int(group["positives"].sum()),
            "positive_queries": total_positive_queries,
            "zero_positive_queries": int(group["zero_positive_queries"].sum()),
            "ndcg_defined_origins": int(group["ndcg_at_100_positive_queries"].notna().sum()),
        }
    b2 = metrics.loc[metrics["model"] == "B2"].set_index("origin_idx").sort_index()
    fusion = metrics.loc[metrics["model"] == "Fusion"].set_index("origin_idx").sort_index()
    for key in ["recall_at_100", "ndcg_at_100_positive_queries", "precision_at_100", "lift_at_100", "pr_auc"]:
        diff = fusion[key] - b2[key]
        # An origin with no positive decision times has undefined NDCG.
        if key != "ndcg_at_100_positive_queries":
            check(diff.notna().all(), f"Fusion-B2 {key} contains missing origin")
        excluded = int(diff.isna().sum())
        diff = diff.dropna()
        check(len(diff) > 0, f"Fusion-B2 {key} has no comparable origins")
        report["fusion_minus_b2"][key] = {
            "mean_delta": float(diff.mean()),
            "median_delta": float(diff.median()),
            "fusion_win_origins": int((diff > 0).sum()),
            "tie_origins": int((diff == 0).sum()),
            "fusion_loss_origins": int((diff < 0).sum()),
            "excluded_undefined_origins": excluded,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    check(not output.exists(), f"Output already exists; source and prior results are never overwritten: {output}")
    source_manifest = SOURCE / "experiment_manifest.json"
    parts_manifest = SOURCE / "full_population_manifest.json"
    original_metrics = SOURCE / "full_period_metrics.csv"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    part_contract = json.loads(parts_manifest.read_text(encoding="utf-8"))
    check(manifest["status"] == "COMPLETED", "Source experiment is not completed")
    check(manifest["target"] == "all_xids", "Source target is not All-XID")
    check(manifest["gpu_count"] == 1992 and manifest["origin_count"] == 64, "Source population/origins changed")
    check(manifest["decision_interval_seconds"] == 1800 and manifest["horizon_hours"] == 24, "Source time contract changed")
    check(manifest["warmup_days"] == 30 and manifest["purge_hours"] == 36 and manifest["feature_cutoff_minutes"] == 10, "Source rolling contract changed")
    parts = [SOURCE / relative for relative in part_contract["full_parts"]]
    check(len(parts) == 64 and all(part.is_file() for part in parts), "Expected 64 complete score parts")
    references = pd.read_csv(original_metrics).sort_values("origin_idx").reset_index(drop=True)
    check(len(references) == 64 and references["origin_idx"].tolist() == list(range(64)), "Original metrics do not contain exactly origins 0..63")
    check(references["gpu_count_evaluated"].eq(1992).all() and references["top_k"].eq(K).all(), "Original metrics population/K mismatch")
    gpu_ids, bins_ns, ground_truth, label_info = read_ground_truth()
    metrics_rows: list[dict] = []
    part_checks: list[dict] = []
    for idx, (part, reference) in enumerate(zip(parts, references.itertuples(index=False))):
        row = pd.Series(reference._asdict())
        check(part.name == f"score_part_{idx:04d}.parquet", f"Unexpected score part: {part.name}")
        rows, check_row = evaluate_part(part, row, gpu_ids, bins_ns, ground_truth)
        metrics_rows.extend(rows)
        part_checks.append(check_row)
        if idx == 0 or (idx + 1) % 8 == 0 or idx == 63:
            print(f"validated {idx + 1}/64 origins", flush=True)
    check(sum(item["rows"] for item in part_checks) == int(part_contract["full_rows"]) == int(manifest["completed_score_rows"]), "Full score row count mismatch")
    check(sum(item["timestamps"] for item in part_checks) == int(part_contract["timestamps"]), "Timestamp count mismatch")
    metrics = pd.DataFrame(metrics_rows)
    report = summarize(metrics, part_checks, label_info)
    contract = {
        "evaluation_type": "saved-score-only_full_population_30m_rolling",
        "source_experiment": str(SOURCE),
        "output_directory": str(output),
        "goal_ids": ["G-004", "G-007", "G-008"],
        "seed": manifest.get("seed", 20260905),
        "source_contract": {key: manifest[key] for key in ["target", "warmup_days", "purge_hours", "horizon_hours", "feature_cutoff_minutes", "decision_interval_seconds", "retrain_cadence_hours", "gpu_count", "origin_count"]},
        "metric_contract": {
            "population": "all 1,992 GPUs at each saved 30-minute decision time, before Top-100 truncation",
            "tie_break": "stable descending score sort in original GPU index order",
            "recall_at_100": "origin-pooled GPU-time top-100 hits divided by all positive GPU-time labels in origin",
            "ndcg_at_100": "binary gain, log2(rank+1) discount, ideal DCG capped at min(positive GPUs at time, 100); zero-positive times excluded",
            "precision_at_100": "top-100 hits divided by decision times times 100",
            "lift_at_100": "precision@100 divided by full-population GPU-time prevalence",
            "pr_auc": "average_precision_score over all GPU-time rows per origin",
            "primary_comparisons": "64 paired origin macro means; pooled micro/query metrics are descriptive",
            "episode_recall": "not estimated here; GPU-time positives can repeat across 30-minute decisions",
        },
        "input_sha256": {
            "experiment_manifest": sha256(source_manifest),
            "full_population_manifest": sha256(parts_manifest),
            "full_period_metrics": sha256(original_metrics),
            "grid_meta": sha256(GRID_META),
            "onset_ledger": sha256(ONSET_LEDGER),
            "score_parts": {part.name: sha256(part) for part in parts},
        },
        "part_checks": part_checks,
        "evaluator_sha256": sha256(Path(__file__).resolve()),
    }
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(output / "top100_origin_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "top100_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "top100_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "models": report["models"], "fusion_minus_b2": report["fusion_minus_b2"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

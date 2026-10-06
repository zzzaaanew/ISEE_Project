"""Independent, full-population audit of Experiment 19's fitted Fusion tape."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

import evaluate_experiment12_top100 as reference


ARCHIVE = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "experiments/2026-10-04_Experiment14_MaturedADST_TemporalMoE_01"
RESULT = ARCHIVE / "results"
GPU_COUNT = 1992
STEP_NS = 30 * 60 * 1_000_000_000
PURGE_NS = 36 * 60 * 60 * 1_000_000_000
HORIZON_NS = 24 * 60 * 60 * 1_000_000_000
EPS = 1e-6
K = 100
ARMS = {
    "no_inertia": (False, False),
    "b1_inertia": (True, False),
    "b2_inertia": (False, True),
    "dual_inertia": (True, True),
}
DISCOUNT = 1.0 / np.log2(np.arange(2, K + 2, dtype=np.float64))
IDEAL_DCG = np.r_[0.0, np.cumsum(DISCOUNT)]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def z(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(probability, dtype=np.float64), EPS, 1 - EPS))


def calibrated(x: np.ndarray, parameters: dict) -> np.ndarray:
    return np.clip(expit(parameters["intercept"] + parameters["slope"] * x / 6.0), EPS, 1 - EPS)


def independent_basis(p1: np.ndarray, p2: np.ndarray, i1: np.ndarray, i2: np.ndarray, arm: str) -> np.ndarray:
    a, b = z(p1) / 6.0, z(p2) / 6.0
    columns = [np.ones(len(a)), a, b, a ** 2, b ** 2, a * b]
    if ARMS[arm][0]:
        c = z(i1) / 6.0
        columns += [c, c ** 2, a * c]
    if ARMS[arm][1]:
        d = z(i2) / 6.0
        columns += [d, d ** 2, b * d]
    return np.column_stack(columns)


def metric(scores: np.ndarray, labels: np.ndarray) -> dict:
    n_times, _ = labels.shape
    selected = np.argsort(-scores, axis=1, kind="stable")[:, :K]
    hit_matrix = np.take_along_axis(labels, selected, axis=1)
    hits = int(hit_matrix.sum())
    positives = int(labels.sum())
    per_query = labels.sum(axis=1)
    valid = per_query > 0
    ndcg_sum = float(np.sum((hit_matrix @ DISCOUNT)[valid] / IDEAL_DCG[np.minimum(per_query[valid], K)]))
    ap = float(average_precision_score(labels.ravel(), scores.ravel())) if positives else 0.0
    return {
        "hits_at_100": hits, "positives": positives, "decision_times": n_times,
        "positive_queries": int(valid.sum()),
        "ndcg_sum_positive_queries": ndcg_sum,
        "pr_auc": ap,
    }


def audit(source: Path, result: Path) -> dict:
    manifest = json.loads((result / "experiment_manifest.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((source / "full_population_manifest.json").read_text(encoding="utf-8"))
    check(manifest["status"] == "COMPLETED", "Partial experiment is not a full audit")
    check(manifest["origin_count"] == 64 and len(source_manifest["full_parts"]) == 64, "Origin count differs")
    check(manifest["source_manifest_sha256"] == sha256(source / "experiment_manifest.json"), "Source run manifest changed")
    check(manifest["source_full_population_manifest_sha256"] == sha256(source / "full_population_manifest.json"), "Source population manifest changed")
    check(manifest["inertia_uses_current_score"] is False, "Non-lagged inertia in contract")
    check(manifest["terminal_heldout_used"] is False and manifest["disagreement_feature_used"] is False, "Forbidden method in manifest")
    gpu_ids, bins_ns, ground_truth, _ = reference.read_ground_truth()
    check(len(gpu_ids) == GPU_COUNT, "GPU count mismatch")
    trace = pd.read_csv(result / "fit_trace.csv")
    params = json.loads((result / "model_parameters.json").read_text(encoding="utf-8"))
    source_hashes = json.loads((result / "input_hashes.json").read_text(encoding="utf-8"))
    recorded_metrics = pd.read_csv(result / "origin_metrics.csv")
    check(len(trace) == 64 * len(ARMS), "Fit trace coverage mismatch")
    check(len(params) == 64 and len(source_hashes) == 64, "Model or source hash coverage mismatch")
    expected_rho1 = 2 ** (-0.5 / manifest["inertia_half_life_hours"]["b1"])
    expected_rho2 = 2 ** (-0.5 / manifest["inertia_half_life_hours"]["b2"])
    old1 = old2 = None
    previous_time = None
    calculated: list[dict] = []
    changed_origins: set[int] = set()
    memory_rows_checked = score_rows = decision_times = fit_count = 0
    for idx, relative in enumerate(source_manifest["full_parts"]):
        source_part = source / relative
        result_part = result / "full_population" / source_part.name
        check(source_part.is_file() and result_part.is_file(), f"Missing part {idx}")
        check(sha256(source_part) == source_hashes[idx]["sha256"], f"Source part {idx} changed")
        src = pq.read_table(source_part, columns=[
            "timestamp", "gpu_uid", "model_state_id", "feature_cutoff_time",
            "b1_probability", "b2_probability", "probability",
        ]).to_pandas()
        dest = pq.read_table(result_part).to_pandas()
        check(not any("label" in col or "target" in col for col in dest.columns), f"Labels leaked into score tape {idx}")
        check(len(src) == len(dest) and len(dest) % GPU_COUNT == 0, f"Row count mismatch {idx}")
        n_time = len(dest) // GPU_COUNT
        check(dest["model_state_id"].eq(f"state_{idx:04d}").all(), f"Model state mismatch {idx}")
        check(np.array_equal(dest["gpu_uid"].astype(str).to_numpy(), src["gpu_uid"].astype(str).to_numpy()), f"GPU IDs changed {idx}")
        check(np.array_equal(dest["timestamp"].to_numpy(), src["timestamp"].to_numpy()), f"Timestamps changed {idx}")
        check(np.array_equal(dest["feature_cutoff_time"].to_numpy(), src["feature_cutoff_time"].to_numpy()), f"Feature cutoff changed {idx}")
        for new, original in [
            ("b1_probability", "b1_probability"),
            ("b2_probability", "b2_probability"),
            ("experiment14_fusion_probability", "probability"),
        ]:
            check(np.array_equal(dest[new].to_numpy(), src[original].to_numpy()), f"{new} differs at origin {idx}")
        time_matrix = pd.to_datetime(dest["timestamp"], utc=True).astype("int64").to_numpy().reshape(n_time, GPU_COUNT)
        check(np.all(time_matrix == time_matrix[:, :1]), f"Mixed times {idx}")
        times = time_matrix[:, 0]
        check(np.all(np.diff(times) == STEP_NS), f"Cadence changed {idx}")
        if previous_time is not None:
            check(int(times[0]) - previous_time == STEP_NS, f"Origin gap {idx}")
        bin_idx = np.searchsorted(bins_ns, times)
        check(np.all(bin_idx < len(bins_ns)) and np.array_equal(bins_ns[bin_idx], times), f"Grid mismatch {idx}")
        labels = ground_truth[bin_idx].astype(np.uint8)
        b1 = src["b1_probability"].to_numpy(dtype=float).reshape(n_time, GPU_COUNT)
        b2 = src["b2_probability"].to_numpy(dtype=float).reshape(n_time, GPU_COUNT)
        mem1 = dest["b1_memory_logit"].to_numpy(dtype=float).reshape(n_time, GPU_COUNT)
        mem2 = dest["b2_memory_logit"].to_numpy(dtype=float).reshape(n_time, GPU_COUNT)
        pi1 = dest["b1_inertia_probability"].to_numpy(dtype=float).reshape(n_time, GPU_COUNT)
        pi2 = dest["b2_inertia_probability"].to_numpy(dtype=float).reshape(n_time, GPU_COUNT)
        parameter = params[idx]
        check(int(parameter["origin_idx"]) == idx, f"Parameter origin mismatch {idx}")
        status = parameter["status"]
        rows = trace.loc[trace["origin_idx"] == idx]
        check(len(rows) == 4 and set(rows["arm"]) == set(ARMS), f"Trace arms incomplete {idx}")
        check(rows["status"].eq(status).all(), f"Trace status mismatch {idx}")
        cutoff = int(times[0]) - PURGE_NS
        trace_cutoff = pd.Timestamp(rows.iloc[0]["cutoff_time"]).value
        check(trace_cutoff == cutoff, f"Feedback cutoff mismatch {idx}")
        check(pd.Timestamp(rows.iloc[0]["calibration_start_time"]).value == cutoff - 14 * 24 * 60 * 60 * 1_000_000_000, f"Calibration window mismatch {idx}")
        check(pd.Timestamp(rows.iloc[0]["fusion_start_time"]).value == cutoff - 7 * 24 * 60 * 60 * 1_000_000_000, f"Fusion window mismatch {idx}")
        if status == "trained":
            latest = pd.Timestamp(rows.iloc[0]["latest_fit_score_time"]).value
            check(latest <= cutoff and latest + HORIZON_NS <= int(times[0]), f"Immature training label {idx}")
            check(int(rows.iloc[0]["calibration_positives"]) >= 20 and int(rows.iloc[0]["fusion_positives"]) >= 20, f"Insufficient trained positives {idx}")
            check(len(parameter["calibrators"]) == 4 and len(parameter["arms"]) == 4, f"Actual model absent {idx}")
            fit_count += 4
        else:
            check(status == "cold_start_b2_fallback", f"Unexpected status {idx}")
        recalculated = {name: np.empty_like(b1) for name in ARMS}
        for t in range(n_time):
            expected1 = np.full(GPU_COUNT, np.nan) if old1 is None else old1
            expected2 = np.full(GPU_COUNT, np.nan) if old2 is None else old2
            check(np.allclose(mem1[t], expected1, rtol=0, atol=1e-12, equal_nan=True), f"B1 memory uses current/future score at {idx}/{t}")
            check(np.allclose(mem2[t], expected2, rtol=0, atol=1e-12, equal_nan=True), f"B2 memory uses current/future score at {idx}/{t}")
            memory_rows_checked += GPU_COUNT
            if status == "trained":
                c = parameter["calibrators"]
                p1 = calibrated(z(b1[t]), c["b1"])
                p2 = calibrated(z(b2[t]), c["b2"])
                expected_pi1 = calibrated(mem1[t], c["i1"])
                expected_pi2 = calibrated(mem2[t], c["i2"])
                check(np.allclose(pi1[t], expected_pi1, rtol=0, atol=1e-12), f"B1 inertia probability mismatch {idx}/{t}")
                check(np.allclose(pi2[t], expected_pi2, rtol=0, atol=1e-12), f"B2 inertia probability mismatch {idx}/{t}")
                offset = z(p2)
                for arm, beta in parameter["arms"].items():
                    value = expit(offset + independent_basis(p1, p2, expected_pi1, expected_pi2, arm) @ np.asarray(beta))
                    recalculated[arm][t] = np.clip(value, EPS, 1 - EPS)
            else:
                check(np.isnan(pi1[t]).all() and np.isnan(pi2[t]).all(), f"Unfitted inertia probability {idx}/{t}")
                for arm in ARMS:
                    recalculated[arm][t] = b2[t]
            current1, current2 = z(b1[t]), z(b2[t])
            old1 = current1.copy() if old1 is None else expected_rho1 * old1 + (1 - expected_rho1) * current1
            old2 = current2.copy() if old2 is None else expected_rho2 * old2 + (1 - expected_rho2) * current2
            previous_time = int(times[t])
        scores = {
            "B1": b1, "B2": b2,
            "Experiment14_Fusion": src["probability"].to_numpy(dtype=float).reshape(n_time, GPU_COUNT),
        }
        for arm in ARMS:
            observed = dest[f"fusion_{arm}"].to_numpy(dtype=float).reshape(n_time, GPU_COUNT)
            check(np.isfinite(observed).all() and np.all((observed >= 0) & (observed <= 1)), f"Invalid Fusion probability {idx}/{arm}")
            check(np.allclose(observed, recalculated[arm], rtol=0, atol=1e-12), f"Fusion formula mismatch {idx}/{arm}")
            if status != "trained":
                check(np.array_equal(observed, b2), f"Fallback is not exact B2 {idx}/{arm}")
            elif not np.array_equal(observed, b2):
                changed_origins.add(idx)
            scores[arm] = observed
        for name, score in scores.items():
            actual = metric(score, labels)
            saved = recorded_metrics.loc[(recorded_metrics["origin_idx"] == idx) & (recorded_metrics["model"] == name)]
            check(len(saved) == 1, f"Origin metric missing {idx}/{name}")
            for key, value in actual.items():
                check(np.isclose(value, float(saved.iloc[0][key]), rtol=1e-10, atol=1e-12), f"Metric {key} differs {idx}/{name}")
            calculated.append({"origin_idx": idx, "model": name, **actual})
        score_rows += len(dest)
        decision_times += n_time
        print(f"audited origin {idx + 1}/64: {status}", flush=True)
    check(score_rows == 6_081_576 and decision_times == 3053, "Full-period coverage mismatch")
    check(manifest["score_rows"] == score_rows and manifest["decision_time_count"] == decision_times, "Manifest coverage mismatch")
    check(manifest["fusion_fit_count"] == fit_count and fit_count > 0, "Actual Fusion fit count mismatch")
    check(len(changed_origins) > 0, "Trained Fusion never changes B2")
    output_metrics = pd.DataFrame(calculated)
    summary = {}
    for name, rows in output_metrics.groupby("model", sort=False):
        summary[name] = {
            "hits_at_100": int(rows["hits_at_100"].sum()),
            "pooled_recall_at_100": float(rows["hits_at_100"].sum() / rows["positives"].sum()),
            "pooled_positive_query_ndcg_at_100": float(rows["ndcg_sum_positive_queries"].sum() / rows["positive_queries"].sum()),
            "origin_mean_pr_auc": float(rows["pr_auc"].mean()),
        }
    saved_summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    for name, values in summary.items():
        for key, value in values.items():
            check(np.isclose(value, saved_summary[name][key], rtol=1e-10, atol=1e-12), f"Summary mismatch {name}/{key}")
    baseline = json.loads((PROJECT / "experiments/2026-10-04_Experiment14_Top100_01/top100_summary.json").read_text(encoding="utf-8"))["models"]
    for name, reference_name in [("B1", "B1"), ("B2", "B2"), ("Experiment14_Fusion", "Fusion")]:
        check(summary[name]["hits_at_100"] == baseline[reference_name]["hits_at_100"], f"Experiment 14 baseline hits changed: {name}")
        check(np.isclose(summary[name]["origin_mean_pr_auc"], baseline[reference_name]["macro_origin_pr_auc"], rtol=0, atol=1e-12), f"Experiment 14 baseline AP changed: {name}")
    return {
        "status": "VERIFIED", "origin_count": 64, "score_rows": score_rows,
        "decision_time_count": decision_times, "memory_rows_checked": memory_rows_checked,
        "fusion_fit_count": fit_count, "fusion_changed_origin_count": len(changed_origins),
        "feedback_cutoff_violations": 0, "memory_lag_violations": 0,
        "source_score_differences": 0, "metric_mismatches": 0,
        "models": summary,
        "interpretation": "Development-period exploratory comparison, not independent generalization proof",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    result = args.result.resolve()
    report = audit(args.source.resolve(), result)
    audit_path = result / "audit.json"
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {audit_path}")
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "score_rows", "fusion_fit_count", "fusion_changed_origin_count")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

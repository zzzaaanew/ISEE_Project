"""Experiment 19: genuinely fitted, prequential dual-inertia Fusion.

Experiment 14's B1/B2 predictions are immutable inputs.  This runner fits four
new B2-offset nonlinear probability models at every eligible rolling origin.
It never calls, copies, or re-labels Experiment 14's old Temporal MoE output as
a newly trained Fusion model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

import evaluate_experiment12_top100 as reference


ARCHIVE = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "experiments/2026-10-04_Experiment14_MaturedADST_TemporalMoE_01"
OUTPUT = ARCHIVE / "rerun"
SEED = 20260905
GPU_COUNT = 1992
K = 100
STEP_NS = 30 * 60 * 1_000_000_000
PURGE_NS = 36 * 60 * 60 * 1_000_000_000
HORIZON_NS = 24 * 60 * 60 * 1_000_000_000
DAY_NS = 24 * 60 * 60 * 1_000_000_000
HALF_LIFE_HOURS = {"b1": 2.0, "b2": 6.0}
RHO = {name: 2 ** (-0.5 / hours) for name, hours in HALF_LIFE_HOURS.items()}
ARM_FEATURES = {
    "no_inertia": (False, False),
    "b1_inertia": (True, False),
    "b2_inertia": (False, True),
    "dual_inertia": (True, True),
}
SCORE_COLUMNS = {name: f"fusion_{name}" for name in ARM_FEATURES}
EPS = 1e-6
REGULARIZATION = 0.01
MIN_POSITIVES = 20
DISCOUNT = 1.0 / np.log2(np.arange(2, K + 2, dtype=np.float64))
IDEAL_DCG = np.r_[0.0, np.cumsum(DISCOUNT)]


@dataclass
class ScoreBlock:
    time_ns: int
    z1: np.ndarray
    z2: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    label: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(probability, dtype=np.float64), EPS, 1.0 - EPS))


def fitted_calibrator(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Monotone Platt head; its positive slope preserves within-query order."""
    xx = np.asarray(x, dtype=np.float64) / 6.0
    yy = np.asarray(y, dtype=np.float64)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * xx
        residual = expit(eta) - yy
        loss = np.mean(np.logaddexp(0.0, eta) - yy * eta)
        loss += 0.0001 * (theta[1] - 6.0) ** 2 / 2.0
        gradient = np.array([
            np.mean(residual),
            np.mean(residual * xx) + 0.0001 * (theta[1] - 6.0),
        ])
        return float(loss), gradient

    fit = minimize(
        objective, np.array([0.0, 6.0]), method="L-BFGS-B", jac=True,
        bounds=[(None, None), (0.01, 20.0)], options={"maxiter": 120},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Probability calibration failed: {fit.message}")
    return {"intercept": float(fit.x[0]), "slope": float(fit.x[1])}


def calibrate(x: np.ndarray, model: dict[str, float]) -> np.ndarray:
    return np.clip(expit(model["intercept"] + model["slope"] * np.asarray(x) / 6.0), EPS, 1.0 - EPS)


def basis(
    p1: np.ndarray, p2: np.ndarray, pi1: np.ndarray, pi2: np.ndarray,
    arm: str,
) -> np.ndarray:
    """Low-capacity nonlinear residual basis; no disagreement statistic."""
    use_i1, use_i2 = ARM_FEATURES[arm]
    z1 = clipped_logit(p1) / 6.0
    z2 = clipped_logit(p2) / 6.0
    columns = [np.ones(len(z1)), z1, z2, z1 * z1, z2 * z2, z1 * z2]
    if use_i1:
        zi1 = clipped_logit(pi1) / 6.0
        columns.extend([zi1, zi1 * zi1, z1 * zi1])
    if use_i2:
        zi2 = clipped_logit(pi2) / 6.0
        columns.extend([zi2, zi2 * zi2, z2 * zi2])
    return np.column_stack(columns)


def fitted_residual(x: np.ndarray, offset: np.ndarray, y: np.ndarray) -> list[float]:
    """Fit logit(P_fusion) = logit(P_B2) + nonlinear correction."""
    yy = np.asarray(y, dtype=np.float64)
    oo = np.asarray(offset, dtype=np.float64)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = oo + x @ beta
        residual = expit(eta) - yy
        loss = np.mean(np.logaddexp(0.0, eta) - yy * eta)
        loss += REGULARIZATION * np.dot(beta[1:], beta[1:]) / 2.0
        gradient = x.T @ residual / len(yy)
        gradient[1:] += REGULARIZATION * beta[1:]
        return float(loss), gradient

    fit = minimize(
        objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
        options={"maxiter": 100, "ftol": 1e-9},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Fusion fit failed: {fit.message}")
    return fit.x.tolist()


def make_arrays(blocks: list[ScoreBlock]) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate(
        [np.column_stack((b.z1, b.z2, b.m1, b.m2)) for b in blocks], axis=0
    )
    labels = np.concatenate([b.label for b in blocks]).astype(np.uint8)
    valid = np.isfinite(features).all(axis=1)
    return features[valid], labels[valid]


def fit_at_origin(history: deque[ScoreBlock], origin_ns: int) -> dict:
    cutoff = origin_ns - PURGE_NS
    calib_start = cutoff - 14 * DAY_NS
    meta_start = cutoff - 7 * DAY_NS
    result = {
        "status": "cold_start_b2_fallback", "cutoff_ns": cutoff,
        "calibration_start_ns": calib_start, "fusion_start_ns": meta_start,
        "calibrators": {}, "arms": {}, "calibration_rows": 0,
        "fusion_rows": 0, "calibration_positives": 0, "fusion_positives": 0,
        "latest_fit_score_ns": None,
    }
    if not history or history[0].time_ns > calib_start + STEP_NS:
        return result
    older = [b for b in history if calib_start <= b.time_ns < meta_start]
    newer = [b for b in history if meta_start <= b.time_ns <= cutoff]
    if not older or not newer:
        return result
    assert max(b.time_ns for b in newer) <= cutoff
    assert max(b.time_ns for b in newer) + HORIZON_NS <= origin_ns
    old_x, old_y = make_arrays(older)
    new_x, new_y = make_arrays(newer)
    result.update({
        "calibration_rows": len(old_y), "fusion_rows": len(new_y),
        "calibration_positives": int(old_y.sum()),
        "fusion_positives": int(new_y.sum()),
        "latest_fit_score_ns": int(newer[-1].time_ns),
    })
    if (min(int(old_y.sum()), int(len(old_y) - old_y.sum()),
            int(new_y.sum()), int(len(new_y) - new_y.sum())) < MIN_POSITIVES):
        return result
    calibrators = {
        name: fitted_calibrator(old_x[:, idx], old_y)
        for idx, name in enumerate(("b1", "b2", "i1", "i2"))
    }
    p1 = calibrate(new_x[:, 0], calibrators["b1"])
    p2 = calibrate(new_x[:, 1], calibrators["b2"])
    pi1 = calibrate(new_x[:, 2], calibrators["i1"])
    pi2 = calibrate(new_x[:, 3], calibrators["i2"])
    offset = clipped_logit(p2)
    arms = {
        arm: fitted_residual(basis(p1, p2, pi1, pi2, arm), offset, new_y)
        for arm in ARM_FEATURES
    }
    result.update({"status": "trained", "calibrators": calibrators, "arms": arms})
    return result


def predict_arms(
    z1: np.ndarray, z2: np.ndarray, m1: np.ndarray, m2: np.ndarray,
    model: dict, raw_b2: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if model["status"] != "trained":
        return {name: raw_b2.copy() for name in ARM_FEATURES}, np.full(len(z1), np.nan), np.full(len(z1), np.nan)
    c = model["calibrators"]
    p1, p2 = calibrate(z1, c["b1"]), calibrate(z2, c["b2"])
    pi1, pi2 = calibrate(m1, c["i1"]), calibrate(m2, c["i2"])
    offset = clipped_logit(p2)
    scores = {
        arm: np.clip(expit(offset + basis(p1, p2, pi1, pi2, arm) @ np.asarray(beta)), EPS, 1.0 - EPS)
        for arm, beta in model["arms"].items()
    }
    return scores, pi1, pi2


def score_metrics(score: np.ndarray, labels: np.ndarray) -> dict:
    times, gpus = labels.shape
    top_idx = np.argsort(-score, axis=1, kind="stable")[:, :K]
    chosen = np.take_along_axis(labels, top_idx, axis=1)
    hits = int(chosen.sum())
    positive_count = int(labels.sum())
    positives_by_time = labels.sum(axis=1)
    valid = positives_by_time > 0
    dcg = chosen @ DISCOUNT
    ndcg_sum = float(np.sum(dcg[valid] / IDEAL_DCG[np.minimum(positives_by_time[valid], K)]))
    return {
        "hits_at_100": hits,
        "positives": positive_count,
        "decision_times": times,
        "gpu_count": gpus,
        "positive_queries": int(valid.sum()),
        "recall_at_100": hits / max(positive_count, 1),
        "ndcg_sum_positive_queries": ndcg_sum,
        "pr_auc": float(average_precision_score(labels.ravel(), score.ravel())) if positive_count else 0.0,
    }


def iso(ns: int | None) -> str | None:
    return pd.Timestamp(ns, unit="ns", tz="UTC").isoformat() if ns is not None else None


def self_test() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=400)
    y = (rng.random(400) < expit(-1.0 + 1.2 * x)).astype(np.uint8)
    cal = fitted_calibrator(x, y)
    p = calibrate(x, cal)
    assert np.isfinite(p).all() and np.all((p > 0) & (p < 1))
    for arm in ARM_FEATURES:
        xx = basis(p, p[::-1], p, p[::-1], arm)
        beta = fitted_residual(xx, clipped_logit(p[::-1]), y)
        assert len(beta) == xx.shape[1]
    prior = np.array([1.0, 2.0])
    current = np.array([3.0, 4.0])
    assert np.array_equal(prior, np.array([1.0, 2.0]))
    assert not np.array_equal(RHO["b1"] * prior + (1 - RHO["b1"]) * current, prior)
    print("Experiment 19 synthetic fit and lag-state self-test passed", flush=True)


def run(source: Path, output: Path, max_origins: int | None) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    source_manifest_path = source / "experiment_manifest.json"
    population_manifest_path = source / "full_population_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    population_manifest = json.loads(population_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETED" or population_manifest.get("status") != "completed":
        raise ValueError("Experiment 14 source is not completed")
    if source_manifest.get("origin_count") != 64 or len(population_manifest["full_parts"]) != 64:
        raise ValueError("Expected 64 Experiment 14 origins")
    if source_manifest.get("target") != "all_xids" or source_manifest.get("purge_hours") != 36:
        raise ValueError("Target or purge contract changed")
    if source_manifest.get("horizon_hours") != 24 or source_manifest.get("decision_interval_seconds") != 1800:
        raise ValueError("Horizon or cadence changed")
    gpu_ids, bins_ns, ground_truth, label_info = reference.read_ground_truth()
    if len(gpu_ids) != GPU_COUNT:
        raise ValueError("GPU population changed")
    parts = [source / name for name in population_manifest["full_parts"]]
    if any(not part.is_file() for part in parts):
        raise FileNotFoundError("An Experiment 14 full-population part is missing")
    selected = parts if max_origins is None else parts[:max_origins]
    output.mkdir(parents=True)
    (output / "full_population").mkdir()
    history: deque[ScoreBlock] = deque()
    memory1: np.ndarray | None = None
    memory2: np.ndarray | None = None
    previous_time_ns: int | None = None
    metrics: list[dict] = []
    traces: list[dict] = []
    parameter_rows: list[dict] = []
    input_hashes: list[dict] = []
    total_rows = total_times = 0
    for origin_idx, part in enumerate(selected):
        columns = ["timestamp", "gpu_uid", "b1_probability", "b2_probability", "probability", "model_state_id", "feature_cutoff_time"]
        frame = pq.read_table(part, columns=columns).to_pandas()
        if len(frame) % GPU_COUNT:
            raise ValueError(f"{part.name}: row count not divisible by GPU count")
        count = len(frame) // GPU_COUNT
        if not np.array_equal(frame["gpu_uid"].astype(str).to_numpy().reshape(count, GPU_COUNT), np.broadcast_to(gpu_ids, (count, GPU_COUNT))):
            raise ValueError(f"{part.name}: GPU order changed")
        if not frame["model_state_id"].eq(f"state_{origin_idx:04d}").all():
            raise ValueError(f"{part.name}: model state changed")
        time_matrix = pd.to_datetime(frame["timestamp"], utc=True).astype("int64").to_numpy().reshape(count, GPU_COUNT)
        if not np.all(time_matrix == time_matrix[:, :1]):
            raise ValueError(f"{part.name}: mixed times")
        times = time_matrix[:, 0]
        cutoff_matrix = pd.to_datetime(frame["feature_cutoff_time"], utc=True).astype("int64").to_numpy().reshape(count, GPU_COUNT)
        if not np.all(time_matrix - cutoff_matrix == 10 * 60 * 1_000_000_000):
            raise ValueError(f"{part.name}: feature cutoff changed")
        if previous_time_ns is not None and times[0] - previous_time_ns != STEP_NS:
            raise ValueError(f"{part.name}: noncontiguous rolling score times")
        if not np.all(np.diff(times) == STEP_NS):
            raise ValueError(f"{part.name}: decision cadence changed")
        bin_idx = np.searchsorted(bins_ns, times)
        if np.any(bin_idx >= len(bins_ns)) or not np.array_equal(bins_ns[bin_idx], times):
            raise ValueError(f"{part.name}: score time off original grid")
        labels = ground_truth[bin_idx].astype(np.uint8)
        b1 = frame["b1_probability"].to_numpy(dtype=np.float64).reshape(count, GPU_COUNT)
        b2 = frame["b2_probability"].to_numpy(dtype=np.float64).reshape(count, GPU_COUNT)
        if not (np.isfinite(b1).all() and np.isfinite(b2).all()):
            raise ValueError(f"{part.name}: invalid Branch score")
        if np.any((b1 < 0) | (b1 > 1) | (b2 < 0) | (b2 > 1)):
            raise ValueError(f"{part.name}: Branch probability outside [0,1]")
        model = fit_at_origin(history, int(times[0]))
        arm_scores = {arm: np.empty_like(b1) for arm in ARM_FEATURES}
        m1_scores = np.full_like(b1, np.nan)
        m2_scores = np.full_like(b2, np.nan)
        pi1_scores = np.full_like(b1, np.nan)
        pi2_scores = np.full_like(b2, np.nan)
        for row, time_ns in enumerate(times):
            z1, z2 = clipped_logit(b1[row]), clipped_logit(b2[row])
            m1 = memory1.copy() if memory1 is not None else np.full(GPU_COUNT, np.nan)
            m2 = memory2.copy() if memory2 is not None else np.full(GPU_COUNT, np.nan)
            if model["status"] == "trained" and (not np.isfinite(m1).all() or not np.isfinite(m2).all()):
                raise ValueError("Trained Fusion has no lagged state")
            scores, pi1, pi2 = predict_arms(z1, z2, m1, m2, model, b2[row])
            for arm in ARM_FEATURES:
                arm_scores[arm][row] = scores[arm]
            m1_scores[row], m2_scores[row] = m1, m2
            pi1_scores[row], pi2_scores[row] = pi1, pi2
            history.append(ScoreBlock(int(time_ns), z1, z2, m1, m2, labels[row]))
            memory1 = z1.copy() if memory1 is None else RHO["b1"] * memory1 + (1 - RHO["b1"]) * z1
            memory2 = z2.copy() if memory2 is None else RHO["b2"] * memory2 + (1 - RHO["b2"]) * z2
            previous_time_ns = int(time_ns)
        keep_after = int(times[-1]) - PURGE_NS - 14 * DAY_NS - STEP_NS
        while history and history[0].time_ns < keep_after:
            history.popleft()
        out = pd.DataFrame({
            "timestamp": frame["timestamp"], "gpu_uid": frame["gpu_uid"],
            "model_state_id": frame["model_state_id"],
            "feature_cutoff_time": frame["feature_cutoff_time"],
            "b1_probability": b1.ravel(), "b2_probability": b2.ravel(),
            "experiment14_fusion_probability": frame["probability"].to_numpy(),
            "b1_memory_logit": m1_scores.ravel(), "b2_memory_logit": m2_scores.ravel(),
            "b1_inertia_probability": pi1_scores.ravel(), "b2_inertia_probability": pi2_scores.ravel(),
            **{SCORE_COLUMNS[arm]: values.ravel() for arm, values in arm_scores.items()},
        })
        out.to_parquet(output / "full_population" / part.name, index=False)
        input_hashes.append({"part": part.name, "sha256": sha256(part)})
        for name, score in {"B1": b1, "B2": b2, "Experiment14_Fusion": frame["probability"].to_numpy().reshape(count, GPU_COUNT), **arm_scores}.items():
            metrics.append({"origin_idx": origin_idx, "origin_time": iso(int(times[0])), "model": name, **score_metrics(score, labels)})
        for arm in ARM_FEATURES:
            traces.append({
                "origin_idx": origin_idx, "origin_time": iso(int(times[0])),
                "arm": arm, "status": model["status"],
                "cutoff_time": iso(model["cutoff_ns"]),
                "calibration_start_time": iso(model["calibration_start_ns"]),
                "fusion_start_time": iso(model["fusion_start_ns"]),
                "latest_fit_score_time": iso(model["latest_fit_score_ns"]),
                "calibration_rows": model["calibration_rows"],
                "fusion_rows": model["fusion_rows"],
                "calibration_positives": model["calibration_positives"],
                "fusion_positives": model["fusion_positives"],
                "coefficient_count": len(model["arms"].get(arm, [])),
            })
        parameter_rows.append({"origin_idx": origin_idx, "origin_time": iso(int(times[0])), **{
            key: value for key, value in model.items() if not key.endswith("_ns")
        }})
        total_rows += len(frame)
        total_times += count
        print(f"origin {origin_idx + 1}/{len(selected)}: {model['status']}, rows={len(frame)}, fit_rows={model['fusion_rows']}", flush=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output / "origin_metrics.csv", index=False)
    pd.DataFrame(traces).to_csv(output / "fit_trace.csv", index=False)
    (output / "model_parameters.json").write_text(json.dumps(parameter_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "input_hashes.json").write_text(json.dumps(input_hashes, indent=2), encoding="utf-8")
    summary = {}
    for model_name, group in metrics_df.groupby("model", sort=False):
        summary[model_name] = {
            "hits_at_100": int(group["hits_at_100"].sum()),
            "pooled_recall_at_100": float(group["hits_at_100"].sum() / group["positives"].sum()),
            "pooled_positive_query_ndcg_at_100": float(group["ndcg_sum_positive_queries"].sum() / group["positive_queries"].sum()),
            "origin_mean_pr_auc": float(group["pr_auc"].mean()),
        }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "status": "COMPLETED" if len(selected) == 64 else "SMOKE_PARTIAL",
        "runner": Path(__file__).name, "runner_sha256": sha256(Path(__file__)),
        "source": str(source), "source_manifest_sha256": sha256(source_manifest_path),
        "source_full_population_manifest_sha256": sha256(population_manifest_path),
        "source_parts": input_hashes, "target": "all_xids", "horizon_hours": 24,
        "purge_hours": 36, "feature_cutoff_minutes": 10,
        "warmup_days": 30, "origin_count": len(selected), "decision_time_count": total_times,
        "score_rows": total_rows, "gpu_count": GPU_COUNT, "decision_interval_seconds": 1800,
        "branch1_branch2": "immutable_Experiment14_fitted_scores",
        "newly_fitted": "four_probability_residual_fusion_arms_and_four_calibrators_per_eligible_origin",
        "fusion_fit_origin_count": int(sum(row["status"] == "trained" for row in parameter_rows)),
        "fusion_fit_count": int(sum(row["status"] == "trained" for row in traces)),
        "calibration_fit_count": 4 * int(sum(row["status"] == "trained" for row in parameter_rows)),
        "inertia_half_life_hours": HALF_LIFE_HOURS,
        "inertia_uses_current_score": False,
        "calibration_window_days": 7, "fusion_window_days": 7,
        "calibration_then_fusion_disjoint": True,
        "regularization": REGULARIZATION, "minimum_positives_per_window": MIN_POSITIVES,
        "negative_sampling": False, "seed": SEED,
        "label_info": label_info, "terminal_heldout_used": False,
        "disagreement_feature_used": False, "blox_used": False,
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-origins", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if args.max_origins is not None and not 1 <= args.max_origins <= 64:
        parser.error("--max-origins must be between 1 and 64")
    with threadpool_limits(limits=args.threads):
        run(args.source.resolve(), args.output.resolve(), args.max_origins)


if __name__ == "__main__":
    main()

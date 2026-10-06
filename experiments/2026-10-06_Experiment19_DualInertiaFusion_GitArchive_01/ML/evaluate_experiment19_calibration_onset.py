"""Operational evaluation of the frozen Experiment 19 full-population tape.

No fitting, parameter choice, or threshold tuning occurs here. The onset-event
eligibility contract matches evaluate_experiment15_boundary_swap.py.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import evaluate_experiment12_top100 as reference


ARCHIVE = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT = ARCHIVE / "results"
GPU_COUNT = 1992
K = 100
DAY_NS = 24 * 60 * 60 * 1_000_000_000
BUFFER_NS = 10 * 60 * 1_000_000_000
SCORE_COLUMNS = {
    "B1": "b1_probability",
    "B2": "b2_probability",
    "Experiment14_Fusion": "experiment14_fusion_probability",
    "no_inertia": "fusion_no_inertia",
    "b1_inertia": "fusion_b1_inertia",
    "b2_inertia": "fusion_b2_inertia",
    "dual_inertia": "fusion_dual_inertia",
}


def evaluate(result: Path) -> dict:
    manifest = json.loads((result / "experiment_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETED" or manifest["origin_count"] != 64:
        raise ValueError("Operational metrics require a completed 64-origin run")
    gpu_ids, bins_ns, truth, label_metadata = reference.read_ground_truth()
    if len(gpu_ids) != GPU_COUNT:
        raise ValueError("Unexpected GPU population")
    sums = {name: {"squared_error": 0.0, "log_loss": 0.0, "prediction": 0.0} for name in SCORE_COLUMNS}
    top_chunks: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_COLUMNS}
    times_chunks: list[np.ndarray] = []
    total_labels = total_rows = 0
    columns = ["timestamp", "gpu_uid", *SCORE_COLUMNS.values()]
    for origin in range(64):
        path = result / "full_population" / f"score_part_{origin:04d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pq.read_table(path, columns=columns).to_pandas()
        if len(frame) % GPU_COUNT:
            raise ValueError(f"Origin {origin} has incomplete GPU rows")
        count = len(frame) // GPU_COUNT
        observed_gpu = frame["gpu_uid"].astype(str).to_numpy().reshape(count, GPU_COUNT)
        if not np.array_equal(observed_gpu, np.broadcast_to(gpu_ids, observed_gpu.shape)):
            raise ValueError(f"Origin {origin} changed GPU ordering")
        observed_times = pd.to_datetime(frame["timestamp"], utc=True).astype("int64").to_numpy().reshape(count, GPU_COUNT)
        if not np.all(observed_times == observed_times[:, :1]):
            raise ValueError(f"Origin {origin} has mixed query times")
        times = observed_times[:, 0]
        bins = np.searchsorted(bins_ns, times)
        if np.any(bins >= len(bins_ns)) or not np.array_equal(bins_ns[bins], times):
            raise ValueError(f"Origin {origin} is not on the ground-truth grid")
        labels = truth[bins].astype(np.float64)
        total_labels += int(labels.sum())
        total_rows += len(frame)
        times_chunks.append(times)
        for name, column in SCORE_COLUMNS.items():
            score = frame[column].to_numpy(dtype=np.float64).reshape(count, GPU_COUNT)
            if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
                raise ValueError(f"Invalid probability at {origin}/{name}")
            safe = np.clip(score, 1e-15, 1 - 1e-15)
            sums[name]["squared_error"] += float(np.square(score - labels).sum())
            sums[name]["log_loss"] += float((-labels * np.log(safe) - (1 - labels) * np.log1p(-safe)).sum())
            sums[name]["prediction"] += float(score.sum())
            top_chunks[name].append(np.argsort(-score, axis=1, kind="stable")[:, :K].astype(np.uint16))
        del frame, observed_gpu, observed_times, labels, score, safe
        gc.collect()
        print(f"operational origin {origin + 1}/64", flush=True)
    if total_rows != 6_081_576:
        raise ValueError(f"Full-population row count changed: {total_rows}")
    times = np.concatenate(times_chunks)
    if len(times) != 3053 or not np.all(np.diff(times) == 30 * 60 * 1_000_000_000):
        raise ValueError("The 30-minute decision-time grid has changed")
    ledger = pq.read_table(reference.ONSET_LEDGER, columns=["gpu_id", "onset_time", "xid_code"]).to_pandas()
    unique = ledger.drop_duplicates(["gpu_id", "onset_time"])
    onset_ns = pd.to_datetime(unique["onset_time"], utc=True).astype("int64").to_numpy()
    gpu_map = {str(gpu): idx for idx, gpu in enumerate(gpu_ids)}
    gpu_indexes = unique["gpu_id"].map(gpu_map).to_numpy()
    eligible: list[tuple[int, int, int, int]] = []
    for ns, gpu in zip(onset_ns, gpu_indexes):
        if pd.isna(gpu) or ns < times[0] + DAY_NS or ns > times[-1] + BUFFER_NS:
            continue
        start = int(np.searchsorted(times, ns - DAY_NS, side="left"))
        stop = int(np.searchsorted(times, ns - BUFFER_NS, side="right"))
        if stop > start:
            eligible.append((start, stop, int(gpu), int(ns)))
    output = {
        "status": "COMPLETED",
        "source_experiment": str(result),
        "rows": total_rows,
        "decision_times": len(times),
        "positive_labels": total_labels,
        "label_metadata": label_metadata,
        "onset_definition": "Unique (gpu_id,onset_time); full preceding 24h scored; alert no later than onset-10min; not reconstructed failure episodes",
        "eligible_onset_events": len(eligible),
        "raw_onset_rows": len(ledger),
        "unique_onsets": len(unique),
        "models": {},
    }
    for name in SCORE_COLUMNS:
        top = np.concatenate(top_chunks[name])
        alert = np.zeros((len(times), GPU_COUNT), dtype=np.bool_)
        alert[np.arange(len(times))[:, None], top] = True
        first_lead_hours = []
        captured = 0
        for start, stop, gpu, onset in eligible:
            alerts = np.flatnonzero(alert[start:stop, gpu])
            if len(alerts):
                captured += 1
                first_lead_hours.append((onset - int(times[start + alerts[0]])) / (60 * 60 * 1_000_000_000))
        leads = np.asarray(first_lead_hours)
        output["models"][name] = {
            "brier_score": sums[name]["squared_error"] / total_rows,
            "log_loss": sums[name]["log_loss"] / total_rows,
            "mean_predicted_probability": sums[name]["prediction"] / total_rows,
            "observed_prevalence": total_labels / total_rows,
            "captured_onset_events": captured,
            "onset_event_capture_rate": captured / len(eligible) if eligible else None,
            "captured_event_first_warning_lead_hours": {
                "median": float(np.median(leads)) if captured else None,
                "p10": float(np.quantile(leads, 0.1)) if captured else None,
                "p90": float(np.quantile(leads, 0.9)) if captured else None,
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    output = args.result / "operational_metrics.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    metrics = evaluate(args.result)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

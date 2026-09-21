"""Branch 1 performance-triggered dynamic training-window experiment.

This runner preserves the GitHub-base data/model contracts and applies the
dynamic training-window controller to Branch 1 only.  Branch 2 remains the
approved strict History-only protocol (7-day training window, no recency
weighting).  The controller uses recent purged validation performance rather
than ADWIN, disagreement, or prevalence triggers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_bidirectional_adst_fusion as base
import run_sliding_adst_v2_1_pooled_adwin as pooled
import run_sliding_adst_v2_branch_specific as v2


STEP_NS = pooled.STEP_NS
DAY_NS = pooled.DAY_NS
HOUR_NS = pooled.HOUR_NS
PURGE_BINS = pooled.PURGE_BINS
VALIDATION_BINS = pooled.VALIDATION_BINS
VALIDATION_BLOCKS = pooled.VALIDATION_BLOCKS
TRAIN_DAYS_GRID = pooled.TRAIN_DAYS_GRID
OBS_HOURS_GRID = pooled.OBS_HOURS_GRID
HALF_LIFE_GRID = pooled.HALF_LIFE_GRID
LAMBDA_GRID = pooled.LAMBDA_GRID
TEST_FRACTION = pooled.TEST_FRACTION
POOL_ORIGINS = pooled.POOL_ORIGINS

B1_FIXED_HALF_LIFE_DAYS = 7
B2_FIXED_TRAIN_DAYS = 7
B2_FIXED_HALF_LIFE_DAYS = 0
WINDOW_CHANGE_RELATIVE_MARGIN = 0.05
WINDOW_CHANGE_ABSOLUTE_MARGIN = 0.002
WINDOW_CHANGE_Q25_TOLERANCE = 0.002
WINDOW_CONFIRMATION_CHECKPOINTS = 2


def _time_text(engine: base.UnifiedDataEngine, bin_index: int) -> str:
    return pd.Timestamp(
        engine.bin_start_ns[int(bin_index)], unit="ns", tz="UTC"
    ).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
    }


class Branch1PerformanceWindowADST(pooled.PooledBatchADST):
    """Pooled v2.1 execution with a validation-only Branch 1 controller."""

    @staticmethod
    def _performance_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["val_pr_auc_mean"]),
            float(row["val_pr_auc_median"]),
            float(row["val_pr_auc_q25"]),
            -float(row["val_pr_auc_std"]),
        )

    @staticmethod
    def _candidate_margin(current_mean: float) -> float:
        return max(
            WINDOW_CHANGE_ABSOLUTE_MARGIN,
            WINDOW_CHANGE_RELATIVE_MARGIN * max(abs(current_mean), 1e-6),
        )

    def _choose_branch(
        self,
        rows: list[dict[str, Any]],
        previous: dict[str, Any] | None,
        train_key: str,
        half_key: str,
    ) -> dict[str, Any]:
        """Choose B1 dynamically and keep the approved B2 control fixed."""
        if train_key == "candidate_b2_train_days":
            fixed = [
                row
                for row in rows
                if int(row.get(train_key, -1)) == B2_FIXED_TRAIN_DAYS
                and int(row.get(half_key, -1)) == B2_FIXED_HALF_LIFE_DAYS
            ]
            if not fixed:
                raise ValueError("Fixed Branch 2 History-only candidate is unavailable.")
            selected = fixed[0].copy()
            selected.update(
                {
                    "b2_window_action": "fixed_control",
                    "b2_window_pending_train_days": None,
                    "b2_window_change_streak": 0,
                }
            )
            return selected

        candidates = [
            row
            for row in rows
            if int(row.get(half_key, -1)) == B1_FIXED_HALF_LIFE_DAYS
        ]
        if not candidates:
            raise ValueError("No Branch 1 candidates with the fixed 7-day half-life.")

        best = max(candidates, key=self._performance_key)
        previous_train = None if previous is None else previous.get("selected_b1_train_days")
        if previous_train is None:
            selected = best.copy()
            selected.update(
                {
                    "b1_window_action": "initial",
                    "b1_window_pending_train_days": None,
                    "b1_window_change_streak": 0,
                    "b1_window_gain": None,
                }
            )
            return selected

        current_train = int(previous_train)
        current_rows = [
            row for row in candidates if int(row[train_key]) == current_train
        ]
        if not current_rows:
            selected = best.copy()
            selected.update(
                {
                    "b1_window_action": "recover_to_available_candidate",
                    "b1_window_pending_train_days": None,
                    "b1_window_change_streak": 0,
                    "b1_window_gain": None,
                }
            )
            return selected
        current = current_rows[0]
        current_mean = float(current["val_pr_auc_mean"])
        required_gain = self._candidate_margin(current_mean)

        neighbors = [
            row
            for row in candidates
            if abs(int(row[train_key]) - current_train) > 0
            and abs(int(row[train_key]) - current_train)
            == min(
                abs(int(item[train_key]) - current_train)
                for item in candidates
                if abs(int(item[train_key]) - current_train) > 0
            )
        ]
        eligible = [
            row
            for row in neighbors
            if float(row["val_pr_auc_mean"]) - current_mean >= required_gain
            and float(row["val_pr_auc_q25"]) + WINDOW_CHANGE_Q25_TOLERANCE
            >= float(current["val_pr_auc_q25"])
        ]
        proposed = max(eligible, key=self._performance_key) if eligible else None
        pending = None if previous is None else previous.get("b1_window_pending_train_days")
        streak = int(previous.get("b1_window_change_streak", 0)) if previous else 0

        if proposed is None:
            selected = current.copy()
            selected.update(
                {
                    "b1_window_action": "hold",
                    "b1_window_pending_train_days": None,
                    "b1_window_change_streak": 0,
                    "b1_window_gain": 0.0,
                }
            )
            return selected

        proposed_train = int(proposed[train_key])
        streak = streak + 1 if pending == proposed_train else 1
        if streak >= WINDOW_CONFIRMATION_CHECKPOINTS:
            direction = "expand" if proposed_train > current_train else "shrink"
            selected = proposed.copy()
            selected.update(
                {
                    "b1_window_action": direction,
                    "b1_window_pending_train_days": None,
                    "b1_window_change_streak": 0,
                    "b1_window_gain": float(
                        float(proposed["val_pr_auc_mean"]) - current_mean
                    ),
                }
            )
            return selected

        selected = current.copy()
        selected.update(
            {
                "b1_window_action": "await_confirmation",
                "b1_window_pending_train_days": proposed_train,
                "b1_window_change_streak": streak,
                "b1_window_gain": float(
                    float(proposed["val_pr_auc_mean"]) - current_mean
                ),
            }
        )
        return selected

    def _compose_selection(
        self,
        contexts: list[dict[str, Any]],
        b1_best: dict[str, Any],
        b2_best: dict[str, Any],
        branch_rows: list[dict[str, Any]],
        previous: dict[str, Any] | None,
        stage: str,
        origin_bin: int,
        validation_mode: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Compose validation selection without ADWIN state or triggers."""
        b1_train = int(b1_best["candidate_b1_train_days"])
        b1_half = int(b1_best["candidate_b1_half_life_days"])
        b2_train = int(b2_best["candidate_b2_train_days"])
        b2_half = int(b2_best["candidate_b2_half_life_days"])

        block_refs: list[dict[str, Any]] = []
        input_aps: dict[str, list[float]] = {str(obs): [] for obs in OBS_HOURS_GRID}
        for context in contexts:
            for block in context["blocks"]:
                labels = block["val_labels"]
                for obs in OBS_HOURS_GRID:
                    input_aps[str(obs)].append(
                        base.safe_average_precision(labels, block["b1_preds"][obs])
                    )
                block_refs.append(block)

        input_mean = {
            key: float(np.mean(values)) for key, values in input_aps.items()
        }
        raw_weights = {
            key: 0.20 + max(value, 0.0) for key, value in input_mean.items()
        }
        total = sum(raw_weights.values())
        b1_weights = {key: float(value / total) for key, value in raw_weights.items()}

        b1_selected: list[np.ndarray] = []
        b2_selected: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        b1_values: list[float] = []
        b2_values: list[float] = []
        b1_rocs: list[float] = []
        b2_rocs: list[float] = []
        for block in block_refs:
            labels = block["val_labels"]
            b1 = sum(
                b1_weights[str(obs)] * block["b1_preds"][obs]
                for obs in OBS_HOURS_GRID
            )
            b2 = block["b2_pred"]
            labels_list.append(labels)
            b1_selected.append(b1)
            b2_selected.append(b2)
            b1_values.append(base.safe_average_precision(labels, b1))
            b2_values.append(base.safe_average_precision(labels, b2))
            b1_rocs.append(base.safe_roc_auc(labels, b1))
            b2_rocs.append(base.safe_roc_auc(labels, b2))

        fusion_rows: list[dict[str, Any]] = []
        for lambda_value in LAMBDA_GRID:
            aps: list[float] = []
            rocs: list[float] = []
            for labels, b1, b2 in zip(labels_list, b1_selected, b2_selected):
                fused = lambda_value * b1 + (1.0 - lambda_value) * b2
                aps.append(base.safe_average_precision(labels, fused))
                rocs.append(base.safe_roc_auc(labels, fused))
            stats = _stats(aps)
            fusion_rows.append(
                {
                    "selection_role": "fusion",
                    "candidate_lambda": float(lambda_value),
                    "val_pr_auc_mean": stats["mean"],
                    "val_pr_auc_median": stats["median"],
                    "val_pr_auc_q25": stats["q25"],
                    "val_pr_auc_std": stats["std"],
                    "val_pr_auc_min": stats["min"],
                    "val_roc_auc_mean": float(np.mean(rocs)),
                }
            )
        fusion_best = max(
            fusion_rows,
            key=lambda row: (
                float(row["val_pr_auc_mean"]),
                float(row["val_pr_auc_median"]),
                float(row["val_pr_auc_q25"]),
                -float(row["val_pr_auc_std"]),
                -abs(float(row["candidate_lambda"]) - 0.5),
            ),
        )
        selected_lambda = float(fusion_best["candidate_lambda"])

        latest_context = max(contexts, key=lambda item: int(item["origin_bin"]))
        latest_block = latest_context["blocks"][0]
        latest_b1 = sum(
            b1_weights[str(obs)] * latest_block["b1_preds"][obs]
            for obs in OBS_HOURS_GRID
        )
        latest_b2 = latest_block["b2_pred"]
        latest_fused = selected_lambda * latest_b1 + (1.0 - selected_lambda) * latest_b2
        latest_ap = base.safe_average_precision(latest_block["val_labels"], latest_fused)
        latest_disagreement = float(np.mean(np.abs(latest_b1 - latest_b2)))
        latest_prevalence = float(np.mean(latest_block["val_labels"]))

        common = {
            "selection_stage": stage,
            "selection_origin_idx": int(origin_bin),
            "selection_origin_time": _time_text(self.engine, origin_bin),
            "purge_hours": int(base.PURGE_NS // HOUR_NS),
            "validation_mode": validation_mode,
            "pool_origin_count": len(contexts),
            "pool_validation_block_count": len(block_refs),
            "adwin_enabled": False,
            "dynamic_selector": "branch1_validation_performance",
        }
        audit_rows: list[dict[str, Any]] = []
        for row in branch_rows + fusion_rows:
            audit_row = dict(common)
            audit_row.update(row)
            audit_rows.append(audit_row)

        previous_latest = None if previous is None else previous.get("latest_val_pr_auc")
        selected = {
            **common,
            "train_end_time": _time_text(self.engine, latest_block["train_end"]),
            "validation_start_time": _time_text(
                self.engine, latest_context["blocks"][-1]["val_start"]
            ),
            "validation_end_time": _time_text(self.engine, latest_block["val_end"]),
            "validation_block_count": len(block_refs),
            "validation_block_days": VALIDATION_BINS * base.STEP_MINUTES / (24 * 60),
            "selected_b1_train_days": b1_train,
            "selected_b1_half_life_days": b1_half,
            "selected_b1_input_hours": list(OBS_HOURS_GRID),
            "selected_b1_input_weights": b1_weights,
            "selected_b2_train_days": b2_train,
            "selected_b2_half_life_days": b2_half,
            "selected_L_train_days": b1_train,
            "selected_L_obs_hours": -1,
            "selected_lambda": selected_lambda,
            "val_pr_auc": float(fusion_best["val_pr_auc_mean"]),
            "val_roc_auc": float(fusion_best["val_roc_auc_mean"]),
            "val_pr_auc_b1": float(np.mean(b1_values)),
            "val_pr_auc_b2": float(np.mean(b2_values)),
            "val_pr_auc_b1_median": float(np.median(b1_values)),
            "val_pr_auc_b2_median": float(np.median(b2_values)),
            "val_roc_auc_b1": float(np.mean(b1_rocs)),
            "val_roc_auc_b2": float(np.mean(b2_rocs)),
            "prediction_disagreement": latest_disagreement,
            "validation_positive_rate": latest_prevalence,
            "latest_val_pr_auc": float(latest_ap),
            "adst_action": b1_best.get("b1_window_action", "hold"),
            "b1_window_action": b1_best.get("b1_window_action", "hold"),
            "b1_window_pending_train_days": b1_best.get(
                "b1_window_pending_train_days"
            ),
            "b1_window_change_streak": int(
                b1_best.get("b1_window_change_streak", 0)
            ),
            "b1_window_gain": b1_best.get("b1_window_gain"),
            "b2_window_action": b2_best.get("b2_window_action", "fixed_control"),
            "previous_val_pr_auc": previous_latest,
            "val_pr_auc_delta": None
            if previous_latest is None
            else float(latest_ap - float(previous_latest)),
            "candidate_count": len(audit_rows),
            "branch1_input_validation_ap": input_mean,
            "window_change_relative_margin": WINDOW_CHANGE_RELATIVE_MARGIN,
            "window_change_absolute_margin": WINDOW_CHANGE_ABSOLUTE_MARGIN,
            "window_change_q25_tolerance": WINDOW_CHANGE_Q25_TOLERANCE,
            "window_confirmation_checkpoints": WINDOW_CONFIRMATION_CHECKPOINTS,
        }
        return selected, audit_rows

    def _generate_summary_report(
        self,
        metrics_df: pd.DataFrame,
        selection_df: pd.DataFrame,
        split_info: dict[str, Any],
    ) -> None:
        def weighted(column: str) -> float:
            weights = metrics_df["positives"].to_numpy(dtype=float)
            return float(
                np.average(
                    metrics_df[column].to_numpy(dtype=float),
                    weights=np.maximum(weights, 1.0),
                )
            )

        changes = 0
        if "b1_window_action" in selection_df:
            changes = int(
                selection_df["b1_window_action"].isin(["shrink", "expand"]).sum()
            )
        lines = [
            "# Branch 1 Performance-Triggered Dynamic Window 보고서",
            "",
            "## Material Passport",
            "- **Status**: COMPLETED",
            f"- **Runner**: `{Path(__file__).name}`",
            "- **Target**: All-XID unified onset, 24-hour horizon",
            "- **Branch 1**: 1h·6h·24h telemetry parallel ensemble",
            "- **Branch 2**: strict History-only, fixed 7-day window, no recency",
            "- **Selector**: Branch 1 validation-performance controller; ADWIN disabled",
            "- **Split**: chronological Sliding Training, 36-hour purge, fixed terminal held-out test",
            "",
            "## Terminal held-out mean",
            f"- Fused PR-AUC: `{metrics_df['pr_auc_fused'].mean():.6f}`",
            f"- Branch 1 PR-AUC: `{metrics_df['pr_auc_b1'].mean():.6f}`",
            f"- Branch 2 PR-AUC: `{metrics_df['pr_auc_b2'].mean():.6f}`",
            f"- Fused normalized PR-AUC: `{metrics_df['pr_auc_fused_normalized'].mean():.6f}`",
            f"- Fused ROC-AUC: `{metrics_df['roc_auc_fused'].mean():.6f}`",
            f"- Recall@100: `{metrics_df['recall_at_100'].mean():.2%}`",
            f"- Lift@100: `{metrics_df['lift_at_100'].mean():.3f}x`",
            f"- B2 positive-weighted PR-AUC: `{weighted('pr_auc_b2'):.6f}`",
            "",
            "## Dynamic window behavior",
            f"- Branch 1 confirmed window changes: `{changes}`",
            f"- Candidate windows: `{list(TRAIN_DAYS_GRID)}` days",
            f"- Fixed Branch 1 half-life: `{B1_FIXED_HALF_LIFE_DAYS}` days",
            f"- Confirmation requirement: `{WINDOW_CONFIRMATION_CHECKPOINTS}` checkpoints",
            f"- Terminal interval: `{split_info['test_start_time']}` ~ `{split_info['test_end_time']}`",
            "- Held-out labels were used only for final reporting, never for window selection.",
            "",
            "## Interpretation boundary",
            "- This experiment tests whether Branch 1 benefits from validation-performance-based training-window changes.",
            "- Branch 2 is a fixed control and is not dynamically adapted.",
            "- The terminal held-out model is fitted once after final selection; this is not test-time adaptive retraining.",
        ]
        (self.output_dir / "branch1_dynamic_window_report.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        started = time.time()
        base.seed_everything(self.seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        warmup_bins, test_start, test_end = self.split_bounds()
        cadence_bins = int(self.retrain_cadence_hours * 60 // base.STEP_MINUTES)
        min_origin = (
            warmup_bins
            + int(21 * DAY_NS // STEP_NS)
            + 2 * PURGE_BINS
            + VALIDATION_BLOCKS * VALIDATION_BINS
        )
        dev_origins = np.arange(min_origin, test_start, cadence_bins, dtype=np.int32)
        self.pool_origin_bins = [int(value) for value in dev_origins[-POOL_ORIGINS:]]
        selection_rows: list[dict[str, Any]] = []
        development_rows: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None

        print("[1/5] Branch 1 performance-dynamic development selection", flush=True)
        print(
            f"  development origins={len(dev_origins):,}, pooled origins={len(self.pool_origin_bins)}, "
            f"validation blocks={VALIDATION_BLOCKS}, purge={int(base.PURGE_NS // HOUR_NS)}h",
            flush=True,
        )
        for selection_idx, origin_bin in enumerate(dev_origins):
            name = f"dev_{selection_idx:04d}"
            loaded = self._load_selection_checkpoint(name)
            if loaded is None:
                base.seed_everything(self.seed + selection_idx)
                best, candidates = self.select_adst_config(
                    int(origin_bin),
                    np.random.default_rng(self.seed + selection_idx),
                    previous,
                    "development",
                )
                self._save_selection_checkpoint(name, best, candidates)
            else:
                best, candidates = loaded
            selection_rows.extend(candidates)
            development_rows.extend(candidates)
            previous = best
            print(
                f"  [{selection_idx + 1}/{len(dev_origins)}] {best['selection_origin_time']} "
                f"-> B1={best['selected_b1_train_days']}d/{best['selected_b1_half_life_days']}d, "
                f"B2={best['selected_b2_train_days']}d/fixed, lambda={best['selected_lambda']:.1f}, "
                f"val PR-AUC={float(best['val_pr_auc']):.4f}, action={best['b1_window_action']}",
                flush=True,
            )

        self.pool_rows = [
            row
            for row in development_rows
            if int(row.get("selection_origin_idx", -1)) in set(self.pool_origin_bins)
        ]
        final_loaded = self._load_selection_checkpoint("final")
        if final_loaded is None:
            final_selected, final_candidates = self.select_adst_config(
                int(test_start),
                np.random.default_rng(self.seed + 900_000),
                previous,
                "heldout_preselection",
            )
            self._save_selection_checkpoint("final", final_selected, final_candidates)
        else:
            final_selected, final_candidates = final_loaded
        selection_rows.extend(final_candidates)
        selection_rows.append(final_selected)
        selection_df = pd.DataFrame(selection_rows)
        selection_df.to_csv(
            self.output_dir / "branch1_dynamic_window_selection_history.csv", index=False
        )

        split_info = {
            "warmup_start_time": _time_text(self.engine, warmup_bins),
            "test_start_time": _time_text(self.engine, test_start),
            "test_end_time": _time_text(self.engine, test_end - 1),
        }
        _write_json_atomic(
            self.output_dir / "experiment_manifest.json",
            {
                "runner": Path(__file__).name,
                "base_runner": "ML/run_bidirectional_adst_fusion.py",
                "reference_runner": "ML/run_sliding_adst_v2_1_pooled_adwin.py",
                "target": "all_xids",
                "branch1_dynamic_window": True,
                "branch1_train_days_grid": list(TRAIN_DAYS_GRID),
                "branch1_half_life_days": B1_FIXED_HALF_LIFE_DAYS,
                "branch1_input_hours": list(OBS_HOURS_GRID),
                "branch1_parallel_ensemble": True,
                "branch2_contract": "history_only_2026_09_10_fixed_control",
                "branch2_train_days": B2_FIXED_TRAIN_DAYS,
                "branch2_half_life_days": B2_FIXED_HALF_LIFE_DAYS,
                "branch2_features": ["xid_count_30d", "days_since_xid"],
                "validation_mode": "recent_3_rolling_blocks_for_dynamic_selection",
                "final_selection_mode": "pooled_latest_6_development_origins",
                "window_change_relative_margin": WINDOW_CHANGE_RELATIVE_MARGIN,
                "window_change_absolute_margin": WINDOW_CHANGE_ABSOLUTE_MARGIN,
                "window_change_q25_tolerance": WINDOW_CHANGE_Q25_TOLERANCE,
                "window_confirmation_checkpoints": WINDOW_CONFIRMATION_CHECKPOINTS,
                "adwin_enabled": False,
                "disagreement_or_prevalence_trigger_enabled": False,
                "retrain_cadence_hours": self.retrain_cadence_hours,
                "negative_ratio": self.negative_ratio,
                "test_stride_bins": self.test_stride_bins,
                "seed": self.seed,
                "purge_hours": int(base.PURGE_NS // HOUR_NS),
                "heldout_test_fraction": TEST_FRACTION,
                "lambda_grid": [float(value) for value in LAMBDA_GRID],
                "warmup_start_time": split_info["warmup_start_time"],
                "test_start_time": split_info["test_start_time"],
                "test_end_time": split_info["test_end_time"],
                "pool_origin_times": [
                    _time_text(self.engine, value) for value in self.pool_origin_bins
                ],
                "final_selection": final_selected,
                "command": " ".join(sys.argv),
            },
        )

        print("[2/5] Fitting final branch-specific models", flush=True)
        models = self._fit_final_models(final_selected, test_start, model_seed=900_001)
        print(
            f"  B1 L={final_selected['selected_b1_train_days']}d/{final_selected['selected_b1_half_life_days']}d, "
            f"B2 L={final_selected['selected_b2_train_days']}d/fixed, "
            f"lambda={final_selected['selected_lambda']:.1f}",
            flush=True,
        )
        metrics_rows: list[dict[str, Any]] = []
        tapes: list[pd.DataFrame] = []
        cycle_starts = np.arange(test_start, test_end, cadence_bins, dtype=np.int32)
        print(f"[3/5] Evaluating {len(cycle_starts):,} common held-out cycles", flush=True)
        for cycle_idx, cycle_start in enumerate(cycle_starts):
            cycle_end = min(int(cycle_start) + cadence_bins, test_end)
            loaded_cycle = self._load_cycle_checkpoint(cycle_idx)
            if loaded_cycle is None:
                metrics, tape = self._predict_cycle(
                    models, final_selected, cycle_idx, int(cycle_start), cycle_end
                )
                self._save_cycle_checkpoint(cycle_idx, metrics, tape)
            else:
                metrics, tape = loaded_cycle
            metrics_rows.append(metrics)
            tapes.append(tape)
            print(
                f"  [{cycle_idx + 1}/{len(cycle_starts)}] PR-AUC fused={float(metrics['pr_auc_fused']):.4f}, "
                f"B1={float(metrics['pr_auc_b1']):.4f}, B2={float(metrics['pr_auc_b2']):.4f}",
                flush=True,
            )
        metrics_df = pd.DataFrame(metrics_rows)
        tape_df = pd.concat(tapes, ignore_index=True) if tapes else pd.DataFrame()
        metrics_df.to_csv(self.output_dir / "branch1_dynamic_window_metrics.csv", index=False)
        tape_df.to_parquet(
            self.output_dir / "branch1_dynamic_window_risk_tape.parquet",
            index=False,
            compression="zstd",
        )
        self._generate_summary_report(metrics_df, selection_df, split_info)
        print(f"[4/5] Saved outputs in {self.output_dir}", flush=True)
        print(f"[5/5] Elapsed: {(time.time() - started) / 60.0:.1f} minutes", flush=True)
        return metrics_df, tape_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Branch 1 validation-performance dynamic training-window ADST"
    )
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/2026-09-14_All-XID_Branch1_PerformanceDynamicWindow_01",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = base.find_data_dir()
    cache_dir = base.PROJECT_ROOT / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    output_dir = base.PROJECT_ROOT / args.output_dir
    engine = base.UnifiedDataEngine(data_dir=data_dir, cache_dir=cache_dir)
    _, history_map, gt_matrix = engine.load_all_xid_ledger()
    pipeline = Branch1PerformanceWindowADST(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        branch2_mode="history_0910",
        resume=args.resume,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

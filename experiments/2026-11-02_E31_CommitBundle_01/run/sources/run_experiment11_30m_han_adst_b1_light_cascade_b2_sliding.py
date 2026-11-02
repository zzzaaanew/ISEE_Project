"""Experiment 11: Han-style ADST + four-model light Cascade + reliability Fusion.

The evaluator is inherited from the approved GitHub-base 30-minute rolling
runner.  This experiment changes only the model/adaptation layers:

* Branch 1: four light model families in the existing soft Cascade.
* Branch 1 ADST: Han-style train-window state with momentum hold/trigger logic.
* Branch 2: strict History-only, fixed Sliding Training.
* Fusion: past-only reliability gate; no ADWIN and no terminal held-out labels.

The original runners and their outputs are not modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_experiment6_30m_adwin_b1_train_window as previous


rolling = previous.rolling
STEP_MINUTES = rolling.STEP_MINUTES
CADENCE_BINS = rolling.CADENCE_BINS
PURGE_BINS = previous.rolling.PURGE_BINS
DAY_NS = previous.rolling.base.DAY_NS
STEP_NS = previous.rolling.base.STEP_NS
HOUR_NS = previous.rolling.base.HOUR_NS
TRAIN_WINDOW_CANDIDATES = (7, 14, 21)
INITIAL_TRAIN_DAYS = 14
FIXED_B2_TRAIN_DAYS = 21
ONE_HOUR_INPUT = (1,)
# The inherited evaluator imported this tuple before the one-hour learner was
# installed.  Keep its audit fields aligned with the actual B1 input contract.
rolling.OBS_HOURS_GRID = ONE_HOUR_INPUT
MODEL_FAMILIES = (
    "logistic",
    "extra_trees",
    "hist_gradient_boosting",
    "cnn",
)
EQUAL_FAMILY_WEIGHTS = {name: 0.25 for name in MODEL_FAMILIES}
RELIABILITY_WINDOW = 3
FUSION_MIN_WEIGHT = 0.20
FUSION_MAX_WEIGHT = 0.80
FUSION_STEP = 0.10
FUSION_DIRECTION_MARGIN = 0.05
FUSION_CONFIDENCE_SCALE = 0.25
FUSION_HYSTERESIS_ORIGINS = 2
FUSION_DRIFT_DROP = 0.20
ADST_DROP_THRESHOLD = 0.15
ADST_STABLE_ORIGINS = 3


class HanOneHourDiversifiedBranchFusion(previous.OneHourDiversifiedBranchFusion):
    """Four-family one-hour learner with two recent purged OOF blocks."""

    def collect_oof(
        self,
        dev_origins: np.ndarray,
        selected: dict[str, Any],
        latest_count: int = 1,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        origins = np.asarray(dev_origins[-latest_count:], dtype=np.int32)
        for origin_idx, origin_bin in enumerate(origins):
            blocks = self._selection_blocks(int(origin_bin))
            if len(blocks) < 2:
                raise ValueError(
                    f"At least two purged validation blocks are required at {origin_bin}."
                )
            for block in blocks[-2:]:
                block_idx = int(block["block_idx"])
                bins, gpus, labels, train_start = self._sample_training(
                    int(block["train_end"]),
                    int(selected["selected_train_days"]),
                    int(origin_bin) + 200_003 * (block_idx + 1),
                )
                fitted = self._fit_families(
                    bins,
                    gpus,
                    labels,
                    int(block["train_end"]),
                    int(selected["selected_train_days"]),
                    int(selected["selected_half_life_days"]),
                    self.seed + 300_000 + origin_idx * 10_000 + block_idx * 100,
                )
                stage1 = self.predict_parallel(
                    fitted, block["val_bins"], block["val_gpus"]
                )
                val_24h, _ = self.engine.extract_branch1_features(
                    block["val_bins"], block["val_gpus"], 24
                )
                records.append(
                    {
                        "origin_bin": int(origin_bin),
                        "block_idx": block_idx,
                        "train_start": int(train_start),
                        "train_end": int(block["train_end"]),
                        "labels": block["val_labels"],
                        "stage1": stage1,
                        # Stage 1 uses the one-hour input contract.  The
                        # existing Cascade stage-2 contract uses 24-hour
                        # context, matching the rolling scorer below.
                        "tabular_24h": val_24h,
                        "val_bins": block["val_bins"],
                        "val_gpus": block["val_gpus"],
                    }
                )
                print(
                    f"    one-hour OOF origin={int(origin_bin)} block={block_idx} "
                    f"n={len(block['val_labels']):,}",
                    flush=True,
                )
        return records


def _safe_norm_ap(ap: float, prevalence: float) -> float:
    denominator = max(1.0 - float(prevalence), 1e-8)
    return float(np.clip((float(ap) - float(prevalence)) / denominator, 0.0, 1.0))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values, ddof=0)) if values else 0.0


class HanADSTLightCascadeRolling(rolling.Experiment6ThirtyMinuteRolling):
    """Full-period 30-minute evaluator with Han-style adaptation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Keep the GitHub-base evaluator, but replace only the learner with the
        # existing one-hour four-family Cascade implementation.
        self.pipeline = HanOneHourDiversifiedBranchFusion(
            engine=self.engine,
            history_map=self.history_map,
            gt_matrix=self.gt,
            output_dir=self.output / "model_checkpoints",
            retrain_cadence_hours=rolling.CADENCE_HOURS,
            negative_ratio=self.negative_ratio,
            test_stride_bins=1,
            seed=self.seed,
            branch2_mode="history_0910",
        )

        self.frozen = {
            "selected_train_days": INITIAL_TRAIN_DAYS,
            # Branch 1 recency weighting must remain positive.  Zero is only
            # valid for the History-only Branch 2 contract.
            "selected_half_life_days": int(
                self.source_selection.get("selected_half_life_days", 7)
            ),
            "selected_input_hours": [1],
            "selected_family_weights": dict(EQUAL_FAMILY_WEIGHTS),
            "selected_input_weights": {"1": 1.0},
            "selected_cascade_alpha": float(
                self.source_selection.get("selected_cascade_alpha", 0.5)
            ),
            "selected_b2_train_days": FIXED_B2_TRAIN_DAYS,
            "selected_b2_half_life_days": 0,
        }

        self.current_train_days = INITIAL_TRAIN_DAYS
        self.pending_adst_action = "initial"
        self.adst_stable_count = 0
        self.adst_history: list[dict[str, Any]] = []

        self.fusion_weight_b1 = 0.5
        self.fusion_pending_direction = 0
        self.fusion_pending_streak = 0
        self.fusion_history: list[dict[str, Any]] = []
        self._restore_controller_state()

    def _restore_controller_state(self) -> None:
        state_files = sorted((self.output / "state_parts").glob("state_*.json"))
        if state_files:
            payload = json.loads(state_files[-1].read_text(encoding="utf-8"))
            self.current_train_days = int(
                payload.get("adst_next_train_days", self.current_train_days)
            )
            if self.current_train_days not in TRAIN_WINDOW_CANDIDATES:
                self.current_train_days = INITIAL_TRAIN_DAYS
            self.fusion_weight_b1 = float(
                payload.get("fusion_weight_b1_next", self.fusion_weight_b1)
            )
            self.fusion_weight_b1 = float(
                np.clip(self.fusion_weight_b1, FUSION_MIN_WEIGHT, FUSION_MAX_WEIGHT)
            )
            self.pending_adst_action = str(
                payload.get("adst_next_action", self.pending_adst_action)
            )
            self.adst_stable_count = int(payload.get("adst_stable_count", 0))

        trace_path = self.output / "fusion_reliability_history.csv"
        if trace_path.exists():
            trace = pd.read_csv(trace_path)
            self.fusion_history = trace.to_dict(orient="records")

        adst_trace_path = self.output / "adst_controller_history.csv"
        if adst_trace_path.exists():
            adst_trace = pd.read_csv(adst_trace_path)
            self.adst_history = adst_trace.to_dict(orient="records")

    def _selection_config(self) -> dict[str, Any]:
        selected = dict(self.frozen)
        selected["selected_train_days"] = int(self.current_train_days)
        selected["selected_lambda"] = float(self.fusion_weight_b1)
        selected["lambda_selection_stage"] = "past_only_reliability_gate"
        selected["lambda_validation_block_count"] = 1
        selected["source_selected_lambda"] = float(
            self.source_selection.get("selected_lambda", 0.5)
        )
        selected["prediction_population"] = "all_gpus_before_top100"
        selected["adst_action"] = self.pending_adst_action
        return selected

    def fit_state(
        self, origin: int, state_idx: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        origin = int(origin)
        train_end = origin - PURGE_BINS
        if train_end <= 0:
            raise ValueError(f"Origin {origin} is before the purge boundary")

        selected = self._selection_config()
        validation_origins = np.asarray([origin], dtype=np.int32)
        b1_records = self.pipeline.collect_oof(
            validation_origins, selected, latest_count=1
        )
        cascade_model, cascade_meta, _ = self.pipeline.fit_cascade(
            b1_records,
            selected["selected_family_weights"],
            selected["selected_input_weights"],
        )
        selected["selected_cascade_alpha"] = float(
            cascade_meta["selected_cascade_alpha"]
        )
        self.pipeline._cascade_validation_predictions(
            b1_records, cascade_model, cascade_meta
        )

        validation_blocks = self.pipeline._selection_blocks(origin)
        validation_end = max(
            int(block.get("val_end", int(block["val_bins"].max()) + 1))
            for block in validation_blocks
        )
        selected["lambda_validation_end_exclusive"] = validation_end
        selected["lambda_validation_label_maturity_latest"] = (
            validation_end + rolling.HORIZON_BINS
        )
        selected["lambda_validation_label_maturity_latest_time"] = rolling.stamp(
            self.engine, selected["lambda_validation_label_maturity_latest"]
        )
        selected["lambda_selection_cutoff_time"] = rolling.stamp(
            self.engine, origin - PURGE_BINS
        )

        final_b1 = self.pipeline._fit_final_stage1(selected, origin)
        final_b2 = self.pipeline._fit_final_b2(FIXED_B2_TRAIN_DAYS, origin)
        state = {
            "model_state_id": f"state_{state_idx:04d}",
            "origin": origin,
            "train_end": train_end,
            "train_start": int(final_b1["metadata"]["train_start"]),
            "b1_models": final_b1,
            "cascade_model": cascade_model,
            "cascade_meta": cascade_meta,
            "b2_model": final_b2,
            "selected": selected,
            "adst_train_window_days": int(self.current_train_days),
            "fusion_weight_b1": float(self.fusion_weight_b1),
        }
        lambda_rows = [
            {
                "state_idx": int(state_idx),
                "origin": origin,
                "origin_time": rolling.stamp(self.engine, origin),
                "selection_stage": "past_only_reliability_gate",
                "selection_role": "dynamic_branch_reliability",
                "candidate_lambda": float(self.fusion_weight_b1),
                "selected_lambda": float(self.fusion_weight_b1),
                "validation_block_count": 1,
                "terminal_heldout_used": False,
            }
        ]
        return state, lambda_rows

    def _branch_reliability(
        self,
        branch: str,
        current_ap: float,
        prevalence: float,
    ) -> tuple[float, float, float, float]:
        values = [
            float(row[f"{branch}_norm_ap"])
            for row in self.fusion_history[-(RELIABILITY_WINDOW - 1) :]
            if f"{branch}_norm_ap" in row
        ]
        current_norm = _safe_norm_ap(current_ap, prevalence)
        values.append(current_norm)
        recent_mean = _safe_mean(values)
        stability = 1.0 / (
            1.0 + _safe_std(values) / max(abs(recent_mean), 1e-4)
        )
        previous_values = [
            float(row[f"{branch}_norm_ap"])
            for row in self.fusion_history[-(2 * RELIABILITY_WINDOW - 1) : -(
                RELIABILITY_WINDOW - 1
            )]
            if f"{branch}_norm_ap" in row
        ]
        previous_mean = _safe_mean(previous_values)
        drop = 0.0
        if previous_values:
            drop = max(
                0.0,
                (previous_mean - recent_mean) / max(abs(previous_mean), 1e-4),
            )
        reliability = float(
            np.clip(0.6 * recent_mean + 0.3 * stability + 0.1 * (1.0 - drop), 0.0, 1.0)
        )
        return reliability, current_norm, stability, drop

    def _update_adst_from_cycle(
        self, b1_ap: float, prevalence: float, state: dict[str, Any]
    ) -> dict[str, Any]:
        current_norm = _safe_norm_ap(b1_ap, prevalence)
        previous = [float(row["b1_norm_ap"]) for row in self.adst_history[-3:]]
        prior = [float(row["b1_norm_ap"]) for row in self.adst_history[-6:-3]]
        recent_mean = _safe_mean(previous + [current_norm])
        prior_mean = _safe_mean(prior)
        drop = (
            max(0.0, (prior_mean - recent_mean) / max(abs(prior_mean), 1e-4))
            if prior
            else 0.0
        )
        idx = TRAIN_WINDOW_CANDIDATES.index(int(self.current_train_days))
        action = "momentum_skip_hold"
        if drop >= ADST_DROP_THRESHOLD and idx > 0:
            idx -= 1
            self.current_train_days = TRAIN_WINDOW_CANDIDATES[idx]
            self.adst_stable_count = 0
            action = "momentum_trigger_rescan_shrink"
        elif drop >= ADST_DROP_THRESHOLD:
            self.adst_stable_count = 0
            action = "momentum_trigger_rescan_hold_min"
        else:
            self.adst_stable_count += 1
            if self.adst_stable_count >= ADST_STABLE_ORIGINS and idx < len(TRAIN_WINDOW_CANDIDATES) - 1:
                idx += 1
                self.current_train_days = TRAIN_WINDOW_CANDIDATES[idx]
                self.adst_stable_count = 0
                action = "momentum_rescan_expand"

        self.pending_adst_action = action
        row = {
            "origin": int(state["origin"]),
            "adst_train_days_used": int(state["selected"]["selected_train_days"]),
            "adst_next_train_days": int(self.current_train_days),
            "adst_action": action,
            "b1_norm_ap": current_norm,
            "adst_recent_drop": float(drop),
            "adst_stable_count": int(self.adst_stable_count),
        }
        self.adst_history.append(row)
        return row

    def _update_fusion_reliability(
        self,
        b1_ap: float,
        b2_ap: float,
        prevalence: float,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        b1_rel, b1_norm, b1_stability, b1_drop = self._branch_reliability(
            "b1", b1_ap, prevalence
        )
        b2_rel, b2_norm, b2_stability, b2_drop = self._branch_reliability(
            "b2", b2_ap, prevalence
        )
        delta = float(b1_rel - b2_rel)
        confidence = float(
            np.clip(abs(delta) / FUSION_CONFIDENCE_SCALE, 0.0, 1.0)
        )
        direction = 0
        if delta >= FUSION_DIRECTION_MARGIN:
            direction = 1
        elif delta <= -FUSION_DIRECTION_MARGIN:
            direction = -1

        drift_branch = "none"
        if b1_drop >= FUSION_DRIFT_DROP and b1_drop > b2_drop:
            drift_branch = "b1"
        elif b2_drop >= FUSION_DRIFT_DROP and b2_drop > b1_drop:
            drift_branch = "b2"

        if drift_branch == "b1":
            direction = -1
        elif drift_branch == "b2":
            direction = 1

        if direction == 0:
            self.fusion_pending_direction = 0
            self.fusion_pending_streak = 0
            target = 0.5
            action = "hold_equal"
        else:
            if direction == self.fusion_pending_direction:
                self.fusion_pending_streak += 1
            else:
                self.fusion_pending_direction = direction
                self.fusion_pending_streak = 1
            confirmed = (
                self.fusion_pending_streak >= FUSION_HYSTERESIS_ORIGINS
                or drift_branch != "none"
            )
            if confirmed:
                target = 0.5 + direction * 0.3 * confidence
                if drift_branch == "b1":
                    target = FUSION_MIN_WEIGHT
                    action = "drift_down_b1"
                elif drift_branch == "b2":
                    target = FUSION_MAX_WEIGHT
                    action = "drift_down_b2"
                else:
                    action = "trust_b1" if direction > 0 else "trust_b2"
            else:
                target = self.fusion_weight_b1
                action = "pending_direction"

        target = float(np.clip(target, FUSION_MIN_WEIGHT, FUSION_MAX_WEIGHT))
        previous_weight = float(self.fusion_weight_b1)
        delta_weight = float(np.clip(target - previous_weight, -FUSION_STEP, FUSION_STEP))
        self.fusion_weight_b1 = float(
            np.clip(previous_weight + delta_weight, FUSION_MIN_WEIGHT, FUSION_MAX_WEIGHT)
        )
        row = {
            "origin": int(state["origin"]),
            "b1_pr_auc": float(b1_ap),
            "b2_pr_auc": float(b2_ap),
            "prevalence": float(prevalence),
            "b1_norm_ap": float(b1_norm),
            "b2_norm_ap": float(b2_norm),
            "b1_reliability": float(b1_rel),
            "b2_reliability": float(b2_rel),
            "b1_stability": float(b1_stability),
            "b2_stability": float(b2_stability),
            "b1_drop": float(b1_drop),
            "b2_drop": float(b2_drop),
            "reliability_delta_b1_minus_b2": delta,
            "confidence": confidence,
            "drift_branch": drift_branch,
            "direction": int(direction),
            "pending_streak": int(self.fusion_pending_streak),
            "fusion_action": action,
            "fusion_weight_b1_used": previous_weight,
            "fusion_weight_b2_used": 1.0 - previous_weight,
            "fusion_weight_b1_next": float(self.fusion_weight_b1),
            "fusion_weight_b2_next": 1.0 - float(self.fusion_weight_b1),
        }
        self.fusion_history.append(row)
        return row

    def score_cycle(
        self, state: dict[str, Any], origin: int, end: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        full_frame, cohort_frame, artifacts = super().score_cycle(state, origin, end)
        cycle_bins = np.arange(int(origin), int(end), 6, dtype=np.int32)
        n_time = len(cycle_bins)
        n_gpu = int(self.engine.num_gpus)
        labels = self.gt[cycle_bins].astype(np.uint8)
        b1 = full_frame["b1_probability"].to_numpy(dtype=float).reshape(n_time, n_gpu)
        b2 = full_frame["b2_probability"].to_numpy(dtype=float).reshape(n_time, n_gpu)
        prevalence = float(labels.mean()) if labels.size else 0.0
        b1_ap = rolling.base.safe_average_precision(labels.ravel(), b1.ravel())
        b2_ap = rolling.base.safe_average_precision(labels.ravel(), b2.ravel())
        adst_row = self._update_adst_from_cycle(b1_ap, prevalence, state)
        fusion_row = self._update_fusion_reliability(
            b1_ap, b2_ap, prevalence, state
        )
        state["adst_next_train_days"] = int(self.current_train_days)
        state["adst_next_action"] = self.pending_adst_action
        state["adst_stable_count"] = int(self.adst_stable_count)
        state["fusion_weight_b1_next"] = float(self.fusion_weight_b1)
        state["fusion_weight_b2_next"] = 1.0 - float(self.fusion_weight_b1)
        artifacts["state"].update(
            {
                "adst_train_window_days": int(state["selected"]["selected_train_days"]),
                "adst_next_train_days": int(self.current_train_days),
                "adst_next_action": self.pending_adst_action,
                "fusion_weight_b1_used": float(fusion_row["fusion_weight_b1_used"]),
                "fusion_weight_b1_next": float(self.fusion_weight_b1),
                "fusion_weight_b2_next": 1.0 - float(self.fusion_weight_b1),
                "fusion_action": fusion_row["fusion_action"],
            }
        )
        artifacts["metric"].update(
            {
                "adst_action": adst_row["adst_action"],
                "adst_next_train_days": int(self.current_train_days),
                "fusion_weight_b1_used": float(fusion_row["fusion_weight_b1_used"]),
                "fusion_weight_b2_used": float(fusion_row["fusion_weight_b2_used"]),
                "fusion_weight_b1_next": float(self.fusion_weight_b1),
                "fusion_weight_b2_next": float(fusion_row["fusion_weight_b2_next"]),
                "fusion_confidence": float(fusion_row["confidence"]),
                "fusion_drift_branch": fusion_row["drift_branch"],
                "fusion_action": fusion_row["fusion_action"],
                "b1_reliability": float(fusion_row["b1_reliability"]),
                "b2_reliability": float(fusion_row["b2_reliability"]),
                "b1_stability": float(fusion_row["b1_stability"]),
                "b2_stability": float(fusion_row["b2_stability"]),
            }
        )
        return full_frame, cohort_frame, artifacts

    def _write_contract(self, origins: list[int], full_population: bool) -> dict[str, Any]:
        contract = super()._write_contract(origins, full_population)
        contract.update(
            {
                "runner_contract": "han_adst_b1_four_model_light_cascade_b2_sliding_reliability_fusion",
                "branch1_model_families": list(MODEL_FAMILIES),
                "branch1_input_hours": [1],
                "branch1_cascade": "four_model_light_soft_residual_hardness_gated",
                "branch1_adaptation": "han_style_adst_train_window_momentum",
                "branch1_adst_train_window_candidates_days": list(TRAIN_WINDOW_CANDIDATES),
                "branch1_adst_initial_train_days": INITIAL_TRAIN_DAYS,
                "branch1_adwin_used": False,
                "branch2_adaptation": "fixed_21d_history_only_sliding_training",
                "branch2_train_days_fixed": FIXED_B2_TRAIN_DAYS,
                "branch2_adst_used": False,
                "fusion": "past_only_branch_reliability_gate",
                "fusion_reliability_window_origins": RELIABILITY_WINDOW,
                "fusion_hysteresis_origins": FUSION_HYSTERESIS_ORIGINS,
                "fusion_weight_bounds": [FUSION_MIN_WEIGHT, FUSION_MAX_WEIGHT],
                "fusion_max_step": FUSION_STEP,
                "fusion_drift_drop_threshold": FUSION_DRIFT_DROP,
                "terminal_heldout_used_for_adaptation": False,
            }
        )
        rolling.write_json(self.output / "export_contract.json", contract)
        return contract

    def run(self, *args: Any, **kwargs: Any) -> None:
        super().run(*args, **kwargs)
        pd.DataFrame(self.fusion_history).to_csv(
            self.output / "fusion_reliability_history.csv", index=False
        )
        pd.DataFrame(self.adst_history).to_csv(
            self.output / "adst_controller_history.csv", index=False
        )
        manifest_path = self.output / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "runner_contract": "han_adst_b1_four_model_light_cascade_b2_sliding_reliability_fusion",
                "branch1_adwin_used": False,
                "branch1_input_hours": [1],
                "branch1_adst": {
                    "candidates_days": list(TRAIN_WINDOW_CANDIDATES),
                    "initial_train_days": INITIAL_TRAIN_DAYS,
                    "controller": "han_style_momentum_skip_trigger",
                    "drop_threshold": ADST_DROP_THRESHOLD,
                },
                "branch2_sliding_train_days": FIXED_B2_TRAIN_DAYS,
                "fusion_reliability": {
                    "window_origins": RELIABILITY_WINDOW,
                    "hysteresis_origins": FUSION_HYSTERESIS_ORIGINS,
                    "weight_bounds": [FUSION_MIN_WEIGHT, FUSION_MAX_WEIGHT],
                    "max_step": FUSION_STEP,
                    "drift_drop_threshold": FUSION_DRIFT_DROP,
                },
                "fusion_trace_rows": len(self.fusion_history),
                "adst_trace_rows": len(self.adst_history),
            }
        )
        rolling.write_json(manifest_path, manifest)
        state_dir = self.output / "state_parts"
        for state_path in sorted(state_dir.glob("state_*.json")):
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            state_payload["branch1_input_hours"] = [1]
            state_path.write_text(
                json.dumps(
                    state_payload,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        state_manifest_path = self.output / "model_state_manifest.parquet"
        if state_manifest_path.exists():
            state_manifest = pd.read_parquet(state_manifest_path)
            state_manifest["branch1_input_hours"] = [[1]] * len(state_manifest)
            state_manifest.to_parquet(
                state_manifest_path, index=False, compression="zstd"
            )
        report = [
            "# Experiment 11: Han ADST + 4-model light Cascade + reliability Fusion",
            "",
            "## Material Passport",
            f"- **Status**: `{manifest.get('status', 'UNKNOWN')}`",
            f"- **Runner**: `{Path(__file__).name}`",
            "- **Evaluation**: GitHub-base 30-minute full-period rolling evaluator",
            "- **BLOX**: not executed",
            "",
            "## 실행 계약",
            "- 30일 warm-up, 24시간 rolling/retrain, 24시간 horizon",
            "- 36시간 purge, 30분 scoring, 전체 GPU population 후 Top-100 tape",
            "- Branch 1: Logistic/ExtraTrees/HGB/CNN 경량 4-model Cascade",
            "- Branch 1: Han-style ADST train-window controller; ADWIN 미적용",
            "- Branch 2: History-only 21일 고정 Sliding Training",
            "- Fusion: development 성능·안정성 기반 past-only reliability gate",
            "- terminal held-out: adaptation 및 Fusion 선택에 사용하지 않음",
            "",
            "## Fusion Drift 규칙",
            "- 최근 3개 matured development origin의 normalized PR-AUC, 안정성, 성능 하락을 결합",
            "- 기본 weight 0.5/0.5, branch별 weight 범위 0.2~0.8",
            "- origin당 weight 변화 최대 0.1, 2회 연속 우세 후 전환",
            "- 최근 성능 20% 이상 하락한 Branch는 drift 대상으로 보고 weight를 단계적으로 낮춤",
            "",
            "## 산출물",
            "- full_period_metrics.csv",
            "- probability_tape.parquet",
            "- evaluation_tape.parquet",
            "- fusion_reliability_history.csv",
            "- adst_controller_history.csv",
            "- model_state_manifest.parquet",
            "- experiment_manifest.json",
        ]
        (self.output / "IMPLEMENTATION_30M_EXPERIMENT11.md").write_text(
            "\n".join(report), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=rolling.DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/2026-09-29_All-XID_HanADST_B1LightCascade_B2Sliding_ReliabilityFusion_01"
        ),
    )
    parser.add_argument("--negative-ratio", type=int, default=rolling.DEFAULT_NEGATIVE_RATIO)
    parser.add_argument("--cohort-size", type=int, default=rolling.COHORT_SIZE)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-full-population", action="store_true")
    parser.add_argument("--max-states", type=int, default=None)
    args = parser.parse_args()

    rolling.base.seed_everything(rolling.DEFAULT_SEED)
    root = Path(__file__).resolve().parents[1]
    source = (root / args.source).resolve()
    output = (root / args.output).resolve()
    cache_dir = root / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = rolling.base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    engine = rolling.base.UnifiedDataEngine(
        data_dir=rolling.base.find_data_dir(),
        cache_dir=cache_dir,
    )
    _, history_map, gt_matrix = engine.load_all_xid_ledger()
    runner = HanADSTLightCascadeRolling(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=output,
        source_dir=source,
        negative_ratio=args.negative_ratio,
        cohort_size=args.cohort_size,
        seed=rolling.DEFAULT_SEED,
        resume=args.resume,
    )
    with threadpool_limits(limits=int(args.threads)):
        runner.run(
            full_population=not args.no_full_population,
            max_states=args.max_states,
        )


if __name__ == "__main__":
    main()

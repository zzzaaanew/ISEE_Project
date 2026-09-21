"""Report-faithful Pareto Fusion with stateful Momentum Skip-Retrain.

This runner is intentionally separate from the earlier Pareto/ADWIN variants.
It keeps the project contract (All-XID, Sliding Training, 36-hour purge,
History-only Branch 2, Cascade Branch 1, and common held-out test) while
implementing the two mechanisms described in the September week-3 report:

* per-GPU Pareto weights use ``w_rep + (w_clean-w_rep)*(1/(1+count))**alpha``;
* Momentum ADST skips the 9-cell window search when the selected pair has
  accumulated confidence, performs one quick validation, and persists the
  decision/state through checkpoints and resume.

The exact report coefficients were not available in the current Git history,
so the runner evaluates the approved sensitivity grid of w_clean, w_rep, and
alpha on the same Branch-1/Branch-2 score matrices.  The primary tape defaults
to (w_clean=0.50, w_rep=0.02, alpha=1.0); all grid metrics are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_bidirectional_adst_fusion as base
import run_branch1_diversified_parallel_cascade as b1mod
import run_fusion_diversified_b1_history_b2 as integrated
from run_pareto_momentum_enhanced_fusion import (
    B2_HALF_LIFE,
    ParetoMomentumFusion,
    _pareto_counts,
)
from branch1_enhanced_features import EnhancedTelemetryEngine


REPORT_W_CLEAN = (0.30, 0.50)
REPORT_W_REP = (0.02, 0.05)
REPORT_ALPHA = (1.0, 1.5)
REPORT_BETA = 0.70
REPORT_CONFIDENCE_THRESHOLD = 0.65
REPORT_DROP_PCT = 0.15


def report_pareto_lambda(
    counts: np.ndarray,
    w_clean: float,
    w_rep: float,
    alpha: float,
) -> np.ndarray:
    """Return the report's GPU-specific B1 weight.

    A clean GPU receives ``w_clean``.  As its trailing 30-day XID count grows,
    the B1 weight approaches ``w_rep`` and the History-only B2 weight grows.
    """

    values = np.asarray(counts, dtype=np.float64)
    result = float(w_rep) + (float(w_clean) - float(w_rep)) * np.power(
        1.0 / (1.0 + values), float(alpha)
    )
    return np.clip(result, 0.0, 1.0)


def _variant_name(w_clean: float, w_rep: float, alpha: float) -> str:
    return f"wc{w_clean:.2f}_wr{w_rep:.2f}_a{alpha:.1f}"


class ReportFaithfulParetoMomentumFusion(ParetoMomentumFusion):
    """Cascade + History-only Fusion using report-faithful adaptation."""

    def __init__(
        self,
        *args: Any,
        w_clean: float = 0.50,
        w_rep: float = 0.02,
        pareto_alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            policy="adst",
            pareto_alpha=float(pareto_alpha),
            momentum_beta=REPORT_BETA,
            momentum_threshold=REPORT_CONFIDENCE_THRESHOLD,
            momentum_drop_pct=REPORT_DROP_PCT,
            **kwargs,
        )
        self.w_clean = float(w_clean)
        self.w_rep = float(w_rep)
        self.variant_specs = self._make_variant_specs()
        self.state_path = self.checkpoint_dir / "momentum_state.json"
        self.state_trace_path = self.output_dir / "momentum_selection_trace.csv"
        self.variant_metrics_path = self.output_dir / "fusion_variant_metrics.csv"
        self._momentum_state = self._load_momentum_state()
        self._momentum_trace = self._load_existing_trace()
        self._variant_rows: list[dict[str, Any]] = self._load_existing_variants()

    def _make_variant_specs(self) -> list[dict[str, Any]]:
        requested = (self.w_clean, self.w_rep, self.pareto_alpha)
        specs: list[tuple[float, float, float]] = [requested]
        for wc in REPORT_W_CLEAN:
            for wr in REPORT_W_REP:
                for alpha in REPORT_ALPHA:
                    specs.append((float(wc), float(wr), float(alpha)))
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for wc, wr, alpha in specs:
            name = _variant_name(wc, wr, alpha)
            if name in seen:
                continue
            seen.add(name)
            unique.append(
                {
                    "variant": name,
                    "w_clean": float(wc),
                    "w_rep": float(wr),
                    "alpha": float(alpha),
                }
            )
        return unique

    @staticmethod
    def _pair_key(train_days: int, half_life_days: int) -> str:
        return f"{int(train_days)}d|{int(half_life_days)}d"

    def _default_momentum_state(self) -> dict[str, Any]:
        confidence = {
            self._pair_key(train, half): 0.0
            for train in b1mod.TRAIN_DAYS_GRID
            for half in b1mod.HALF_LIFE_GRID
        }
        return {
            "version": 1,
            "beta": REPORT_BETA,
            "confidence_threshold": REPORT_CONFIDENCE_THRESHOLD,
            "performance_drop_trigger": REPORT_DROP_PCT,
            "confidence": confidence,
            "momentum": 0.0,
            "previous_validation_pr_auc": None,
            "current_pair": None,
            "last_origin_bin": None,
            "processed_origins": [],
            "skip_count": 0,
            "full_rescan_count": 0,
        }

    def _load_momentum_state(self) -> dict[str, Any]:
        state = self._default_momentum_state()
        if self.resume and self.state_path.exists():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
                state.update(loaded)
                merged = state["confidence"]
                for key in self._default_momentum_state()["confidence"]:
                    merged.setdefault(key, 0.0)
                state["confidence"] = merged
                return state
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                # A malformed state must not make a recoverable experiment
                # unusable; the selection checkpoints remain authoritative.
                pass
        return state

    def _load_existing_trace(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self.resume and self.state_trace_path.exists():
            try:
                frame = pd.read_csv(self.state_trace_path)
                rows = frame.replace({np.nan: None}).to_dict("records")
            except (OSError, ValueError, pd.errors.ParserError):
                rows = []

        # A process can stop after a selection checkpoint is written but
        # before the final trace CSV is flushed. Rebuild missing decisions
        # from those durable per-origin checkpoints on resume.
        known_origins = {int(row.get("origin_bin", -1)) for row in rows}
        if self.resume and self.checkpoint_dir.exists():
            for checkpoint in sorted(self.checkpoint_dir.glob("selection_branch1_dev_*.json")):
                try:
                    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                    best = payload.get("best", {})
                    origin = int(best.get("selection_origin_idx"))
                    if origin in known_origins:
                        continue
                    rows.append(
                        {
                            "origin_bin": origin,
                            "policy": "momentum_skip_retrain",
                            "action": best.get("adst_action"),
                            "selected_train_days": best.get("selected_train_days"),
                            "selected_half_life_days": best.get("selected_half_life_days"),
                            "candidate_count": best.get("candidate_count"),
                            "fit_plan": best.get("momentum_fit_plan"),
                            "quick_validation_pr_auc": best.get("quick_validation_pr_auc"),
                            "performance_drop": best.get("performance_drop"),
                            "pair_confidence": best.get("momentum_pair_confidence"),
                            "momentum": best.get("momentum_value"),
                            "validation_pr_auc": best.get("val_pr_auc_mean"),
                        }
                    )
                    known_origins.add(origin)
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    continue
        return sorted(rows, key=lambda row: int(row.get("origin_bin", -1)))
    def _load_existing_variants(self) -> list[dict[str, Any]]:
        if not self.resume or not self.variant_metrics_path.exists():
            return []
        try:
            frame = pd.read_csv(self.variant_metrics_path)
            return frame.replace({np.nan: None}).to_dict("records")
        except (OSError, ValueError, pd.errors.ParserError):
            return []

    def _save_momentum_state(self) -> None:
        b1mod._write_json_atomic(self.state_path, self._momentum_state)

    def _record_trace(self, row: dict[str, Any]) -> None:
        origin = int(row["origin_bin"])
        self._momentum_trace = [
            old for old in self._momentum_trace if int(old.get("origin_bin", -1)) != origin
        ]
        self._momentum_trace.append(row)
        self._momentum_trace.sort(key=lambda old: int(old.get("origin_bin", -1)))

    def _update_momentum_state(self, selected: dict[str, Any]) -> tuple[float, float, float | None]:
        train_days = int(selected["selected_train_days"])
        half_life = int(selected["selected_half_life_days"])
        key = self._pair_key(train_days, half_life)
        confidence = self._momentum_state.setdefault("confidence", {})
        for candidate_key in list(confidence):
            confidence[candidate_key] = REPORT_BETA * float(confidence[candidate_key]) + (
                1.0 - REPORT_BETA
            ) * float(candidate_key == key)
        current_ap = float(selected["val_pr_auc_mean"])
        previous_ap = self._momentum_state.get("previous_validation_pr_auc")
        delta = None if previous_ap is None else current_ap - float(previous_ap)
        momentum = REPORT_BETA * float(self._momentum_state.get("momentum", 0.0)) + (
            1.0 - REPORT_BETA
        ) * (0.0 if delta is None else float(delta))
        self._momentum_state["momentum"] = float(momentum)
        self._momentum_state["previous_validation_pr_auc"] = current_ap
        self._momentum_state["current_pair"] = key
        return float(confidence[key]), float(momentum), delta

    def _persist_selection_state(
        self,
        origin_bin: int,
        selected: dict[str, Any],
        action: str,
        candidate_count: int,
        quick_ap: float | None,
        performance_drop: float | None,
        fit_plan: str,
    ) -> None:
        train_conf, momentum, delta = self._update_momentum_state(selected)
        key = self._pair_key(
            int(selected["selected_train_days"]),
            int(selected["selected_half_life_days"]),
        )
        self._momentum_state["last_origin_bin"] = int(origin_bin)
        processed = {int(value) for value in self._momentum_state.get("processed_origins", [])}
        processed.add(int(origin_bin))
        self._momentum_state["processed_origins"] = sorted(processed)
        if action == "momentum_skip_hold":
            self._momentum_state["skip_count"] = int(self._momentum_state.get("skip_count", 0)) + 1
        elif "rescan" in action or "initial" in action:
            self._momentum_state["full_rescan_count"] = int(
                self._momentum_state.get("full_rescan_count", 0)
            ) + 1
        selected["momentum_pair_confidence"] = float(self._momentum_state["confidence"][key])
        selected["momentum_value"] = float(momentum)
        selected["quick_validation_pr_auc"] = quick_ap
        selected["performance_drop"] = performance_drop
        selected["momentum_fit_plan"] = fit_plan
        selected["candidate_count"] = int(candidate_count)
        selected["adst_action"] = action
        selected["previous_val_pr_auc_mean"] = (
            None if delta is None else float(selected["val_pr_auc_mean"] - delta)
        )
        selected["val_pr_auc_delta"] = delta
        self._record_trace(
            {
                "origin_bin": int(origin_bin),
                "policy": "momentum_skip_retrain",
                "action": action,
                "selected_train_days": int(selected["selected_train_days"]),
                "selected_half_life_days": int(selected["selected_half_life_days"]),
                "candidate_count": int(candidate_count),
                "fit_plan": fit_plan,
                "quick_validation_pr_auc": quick_ap,
                "performance_drop": performance_drop,
                "pair_confidence": float(self._momentum_state["confidence"][key]),
                "momentum": float(momentum),
                "validation_pr_auc": float(selected["val_pr_auc_mean"]),
            }
        )
        self._save_momentum_state()
        # Flush the trace at every decision so a later resume cannot lose
        # already completed origins if the process is interrupted.
        pd.DataFrame(self._momentum_trace).to_csv(self.state_trace_path, index=False)

    def _skip_candidate(
        self,
        origin_bin: int,
        previous: dict[str, Any],
        stage: str,
        quick_ap: float,
    ) -> dict[str, Any]:
        return {
            "selection_role": "branch1_proxy",
            "selection_stage": stage,
            "selection_origin_idx": int(origin_bin),
            "selection_origin_time": integrated._time_text(self.engine, origin_bin),
            "candidate_train_days": int(previous["selected_train_days"]),
            "candidate_half_life_days": int(previous["selected_half_life_days"]),
            "candidate_input_hours": "all_1_6_24",
            "validation_block_count": integrated.VALIDATION_BLOCKS,
            "validation_block_days": 3,
            "purge_hours": int(base.PURGE_NS // integrated.HOUR_NS),
            "val_pr_auc_mean": float(quick_ap),
            "val_pr_auc_median": float(quick_ap),
            "val_pr_auc_q25": float(quick_ap),
            "val_pr_auc_std": 0.0,
            "val_pr_auc_min": float(quick_ap),
            "val_roc_auc_mean": np.nan,
            "candidate_count": 1,
        }

    def select_branch1_window(
        self,
        origin_bin: int,
        previous: dict[str, Any] | None,
        stage: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run full 9-grid search or one quick validation according to state."""

        if previous is None:
            best, candidates = b1mod.Branch1Diversified.select_branch1_window(
                self, origin_bin, previous, stage
            )
            self._persist_selection_state(
                origin_bin, best, "momentum_initial_full_search", len(candidates), None, None, "full_9_grid"
            )
            return best, candidates

        previous_key = self._pair_key(
            int(previous["selected_train_days"]), int(previous["selected_half_life_days"])
        )
        pair_confidence = float(self._momentum_state["confidence"].get(previous_key, 0.0))
        if pair_confidence >= REPORT_CONFIDENCE_THRESHOLD:
            quick_ap = float(self._quick_validate(origin_bin, previous))
            previous_ap = float(
                self._momentum_state.get("previous_validation_pr_auc")
                or previous.get("val_pr_auc_mean", quick_ap)
            )
            drop = (previous_ap - quick_ap) / max(previous_ap, 1e-9)
            if drop < REPORT_DROP_PCT:
                best = dict(previous)
                best["val_pr_auc_mean"] = quick_ap
                best["val_pr_auc_median"] = quick_ap
                best["val_pr_auc_q25"] = quick_ap
                best["val_pr_auc_std"] = 0.0
                best["val_pr_auc_min"] = quick_ap
                best["selection_stage"] = stage
                candidate = self._skip_candidate(origin_bin, previous, stage, quick_ap)
                self._persist_selection_state(
                    origin_bin,
                    best,
                    "momentum_skip_hold",
                    1,
                    quick_ap,
                    drop,
                    "single_selected_config",
                )
                return best, [candidate]
            # The report's drop trigger resets momentum before a full rescan.
            self._momentum_state["momentum"] = 0.0
            action = "momentum_trigger_rescan"
        else:
            action = "momentum_full_rescan"
            quick_ap = None
            drop = None

        best, candidates = b1mod.Branch1Diversified.select_branch1_window(
            self, origin_bin, previous, stage
        )
        self._persist_selection_state(
            origin_bin,
            best,
            action,
            len(candidates),
            quick_ap,
            drop,
            "full_9_grid",
        )
        return best, candidates

    def _score_metrics(
        self,
        fused_scores: np.ndarray,
        b1_scores: np.ndarray,
        b2_scores: np.ndarray,
        gt: np.ndarray,
        cycle_idx: int,
        cycle_start: int,
        cycle_end: int,
        selected: dict[str, Any],
        cascade_meta: dict[str, Any],
        spec: dict[str, Any],
        lambda_matrix: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray]:
        y = gt.ravel()
        fused_rank = integrated._rank_matrix(fused_scores)
        b1_rank = integrated._rank_matrix(b1_scores)
        b2_rank = integrated._rank_matrix(b2_scores)
        top_k = min(100, self.engine.num_gpus)
        positives = int(gt.sum())
        prevalence = float(gt.mean())

        def top_metrics(ranks: np.ndarray) -> tuple[float, float]:
            hits = int((gt & (ranks <= top_k)).sum())
            return hits / max(positives, 1), (hits / max(len(gt) * top_k, 1)) / max(prevalence, 1e-12)

        recall_fused, lift_fused = top_metrics(fused_rank)
        recall_b1, lift_b1 = top_metrics(b1_rank)
        recall_b2, lift_b2 = top_metrics(b2_rank)
        ap_fused = base.safe_average_precision(y, fused_scores.ravel())
        metrics = {
            "origin_idx": int(cycle_idx),
            "model_origin_time": integrated._time_text(self.engine, cycle_start),
            "test_start_time": integrated._time_text(self.engine, cycle_start),
            "test_end_time": integrated._time_text(self.engine, cycle_end - 1),
            "evaluation_scope": "common_terminal_heldout",
            "prediction_population": "all_gpus_before_top100",
            "policy": "momentum_skip_retrain",
            "selected_b1_train_days": int(selected["selected_train_days"]),
            "selected_b1_half_life_days": int(selected["selected_half_life_days"]),
            "selected_b2_train_days": int(selected["selected_b2_train_days"]),
            "selected_b2_half_life_days": B2_HALF_LIFE,
            "selected_b1_input_hours": "1,6,24",
            "selected_cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
            # Kept numeric for compatibility with the parent progress printer;
            # the real Fusion weight is GPU-time-specific and is in the tape.
            "selected_lambda": float(spec["alpha"]),
            "lambda_mode": "gpu_dynamic_pareto",
            "pareto_variant": spec["variant"],
            "pareto_w_clean": float(spec["w_clean"]),
            "pareto_w_rep": float(spec["w_rep"]),
            "pareto_alpha": float(spec["alpha"]),
            "pareto_lambda_mean": float(lambda_matrix.mean()),
            "pareto_lambda_min": float(lambda_matrix.min()),
            "pareto_lambda_max": float(lambda_matrix.max()),
            "pareto_history_cutoff": "decision_time_minus_10min",
            "fusion_method": "report_pareto_formula_full_gpu_before_top100",
            "pr_auc_fused": ap_fused,
            "pr_auc_b1": base.safe_average_precision(y, b1_scores.ravel()),
            "pr_auc_b2": base.safe_average_precision(y, b2_scores.ravel()),
            "pr_auc_fused_normalized": integrated._normalized_pr_auc(ap_fused, prevalence),
            "roc_auc_fused": base.safe_roc_auc(y, fused_scores.ravel()),
            "roc_auc_b1": base.safe_roc_auc(y, b1_scores.ravel()),
            "roc_auc_b2": base.safe_roc_auc(y, b2_scores.ravel()),
            "recall_at_100": recall_fused,
            "lift_at_100": lift_fused,
            "recall_at_100_b1": recall_b1,
            "lift_at_100_b1": lift_b1,
            "recall_at_100_b2": recall_b2,
            "lift_at_100_b2": lift_b2,
            "positives": positives,
            "test_prevalence": prevalence,
            "gpu_count_evaluated": int(self.engine.num_gpus),
            "decision_time_count": int(len(gt)),
            "top_k": int(top_k),
        }
        rel_idx, gpu_idx = np.where(fused_rank <= top_k)
        tape = pd.DataFrame(
            {
                "decision_time": pd.to_datetime(
                    self.engine.bin_start_ns[np.arange(cycle_start, cycle_end, self.test_stride_bins, dtype=np.int32)][rel_idx],
                    unit="ns",
                    utc=True,
                ),
                "gpu_id": self.engine.gpu_ids[gpu_idx],
                "fused_risk": fused_scores[rel_idx, gpu_idx],
                "b1_risk": b1_scores[rel_idx, gpu_idx],
                "b2_risk": b2_scores[rel_idx, gpu_idx],
                "pareto_lambda": lambda_matrix[rel_idx, gpu_idx],
                "fused_rank": fused_rank[rel_idx, gpu_idx],
                "b1_rank": b1_rank[rel_idx, gpu_idx],
                "b2_rank": b2_rank[rel_idx, gpu_idx],
                "pareto_variant": spec["variant"],
                "pareto_w_clean": float(spec["w_clean"]),
                "pareto_w_rep": float(spec["w_rep"]),
                "pareto_alpha": float(spec["alpha"]),
                "b1_train_days": int(selected["selected_train_days"]),
                "b1_half_life_days": int(selected["selected_half_life_days"]),
                "b2_train_days": int(selected["selected_b2_train_days"]),
                "b2_half_life_days": B2_HALF_LIFE,
                "cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
                "target_24h": gt[rel_idx, gpu_idx].astype(np.uint8),
            }
        )
        return metrics, tape

    def _predict_fusion_cycle(self, *args: Any, **kwargs: Any):
        b1_models, cascade_model, cascade_meta, b2_model, selected, cycle_idx, cycle_start, cycle_end = args
        test_bins = np.arange(cycle_start, cycle_end, self.test_stride_bins, dtype=np.int32)
        if len(test_bins) == 0:
            raise ValueError(f"Empty held-out test cycle {cycle_idx}.")
        n_gpu = self.engine.num_gpus
        b1_scores = np.zeros((len(test_bins), n_gpu), dtype=np.float64)
        b2_scores = np.zeros_like(b1_scores)
        lambda_matrices = {
            spec["variant"]: np.zeros_like(b1_scores) for spec in self.variant_specs
        }
        fused_matrices = {
            spec["variant"]: np.zeros_like(b1_scores) for spec in self.variant_specs
        }
        for rel_idx, current_bin in enumerate(test_bins):
            bins = np.full(n_gpu, int(current_bin), dtype=np.int32)
            gpus = np.arange(n_gpu, dtype=np.int32)
            stage1 = self.predict_parallel(
                b1_models,
                bins,
                gpus,
                selected["selected_family_weights"],
                selected["selected_input_weights"],
            )
            tabular_24h, _ = self.engine.extract_branch1_features(bins, gpus, 24)
            stage2_x = self.stage2_features(
                tabular_24h,
                stage1["parallel"],
                stage1["disagreement"],
                stage1["uncertainty"],
                stage1["hardness"],
            )
            p_stage2 = base.adjusted_probability(
                cascade_model.predict_proba(stage2_x)[:, 1],
                float(cascade_meta["stage2_true_prior"]),
                float(cascade_meta["stage2_sample_prior"]),
            )
            b1 = np.clip(
                stage1["parallel"]
                + float(cascade_meta["selected_cascade_alpha"])
                * stage1["hardness"]
                * (p_stage2 - stage1["parallel"]),
                0.0,
                1.0,
            )
            b2_df = self.engine.extract_branch2_features(bins, gpus, self.history_map)
            raw_b2 = b2_model["model"].predict_proba(b2_df)[:, 1]
            b2 = base.adjusted_probability(
                raw_b2, float(b2_model["true_prior"]), float(b2_model["sample_prior"])
            )
            counts = _pareto_counts(
                self.history_map,
                int(self.engine.bin_start_ns[current_bin]),
                n_gpu,
            )
            b1_scores[rel_idx] = b1
            b2_scores[rel_idx] = b2
            for spec in self.variant_specs:
                lambdas = report_pareto_lambda(
                    counts, spec["w_clean"], spec["w_rep"], spec["alpha"]
                )
                lambda_matrices[spec["variant"]][rel_idx] = lambdas
                fused_matrices[spec["variant"]][rel_idx] = lambdas * b1 + (1.0 - lambdas) * b2

        gt = self.gt_matrix[test_bins]
        primary = self.variant_specs[0]
        primary_metrics, primary_tape = self._score_metrics(
            fused_matrices[primary["variant"]],
            b1_scores,
            b2_scores,
            gt,
            cycle_idx,
            cycle_start,
            cycle_end,
            selected,
            cascade_meta,
            primary,
            lambda_matrices[primary["variant"]],
        )
        for spec in self.variant_specs:
            variant_metrics, _ = self._score_metrics(
                fused_matrices[spec["variant"]],
                b1_scores,
                b2_scores,
                gt,
                cycle_idx,
                cycle_start,
                cycle_end,
                selected,
                cascade_meta,
                spec,
                lambda_matrices[spec["variant"]],
            )
            self._variant_rows = [
                row
                for row in self._variant_rows
                if not (
                    int(row.get("origin_idx", -1)) == int(cycle_idx)
                    and row.get("pareto_variant") == spec["variant"]
                )
            ]
            self._variant_rows.append(variant_metrics)
        return primary_metrics, primary_tape

    def _write_manifest(self, *args: Any, **kwargs: Any) -> None:
        super()._write_manifest(*args, **kwargs)
        path = self.output_dir / "experiment_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        payload.update(
            {
                "runner": Path(__file__).name,
                "pareto": {
                    "formula": "w_rep + (w_clean-w_rep)*(1/(1+count_30d))^alpha",
                    "primary_w_clean": self.w_clean,
                    "primary_w_rep": self.w_rep,
                    "primary_alpha": self.pareto_alpha,
                    "sensitivity_grid": self.variant_specs,
                    "population": "all_gpu_time_before_top100",
                    "history_cutoff": "decision_time_minus_10min",
                },
                "adst": {
                    "method": "momentum_skip_retrain",
                    "adwin_used": False,
                    "beta": REPORT_BETA,
                    "confidence_threshold": REPORT_CONFIDENCE_THRESHOLD,
                    "performance_drop_trigger": REPORT_DROP_PCT,
                    "skip_state_checkpoint": str(self.state_path),
                    "skip_state_persisted": True,
                    "full_search_candidates": 9,
                    "skip_search_candidates": 1,
                },
                "branch2_contract": "history_only_2026_09_10_without_1h_telemetry_summary",
            }
        )
        b1mod._write_json_atomic(path, payload)

    def run(self):
        metrics, tape = super().run()
        if self._momentum_trace:
            pd.DataFrame(self._momentum_trace).to_csv(self.state_trace_path, index=False)
        if self._variant_rows:
            pd.DataFrame(self._variant_rows).sort_values(
                ["origin_idx", "pareto_variant"]
            ).to_csv(self.variant_metrics_path, index=False)
        primary = self.variant_specs[0]
        variant_frame = pd.DataFrame(self._variant_rows)
        report = [
            "# All-XID Report-faithful Pareto Fusion + Momentum Skip-Retrain",
            "",
            "## 실험 계약",
            "- All-XID `[13, 31, 43, 45, 94]`, 24시간 horizon, 오류 직전 10분 제외",
            "- 시간순 Sliding Training, 36시간 purge, 공통 terminal held-out test",
            "- Branch 1: Parallel Ensemble + soft residual Cascade 유지",
            "- Branch 2: 1시간 telemetry 요약값을 제외한 strict History-only",
            "- 전체 GPU-time 점수 → Pareto Fusion → Top-100 risk tape",
            "",
            "## 적용한 보고서 방식",
            f"- Pareto: `w_rep + (w_clean-w_rep)*(1/(1+count_30d))^alpha`; primary=`{primary['variant']}`",
            "- clean GPU는 B1 비중이 높고, 반복 XID GPU는 History-only B2 비중이 높아진다.",
            "- Momentum: beta=.70, pair confidence>=.65이면 9-grid 대신 selected pair 1회 quick validation",
            "- quick validation PR-AUC가 직전 대비 15% 이상 하락하면 full 9-grid rescan",
            "- confidence/momentum/skip count는 checkpoints/momentum_state.json에 저장되어 resume 후 복원된다.",
            "",
            "## Terminal held-out 평균 (primary)",
            f"- Fusion PR-AUC: `{float(metrics['pr_auc_fused'].mean()):.6f}`",
            f"- Branch 1 Cascade PR-AUC: `{float(metrics['pr_auc_b1'].mean()):.6f}`",
            f"- Branch 2 History-only PR-AUC: `{float(metrics['pr_auc_b2'].mean()):.6f}`",
            f"- Fusion ROC-AUC: `{float(metrics['roc_auc_fused'].mean()):.6f}`",
            f"- Recall@100: `{float(metrics['recall_at_100'].mean()):.2%}`",
            f"- Lift@100: `{float(metrics['lift_at_100'].mean()):.3f}x`",
            f"- Mean Pareto B1 weight: `{float(metrics['pareto_lambda_mean'].mean()):.4f}`",
            "",
            "## Pareto sensitivity grid",
        ]
        if not variant_frame.empty:
            summary = variant_frame.groupby("pareto_variant", as_index=False).agg(
                pr_auc_fused=("pr_auc_fused", "mean"),
                roc_auc_fused=("roc_auc_fused", "mean"),
                pareto_lambda_mean=("pareto_lambda_mean", "mean"),
            )
            for row in summary.sort_values("pr_auc_fused", ascending=False).itertuples():
                report.append(
                    f"- `{row.pareto_variant}`: PR-AUC `{row.pr_auc_fused:.6f}`, "
                    f"ROC-AUC `{row.roc_auc_fused:.6f}`, mean B1 weight `{row.pareto_lambda_mean:.4f}`"
                )
        report += [
            "",
            "## 산출물",
            "- fusion_metrics.csv (primary variant)",
            "- fusion_variant_metrics.csv (approved sensitivity grid)",
            "- fusion_risk_tape.parquet (primary variant, full GPU before Top-100)",
            "- momentum_selection_trace.csv",
            "- checkpoints/momentum_state.json",
            "- experiment_manifest.json",
            "",
            "정확한 보고서 원본 commit의 w_clean/w_rep 값은 현재 원격 main에서 복구되지 않아, 승인된 2×2×2 sensitivity grid로 대체 검증했다.",
        ]
        (self.output_dir / "fusion_report.md").write_text("\n".join(report), encoding="utf-8")
        return metrics, tape


def _self_test() -> None:
    counts = np.array([0.0, 1.0, 4.0, 100.0])
    values = report_pareto_lambda(counts, 0.50, 0.02, 1.0)
    assert np.isclose(values[0], 0.50)
    assert values[0] > values[1] > values[2] > values[3] > 0.02
    values_alpha = report_pareto_lambda(counts, 0.50, 0.02, 1.5)
    assert np.isclose(values_alpha[0], 0.50)
    assert values_alpha[-1] < values[-1]
    assert len({
        _variant_name(wc, wr, alpha)
        for wc in REPORT_W_CLEAN
        for wr in REPORT_W_REP
        for alpha in REPORT_ALPHA
    }) == 8
    print("[Self-test] Pareto formula and sensitivity grid passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report-faithful Pareto + Momentum Skip-Retrain Fusion")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument("--w-clean", type=float, default=0.50)
    parser.add_argument("--w-rep", type=float, default=0.02)
    parser.add_argument("--pareto-alpha", type=float, default=1.0)
    parser.add_argument("--include-topology", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/2026-09-21_All-XID_ReportFaithful_ParetoMomentumCascade_01",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return

    data_dir = base.find_data_dir()
    cache_dir = base.PROJECT_ROOT / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    engine = EnhancedTelemetryEngine(
        data_dir=data_dir,
        cache_dir=cache_dir,
        include_topology=args.include_topology,
    )
    _, history_map, gt_matrix = engine.load_all_xid_ledger()
    pipeline = ReportFaithfulParetoMomentumFusion(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=base.PROJECT_ROOT / args.output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        resume=args.resume,
        branch2_mode="history_0910",
        w_clean=args.w_clean,
        w_rep=args.w_rep,
        pareto_alpha=args.pareto_alpha,
        include_topology=args.include_topology,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

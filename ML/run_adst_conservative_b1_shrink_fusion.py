"""Conservative Momentum-ADST with B1-shrunk Pareto Fusion.

This runner is deliberately separate from the committed Pareto runners.  It
keeps the All-XID, History-only Branch 2, purged Sliding Training, and common
terminal held-out contract while changing two experimental factors:

1. Momentum-ADST uses a two-block robust quick check, slower confidence
   updates, a short-window promotion guard, and a two-origin cooldown.
2. The primary Fusion tape uses Pareto weights capped at 0.6 for Branch 1.
   Current Pareto, fixed Lambda, B1-only, and B2-only variants are measured
   from the same full-GPU score matrices for an auditable ablation.

The terminal held-out labels are never used for selection.
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
import run_pareto_momentum_enhanced_fusion as pareto
from branch1_enhanced_features import EnhancedTelemetryEngine, enhanced_feature_manifest


PRIMARY_VARIANT = "pareto_cap_0.6"
DEFAULT_LAMBDA_CAP = 0.6
QUICK_VALIDATION_BLOCKS = 2
COOLDOWN_ORIGINS = 2
SHORT_WINDOW_MIN_ABS_GAIN = 0.005
SHORT_WINDOW_MIN_REL_GAIN = 0.05


class ConservativeReliabilityFusion(pareto.ParetoMomentumFusion):
    """ADST/Fusion variant with conservative B1 trust and auditable controls."""

    def __init__(
        self,
        *args: Any,
        policy: str = "adst",
        pareto_alpha: float = 1.0,
        include_topology: bool = False,
        lambda_cap: float = DEFAULT_LAMBDA_CAP,
        quick_validation_blocks: int = QUICK_VALIDATION_BLOCKS,
        cooldown_origins: int = COOLDOWN_ORIGINS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            policy=policy,
            pareto_alpha=pareto_alpha,
            include_topology=include_topology,
            momentum_beta=0.85,
            momentum_threshold=0.70,
            momentum_drop_pct=0.20,
            **kwargs,
        )
        if not 0.0 < float(lambda_cap) <= 1.0:
            raise ValueError("lambda_cap must be in (0, 1].")
        self.lambda_cap = float(lambda_cap)
        self.quick_validation_blocks = max(1, int(quick_validation_blocks))
        self.cooldown_origins = max(0, int(cooldown_origins))
        self._adst_call_index = -1
        self._last_change_call: int | None = None
        self._last_quick_block_aps: list[float] = []
        self._variant_rows: list[dict[str, Any]] = []

    def _quick_validate(self, origin_bin: int, previous: dict[str, Any]) -> float:
        """Use the median of the latest two validation blocks.

        The model proxy intentionally remains the existing lightweight
        Logistic/three-observation proxy in this run so the ADST setting
        change is isolated.  An architecture-matched quick proxy is a later
        ablation because it changes a second factor.
        """

        blocks = self._selection_blocks(int(origin_bin))
        if not blocks:
            self._last_quick_block_aps = []
            return float(previous.get("val_pr_auc_mean", 0.0))
        blocks = blocks[-self.quick_validation_blocks :]
        train_days = int(previous["selected_train_days"])
        half_life = int(previous["selected_half_life_days"])
        aps: list[float] = []
        for block_idx, block in enumerate(blocks):
            train_bins, train_gpus, train_labels, _ = self._sample_training(
                int(block["train_end"]),
                train_days,
                int(origin_bin) + 818_181 + 997 * block_idx,
            )
            weights = b1mod._recency_weights(
                train_bins, int(block["train_end"]), half_life
            )
            scores: list[np.ndarray] = []
            for obs in b1mod.OBS_HOURS_GRID:
                train_x = self.engine.extract_branch1_features(
                    train_bins, train_gpus, obs
                )[0]
                val_x = self.engine.extract_branch1_features(
                    block["val_bins"], block["val_gpus"], obs
                )[0]
                model = b1mod._fit_logistic(
                    train_x,
                    train_labels,
                    weights,
                    self.seed + int(origin_bin) + int(obs) + 71_001 + block_idx,
                )
                scores.append(model.predict_proba(val_x)[:, 1])
            aps.append(
                base.safe_average_precision(
                    block["val_labels"], np.mean(scores, axis=0)
                )
            )
        self._last_quick_block_aps = [float(value) for value in aps]
        return float(np.median(aps))

    def select_branch1_window(
        self,
        origin_bin: int,
        previous: dict[str, Any] | None,
        stage: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._adst_call_index += 1
        best, candidates = super().select_branch1_window(
            int(origin_bin), previous, stage
        )
        action_before_guard = str(best.get("adst_action", "unknown"))
        reverted_reason: str | None = None

        if previous is not None:
            changed = (
                int(best["selected_train_days"])
                != int(previous["selected_train_days"])
                or int(best["selected_half_life_days"])
                != int(previous["selected_half_life_days"])
            )
            in_cooldown = (
                changed
                and self._last_change_call is not None
                and self._adst_call_index - int(self._last_change_call)
                <= self.cooldown_origins
            )
            short_window_change = (
                changed
                and int(best["selected_train_days"]) == 7
                and int(previous["selected_train_days"]) != 7
            )
            previous_ap = float(previous.get("val_pr_auc_mean", 0.0))
            candidate_ap = float(best.get("val_pr_auc_mean", 0.0))
            required_gain = max(
                SHORT_WINDOW_MIN_ABS_GAIN,
                abs(previous_ap) * SHORT_WINDOW_MIN_REL_GAIN,
            )
            short_window_guard = short_window_change and (
                candidate_ap - previous_ap < required_gain
            )

            if in_cooldown:
                reverted_reason = "cooldown"
            elif short_window_guard:
                reverted_reason = "short_window_promotion_guard"

            if reverted_reason is not None:
                best = dict(previous)
                best["adst_action"] = f"conservative_{reverted_reason}_hold"
                best["conservative_reverted_action"] = action_before_guard
                best["conservative_required_gain"] = float(required_gain)
                train_conf, half_conf = self._update_momentum(best)
                best["momentum_train_confidence"] = train_conf
                best["momentum_half_confidence"] = half_conf
            elif changed:
                self._last_change_call = self._adst_call_index

        if self._momentum_trace:
            trace = self._momentum_trace[-1]
            trace["conservative_action_before_guard"] = action_before_guard
            trace["conservative_reverted_reason"] = reverted_reason or "none"
            trace["quick_validation_block_count"] = len(self._last_quick_block_aps)
            trace["quick_validation_block_pr_auc"] = json.dumps(
                self._last_quick_block_aps
            )
            trace["cooldown_origins"] = self.cooldown_origins
            trace["short_window_min_abs_gain"] = SHORT_WINDOW_MIN_ABS_GAIN
            trace["short_window_min_rel_gain"] = SHORT_WINDOW_MIN_REL_GAIN
        best["conservative_quick_validation_blocks"] = self.quick_validation_blocks
        best["conservative_cooldown_origins"] = self.cooldown_origins
        return best, candidates

    def _select_lambda(
        self,
        b1_records: list[dict[str, Any]],
        b2_records: list[dict[str, Any]],
        selected_b2_train_days: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        selected, rows = super()._select_lambda(
            b1_records, b2_records, selected_b2_train_days
        )
        selected["selected_lambda"] = self.lambda_cap
        selected["selected_fusion_variant"] = PRIMARY_VARIANT
        selected["lambda_cap"] = self.lambda_cap
        for row in rows:
            row["selected_lambda"] = self.lambda_cap
            row["selected_fusion_variant"] = PRIMARY_VARIANT
            row["lambda_cap"] = self.lambda_cap
        return selected, rows

    def _variant_specs(
        self, pareto_lambda: np.ndarray
    ) -> dict[str, tuple[np.ndarray, str]]:
        return {
            "pareto_current": (
                pareto_lambda,
                "current_pareto_full_gpu",
            ),
            PRIMARY_VARIANT: (
                np.minimum(pareto_lambda, self.lambda_cap),
                "pareto_b1_capped_full_gpu",
            ),
            "fixed_lambda_0.6": (
                np.full_like(pareto_lambda, 0.6, dtype=np.float64),
                "fixed_binary_lambda_0.6_full_gpu",
            ),
            "b2_only_reference": (
                np.zeros_like(pareto_lambda, dtype=np.float64),
                "b2_only_reference",
            ),
            "b1_only_reference": (
                np.ones_like(pareto_lambda, dtype=np.float64),
                "b1_only_reference",
            ),
        }

    def _variant_metrics(
        self,
        variant: str,
        method: str,
        lambdas: np.ndarray,
        b1_scores: np.ndarray,
        b2_scores: np.ndarray,
        gt: np.ndarray,
        selected: dict[str, Any],
        cascade_meta: dict[str, Any],
        cycle_idx: int,
        cycle_start: int,
        cycle_end: int,
        pareto_lambda: np.ndarray,
    ) -> dict[str, Any]:
        fused_scores = lambdas * b1_scores + (1.0 - lambdas) * b2_scores
        y = gt.ravel()
        fused_rank = integrated._rank_matrix(fused_scores)
        b1_rank = integrated._rank_matrix(b1_scores)
        b2_rank = integrated._rank_matrix(b2_scores)
        top_k = min(100, self.engine.num_gpus)
        positives = int(gt.sum())
        prevalence = float(gt.mean())

        def top_metrics(ranks: np.ndarray) -> tuple[float, float]:
            hits = int((gt & (ranks <= top_k)).sum())
            recall = hits / max(positives, 1)
            lift = (hits / max(len(gt) * top_k, 1)) / max(prevalence, 1e-12)
            return recall, lift

        ap_fused = base.safe_average_precision(y, fused_scores.ravel())
        recall_fused, lift_fused = top_metrics(fused_rank)
        recall_b1, lift_b1 = top_metrics(b1_rank)
        recall_b2, lift_b2 = top_metrics(b2_rank)
        return {
            "origin_idx": int(cycle_idx),
            "model_origin_time": integrated._time_text(self.engine, cycle_start),
            "test_start_time": integrated._time_text(self.engine, cycle_start),
            "test_end_time": integrated._time_text(self.engine, cycle_end - 1),
            "evaluation_scope": "common_terminal_heldout",
            "prediction_population": "all_gpus_before_top100",
            "policy": self.policy,
            "fusion_variant": variant,
            "fusion_method": method,
            "selected_b1_train_days": int(selected["selected_train_days"]),
            "selected_b1_half_life_days": int(selected["selected_half_life_days"]),
            "selected_b2_train_days": int(selected["selected_b2_train_days"]),
            "selected_b2_half_life_days": pareto.B2_HALF_LIFE,
            "selected_b1_input_hours": "1,6,24",
            "selected_cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
            "selected_lambda": float(self.lambda_cap if variant == PRIMARY_VARIANT else (self.pareto_alpha if variant == "pareto_current" else lambdas.flat[0])),
            "pareto_alpha": float(self.pareto_alpha),
            "pareto_lambda_mean": float(pareto_lambda.mean()),
            "pareto_lambda_min": float(pareto_lambda.min()),
            "pareto_lambda_max": float(pareto_lambda.max()),
            "effective_lambda_mean": float(lambdas.mean()),
            "effective_lambda_min": float(lambdas.min()),
            "effective_lambda_max": float(lambdas.max()),
            "pareto_history_cutoff": "decision_time_minus_10min",
            "lambda_cap": self.lambda_cap,
            "pr_auc_fused": float(ap_fused),
            "pr_auc_b1": float(base.safe_average_precision(y, b1_scores.ravel())),
            "pr_auc_b2": float(base.safe_average_precision(y, b2_scores.ravel())),
            "pr_auc_fused_normalized": float(
                integrated._normalized_pr_auc(ap_fused, prevalence)
            ),
            "roc_auc_fused": float(base.safe_roc_auc(y, fused_scores.ravel())),
            "roc_auc_b1": float(base.safe_roc_auc(y, b1_scores.ravel())),
            "roc_auc_b2": float(base.safe_roc_auc(y, b2_scores.ravel())),
            "recall_at_100": float(recall_fused),
            "lift_at_100": float(lift_fused),
            "recall_at_100_b1": float(recall_b1),
            "lift_at_100_b1": float(lift_b1),
            "recall_at_100_b2": float(recall_b2),
            "lift_at_100_b2": float(lift_b2),
            "positives": positives,
            "test_prevalence": prevalence,
            "gpu_count_evaluated": int(self.engine.num_gpus),
            "decision_time_count": int(len(gt)),
            "top_k": int(top_k),
        }

    def _load_cycle_checkpoint(self, cycle_idx: int):
        loaded = super()._load_cycle_checkpoint(cycle_idx)
        if loaded is not None:
            metrics, tape = loaded
            metrics.setdefault("selected_lambda", float(self.lambda_cap))
            metrics.setdefault("fusion_variant", PRIMARY_VARIANT)
            path = self.checkpoint_dir / f"variant_metrics_cycle_{cycle_idx:04d}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._variant_rows.extend(payload.get("rows", []))
            loaded = (metrics, tape)
        return loaded

    def _predict_fusion_cycle(
        self,
        b1_models: dict[str, Any],
        cascade_model: Any,
        cascade_meta: dict[str, Any],
        b2_model: dict[str, Any],
        selected: dict[str, Any],
        cycle_idx: int,
        cycle_start: int,
        cycle_end: int,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        test_bins = np.arange(
            cycle_start, cycle_end, self.test_stride_bins, dtype=np.int32
        )
        if len(test_bins) == 0:
            raise ValueError(f"Empty held-out test cycle {cycle_idx}.")
        n_gpu = self.engine.num_gpus
        b1_scores = np.zeros((len(test_bins), n_gpu), dtype=np.float64)
        b2_scores = np.zeros_like(b1_scores)
        pareto_lambda = np.zeros_like(b1_scores)

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
            b2_df = self.engine.extract_branch2_features(
                bins, gpus, self.history_map
            )
            raw_b2 = b2_model["model"].predict_proba(b2_df)[:, 1]
            b2 = base.adjusted_probability(
                raw_b2,
                float(b2_model["true_prior"]),
                float(b2_model["sample_prior"]),
            )
            counts = pareto._pareto_counts(
                self.history_map,
                int(self.engine.bin_start_ns[current_bin]),
                n_gpu,
            )
            b1_scores[rel_idx] = b1
            b2_scores[rel_idx] = b2
            pareto_lambda[rel_idx] = pareto._pareto_lambda(
                counts, alpha=self.pareto_alpha
            )

        gt = self.gt_matrix[test_bins]
        specs = self._variant_specs(pareto_lambda)
        rows: list[dict[str, Any]] = []
        for variant, (lambdas, method) in specs.items():
            row = self._variant_metrics(
                variant,
                method,
                lambdas,
                b1_scores,
                b2_scores,
                gt,
                selected,
                cascade_meta,
                cycle_idx,
                cycle_start,
                cycle_end,
                pareto_lambda,
            )
            rows.append(row)
        self._variant_rows.extend(rows)
        integrated._write_json_atomic(
            self.checkpoint_dir / f"variant_metrics_cycle_{cycle_idx:04d}.json",
            {"cycle_idx": int(cycle_idx), "rows": rows},
        )

        primary_lambdas = specs[PRIMARY_VARIANT][0]
        primary_scores = (
            primary_lambdas * b1_scores + (1.0 - primary_lambdas) * b2_scores
        )
        primary_row = next(row for row in rows if row["fusion_variant"] == PRIMARY_VARIANT)
        primary_rank = integrated._rank_matrix(primary_scores)
        b1_rank = integrated._rank_matrix(b1_scores)
        b2_rank = integrated._rank_matrix(b2_scores)
        top_k = min(100, n_gpu)
        rel_idx, gpu_idx = np.where(primary_rank <= top_k)
        tape = pd.DataFrame(
            {
                "decision_time": pd.to_datetime(
                    self.engine.bin_start_ns[test_bins][rel_idx],
                    unit="ns",
                    utc=True,
                ),
                "gpu_id": self.engine.gpu_ids[gpu_idx],
                "fusion_variant": PRIMARY_VARIANT,
                "fused_risk": primary_scores[rel_idx, gpu_idx],
                "b1_risk": b1_scores[rel_idx, gpu_idx],
                "b2_risk": b2_scores[rel_idx, gpu_idx],
                "pareto_lambda": pareto_lambda[rel_idx, gpu_idx],
                "effective_lambda": primary_lambdas[rel_idx, gpu_idx],
                "fused_rank": primary_rank[rel_idx, gpu_idx],
                "b1_rank": b1_rank[rel_idx, gpu_idx],
                "b2_rank": b2_rank[rel_idx, gpu_idx],
                "b1_train_days": int(selected["selected_train_days"]),
                "b1_half_life_days": int(selected["selected_half_life_days"]),
                "b2_train_days": int(selected["selected_b2_train_days"]),
                "b2_half_life_days": pareto.B2_HALF_LIFE,
                "cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
                "pareto_alpha": float(self.pareto_alpha),
                "lambda_cap": self.lambda_cap,
                "target_24h": gt[rel_idx, gpu_idx].astype(np.uint8),
            }
        )
        return primary_row, tape

    def _write_manifest(self, *args: Any, **kwargs: Any) -> None:
        super()._write_manifest(*args, **kwargs)
        path = self.output_dir / "experiment_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["runner"] = "run_adst_conservative_b1_shrink_fusion.py"
        payload["adst"].update(
            {
                "method": "conservative_decoupled_momentum",
                "beta": self.momentum_beta,
                "confidence_threshold": self.momentum_threshold,
                "performance_drop_trigger": self.momentum_drop_pct,
                "quick_validation_blocks": self.quick_validation_blocks,
                "cooldown_origins": self.cooldown_origins,
                "short_window_min_abs_gain": SHORT_WINDOW_MIN_ABS_GAIN,
                "short_window_min_rel_gain": SHORT_WINDOW_MIN_REL_GAIN,
                "architecture_matched_quick_proxy": False,
            }
        )
        payload["fusion_variants"] = {
            "primary": PRIMARY_VARIANT,
            "lambda_cap": self.lambda_cap,
            "variants": [],
        }
        payload["fusion_variants"]["variants"] = [
            "pareto_current",
            PRIMARY_VARIANT,
            "fixed_lambda_0.6",
            "b2_only_reference",
            "b1_only_reference",
        ]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def run(self):
        metrics, tape = super().run()
        variant_df = pd.DataFrame(self._variant_rows)
        variant_df.to_csv(
            self.output_dir / "fusion_variant_metrics.csv", index=False
        )
        grouped = (
            variant_df.groupby("fusion_variant", sort=False)
            .agg(
                pr_auc_fused_mean=("pr_auc_fused", "mean"),
                pr_auc_fused_median=("pr_auc_fused", "median"),
                pr_auc_fused_q25=("pr_auc_fused", lambda s: float(s.quantile(0.25))),
                normalized_pr_auc_mean=("pr_auc_fused_normalized", "mean"),
                roc_auc_fused_mean=("roc_auc_fused", "mean"),
                recall_at_100_mean=("recall_at_100", "mean"),
                lift_at_100_mean=("lift_at_100", "mean"),
                effective_lambda_mean=("effective_lambda_mean", "mean"),
            )
            .reset_index()
        )
        grouped.to_csv(
            self.output_dir / "fusion_variant_summary.csv", index=False
        )
        report = [
            "# Conservative Momentum-ADST + B1-shrunk Pareto Fusion",
            "",
            "## Material Passport",
            "- **Status**: COMPLETED",
            "- **Goal IDs**: G-004, G-007, G-008",
            "- **Target**: All-XID `[13, 31, 43, 45, 94]`, 24-hour horizon",
            "- **Split**: time-ordered Sliding Training, 36-hour purge, common terminal held-out 20%",
            "- **Branch 2**: strict History-only (`xid_count_30d`, `days_since_xid`)",
            "- **Primary evaluation**: full GPU-time scores → B1-capped Pareto Fusion → Top-100",
            "",
            "## Conservative ADST changes",
            f"- Momentum beta: `{self.momentum_beta}`",
            f"- Confidence threshold: `{self.momentum_threshold}`",
            f"- Performance-drop trigger: `{self.momentum_drop_pct}`",
            f"- Quick validation: latest `{self.quick_validation_blocks}` blocks, median PR-AUC",
            f"- Window-change cooldown: `{self.cooldown_origins}` origins",
            f"- 7-day promotion guard: `+{SHORT_WINDOW_MIN_ABS_GAIN}` absolute and `+{SHORT_WINDOW_MIN_REL_GAIN:.0%}` relative requirement",
            "- The lightweight Logistic proxy is retained in this run to isolate setting changes; architecture-matched ADST proxy remains a separate ablation.",
            "",
            "## Fusion variants",
            f"- Primary: `{PRIMARY_VARIANT}` with effective Lambda `min(Pareto Lambda, {self.lambda_cap})`",
            "- Controls: current Pareto, fixed Lambda 0.6, B2-only, B1-only",
            "- No terminal held-out label was used to select a variant.",
            "",
            "## Primary held-out result",
            f"- **Fusion PR-AUC**: `{metrics['pr_auc_fused'].mean():.6f}`",
            f"- **Branch 1 PR-AUC**: `{metrics['pr_auc_b1'].mean():.6f}`",
            f"- **Branch 2 PR-AUC**: `{metrics['pr_auc_b2'].mean():.6f}`",
            f"- **Fusion ROC-AUC**: `{metrics['roc_auc_fused'].mean():.6f}`",
            f"- **Recall@100**: `{metrics['recall_at_100'].mean():.2%}`",
            f"- **Lift@100**: `{metrics['lift_at_100'].mean():.3f}x`",
            "",
            "## Output",
            "- `fusion_metrics.csv` (primary capped Pareto tape metrics)",
            "- `fusion_variant_metrics.csv` (all variants by held-out cycle)",
            "- `fusion_variant_summary.csv` (variant means/medians)",
            "- `fusion_risk_tape.parquet` (primary Top-100 tape)",
            "- `momentum_selection_trace.csv`",
            "- `experiment_manifest.json`",
            "- `checkpoints/`",
        ]
        (self.output_dir / "fusion_report.md").write_text(
            "\n".join(report), encoding="utf-8"
        )
        return metrics, tape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservative Momentum-ADST with B1-shrunk Pareto Fusion"
    )
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument("--pareto-alpha", type=float, default=1.0)
    parser.add_argument("--lambda-cap", type=float, default=DEFAULT_LAMBDA_CAP)
    parser.add_argument("--include-topology", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

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
    pipeline = ConservativeReliabilityFusion(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=base.PROJECT_ROOT / args.output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        resume=args.resume,
        branch2_mode="history_0910",
        policy="adst",
        pareto_alpha=args.pareto_alpha,
        lambda_cap=args.lambda_cap,
        include_topology=args.include_topology,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

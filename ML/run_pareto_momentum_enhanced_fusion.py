"""All-XID Pareto Fusion with separate Sliding and Momentum-ADST policies.

The runner reuses the existing diversified Branch 1 + History-only Branch 2
pipeline, but replaces the final global Lambda Fusion with the GitHub Pareto
formula applied to the full GPU-time score matrix before ranking.

``--policy sliding`` performs a fixed terminal policy after development
selection. ``--policy adst`` uses a decoupled EMA state for Branch 1's
training window and recency half-life during development selection. The
terminal held-out interval remains frozen for both policies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_bidirectional_adst_fusion as base
import run_branch1_diversified_parallel_cascade as b1mod
import run_fusion_diversified_b1_history_b2 as integrated
from branch1_enhanced_features import EnhancedTelemetryEngine, enhanced_feature_manifest


DAY_NS = integrated.DAY_NS
STEP_NS = integrated.STEP_NS
PURGE_BINS = integrated.PURGE_BINS
VALIDATION_BLOCKS = integrated.VALIDATION_BLOCKS
VALIDATION_BINS = integrated.VALIDATION_BINS
POOL_ORIGINS = integrated.POOL_ORIGINS
B2_HALF_LIFE = integrated.B2_HALF_LIFE


def _pareto_counts(history_map: dict[int, np.ndarray], decision_time_ns: int, n_gpu: int) -> np.ndarray:
    """Count prior XID events using the same 10-minute unavailable-data cutoff."""

    cutoff_ns = int(decision_time_ns) - int(base.BUFFER_MASK_BINS) * int(STEP_NS)
    start_ns = cutoff_ns - int(30 * DAY_NS)
    counts = np.zeros(n_gpu, dtype=np.float64)
    for gpu in range(n_gpu):
        events = history_map.get(gpu)
        if events is None or len(events) == 0:
            continue
        right = int(np.searchsorted(events, cutoff_ns, side="left"))
        left = int(np.searchsorted(events, start_ns, side="left"))
        counts[gpu] = max(0, right - left)
    return counts


def _pareto_lambda(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    return np.clip(np.power(1.0 / (counts + 1.0), float(alpha)), 0.05, 1.0)

class ParetoMomentumFusion(integrated.DiversifiedBranchFusion):
    """Existing diversified Fusion pipeline with valid full-population Pareto."""

    def __init__(
        self,
        *args,
        policy: str,
        pareto_alpha: float = 1.0,
        include_topology: bool = False,
        momentum_beta: float = 0.7,
        momentum_threshold: float = 0.6,
        momentum_drop_pct: float = 0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if policy not in {"sliding", "adst"}:
            raise ValueError(f"Unsupported policy: {policy}")
        self.policy = policy
        self.pareto_alpha = float(pareto_alpha)
        self.include_topology = bool(include_topology)
        self.momentum_beta = float(momentum_beta)
        self.momentum_threshold = float(momentum_threshold)
        self.momentum_drop_pct = float(momentum_drop_pct)
        self._train_confidence = {int(v): 0.0 for v in b1mod.TRAIN_DAYS_GRID}
        self._half_confidence = {int(v): 0.0 for v in b1mod.HALF_LIFE_GRID}
        self._momentum_trace: list[dict[str, Any]] = []

    def _quick_validate(self, origin_bin: int, previous: dict[str, Any]) -> float:
        """One recent validation block, using the previous configuration only."""

        blocks = self._selection_blocks(int(origin_bin))
        if not blocks:
            return float(previous.get("val_pr_auc_mean", 0.0))
        block = blocks[-1]
        train_days = int(previous["selected_train_days"])
        half_life = int(previous["selected_half_life_days"])
        train_bins, train_gpus, train_labels, _ = self._sample_training(
            int(block["train_end"]), train_days, int(origin_bin) + 818_181
        )
        scores: list[np.ndarray] = []
        weights = b1mod._recency_weights(train_bins, int(block["train_end"]), half_life)
        for obs in b1mod.OBS_HOURS_GRID:
            train_x = self.engine.extract_branch1_features(train_bins, train_gpus, obs)[0]
            val_x = self.engine.extract_branch1_features(block["val_bins"], block["val_gpus"], obs)[0]
            model = b1mod._fit_logistic(
                train_x,
                train_labels,
                weights,
                self.seed + int(origin_bin) + int(obs) + 71_001,
            )
            scores.append(model.predict_proba(val_x)[:, 1])
        return base.safe_average_precision(block["val_labels"], np.mean(scores, axis=0))

    def _restricted_window_search(
        self,
        origin_bin: int,
        previous: dict[str, Any],
        fixed_train: bool,
        fixed_half: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Reuse the proven selector with one dimension held fixed."""

        old_train = b1mod.TRAIN_DAYS_GRID
        old_half = b1mod.HALF_LIFE_GRID
        try:
            if fixed_train:
                b1mod.TRAIN_DAYS_GRID = (int(previous["selected_train_days"]),)
            if fixed_half:
                b1mod.HALF_LIFE_GRID = (int(previous["selected_half_life_days"]),)
            return super().select_branch1_window(int(origin_bin), previous, "development")
        finally:
            b1mod.TRAIN_DAYS_GRID = old_train
            b1mod.HALF_LIFE_GRID = old_half

    def _update_momentum(self, selected: dict[str, Any]) -> tuple[float, float]:
        train_choice = int(selected["selected_train_days"])
        half_choice = int(selected["selected_half_life_days"])
        for value in self._train_confidence:
            self._train_confidence[value] = self.momentum_beta * self._train_confidence[value] + (1.0 - self.momentum_beta) * float(value == train_choice)
        for value in self._half_confidence:
            self._half_confidence[value] = self.momentum_beta * self._half_confidence[value] + (1.0 - self.momentum_beta) * float(value == half_choice)
        return self._train_confidence[train_choice], self._half_confidence[half_choice]

    def select_branch1_window(self, origin_bin: int, previous: dict[str, Any] | None, stage: str):
        if self.policy == "sliding" or previous is None:
            best, candidates = super().select_branch1_window(origin_bin, previous, stage)
            action = "sliding_full_development_search" if self.policy == "sliding" else "momentum_initial_full_search"
            best["adst_action"] = action
            best["momentum_fit_plan"] = "full_grid"
            train_conf, half_conf = self._update_momentum(best)
            best["momentum_train_confidence"] = train_conf
            best["momentum_half_confidence"] = half_conf
            self._momentum_trace.append(
                {
                    "origin_bin": int(origin_bin),
                    "policy": self.policy,
                    "action": action,
                    "selected_train_days": int(best["selected_train_days"]),
                    "selected_half_life_days": int(best["selected_half_life_days"]),
                    "candidate_count": len(candidates),
                    "quick_validation_pr_auc": np.nan,
                    "performance_drop": np.nan,
                    "train_confidence": train_conf,
                    "half_confidence": half_conf,
                }
            )
            return best, candidates

        train_conf = self._train_confidence.get(int(previous["selected_train_days"]), 0.0)
        half_conf = self._half_confidence.get(int(previous["selected_half_life_days"]), 0.0)
        action = "momentum_partial_search"
        quick_ap = np.nan
        drop = np.nan

        if train_conf >= self.momentum_threshold and half_conf >= self.momentum_threshold:
            quick_ap = self._quick_validate(origin_bin, previous)
            previous_ap = float(previous.get("val_pr_auc_mean", quick_ap))
            drop = (previous_ap - quick_ap) / max(previous_ap, 1e-9)
            if drop <= self.momentum_drop_pct:
                best = dict(previous)
                best["adst_action"] = "momentum_skip_hold"
                best["momentum_fit_plan"] = "quick_validate_only"
                candidate = {
                    "selection_role": "branch1_proxy",
                    "selection_stage": stage,
                    "selection_origin_idx": int(origin_bin),
                    "selection_origin_time": integrated._time_text(self.engine, origin_bin),
                    "candidate_train_days": int(previous["selected_train_days"]),
                    "candidate_half_life_days": int(previous["selected_half_life_days"]),
                    "candidate_input_hours": "all_1_6_24",
                    "validation_block_count": VALIDATION_BLOCKS,
                    "validation_block_days": 3,
                    "purge_hours": int(base.PURGE_NS // integrated.HOUR_NS),
                    "val_pr_auc_mean": float(quick_ap),
                    "val_pr_auc_median": float(quick_ap),
                    "val_pr_auc_q25": float(quick_ap),
                    "val_pr_auc_std": 0.0,
                    "val_pr_auc_min": float(quick_ap),
                    "val_roc_auc_mean": np.nan,
                }
                train_conf, half_conf = self._update_momentum(best)
                best["val_pr_auc_mean"] = float(quick_ap)
                best["momentum_train_confidence"] = train_conf
                best["momentum_half_confidence"] = half_conf
                self._momentum_trace.append(
                    {
                        "origin_bin": int(origin_bin),
                        "policy": self.policy,
                        "action": "momentum_skip_hold",
                        "selected_train_days": int(best["selected_train_days"]),
                        "selected_half_life_days": int(best["selected_half_life_days"]),
                        "candidate_count": 0,
                        "quick_validation_pr_auc": float(quick_ap),
                        "performance_drop": float(drop),
                        "train_confidence": train_conf,
                        "half_confidence": half_conf,
                    }
                )
                return best, [candidate]
            action = "momentum_trigger_rescan"

        fixed_train = train_conf >= self.momentum_threshold and half_conf < self.momentum_threshold
        fixed_half = half_conf >= self.momentum_threshold and train_conf < self.momentum_threshold
        if fixed_train:
            action = "momentum_partial_search_hold_train"
        elif fixed_half:
            action = "momentum_partial_search_hold_half"
        else:
            action = "momentum_full_rescan"
        best, candidates = (
            self._restricted_window_search(origin_bin, previous, fixed_train, fixed_half)
            if action != "momentum_full_rescan"
            else super().select_branch1_window(origin_bin, previous, stage)
        )
        best["adst_action"] = action
        best["momentum_fit_plan"] = "partial_grid" if "partial" in action else "full_grid"
        train_conf, half_conf = self._update_momentum(best)
        best["momentum_train_confidence"] = train_conf
        best["momentum_half_confidence"] = half_conf
        self._momentum_trace.append(
            {
                "origin_bin": int(origin_bin),
                "policy": self.policy,
                "action": action,
                "selected_train_days": int(best["selected_train_days"]),
                "selected_half_life_days": int(best["selected_half_life_days"]),
                "candidate_count": len(candidates),
                "quick_validation_pr_auc": quick_ap,
                "performance_drop": drop,
                "train_confidence": train_conf,
                "half_confidence": half_conf,
            }
        )
        return best, candidates

    def _select_lambda(self, b1_records, b2_records, selected_b2_train_days):
        # ``selected_lambda`` is retained for compatibility with the parent
        # manifest; the actual per-GPU weight is computed in the cycle method.
        row = {
            "selection_role": "fusion_pareto_alpha",
            "candidate_pareto_alpha": self.pareto_alpha,
            "selected_lambda": self.pareto_alpha,
            "candidate_b2_train_days": int(selected_b2_train_days),
            "candidate_b2_half_life_days": B2_HALF_LIFE,
            "val_pr_auc_mean": np.nan,
            "val_roc_auc_mean": np.nan,
            "pool_origin_count": POOL_ORIGINS,
            "pool_validation_block_count": len(b1_records),
            # Parent-runner compatibility: Pareto alpha is code-fixed,
            # while selection still uses paired full-GPU development blocks.
            "paired_validation_block_count": len(b1_records),
            "selection_stage": "code_faithful_pareto_formula",
        }
        return dict(row), [row]

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
    ):
        test_bins = np.arange(cycle_start, cycle_end, self.test_stride_bins, dtype=np.int32)
        if len(test_bins) == 0:
            raise ValueError(f"Empty held-out test cycle {cycle_idx}.")
        n_gpu = self.engine.num_gpus
        b1_scores = np.zeros((len(test_bins), n_gpu), dtype=np.float64)
        b2_scores = np.zeros_like(b1_scores)
        fused_scores = np.zeros_like(b1_scores)
        lambda_matrix = np.zeros_like(b1_scores)
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
            b2 = base.adjusted_probability(raw_b2, float(b2_model["true_prior"]), float(b2_model["sample_prior"]))
            counts = _pareto_counts(self.history_map, int(self.engine.bin_start_ns[current_bin]), n_gpu)
            lambdas = _pareto_lambda(counts, alpha=self.pareto_alpha)
            b1_scores[rel_idx] = b1
            b2_scores[rel_idx] = b2
            lambda_matrix[rel_idx] = lambdas
            fused_scores[rel_idx] = lambdas * b1 + (1.0 - lambdas) * b2

        gt = self.gt_matrix[test_bins]
        y = gt.ravel()
        fused_rank = integrated._rank_matrix(fused_scores)
        b1_rank = integrated._rank_matrix(b1_scores)
        b2_rank = integrated._rank_matrix(b2_scores)
        top_k = min(100, n_gpu)
        positives = int(gt.sum())
        prevalence = float(gt.mean())

        def top_metrics(ranks):
            hits = int((gt & (ranks <= top_k)).sum())
            return hits / max(positives, 1), (hits / max(len(test_bins) * top_k, 1)) / max(prevalence, 1e-12)

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
            "policy": self.policy,
            "selected_b1_train_days": int(selected["selected_train_days"]),
            "selected_b1_half_life_days": int(selected["selected_half_life_days"]),
            "selected_b2_train_days": int(selected["selected_b2_train_days"]),
            "selected_b2_half_life_days": B2_HALF_LIFE,
            "selected_b1_input_hours": "1,6,24",
            "selected_cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
            "selected_lambda": float(self.pareto_alpha),
            "pareto_alpha": float(self.pareto_alpha),
            "pareto_lambda_mean": float(lambda_matrix.mean()),
            "pareto_lambda_min": float(lambda_matrix.min()),
            "pareto_lambda_max": float(lambda_matrix.max()),
            "pareto_history_cutoff": "decision_time_minus_10min",
            "fusion_method": "github_pareto_formula_full_gpu_before_top100",
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
            "gpu_count_evaluated": int(n_gpu),
            "decision_time_count": int(len(test_bins)),
            "top_k": int(top_k),
        }
        rel_idx, gpu_idx = np.where(fused_rank <= top_k)
        tape = pd.DataFrame(
            {
                "decision_time": pd.to_datetime(self.engine.bin_start_ns[test_bins][rel_idx], unit="ns", utc=True),
                "gpu_id": self.engine.gpu_ids[gpu_idx],
                "fused_risk": fused_scores[rel_idx, gpu_idx],
                "b1_risk": b1_scores[rel_idx, gpu_idx],
                "b2_risk": b2_scores[rel_idx, gpu_idx],
                "pareto_lambda": lambda_matrix[rel_idx, gpu_idx],
                "fused_rank": fused_rank[rel_idx, gpu_idx],
                "b1_rank": b1_rank[rel_idx, gpu_idx],
                "b2_rank": b2_rank[rel_idx, gpu_idx],
                "b1_train_days": int(selected["selected_train_days"]),
                "b1_half_life_days": int(selected["selected_half_life_days"]),
                "b2_train_days": int(selected["selected_b2_train_days"]),
                "b2_half_life_days": B2_HALF_LIFE,
                "cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
                "pareto_alpha": float(self.pareto_alpha),
                "target_24h": gt[rel_idx, gpu_idx].astype(np.uint8),
            }
        )
        return metrics, tape

    def _write_manifest(self, *args, **kwargs):
        super()._write_manifest(*args, **kwargs)
        path = self.output_dir / "experiment_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        payload.update(
            {
                "policy": self.policy,
                "pareto": {
                    "formula": "clip((1/(count_30d+1))^alpha, 0.05, 1.0)",
                    "alpha": self.pareto_alpha,
                    "population": "all_gpu_time_before_top100",
                    "history_cutoff": "decision_time_minus_10min",
                },
                "adst": {
                    "method": "decoupled_momentum" if self.policy == "adst" else "fixed_sliding_terminal_policy",
                    "adwin_used": False,
                    "beta": self.momentum_beta,
                    "confidence_threshold": self.momentum_threshold,
                    "performance_drop_trigger": self.momentum_drop_pct,
                },
                "branch1_enhanced": enhanced_feature_manifest(self.engine),
            }
        )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def run(self):
        metrics, tape = super().run()
        if self._momentum_trace:
            pd.DataFrame(self._momentum_trace).to_csv(
                self.output_dir / "momentum_selection_trace.csv", index=False
            )
        report = [
            "# All-XID Enhanced Branch 1 + History-only Branch 2 Pareto Fusion",
            "",
            "## Material Passport",
            "- **Status**: COMPLETED",
            f"- **Policy**: `{self.policy}`",
            "- **Target**: All-XID `[13, 31, 43, 45, 94]`, 24-hour horizon",
            "- **Split**: Sliding Training, 36-hour purge, common terminal held-out 20%",
            "- **Evaluation**: full GPU-time scores → Pareto Fusion → Top-100 ranking",
            "- **Branch 2**: strict History-only (`xid_count_30d`, `days_since_xid`)",
            "- **ADWIN**: not used; previous ADWIN experiments remain historical artifacts",
            "",
            "## Branch 1 feature contract",
            "- Existing 1/6/24-hour telemetry inputs retained.",
            f"- Cross-Metric features: `{', '.join(enhanced_feature_manifest(self.engine)['cross_metric_features'])}`",
            f"- Topology features: `{len(enhanced_feature_manifest(self.engine)['topology_features'])}`; source `{self.engine.topology_source if self.include_topology else 'disabled'}`",
            "- Excluded from Branch 1: `xid_count_30d`, `days_since_xid`, XID filters, calendar features, and learned GNN layers.",
            "",
            "## Pareto limitations corrected",
            "- The GitHub post-hoc analyzer operates on an existing Top-100 tape; this runner computes Pareto weights for every evaluated GPU-time first.",
            "- Rankings are recomputed after Fusion, so GPUs previously outside Top-100 can enter the tape.",
            "- History counts use the decision-time minus 10-minute unavailable-data cutoff.",
            "- Alpha is fixed at the code-faithful `1.0`; held-out labels are not used for tuning.",
            "",
            "## Terminal held-out mean",
            f"- **Fusion PR-AUC**: `{metrics['pr_auc_fused'].mean():.6f}`",
            f"- **Branch 1 PR-AUC**: `{metrics['pr_auc_b1'].mean():.6f}`",
            f"- **Branch 2 PR-AUC**: `{metrics['pr_auc_b2'].mean():.6f}`",
            f"- **Fusion ROC-AUC**: `{metrics['roc_auc_fused'].mean():.6f}`",
            f"- **Recall@100**: `{metrics['recall_at_100'].mean():.2%}`",
            f"- **Lift@100**: `{metrics['lift_at_100'].mean():.3f}x`",
            f"- **Mean Pareto lambda**: `{metrics['pareto_lambda_mean'].mean():.4f}`",
            f"- **Evaluated GPUs per decision time**: `{int(metrics['gpu_count_evaluated'].iloc[0])}`",
            f"- **Held-out cycles**: `{len(metrics)}`",
            "",
            "## Output",
            "- `fusion_metrics.csv`",
            "- `fusion_risk_tape.parquet`",
            "- `fusion_selection_history.csv`",
            "- `momentum_selection_trace.csv`",
            "- `experiment_manifest.json`",
            "- `checkpoints/`",
        ]
        (self.output_dir / "fusion_report.md").write_text("\n".join(report), encoding="utf-8")
        return metrics, tape


def main() -> None:
    parser = argparse.ArgumentParser(description="All-XID Sliding/Momentum-ADST Pareto Fusion")
    parser.add_argument("--policy", choices=["sliding", "adst"], required=True)
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument("--pareto-alpha", type=float, default=1.0)
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
    pipeline = ParetoMomentumFusion(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=base.PROJECT_ROOT / args.output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        resume=args.resume,
        branch2_mode="history_0910",
        policy=args.policy,
        pareto_alpha=args.pareto_alpha,
        include_topology=args.include_topology,
    )
    pipeline.run()


if __name__ == "__main__":
    main()


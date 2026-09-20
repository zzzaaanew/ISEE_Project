"""All-XID integrated Fusion with the improved Branch 1 cascade.

The GitHub-base execution order is preserved:

    Branch 1 and Branch 2 full-GPU predictions -> Lambda Fusion -> Top-100 tape

Branch 1 is the diversified parallel ensemble plus soft residual cascade from
``run_branch1_diversified_parallel_cascade.py``.  Branch 2 is the 2026-09-10
History-only historical logistic contract.  This runner is deliberately
separate so the earlier branch-only experiments remain unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_bidirectional_adst_fusion as base
import run_sliding_adst_v2_branch_specific as v2
from run_branch1_diversified_parallel_cascade import (
    Branch1Diversified,
    CASCADE_ALPHA_GRID,
    DAY_NS,
    HOUR_NS,
    HALF_LIFE_GRID,
    OBS_HOURS_GRID,
    PURGE_BINS,
    STEP_NS,
    TEST_FRACTION,
    TRAIN_DAYS_GRID,
    VALIDATION_BINS,
    VALIDATION_BLOCKS,
    MODEL_FAMILIES,
    _aggregate,
    _normalized_pr_auc,
    _time_text,
    _write_json_atomic,
)


LAMBDA_GRID = tuple(float(value) for value in v2.LAMBDA_GRID)
B2_HALF_LIFE = 0
POOL_ORIGINS = 6


def _rank_matrix(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.uint16)
    np.put_along_axis(
        ranks,
        order,
        np.arange(1, scores.shape[1] + 1, dtype=np.uint16)[None, :],
        axis=1,
    )
    return ranks


class DiversifiedBranchFusion(Branch1Diversified):
    """Run the improved B1 and History-only B2 on one full-GPU population."""

    def _b2_validation_records(
        self,
        dev_origins: np.ndarray,
        latest_count: int = POOL_ORIGINS,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for origin_bin in dev_origins[-latest_count:]:
            for block in self._selection_blocks(int(origin_bin)):
                block_idx = int(block["block_idx"])
                train_end = int(block["train_end"])
                val_bins = np.asarray(block["val_bins"], dtype=np.int32)
                val_gpus = np.asarray(block["val_gpus"], dtype=np.int32)
                val_labels = np.asarray(block["val_labels"], dtype=np.uint8)
                val_df = self.engine.extract_branch2_features(
                    val_bins, val_gpus, self.history_map
                )
                predictions: dict[int, np.ndarray] = {}
                for train_days in TRAIN_DAYS_GRID:
                    train_start = max(
                        0, int(train_end) - int(train_days * DAY_NS // STEP_NS)
                    )
                    rng = np.random.default_rng(
                        self.seed
                        + int(origin_bin)
                        + 100_003 * (block_idx + 1)
                        + 1_009 * int(train_days)
                    )
                    train_bins, train_gpus, train_labels = self.sample_indices(
                        train_start, train_end, rng
                    )
                    if len(train_labels) == 0 or np.unique(train_labels).size < 2:
                        raise ValueError(
                            f"Branch 2 training block has insufficient classes at {origin_bin}."
                        )
                    train_df = self.engine.extract_branch2_features(
                        train_bins, train_gpus, self.history_map
                    )
                    model = v2._fit_weighted_logistic(
                        train_df,
                        train_labels,
                        np.ones(len(train_labels), dtype=np.float64),
                        self.seed
                        + int(origin_bin)
                        + block_idx
                        + int(train_days)
                        + B2_HALF_LIFE,
                    )
                    predictions[int(train_days)] = model.predict_proba(val_df)[:, 1]
                records.append(
                    {
                        "origin_bin": int(origin_bin),
                        "block_idx": block_idx,
                        "val_bins": val_bins,
                        "val_gpus": val_gpus,
                        "labels": val_labels,
                        "predictions": predictions,
                    }
                )
                print(
                    f"    B2 validation origin={int(origin_bin)} block={block_idx} "
                    f"n={len(val_labels):,}",
                    flush=True,
                )
        return records

    def _select_b2_window(
        self, records: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        for train_days in TRAIN_DAYS_GRID:
            aps = [
                base.safe_average_precision(
                    record["labels"], record["predictions"][int(train_days)]
                )
                for record in records
            ]
            rocs = [
                base.safe_roc_auc(
                    record["labels"], record["predictions"][int(train_days)]
                )
                for record in records
            ]
            summary = _aggregate(aps)
            rows.append(
                {
                    "selection_role": "branch2_history_only",
                    "candidate_b2_train_days": int(train_days),
                    "candidate_b2_half_life_days": B2_HALF_LIFE,
                    "candidate_L_obs_hours": "history_only",
                    "val_pr_auc_mean": summary["mean"],
                    "val_pr_auc_median": summary["median"],
                    "val_pr_auc_q25": summary["q25"],
                    "val_pr_auc_std": summary["std"],
                    "val_pr_auc_min": summary["min"],
                    "val_roc_auc_mean": float(np.mean(rocs)),
                    "pool_origin_count": POOL_ORIGINS,
                    "pool_validation_block_count": len(records),
                }
            )
        best = max(
            rows,
            key=lambda row: (
                float(row["val_pr_auc_mean"]),
                float(row["val_pr_auc_median"]),
                float(row["val_pr_auc_q25"]),
                -float(row["val_pr_auc_std"]),
                -int(row["candidate_b2_train_days"]),
            ),
        ).copy()
        best["selected_b2_train_days"] = int(best["candidate_b2_train_days"])
        best["selected_b2_half_life_days"] = B2_HALF_LIFE
        best["selection_stage"] = "pooled_development"
        return best, rows

    @staticmethod
    def _cascade_validation_predictions(
        records: list[dict[str, Any]],
        cascade_model: Any,
        cascade_meta: dict[str, Any],
    ) -> None:
        for record in records:
            stage1 = record["stage1"]
            p_stage2 = base.adjusted_probability(
                cascade_model.predict_proba(record["stage2_x"])[:, 1],
                float(cascade_meta["stage2_true_prior"]),
                float(cascade_meta["stage2_sample_prior"]),
            )
            record["cascade"] = np.clip(
                stage1["parallel"]
                + float(cascade_meta["selected_cascade_alpha"])
                * stage1["hardness"]
                * (p_stage2 - stage1["parallel"]),
                0.0,
                1.0,
            )

    def _select_lambda(
        self,
        b1_records: list[dict[str, Any]],
        b2_records: list[dict[str, Any]],
        selected_b2_train_days: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        b2_by_key = {
            (int(record["origin_bin"]), int(record["block_idx"])): record
            for record in b2_records
        }
        paired: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for record in b1_records:
            key = (int(record["origin_bin"]), int(record["block_idx"]))
            b2 = b2_by_key.get(key)
            if b2 is None:
                raise ValueError(f"Missing Branch 2 validation block for key={key}.")
            if not np.array_equal(record["labels"], b2["labels"]):
                raise ValueError(f"Branch validation labels are not aligned for key={key}.")
            paired.append(
                (
                    np.asarray(record["labels"], dtype=np.uint8),
                    np.asarray(record["cascade"], dtype=np.float64),
                    np.asarray(
                        b2["predictions"][int(selected_b2_train_days)],
                        dtype=np.float64,
                    ),
                )
            )
        rows: list[dict[str, Any]] = []
        for lambda_value in LAMBDA_GRID:
            aps: list[float] = []
            rocs: list[float] = []
            for labels, b1, b2 in paired:
                fused = lambda_value * b1 + (1.0 - lambda_value) * b2
                aps.append(base.safe_average_precision(labels, fused))
                rocs.append(base.safe_roc_auc(labels, fused))
            summary = _aggregate(aps)
            rows.append(
                {
                    "selection_role": "fusion_lambda",
                    "candidate_lambda": float(lambda_value),
                    "candidate_b2_train_days": int(selected_b2_train_days),
                    "candidate_b2_half_life_days": B2_HALF_LIFE,
                    "val_pr_auc_mean": summary["mean"],
                    "val_pr_auc_median": summary["median"],
                    "val_pr_auc_q25": summary["q25"],
                    "val_pr_auc_std": summary["std"],
                    "val_pr_auc_min": summary["min"],
                    "val_roc_auc_mean": float(np.mean(rocs)),
                    "pool_origin_count": POOL_ORIGINS,
                    "pool_validation_block_count": len(paired),
                }
            )
        best = max(
            rows,
            key=lambda row: (
                float(row["val_pr_auc_mean"]),
                float(row["val_pr_auc_median"]),
                float(row["val_pr_auc_q25"]),
                -float(row["val_pr_auc_std"]),
                -abs(float(row["candidate_lambda"]) - 0.5),
            ),
        ).copy()
        best["selected_lambda"] = float(best["candidate_lambda"])
        best["selection_stage"] = "pooled_development"
        best["paired_validation_block_count"] = len(paired)
        return best, rows

    def _fit_final_b2(self, train_days: int, test_start: int) -> dict[str, Any]:
        train_end = int(test_start - PURGE_BINS)
        train_start = max(0, train_end - int(train_days * DAY_NS // STEP_NS))
        rng = np.random.default_rng(self.seed + 910_001)
        bins, gpus, labels = self.sample_indices(train_start, train_end, rng)
        train_df = self.engine.extract_branch2_features(
            bins, gpus, self.history_map
        )
        model = v2._fit_weighted_logistic(
            train_df,
            labels,
            np.ones(len(labels), dtype=np.float64),
            self.seed + 910_002,
        )
        return {
            "model": model,
            "train_start": int(train_start),
            "train_end": int(train_end),
            "true_prior": float(self.gt_matrix[train_start:train_end].mean()),
            "sample_prior": float(labels.mean()),
        }

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
        fused_scores = np.zeros_like(b1_scores)
        lambda_value = float(selected["selected_lambda"])
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
            b1_scores[rel_idx] = b1
            b2_scores[rel_idx] = b2
            fused_scores[rel_idx] = lambda_value * b1 + (1.0 - lambda_value) * b2

        gt = self.gt_matrix[test_bins]
        y = gt.ravel()
        fused_rank = _rank_matrix(fused_scores)
        b1_rank = _rank_matrix(b1_scores)
        b2_rank = _rank_matrix(b2_scores)
        top_k = min(100, n_gpu)
        positives = int(gt.sum())
        prevalence = float(gt.mean())

        def top_metrics(ranks: np.ndarray) -> tuple[float, float]:
            hits = int((gt & (ranks <= top_k)).sum())
            recall = hits / max(positives, 1)
            lift = (hits / max(len(test_bins) * top_k, 1)) / max(prevalence, 1e-12)
            return recall, lift

        recall_fused, lift_fused = top_metrics(fused_rank)
        recall_b1, lift_b1 = top_metrics(b1_rank)
        recall_b2, lift_b2 = top_metrics(b2_rank)
        metrics = {
            "origin_idx": int(cycle_idx),
            "model_origin_time": _time_text(self.engine, cycle_start),
            "test_start_time": _time_text(self.engine, cycle_start),
            "test_end_time": _time_text(self.engine, cycle_end - 1),
            "evaluation_scope": "common_terminal_heldout",
            "prediction_population": "all_gpus_before_top100",
            "selected_b1_train_days": int(selected["selected_train_days"]),
            "selected_b1_half_life_days": int(selected["selected_half_life_days"]),
            "selected_b2_train_days": int(selected["selected_b2_train_days"]),
            "selected_b2_half_life_days": B2_HALF_LIFE,
            "selected_b1_input_hours": "1,6,24",
            "selected_cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
            "selected_lambda": lambda_value,
            "pr_auc_fused": base.safe_average_precision(y, fused_scores.ravel()),
            "pr_auc_b1": base.safe_average_precision(y, b1_scores.ravel()),
            "pr_auc_b2": base.safe_average_precision(y, b2_scores.ravel()),
            "pr_auc_fused_normalized": _normalized_pr_auc(
                base.safe_average_precision(y, fused_scores.ravel()), prevalence
            ),
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
        top_mask = fused_rank <= top_k
        rel_idx, gpu_idx = np.where(top_mask)
        tape = pd.DataFrame(
            {
                "decision_time": pd.to_datetime(
                    self.engine.bin_start_ns[test_bins][rel_idx],
                    unit="ns",
                    utc=True,
                ),
                "gpu_id": self.engine.gpu_ids[gpu_idx],
                "fused_risk": fused_scores[rel_idx, gpu_idx],
                "b1_risk": b1_scores[rel_idx, gpu_idx],
                "b2_risk": b2_scores[rel_idx, gpu_idx],
                "fused_rank": fused_rank[rel_idx, gpu_idx],
                "b1_rank": b1_rank[rel_idx, gpu_idx],
                "b2_rank": b2_rank[rel_idx, gpu_idx],
                "b1_train_days": int(selected["selected_train_days"]),
                "b1_half_life_days": int(selected["selected_half_life_days"]),
                "b2_train_days": int(selected["selected_b2_train_days"]),
                "b2_half_life_days": B2_HALF_LIFE,
                "cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
                "lambda_fusion": lambda_value,
                "target_24h": gt[rel_idx, gpu_idx].astype(np.uint8),
            }
        )
        return metrics, tape

    def _write_manifest(
        self,
        warmup_bins: int,
        test_start: int,
        test_end: int,
        dev_origins: np.ndarray,
        selected: dict[str, Any],
        completed_cycles: int,
    ) -> None:
        payload = {
            "runner": Path(__file__).name,
            "base_runner": "ML/run_bidirectional_adst_fusion.py",
            "branch1_runner": "ML/run_branch1_diversified_parallel_cascade.py",
            "branch2_runner": "ML/run_sliding_adst_v2_1_pooled_adwin.py",
            "experiment_scope": "integrated_full_gpu_fusion",
            "target": "all_xids",
            "horizon_hours": 24,
            "error_before_onset_exclusion_minutes": 10,
            "branch1_model_families": list(MODEL_FAMILIES),
            "branch1_input_hours": list(OBS_HOURS_GRID),
            "branch1_cascade": "soft_residual_hardness_gated",
            "branch2_contract": "history_only_2026_09_10",
            "branch2_features": ["xid_count_30d", "days_since_xid"],
            "branch2_model": "historical_logistic",
            "branch2_recency": "none",
            "fusion": "lambda_binary_weighted_full_gpu_before_top100",
            "lambda_grid": list(LAMBDA_GRID),
            "training_windows_days": list(TRAIN_DAYS_GRID),
            "recency_half_life_days": list(HALF_LIFE_GRID),
            "validation_blocks": VALIDATION_BLOCKS,
            "validation_block_days": 3,
            "pooled_origin_count": POOL_ORIGINS,
            "purge_hours": int(base.PURGE_NS // HOUR_NS),
            "retrain_cadence_hours": self.retrain_cadence_hours,
            "negative_ratio": self.negative_ratio,
            "test_stride_bins": self.test_stride_bins,
            "seed": self.seed,
            "heldout_test_fraction": TEST_FRACTION,
            "warmup_start_time": _time_text(self.engine, warmup_bins),
            "test_start_time": _time_text(self.engine, test_start),
            "test_end_time": _time_text(self.engine, test_end - 1),
            "development_origin_count": int(len(dev_origins)),
            "selected": selected,
            "completed_cycle_count": int(completed_cycles),
            "command": " ".join(sys.argv),
        }
        _write_json_atomic(self.output_dir / "experiment_manifest.json", payload)

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
        if len(dev_origins) == 0:
            raise ValueError("No development origins are available.")

        rows: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None
        print("[1/7] Selecting diversified Branch 1 configuration", flush=True)
        for idx, origin_bin in enumerate(dev_origins):
            loaded = self._load_selection_checkpoint(f"branch1_dev_{idx:04d}")
            if loaded is None:
                best, candidates = self.select_branch1_window(
                    int(origin_bin), previous, "development"
                )
                self._save_selection_checkpoint(
                    f"branch1_dev_{idx:04d}", best, candidates
                )
            else:
                best, candidates = loaded
            rows.extend(candidates)
            previous = best
            print(
                f"  [{idx + 1}/{len(dev_origins)}] "
                f"B1 L={best['selected_train_days']}d "
                f"half={best['selected_half_life_days']}d "
                f"val_PR_AUC={float(best['val_pr_auc_mean']):.4f}",
                flush=True,
            )
        final_loaded = self._load_selection_checkpoint("branch1_final")
        if final_loaded is None:
            selected_b1, pooled_candidates = self.pooled_window(dev_origins, rows, POOL_ORIGINS)
            self._save_selection_checkpoint(
                "branch1_final", selected_b1, pooled_candidates
            )
        else:
            selected_b1, pooled_candidates = final_loaded
        rows.extend(pooled_candidates)

        print("[2/7] Fitting pooled diversified Branch 1 OOF models", flush=True)
        b1_records = self.collect_oof(dev_origins, selected_b1, POOL_ORIGINS)
        family_weights, input_weights, weight_rows = self.derive_weights(b1_records)
        print(
            f"  family_weights={family_weights}; input_weights={input_weights}",
            flush=True,
        )
        print("[3/7] Fitting Branch 1 soft residual cascade", flush=True)
        cascade_model, cascade_meta, alpha_rows = self.fit_cascade(
            b1_records, family_weights, input_weights
        )
        self._cascade_validation_predictions(b1_records, cascade_model, cascade_meta)

        print("[4/7] Selecting Branch 2 History-only window", flush=True)
        b2_records = self._b2_validation_records(dev_origins, POOL_ORIGINS)
        selected_b2, b2_rows = self._select_b2_window(b2_records)
        print(
            f"  B2 L={selected_b2['selected_b2_train_days']}d "
            f"val_PR_AUC={float(selected_b2['val_pr_auc_mean']):.4f}",
            flush=True,
        )

        print("[5/7] Selecting Lambda on paired full-GPU validation blocks", flush=True)
        lambda_best, lambda_rows = self._select_lambda(
            b1_records,
            b2_records,
            int(selected_b2["selected_b2_train_days"]),
        )
        selected = {
            **selected_b1,
            "selected_family_weights": family_weights,
            "selected_input_weights": input_weights,
            "selected_cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
            "selected_cascade_validation_pr_auc": float(
                cascade_meta["stage2_alpha_validation_pr_auc"]
            ),
            "selected_b2_train_days": int(selected_b2["selected_b2_train_days"]),
            "selected_b2_half_life_days": B2_HALF_LIFE,
            "selected_lambda": float(lambda_best["selected_lambda"]),
            "fusion_validation_pr_auc": float(lambda_best["val_pr_auc_mean"]),
            "fusion_validation_roc_auc": float(lambda_best["val_roc_auc_mean"]),
            "paired_validation_block_count": int(lambda_best["paired_validation_block_count"]),
            "pool_origin_count": POOL_ORIGINS,
            "pool_validation_block_count": len(b1_records),
            "prediction_population": "all_gpus_before_top100",
        }
        selection_df = pd.DataFrame(
            rows
            + weight_rows
            + alpha_rows
            + b2_rows
            + lambda_rows
            + [selected]
        )
        selection_df.to_csv(
            self.output_dir / "fusion_selection_history.csv", index=False
        )
        _write_json_atomic(
            self.checkpoint_dir / "fusion_selection.json", selected
        )

        print("[6/7] Fitting final Branch 1 and Branch 2 models", flush=True)
        final_b1 = self._fit_final_stage1(selected, test_start)
        final_b2 = self._fit_final_b2(int(selected["selected_b2_train_days"]), test_start)
        self._write_manifest(
            warmup_bins,
            test_start,
            test_end,
            dev_origins,
            selected,
            0,
        )

        metrics_rows: list[dict[str, Any]] = []
        tapes: list[pd.DataFrame] = []
        cycle_starts = np.arange(test_start, test_end, cadence_bins, dtype=np.int32)
        print(
            f"[7/7] Evaluating {len(cycle_starts)} full-GPU held-out cycles before Top-100 tape",
            flush=True,
        )
        for cycle_idx, cycle_start in enumerate(cycle_starts):
            cycle_end = min(int(cycle_start) + cadence_bins, test_end)
            loaded = self._load_cycle_checkpoint(cycle_idx)
            if loaded is None:
                metrics, tape = self._predict_fusion_cycle(
                    final_b1,
                    cascade_model,
                    cascade_meta,
                    final_b2,
                    selected,
                    cycle_idx,
                    int(cycle_start),
                    cycle_end,
                )
                self._save_cycle_checkpoint(cycle_idx, metrics, tape)
            else:
                metrics, tape = loaded
            metrics_rows.append(metrics)
            tapes.append(tape)
            print(
                f"  [{cycle_idx + 1}/{len(cycle_starts)}] "
                f"fused={float(metrics['pr_auc_fused']):.4f} "
                f"B1={float(metrics['pr_auc_b1']):.4f} "
                f"B2={float(metrics['pr_auc_b2']):.4f} "
                f"lambda={float(metrics['selected_lambda']):.1f}",
                flush=True,
            )

        metrics_df = pd.DataFrame(metrics_rows)
        tape_df = pd.concat(tapes, ignore_index=True) if tapes else pd.DataFrame()
        metrics_df.to_csv(self.output_dir / "fusion_metrics.csv", index=False)
        tape_df.to_parquet(
            self.output_dir / "fusion_risk_tape.parquet",
            index=False,
            compression="zstd",
        )
        selected["completed_cycle_count"] = int(len(metrics_df))
        self._write_manifest(
            warmup_bins,
            test_start,
            test_end,
            dev_origins,
            selected,
            len(metrics_df),
        )

        report_lines = [
            "# All-XID diversified Branch 1 + History-only Branch 2 Fusion 보고서",
            "",
            "## Material Passport",
            "- **Status**: COMPLETED",
            f"- **Runner**: `{Path(__file__).name}`",
            "- **Target**: All-XID unified onset, 24-hour horizon",
            "- **Evaluation**: full GPU-time score matrices, then Fusion, then Top-100 tape",
            f"- **Seed**: `{self.seed}`",
            "",
            "## 실험 계약",
            "- 시간순 Sliding Training, 36시간 purge, 공통 terminal held-out 20%",
            "- Branch 1: 1h·6h·24h telemetry, 4-family diversified ensemble, soft cascade",
            "- Branch 2: History-only `xid_count_30d`, `days_since_xid`, historical logistic",
            "- Lambda: paired pooled Validation PR-AUC 선택, held-out label 미사용",
            "- Top-100: Fusion score로 전체 GPU를 정렬한 후 저장",
            "",
            "## Terminal held-out 평균",
            f"- **Fusion PR-AUC**: `{float(metrics_df['pr_auc_fused'].mean()):.6f}`",
            f"- **Branch 1 cascade PR-AUC**: `{float(metrics_df['pr_auc_b1'].mean()):.6f}`",
            f"- **Branch 2 History-only PR-AUC**: `{float(metrics_df['pr_auc_b2'].mean()):.6f}`",
            f"- **Fusion ROC-AUC**: `{float(metrics_df['roc_auc_fused'].mean()):.6f}`",
            f"- **Fusion Recall@100**: `{float(metrics_df['recall_at_100'].mean()):.2%}`",
            f"- **Fusion Lift@100**: `{float(metrics_df['lift_at_100'].mean()):.3f}x`",
            f"- **Selected Lambda**: `{float(selected['selected_lambda']):.1f}`",
            f"- **Evaluated GPUs per decision time**: `{int(metrics_df['gpu_count_evaluated'].iloc[0])}`",
            f"- **Held-out cycles**: `{len(metrics_df)}`",
            "",
            "## 해석 경계",
            "- PR-AUC·ROC-AUC는 Top-100으로 자르기 전 전체 GPU-time score를 사용했다.",
            "- Risk tape는 Fusion rank 기준 Top-100만 저장한 운영 후보 목록이다.",
            "- Lambda가 0 또는 1이면 해당 결과는 사실상 한 Branch 중심이며, Fusion synergy가 확인된 것으로 해석하지 않는다.",
            "",
            "## 산출물",
            "- fusion_metrics.csv",
            "- fusion_risk_tape.parquet",
            "- fusion_selection_history.csv",
            "- experiment_manifest.json",
            "- checkpoints/",
        ]
        (self.output_dir / "fusion_report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )
        print(
            f"[Done] Saved integrated Fusion outputs in {self.output_dir}; "
            f"elapsed={(time.time() - started) / 60.0:.1f} minutes",
            flush=True,
        )
        return metrics_df, tape_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-XID diversified Branch 1 + History-only Branch 2 Fusion"
    )
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/2026-09-18_All-XID_Fusion_DiversifiedB1_HistoryB2_01",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = base.find_data_dir()
    cache_dir = base.PROJECT_ROOT / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    engine = base.UnifiedDataEngine(data_dir=data_dir, cache_dir=cache_dir)
    _, history_map, gt_matrix = engine.load_all_xid_ledger()
    pipeline = DiversifiedBranchFusion(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=base.PROJECT_ROOT / args.output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        resume=args.resume,
        branch2_mode="history_0910",
    )
    pipeline.run()


if __name__ == "__main__":
    main()

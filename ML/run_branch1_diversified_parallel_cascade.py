"""All-XID Branch 1 diversified parallel ensemble and soft cascade experiment.

This file is intentionally separate from the GitHub-base runner. It reuses
the existing data engine, label construction, Sliding Training split,
36-hour purge, common terminal held-out test, and checkpoint helpers.
Branch 2 and Lambda fusion are not executed.
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
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_bidirectional_adst_fusion as base
from run_sliding_adst_v2_branch_specific import (
    BranchSpecificADST,
    HALF_LIFE_GRID,
    OBS_HOURS_GRID,
    TRAIN_DAYS_GRID,
    VALIDATION_BLOCKS,
    VALIDATION_BINS,
    _normalized_pr_auc,
    _recency_weights,
    _time_text,
    _write_json_atomic,
)

MODEL_FAMILIES = (
    "logistic",
    "extra_trees",
    "hist_gradient_boosting",
    "cnn",
)
PURGE_BINS = max(1, int(base.PURGE_NS // base.STEP_NS))
DAY_NS = base.DAY_NS
HOUR_NS = base.HOUR_NS
STEP_NS = base.STEP_NS
TEST_FRACTION = 0.20
CASCADE_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CNN_EPOCHS = 5


def _aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
    }


def _normalise(raw: dict[str, float]) -> dict[str, float]:
    values = {key: max(float(value), 1e-8) for key, value in raw.items()}
    total = sum(values.values())
    return {key: float(value / total) for key, value in values.items()}


def _fit_logistic(
    features: pd.DataFrame,
    labels: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=250,
            class_weight="balanced",
            random_state=int(seed),
        ),
    )
    model.fit(features, labels, logisticregression__sample_weight=weights)
    return model


class Branch1Diversified(BranchSpecificADST):
    """Only Branch 1 is fitted, selected, evaluated, and reported."""

    def _sample_training(
        self,
        train_end: int,
        train_days: int,
        seed_offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        train_start = max(
            0, int(train_end) - int(train_days * DAY_NS // STEP_NS)
        )
        bins, gpus, labels = self.sample_indices(
            train_start,
            int(train_end),
            np.random.default_rng(self.seed + int(seed_offset)),
        )
        if len(labels) == 0 or np.unique(labels).size < 2:
            raise ValueError(
                f"Training block has insufficient classes: {train_start}:{train_end}"
            )
        return bins, gpus, labels, train_start

    def select_branch1_window(
        self,
        origin_bin: int,
        previous: dict[str, Any] | None,
        stage: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        blocks = self._selection_blocks(origin_bin)
        candidates: list[dict[str, Any]] = []
        for train_days in TRAIN_DAYS_GRID:
            block_cache: list[dict[str, Any]] = []
            for block in blocks:
                block_idx = int(block["block_idx"])
                train_bins, train_gpus, train_labels, _ = self._sample_training(
                    int(block["train_end"]),
                    int(train_days),
                    int(origin_bin) + 100_003 * (block_idx + 1) + 1_009 * train_days,
                )
                train_features: dict[int, pd.DataFrame] = {}
                val_features: dict[int, pd.DataFrame] = {}
                for obs in OBS_HOURS_GRID:
                    train_features[obs] = self.engine.extract_branch1_features(
                        train_bins, train_gpus, obs
                    )[0]
                    val_features[obs] = self.engine.extract_branch1_features(
                        block["val_bins"], block["val_gpus"], obs
                    )[0]
                block_cache.append(
                    {
                        "block": block,
                        "train_bins": train_bins,
                        "train_labels": train_labels,
                        "train_features": train_features,
                        "val_features": val_features,
                    }
                )
            for half_life in HALF_LIFE_GRID:
                aps: list[float] = []
                rocs: list[float] = []
                for cache in block_cache:
                    block = cache["block"]
                    block_idx = int(block["block_idx"])
                    val_predictions: list[np.ndarray] = []
                    for obs in OBS_HOURS_GRID:
                        model = _fit_logistic(
                            cache["train_features"][obs],
                            cache["train_labels"],
                            _recency_weights(
                                cache["train_bins"], int(block["train_end"]), int(half_life)
                            ),
                            self.seed + int(origin_bin) + block_idx + obs + train_days,
                        )
                        val_predictions.append(
                            model.predict_proba(cache["val_features"][obs])[:, 1]
                        )
                    parallel = np.mean(val_predictions, axis=0)
                    aps.append(
                        base.safe_average_precision(block["val_labels"], parallel)
                    )
                    rocs.append(base.safe_roc_auc(block["val_labels"], parallel))
                agg = _aggregate(aps)
                candidates.append(
                    {
                        "selection_role": "branch1_proxy",
                        "selection_stage": stage,
                        "selection_origin_idx": int(origin_bin),
                        "selection_origin_time": _time_text(self.engine, origin_bin),
                        "candidate_train_days": int(train_days),
                        "candidate_half_life_days": int(half_life),
                        "candidate_input_hours": "all_1_6_24",
                        "validation_block_count": VALIDATION_BLOCKS,
                        "validation_block_days": 3,
                        "purge_hours": int(base.PURGE_NS // HOUR_NS),
                        "val_pr_auc_mean": agg["mean"],
                        "val_pr_auc_median": agg["median"],
                        "val_pr_auc_q25": agg["q25"],
                        "val_pr_auc_std": agg["std"],
                        "val_pr_auc_min": agg["min"],
                        "val_roc_auc_mean": float(np.mean(rocs)),
                    }
                )
        best = max(
            candidates,
            key=lambda row: (
                float(row["val_pr_auc_mean"]),
                float(row["val_pr_auc_median"]),
                float(row["val_pr_auc_q25"]),
                -float(row["val_pr_auc_std"]),
                -int(row["candidate_train_days"]),
            ),
        ).copy()
        best["selected_train_days"] = int(best.pop("candidate_train_days"))
        best["selected_half_life_days"] = int(best.pop("candidate_half_life_days"))
        best["selected_input_hours"] = list(OBS_HOURS_GRID)
        previous_score = None if previous is None else float(previous["val_pr_auc_mean"])
        previous_days = None if previous is None else int(previous["selected_train_days"])
        previous_half = None if previous is None else int(previous["selected_half_life_days"])
        if previous is None:
            action = "initial"
        elif best["selected_train_days"] != previous_days:
            action = (
                "expand_train_window"
                if best["selected_train_days"] > previous_days
                else "shrink_train_window"
            )
        elif best["selected_half_life_days"] != previous_half:
            action = "change_recency_half_life"
        else:
            action = "hold"
        best["adst_action"] = action
        best["previous_val_pr_auc_mean"] = previous_score
        best["val_pr_auc_delta"] = (
            None
            if previous_score is None
            else float(best["val_pr_auc_mean"] - previous_score)
        )
        best["candidate_count"] = len(candidates)
        return best, candidates

    def pooled_window(
        self,
        dev_origins: np.ndarray,
        rows: list[dict[str, Any]],
        latest_count: int = 6,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        allowed = {int(value) for value in dev_origins[-latest_count:]}
        selected_rows = [
            row
            for row in rows
            if row.get("selection_role") == "branch1_proxy"
            and int(row.get("selection_origin_idx", -1)) in allowed
        ]
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in selected_rows:
            key = (
                int(row["candidate_train_days"]),
                int(row["candidate_half_life_days"]),
            )
            grouped.setdefault(key, []).append(row)
        candidates: list[dict[str, Any]] = []
        for (train_days, half_life), items in grouped.items():
            agg = _aggregate([float(item["val_pr_auc_mean"]) for item in items])
            candidates.append(
                {
                    "selection_role": "branch1_proxy_pooled",
                    "selection_stage": "pooled_development",
                    "selection_origin_count": len(items),
                    "candidate_train_days": train_days,
                    "candidate_half_life_days": half_life,
                    "candidate_input_hours": "all_1_6_24",
                    "val_pr_auc_mean": agg["mean"],
                    "val_pr_auc_median": agg["median"],
                    "val_pr_auc_q25": agg["q25"],
                    "val_pr_auc_std": agg["std"],
                    "val_pr_auc_min": agg["min"],
                    "val_roc_auc_mean": float(
                        np.mean([float(item["val_roc_auc_mean"]) for item in items])
                    ),
                }
            )
        best = max(
            candidates,
            key=lambda row: (
                float(row["val_pr_auc_mean"]),
                float(row["val_pr_auc_median"]),
                float(row["val_pr_auc_q25"]),
                -float(row["val_pr_auc_std"]),
                -int(row["candidate_train_days"]),
            ),
        ).copy()
        best["selection_stage"] = "heldout_preselection"
        best["selection_origin_idx"] = int(dev_origins[-1])
        best["selection_origin_time"] = _time_text(self.engine, int(dev_origins[-1]))
        best["selected_train_days"] = int(best.pop("candidate_train_days"))
        best["selected_half_life_days"] = int(best.pop("candidate_half_life_days"))
        best["selected_input_hours"] = list(OBS_HOURS_GRID)
        best["validation_origin_count"] = len(allowed)
        best["validation_block_count"] = VALIDATION_BLOCKS
        best["purge_hours"] = int(base.PURGE_NS // HOUR_NS)
        best["adst_action"] = "pooled_latest_development_origins"
        best["candidate_count"] = len(candidates)
        return best, candidates

    def _fit_cnn(
        self,
        tensor: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        seed: int,
    ) -> Any:
        base.seed_everything(int(seed))
        model = base.TemporalCNN1D(in_channels=7, hidden_channels=32, dropout=0.2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        positive_weight = max(
            1.0, (labels == 0).sum() / max(1, (labels == 1).sum())
        )
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight], dtype=torch.float32),
            reduction="none",
        )
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(tensor, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
        )
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=4096, shuffle=True, generator=generator
        )
        model.train()
        for _ in range(CNN_EPOCHS):
            for batch_x, batch_y, batch_w in loader:
                optimizer.zero_grad()
                loss = (criterion(model(batch_x), batch_y) * batch_w).mean()
                loss.backward()
                optimizer.step()
        model.eval()
        return model

    def _fit_families(
        self,
        bins: np.ndarray,
        gpus: np.ndarray,
        labels: np.ndarray,
        train_end: int,
        train_days: int,
        half_life: int,
        seed: int,
    ) -> dict[str, Any]:
        weights = _recency_weights(bins, train_end, half_life)
        models: dict[str, dict[str, Any]] = {}
        for obs_idx, obs in enumerate(OBS_HOURS_GRID):
            tabular, tensor = self.engine.extract_branch1_features(bins, gpus, obs)
            family: dict[str, Any] = {}
            family["logistic"] = _fit_logistic(
                tabular, labels, weights, seed + 101 * obs_idx + 1
            )
            tree = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed + 101 * obs_idx + 2,
            )
            tree.fit(tabular, labels, sample_weight=weights)
            family["extra_trees"] = tree
            hgb = HistGradientBoostingClassifier(
                max_iter=120,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=seed + 101 * obs_idx + 3,
            )
            hgb.fit(tabular, labels, sample_weight=weights)
            family["hist_gradient_boosting"] = hgb
            family["cnn"] = self._fit_cnn(
                tensor, labels, weights, seed + 101 * obs_idx + 4
            )
            models[str(obs)] = family
        return {
            "models": models,
            "metadata": {
                "train_start": int(bins.min()),
                "train_end": int(train_end),
                "train_days": int(train_days),
                "half_life_days": int(half_life),
                "true_prior": float(
                    self.gt_matrix[int(bins.min()):int(train_end)].mean()
                ),
                "sample_prior": float(labels.mean()),
            },
        }

    def predict_parallel(
        self,
        fitted: dict[str, Any],
        bins: np.ndarray,
        gpus: np.ndarray,
        family_weights: dict[str, float] | None = None,
        input_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        family_weights = family_weights or {
            key: 1.0 / len(MODEL_FAMILIES) for key in MODEL_FAMILIES
        }
        input_weights = input_weights or {
            str(obs): 1.0 / len(OBS_HOURS_GRID) for obs in OBS_HOURS_GRID
        }
        scores: dict[str, dict[str, np.ndarray]] = {}
        meta = fitted["metadata"]
        for obs in OBS_HOURS_GRID:
            tabular, tensor = self.engine.extract_branch1_features(bins, gpus, obs)
            scores[str(obs)] = {}
            for family in MODEL_FAMILIES:
                model = fitted["models"][str(obs)][family]
                if family == "cnn":
                    with torch.no_grad():
                        raw = torch.sigmoid(
                            model(torch.tensor(tensor, dtype=torch.float32))
                        ).cpu().numpy()
                else:
                    raw = model.predict_proba(tabular)[:, 1]
                scores[str(obs)][family] = base.adjusted_probability(
                    np.asarray(raw, dtype=np.float64),
                    float(meta["true_prior"]),
                    float(meta["sample_prior"]),
                )
        all_scores = np.vstack(
            [
                scores[str(obs)][family]
                for obs in OBS_HOURS_GRID
                for family in MODEL_FAMILIES
            ]
        )
        parallel = np.zeros(all_scores.shape[1], dtype=np.float64)
        idx = 0
        for obs in OBS_HOURS_GRID:
            for family in MODEL_FAMILIES:
                parallel += (
                    float(input_weights[str(obs)])
                    * float(family_weights[family])
                    * all_scores[idx]
                )
                idx += 1
        disagreement = np.mean(np.abs(all_scores - parallel[None, :]), axis=0)
        uncertainty = np.clip(4.0 * parallel * (1.0 - parallel), 0.0, 1.0)
        scale = float(np.quantile(disagreement, 0.95)) if len(disagreement) else 1.0
        hardness = np.clip(
            0.5 * disagreement / max(scale, 1e-8) + 0.5 * uncertainty,
            0.0,
            1.0,
        )
        return {
            "family_scores": scores,
            "parallel": parallel,
            "disagreement": disagreement,
            "uncertainty": uncertainty,
            "hardness": hardness,
        }

    @staticmethod
    def stage2_features(
        tabular: pd.DataFrame,
        parallel: np.ndarray,
        disagreement: np.ndarray,
        uncertainty: np.ndarray,
        hardness: np.ndarray,
    ) -> pd.DataFrame:
        result = tabular.copy()
        result["parallel_score"] = parallel
        result["prediction_disagreement"] = disagreement
        result["parallel_uncertainty"] = uncertainty
        result["sample_hardness"] = hardness
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def collect_oof(
        self,
        dev_origins: np.ndarray,
        selected: dict[str, Any],
        latest_count: int = 6,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for origin_idx, origin_bin in enumerate(dev_origins[-latest_count:]):
            for block in self._selection_blocks(int(origin_bin)):
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
                        "tabular_24h": val_24h,
                    }
                )
                print(
                    f"    pooled OOF origin={int(origin_bin)} block={block_idx} "
                    f"n={len(block['val_labels']):,}",
                    flush=True,
                )
        return records

    def derive_weights(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
        family_ap = {family: [] for family in MODEL_FAMILIES}
        input_ap = {str(obs): [] for obs in OBS_HOURS_GRID}
        for record in records:
            labels = record["labels"]
            scores = record["stage1"]["family_scores"]
            for family in MODEL_FAMILIES:
                family_ap[family].append(
                    float(
                        np.mean(
                            [
                                base.safe_average_precision(
                                    labels, scores[str(obs)][family]
                                )
                                for obs in OBS_HOURS_GRID
                            ]
                        )
                    )
                )
            for obs in OBS_HOURS_GRID:
                input_ap[str(obs)].append(
                    base.safe_average_precision(
                        labels,
                        np.mean(
                            [scores[str(obs)][family] for family in MODEL_FAMILIES],
                            axis=0,
                        ),
                    )
                )
        family_mean = {
            key: float(np.mean(value)) for key, value in family_ap.items()
        }
        input_mean = {key: float(np.mean(value)) for key, value in input_ap.items()}
        family_weights = _normalise(
            {key: 0.25 + value for key, value in family_mean.items()}
        )
        input_weights = _normalise(
            {key: 0.20 + value for key, value in input_mean.items()}
        )
        return family_weights, input_weights, [
            {
                "selection_role": "pooled_parallel_weight_summary",
                "family_validation_pr_auc": family_mean,
                "input_validation_pr_auc": input_mean,
                "selected_family_weights": family_weights,
                "selected_input_weights": input_weights,
            }
        ]

    def attach_parallel(
        self,
        records: list[dict[str, Any]],
        family_weights: dict[str, float],
        input_weights: dict[str, float],
    ) -> None:
        for record in records:
            stage1 = record["stage1"]
            family_scores = stage1["family_scores"]
            all_scores = np.vstack(
                [
                    family_scores[str(obs)][family]
                    for obs in OBS_HOURS_GRID
                    for family in MODEL_FAMILIES
                ]
            )
            parallel = np.zeros(all_scores.shape[1], dtype=np.float64)
            idx = 0
            for obs in OBS_HOURS_GRID:
                for family in MODEL_FAMILIES:
                    parallel += (
                        float(input_weights[str(obs)])
                        * float(family_weights[family])
                        * all_scores[idx]
                    )
                    idx += 1
            disagreement = np.mean(np.abs(all_scores - parallel[None, :]), axis=0)
            uncertainty = np.clip(4.0 * parallel * (1.0 - parallel), 0.0, 1.0)
            scale = float(np.quantile(disagreement, 0.95)) if len(disagreement) else 1.0
            hardness = np.clip(
                0.5 * disagreement / max(scale, 1e-8) + 0.5 * uncertainty,
                0.0,
                1.0,
            )
            stage1.update(
                {
                    "parallel": parallel,
                    "disagreement": disagreement,
                    "uncertainty": uncertainty,
                    "hardness": hardness,
                }
            )
            record["stage2_x"] = self.stage2_features(
                record["tabular_24h"],
                parallel,
                disagreement,
                uncertainty,
                hardness,
            )

    def fit_cascade(
        self,
        records: list[dict[str, Any]],
        family_weights: dict[str, float],
        input_weights: dict[str, float],
    ) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
        self.attach_parallel(records, family_weights, input_weights)
        records.sort(key=lambda row: (int(row["origin_bin"]), int(row["block_idx"])))
        split_at = max(1, int(np.floor(len(records) * 2 / 3)))
        older, latest = records[:split_at], records[split_at:]
        x_old = pd.concat([row["stage2_x"] for row in older], ignore_index=True)
        y_old = np.concatenate([row["labels"] for row in older])
        h_old = np.concatenate([row["stage1"]["hardness"] for row in older])
        model = HistGradientBoostingClassifier(
            max_iter=160,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=self.seed + 700_001,
        )
        model.fit(x_old, y_old, sample_weight=0.5 + h_old)
        x_latest = pd.concat([row["stage2_x"] for row in latest], ignore_index=True)
        y_latest = np.concatenate([row["labels"] for row in latest])
        p_parallel = np.concatenate([row["stage1"]["parallel"] for row in latest])
        p_hardness = np.concatenate([row["stage1"]["hardness"] for row in latest])
        p_stage2 = base.adjusted_probability(
            model.predict_proba(x_latest)[:, 1],
            float(np.mean([row["stage1"]["parallel"].mean() for row in records])),
            float(y_old.mean()),
        )
        alpha_rows: list[dict[str, Any]] = []
        for alpha in CASCADE_ALPHA_GRID:
            cascade = p_parallel + float(alpha) * p_hardness * (p_stage2 - p_parallel)
            alpha_rows.append(
                {
                    "selection_role": "cascade_alpha",
                    "candidate_cascade_alpha": float(alpha),
                    "val_pr_auc_mean": base.safe_average_precision(y_latest, cascade),
                    "val_roc_auc_mean": base.safe_roc_auc(y_latest, cascade),
                    "validation_scope": "latest_pooled_oof_blocks",
                    "validation_samples": int(len(y_latest)),
                }
            )
        best_alpha = max(
            alpha_rows,
            key=lambda row: (
                float(row["val_pr_auc_mean"]),
                float(row["val_roc_auc_mean"]),
                -abs(float(row["candidate_cascade_alpha"]) - 0.5),
            ),
        )
        x_all = pd.concat([row["stage2_x"] for row in records], ignore_index=True)
        y_all = np.concatenate([row["labels"] for row in records])
        h_all = np.concatenate([row["stage1"]["hardness"] for row in records])
        final_model = HistGradientBoostingClassifier(
            max_iter=160,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=self.seed + 700_002,
        )
        final_model.fit(x_all, y_all, sample_weight=0.5 + h_all)
        metadata = {
            "stage2_true_prior": float(
                np.mean([row["stage1"]["parallel"].mean() for row in records])
            ),
            "stage2_sample_prior": float(y_all.mean()),
            "selected_cascade_alpha": float(best_alpha["candidate_cascade_alpha"]),
            "stage2_training_record_count": len(records),
            "stage2_alpha_selection_record_count": len(latest),
            "stage2_alpha_validation_pr_auc": float(best_alpha["val_pr_auc_mean"]),
        }
        return final_model, metadata, alpha_rows

    def _fit_final_stage1(self, selected: dict[str, Any], test_start: int) -> dict[str, Any]:
        train_end = int(test_start - PURGE_BINS)
        bins, gpus, labels, train_start = self._sample_training(
            train_end, int(selected["selected_train_days"]), 900_001
        )
        fitted = self._fit_families(
            bins,
            gpus,
            labels,
            train_end,
            int(selected["selected_train_days"]),
            int(selected["selected_half_life_days"]),
            self.seed + 900_010,
        )
        fitted["metadata"]["train_start"] = int(train_start)
        return fitted

    def _predict_cycle(
        self,
        fitted: dict[str, Any],
        cascade_model: Any,
        cascade_meta: dict[str, Any],
        selected: dict[str, Any],
        cycle_idx: int,
        cycle_start: int,
        cycle_end: int,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        test_bins = np.arange(
            cycle_start, cycle_end, self.test_stride_bins, dtype=np.int32
        )
        n_gpu = self.engine.num_gpus
        parallel = np.zeros((len(test_bins), n_gpu), dtype=np.float64)
        cascade = np.zeros_like(parallel)
        disagreement = np.zeros_like(parallel)
        hardness = np.zeros_like(parallel)
        for idx, current_bin in enumerate(test_bins):
            bins = np.full(n_gpu, int(current_bin), dtype=np.int32)
            gpus = np.arange(n_gpu, dtype=np.int32)
            stage1 = self.predict_parallel(
                fitted,
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
            parallel[idx] = stage1["parallel"]
            cascade[idx] = np.clip(
                stage1["parallel"]
                + float(cascade_meta["selected_cascade_alpha"])
                * stage1["hardness"]
                * (p_stage2 - stage1["parallel"]),
                0.0,
                1.0,
            )
            disagreement[idx] = stage1["disagreement"]
            hardness[idx] = stage1["hardness"]
        gt = self.gt_matrix[test_bins]
        y = gt.ravel()
        p_flat, c_flat = parallel.ravel(), cascade.ravel()
        p_order = np.argsort(-parallel, axis=1, kind="stable")
        c_order = np.argsort(-cascade, axis=1, kind="stable")
        p_rank = np.empty_like(p_order, dtype=np.uint16)
        c_rank = np.empty_like(c_order, dtype=np.uint16)
        np.put_along_axis(
            p_rank,
            p_order,
            np.arange(1, n_gpu + 1, dtype=np.uint16)[None, :],
            axis=1,
        )
        np.put_along_axis(
            c_rank,
            c_order,
            np.arange(1, n_gpu + 1, dtype=np.uint16)[None, :],
            axis=1,
        )
        top_k = min(100, n_gpu)
        positives = int(gt.sum())
        p_hits = int((gt & (p_rank <= top_k)).sum())
        c_hits = int((gt & (c_rank <= top_k)).sum())
        prevalence = float(gt.mean())
        p_ap = base.safe_average_precision(y, p_flat)
        c_ap = base.safe_average_precision(y, c_flat)
        metrics = {
            "origin_idx": int(cycle_idx),
            "model_origin_time": _time_text(self.engine, cycle_start),
            "test_start_time": _time_text(self.engine, cycle_start),
            "test_end_time": _time_text(self.engine, cycle_end - 1),
            "evaluation_scope": "common_terminal_heldout",
            "branch1_only": True,
            "branch2_executed": False,
            "selected_train_days": int(selected["selected_train_days"]),
            "selected_half_life_days": int(selected["selected_half_life_days"]),
            "selected_cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
            "pr_auc_parallel": p_ap,
            "pr_auc_cascade": c_ap,
            "pr_auc_parallel_normalized": _normalized_pr_auc(p_ap, prevalence),
            "pr_auc_cascade_normalized": _normalized_pr_auc(c_ap, prevalence),
            "roc_auc_parallel": base.safe_roc_auc(y, p_flat),
            "roc_auc_cascade": base.safe_roc_auc(y, c_flat),
            "recall_at_100_parallel": p_hits / max(positives, 1),
            "recall_at_100_cascade": c_hits / max(positives, 1),
            "lift_at_100_parallel": (p_hits / max(len(test_bins) * top_k, 1))
            / max(prevalence, 1e-12),
            "lift_at_100_cascade": (c_hits / max(len(test_bins) * top_k, 1))
            / max(prevalence, 1e-12),
            "positives": positives,
            "test_prevalence": prevalence,
            "mean_prediction_disagreement": float(disagreement.mean()),
            "mean_sample_hardness": float(hardness.mean()),
        }
        mask = c_rank <= top_k
        rel_idx, gpu_idx = np.where(mask)
        tape = pd.DataFrame(
            {
                "decision_time": pd.to_datetime(
                    self.engine.bin_start_ns[test_bins][rel_idx],
                    unit="ns",
                    utc=True,
                ),
                "gpu_id": self.engine.gpu_ids[gpu_idx],
                "cascade_risk": cascade[rel_idx, gpu_idx],
                "parallel_risk": parallel[rel_idx, gpu_idx],
                "prediction_disagreement": disagreement[rel_idx, gpu_idx],
                "sample_hardness": hardness[rel_idx, gpu_idx],
                "cascade_rank": c_rank[rel_idx, gpu_idx],
                "parallel_rank": p_rank[rel_idx, gpu_idx],
                "L_train_days": int(selected["selected_train_days"]),
                "L_obs_hours": "1_6_24",
                "cascade_alpha": float(cascade_meta["selected_cascade_alpha"]),
                "target_24h": gt[rel_idx, gpu_idx].astype(np.uint8),
            }
        )
        return metrics, tape

    def _manifest(
        self,
        warmup_bins: int,
        test_start: int,
        test_end: int,
        dev_origins: np.ndarray,
        selected: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "runner": Path(__file__).name,
            "base_runner": "ML/run_bidirectional_adst_fusion.py",
            "reference_runner": "ML/run_sliding_adst_v2_branch_specific.py",
            "experiment_scope": "Branch1_only",
            "branch1_only": True,
            "branch2_executed": False,
            "lambda_fusion_executed": False,
            "target": "all_xids",
            "horizon_hours": 24,
            "error_before_onset_exclusion_minutes": 10,
            "branch1_input_hours": list(OBS_HOURS_GRID),
            "model_families": list(MODEL_FAMILIES),
            "parallel_ensemble": True,
            "cascade": "soft_residual_hardness_gated",
            "cascade_alpha_grid": list(CASCADE_ALPHA_GRID),
            "training_windows_days": list(TRAIN_DAYS_GRID),
            "recency_half_life_days": list(HALF_LIFE_GRID),
            "validation_blocks": VALIDATION_BLOCKS,
            "validation_block_days": 3,
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
            "pooled_validation_origin_count": min(6, len(dev_origins)),
            "selected": selected,
            "command": " ".join(sys.argv),
        }
        if extra:
            payload.update(extra)
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
        print("[1/6] Branch 1-only development window selection", flush=True)
        print(
            f"  origins={len(dev_origins)}, validation_blocks={VALIDATION_BLOCKS}, "
            f"purge={int(base.PURGE_NS // HOUR_NS)}h",
            flush=True,
        )
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
                f"L={best['selected_train_days']}d half={best['selected_half_life_days']}d "
                f"val_PR_AUC={float(best['val_pr_auc_mean']):.4f} "
                f"action={best['adst_action']}",
                flush=True,
            )
        final_loaded = self._load_selection_checkpoint("branch1_final")
        if final_loaded is None:
            selected, pooled_candidates = self.pooled_window(dev_origins, rows, 6)
            self._save_selection_checkpoint(
                "branch1_final", selected, pooled_candidates
            )
        else:
            selected, pooled_candidates = final_loaded
        rows.extend(pooled_candidates)
        rows.append(selected)
        selection_df = pd.DataFrame(rows)
        print("[2/6] Fitting pooled OOF diversified parallel models", flush=True)
        records = self.collect_oof(dev_origins, selected, 6)
        if len(records) < 3:
            raise ValueError("Too few pooled OOF records for cascade fitting.")
        family_weights, input_weights, weight_rows = self.derive_weights(records)
        print(
            f"  family_weights={family_weights}; input_weights={input_weights}",
            flush=True,
        )
        print("[3/6] Fitting soft residual cascade", flush=True)
        cascade_model, cascade_meta, alpha_rows = self.fit_cascade(
            records, family_weights, input_weights
        )
        selected = dict(selected)
        selected["selected_family_weights"] = family_weights
        selected["selected_input_weights"] = input_weights
        selected["selected_cascade_alpha"] = float(cascade_meta["selected_cascade_alpha"])
        selected["cascade_validation_pr_auc"] = float(
            cascade_meta["stage2_alpha_validation_pr_auc"]
        )
        selection_df = pd.concat(
            [
                selection_df,
                pd.DataFrame(weight_rows),
                pd.DataFrame(alpha_rows),
                pd.DataFrame([selected]),
            ],
            ignore_index=True,
        )
        selection_df.to_csv(
            self.output_dir / "branch1_diversified_selection_history.csv", index=False
        )
        self._manifest(
            warmup_bins,
            test_start,
            test_end,
            dev_origins,
            selected,
            {
                "selected_family_weights": family_weights,
                "selected_input_weights": input_weights,
                "cascade_metadata": cascade_meta,
            },
        )
        print("[4/6] Fitting final Branch 1 models", flush=True)
        final_stage1 = self._fit_final_stage1(selected, test_start)
        metrics_rows: list[dict[str, Any]] = []
        tapes: list[pd.DataFrame] = []
        cycle_starts = np.arange(test_start, test_end, cadence_bins, dtype=np.int32)
        print(f"[5/6] Evaluating {len(cycle_starts)} common held-out cycles", flush=True)
        for cycle_idx, cycle_start in enumerate(cycle_starts):
            cycle_end = min(int(cycle_start) + cadence_bins, test_end)
            loaded_cycle = self._load_cycle_checkpoint(cycle_idx)
            if loaded_cycle is None:
                metrics, tape = self._predict_cycle(
                    final_stage1,
                    cascade_model,
                    cascade_meta,
                    selected,
                    cycle_idx,
                    int(cycle_start),
                    cycle_end,
                )
                self._save_cycle_checkpoint(cycle_idx, metrics, tape)
            else:
                metrics, tape = loaded_cycle
            metrics_rows.append(metrics)
            tapes.append(tape)
            print(
                f"  [{cycle_idx + 1}/{len(cycle_starts)}] "
                f"parallel={float(metrics['pr_auc_parallel']):.4f} "
                f"cascade={float(metrics['pr_auc_cascade']):.4f}",
                flush=True,
            )
        metrics_df = pd.DataFrame(metrics_rows)
        tape_df = pd.concat(tapes, ignore_index=True) if tapes else pd.DataFrame()
        metrics_df.to_csv(
            self.output_dir / "branch1_diversified_metrics.csv", index=False
        )
        tape_df.to_parquet(
            self.output_dir / "branch1_diversified_risk_tape.parquet",
            index=False,
            compression="zstd",
        )
        mean_parallel = float(metrics_df["pr_auc_parallel"].mean())
        mean_cascade = float(metrics_df["pr_auc_cascade"].mean())
        report = [
            "# Branch 1 diversified parallel + cascade 실험 보고서",
            "",
            "## 실험 계약",
            "- All-XID 통합 onset, 24시간 horizon, 오류 직전 10분 제외",
            "- 시간순 Sliding Training, 36시간 purge, 최근 development origin pooled validation",
            "- Branch 1 telemetry 입력 1시간·6시간·24시간 모두 사용",
            "- Logistic, Extra Trees, HistGradientBoosting, TemporalCNN1D 병렬 앙상블",
            "- prediction disagreement·uncertainty 기반 soft residual cascade",
            "- Branch 2와 Lambda fusion은 실행하지 않음",
            "",
            "## Terminal held-out 평균",
            f"- Parallel PR-AUC: {mean_parallel:.6f}",
            f"- Cascade PR-AUC: {mean_cascade:.6f}",
            f"- Parallel median PR-AUC: {float(metrics_df['pr_auc_parallel'].median()):.6f}",
            f"- Cascade median PR-AUC: {float(metrics_df['pr_auc_cascade'].median()):.6f}",
            f"- Parallel normalized PR-AUC: {float(metrics_df['pr_auc_parallel_normalized'].mean()):.6f}",
            f"- Cascade normalized PR-AUC: {float(metrics_df['pr_auc_cascade_normalized'].mean()):.6f}",
            f"- Parallel ROC-AUC: {float(metrics_df['roc_auc_parallel'].mean()):.6f}",
            f"- Cascade ROC-AUC: {float(metrics_df['roc_auc_cascade'].mean()):.6f}",
            f"- Cascade가 Parallel보다 높은 cycle: {int((metrics_df['pr_auc_cascade'] > metrics_df['pr_auc_parallel']).sum())}/{len(metrics_df)}",
            "",
            "## 기준선",
            "- 동일 terminal held-out Branch 1 v1 PR-AUC: 0.023381",
            "- GitHub-base v2.1 Branch 1 PR-AUC: 0.022086",
            f"- Cascade가 v1보다 높은지: {mean_cascade > 0.023381}",
            "",
            "## 산출물",
            "- branch1_diversified_metrics.csv",
            "- branch1_diversified_risk_tape.parquet",
            "- branch1_diversified_selection_history.csv",
            "- experiment_manifest.json",
            "- checkpoints/",
        ]
        (self.output_dir / "branch1_diversified_report.md").write_text(
            "\n".join(report), encoding="utf-8"
        )
        self._manifest(
            warmup_bins,
            test_start,
            test_end,
            dev_origins,
            selected,
            {
                "selected_family_weights": family_weights,
                "selected_input_weights": input_weights,
                "cascade_metadata": cascade_meta,
                "completed_cycle_count": int(len(metrics_df)),
                "mean_pr_auc_parallel": mean_parallel,
                "mean_pr_auc_cascade": mean_cascade,
            },
        )
        print(f"[6/6] Saved outputs in {self.output_dir}", flush=True)
        print(f"  elapsed={(time.time() - started) / 60.0:.1f} minutes", flush=True)
        return metrics_df, tape_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-XID Branch 1 diversified parallel ensemble and soft cascade"
    )
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/2026-09-15_All-XID_Branch1_Diversified_Parallel_Cascade_01",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    data_dir = base.find_data_dir()
    cache_dir = base.PROJECT_ROOT / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    engine = base.UnifiedDataEngine(data_dir=data_dir, cache_dir=cache_dir)
    _, history_map, gt_matrix = engine.load_all_xid_ledger()
    pipeline = Branch1Diversified(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=base.PROJECT_ROOT / args.output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        resume=args.resume,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

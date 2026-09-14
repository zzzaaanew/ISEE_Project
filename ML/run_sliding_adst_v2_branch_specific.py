"""
[All-XID ADST v2: branch-specific windows, recency weighting, and ADWIN]

This runner is intentionally separate from the GitHub-base runner and from
the first purged held-out experiment.  It keeps the data/label contracts and
the terminal held-out split, while changing only the ADST selection layer:

* Branch 1 keeps all 1h/6h/24h telemetry inputs in a parallel ensemble.
* Branch 2 remains strictly History-only.
* Branch 1 and Branch 2 select training windows independently.
* Training samples receive validation-selected recency weights.
* A small, dependency-free ADWIN implementation monitors matured validation
  loss, prediction disagreement, and positive prevalence.
* Lambda is selected only after the two branch predictions are fixed.

The primary run is a fixed terminal held-out benchmark.  No terminal-test
labels are used to change a model or a selection decision.
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
import run_sliding_adst_lambda_heldout as legacy


STEP_NS = base.STEP_NS
DAY_NS = base.DAY_NS
HOUR_NS = base.HOUR_NS
PURGE_BINS = max(1, int(base.PURGE_NS // STEP_NS))
VALIDATION_BINS = int(3 * DAY_NS // STEP_NS)
VALIDATION_BLOCKS = 3
TEST_FRACTION = 0.20
TRAIN_DAYS_GRID = (7, 14, 21)
OBS_HOURS_GRID = (1, 6, 24)
HALF_LIFE_GRID = (3, 7, 14)
LAMBDA_GRID = base.LAMBDA_GRID
ADWIN_DELTA = 0.01
ADWIN_MIN_SUBWINDOW = 4


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


def _recency_weights(
    train_bins: np.ndarray,
    train_end: int,
    half_life_days: int,
) -> np.ndarray:
    """Half-life weighting; the most recent training bin has weight 1."""
    age_days = np.maximum(
        (int(train_end) - train_bins.astype(np.int64)) * STEP_NS / DAY_NS,
        0.0,
    )
    return np.power(0.5, age_days / float(half_life_days)).astype(np.float64)


def _fit_weighted_logistic(
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


def _normalized_pr_auc(ap: float, prevalence: float) -> float:
    """PR-AUC skill above the prevalence baseline, bounded for reporting."""
    if prevalence >= 1.0:
        return 0.0
    return float(np.clip((float(ap) - float(prevalence)) / (1.0 - float(prevalence)), -1.0, 1.0))


class ADWINLite:
    """Small bounded-stream ADWIN detector for local experiment control.

    The detector searches valid historical/new cuts and removes the older
    side after a statistically supported mean shift.  It is used only on
    development/validation feedback; terminal held-out labels never enter it.
    """

    def __init__(
        self,
        delta: float = ADWIN_DELTA,
        min_subwindow: int = ADWIN_MIN_SUBWINDOW,
        max_window: int = 64,
        values: list[float] | None = None,
    ) -> None:
        self.delta = float(delta)
        self.min_subwindow = int(min_subwindow)
        self.max_window = int(max_window)
        self.values = [float(v) for v in (values or [])]
        self.last_cut: int | None = None
        self.last_difference: float | None = None

    def update(self, value: float) -> bool:
        self.values.append(float(np.clip(value, 0.0, 1.0)))
        self.last_cut = None
        self.last_difference = None
        if len(self.values) < 2 * self.min_subwindow:
            return False
        if len(self.values) > self.max_window:
            self.values = self.values[-self.max_window :]

        values = np.asarray(self.values, dtype=np.float64)
        n = len(values)
        # Check a compact set of cuts to keep selection CPU bounded.
        cuts = sorted(
            set(
                [
                    self.min_subwindow,
                    n // 4,
                    n // 3,
                    n // 2,
                    (2 * n) // 3,
                    (3 * n) // 4,
                    n - self.min_subwindow,
                ]
            )
        )
        log_term = np.log(max(4.0, 2.0 * np.log(max(3, n)) / self.delta))
        for cut in cuts:
            if cut < self.min_subwindow or n - cut < self.min_subwindow:
                continue
            old = values[:cut]
            new = values[cut:]
            difference = abs(float(old.mean()) - float(new.mean()))
            epsilon = np.sqrt(0.5 * log_term * (1.0 / len(old) + 1.0 / len(new)))
            if difference > epsilon:
                self.last_cut = int(cut)
                self.last_difference = difference
                self.values = values[cut:].tolist()
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "min_subwindow": self.min_subwindow,
            "max_window": self.max_window,
            "values": self.values,
            "last_cut": self.last_cut,
            "last_difference": self.last_difference,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ADWINLite":
        if not payload:
            return cls()
        return cls(
            delta=float(payload.get("delta", ADWIN_DELTA)),
            min_subwindow=int(payload.get("min_subwindow", ADWIN_MIN_SUBWINDOW)),
            max_window=int(payload.get("max_window", 64)),
            values=[float(v) for v in payload.get("values", [])],
        )


class BranchSpecificADST(legacy.SlidingADSTLambdaHeldout):
    """ADST v2 pipeline with branch-specific adaptation."""

    def __init__(self, *args: Any, branch2_mode: str = "current", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if branch2_mode not in {"current", "history_0910"}:
            raise ValueError("Unsupported branch2_mode; use 'current' or 'history_0910'.")
        self.branch2_mode = branch2_mode

    def _selection_blocks(self, origin_bin: int) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block_idx in range(VALIDATION_BLOCKS):
            val_end = int(origin_bin - PURGE_BINS - block_idx * VALIDATION_BINS)
            val_start = int(val_end - VALIDATION_BINS)
            train_end = int(val_start - PURGE_BINS)
            if train_end <= 0 or val_start <= 0 or val_end <= val_start:
                raise ValueError(
                    f"Insufficient history for validation block {block_idx} at {origin_bin}."
                )
            rng = np.random.default_rng(
                self.seed + int(origin_bin) + 10_007 * (block_idx + 1)
            )
            val_bins, val_gpus, val_labels = self.sample_indices(val_start, val_end, rng)
            if len(val_labels) == 0 or np.unique(val_labels).size < 2:
                raise ValueError(
                    f"Validation block {block_idx} at origin {origin_bin} has insufficient classes."
                )
            blocks.append(
                {
                    "block_idx": block_idx,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "val_bins": val_bins,
                    "val_gpus": val_gpus,
                    "val_labels": val_labels,
                }
            )
        return blocks

    @staticmethod
    def _aggregate(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
        }

    @staticmethod
    def _candidate_key(
        row: dict[str, Any],
        drift_hint: bool,
        train_key: str,
        half_life_key: str,
    ) -> tuple[float, float, float, float, float]:
        """Mean AP is primary; stability and ADWIN direction are tie-breakers."""
        train_days = float(row[train_key])
        half_life = float(row[half_life_key])
        direction = -(train_days + half_life) if drift_hint else (train_days + half_life)
        return (
            float(row["val_pr_auc_mean"]),
            float(row["val_pr_auc_median"]),
            float(row["val_pr_auc_q25"]),
            -float(row["val_pr_auc_std"]),
            direction,
        )

    def _apply_adwin_policy(
        self,
        rows: list[dict[str, Any]],
        previous: dict[str, Any] | None,
        drift_hint: bool,
        train_key: str,
        half_life_key: str,
    ) -> list[dict[str, Any]]:
        if previous is None:
            return rows
        previous_train = int(previous.get(train_key, previous.get("selected_L_train_days", 21)))
        previous_half = int(previous.get(half_life_key, previous.get("selected_half_life_days", 14)))
        if drift_hint:
            preferred = [
                row
                for row in rows
                if int(row[train_key]) <= previous_train
                and int(row[half_life_key]) <= previous_half
            ]
        elif bool(previous.get("adwin_stable", False)):
            preferred = [
                row
                for row in rows
                if int(row[train_key]) >= previous_train
                and int(row[half_life_key]) >= previous_half
            ]
        else:
            preferred = []
        # Keep a meaningful candidate pool if the direction is impossible.
        return preferred if len(preferred) >= 2 else rows

    def _fit_selection_model(
        self,
        branch: str,
        train_df: pd.DataFrame,
        train_tensor: np.ndarray | None,
        train_labels: np.ndarray,
        train_bins: np.ndarray,
        train_end: int,
        half_life_days: int,
        model_seed: int,
    ) -> Any:
        if branch == "b2" and self.branch2_mode == "history_0910":
            weights = np.ones(len(train_labels), dtype=np.float64)
        else:
            weights = _recency_weights(train_bins, train_end, half_life_days)
        return _fit_weighted_logistic(train_df, train_labels, weights, model_seed)

    def select_adst_config(
        self,
        origin_bin: int,
        rng: np.random.Generator,
        previous: dict[str, Any] | None,
        stage: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        del rng  # Candidate sampling is deterministic per origin/block/config.
        blocks = self._selection_blocks(origin_bin)
        block_b1: dict[tuple[int, int, int, int], np.ndarray] = {}
        block_b2: dict[tuple[int, int, int], np.ndarray] = {}
        b1_metrics: dict[tuple[int, int], list[dict[str, Any]]] = {}
        b2_metrics: dict[tuple[int, int], list[dict[str, Any]]] = {}

        for block in blocks:
            block_idx = int(block["block_idx"])
            train_end = int(block["train_end"])
            val_bins = block["val_bins"]
            val_gpus = block["val_gpus"]
            val_labels = block["val_labels"]
            val_b2_df = self.engine.extract_branch2_features(
                val_bins, val_gpus, self.history_map
            )
            val_b1_features = {
                obs: self.engine.extract_branch1_features(val_bins, val_gpus, obs)[0]
                for obs in OBS_HOURS_GRID
            }

            train_cache_b2: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}
            train_cache_b1: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}
            for train_days in TRAIN_DAYS_GRID:
                train_start = max(0, train_end - int(train_days * DAY_NS // STEP_NS))
                local_rng = np.random.default_rng(
                    self.seed
                    + int(origin_bin)
                    + 100_003 * (block_idx + 1)
                    + 1_009 * train_days
                )
                train_bins, train_gpus, train_labels = self.sample_indices(
                    train_start, train_end, local_rng
                )
                if len(train_labels) == 0 or np.unique(train_labels).size < 2:
                    continue
                train_b2_df = self.engine.extract_branch2_features(
                    train_bins, train_gpus, self.history_map
                )
                train_cache_b2[train_days] = (
                    train_bins,
                    train_gpus,
                    train_labels,
                    train_b2_df,
                )
                for obs in OBS_HOURS_GRID:
                    train_b1_df, _ = self.engine.extract_branch1_features(
                        train_bins, train_gpus, obs
                    )
                    train_cache_b1[(train_days, obs)] = (
                        train_bins,
                        train_gpus,
                        train_labels,
                        train_b1_df,
                    )

            for train_days in TRAIN_DAYS_GRID:
                if train_days not in train_cache_b2:
                    continue
                train_bins, _, train_labels, train_b2_df = train_cache_b2[train_days]
                for half_life in HALF_LIFE_GRID:
                    model = self._fit_selection_model(
                        "b2",
                        train_b2_df,
                        None,
                        train_labels,
                        train_bins,
                        train_end,
                        half_life,
                        self.seed + origin_bin + block_idx + train_days + half_life,
                    )
                    pred = model.predict_proba(val_b2_df)[:, 1]
                    key = (block_idx, train_days, half_life)
                    block_b2[key] = pred
                    b2_metrics.setdefault((train_days, half_life), []).append(
                        {
                            "ap": base.safe_average_precision(val_labels, pred),
                            "roc": base.safe_roc_auc(val_labels, pred),
                        }
                    )

                for half_life in HALF_LIFE_GRID:
                    for obs in OBS_HOURS_GRID:
                        train_bins_1, _, train_labels_1, train_b1_df = train_cache_b1[
                            (train_days, obs)
                        ]
                        model = self._fit_selection_model(
                            "b1",
                            train_b1_df,
                            None,
                            train_labels_1,
                            train_bins_1,
                            train_end,
                            half_life,
                            self.seed + origin_bin + block_idx + train_days + half_life + obs,
                        )
                        pred = model.predict_proba(val_b1_features[obs])[:, 1]
                        block_b1[(block_idx, train_days, half_life, obs)] = pred

            # Temporary caches are deliberately block-scoped to control memory.
            del train_cache_b1, train_cache_b2, val_b1_features, val_b2_df

        b1_rows: list[dict[str, Any]] = []
        for train_days in TRAIN_DAYS_GRID:
            for half_life in HALF_LIFE_GRID:
                values: list[dict[str, Any]] = []
                for block_idx in range(VALIDATION_BLOCKS):
                    preds = [
                        block_b1[(block_idx, train_days, half_life, obs)]
                        for obs in OBS_HOURS_GRID
                    ]
                    labels = blocks[block_idx]["val_labels"]
                    equal_pred = np.mean(preds, axis=0)
                    values.append(
                        {
                            "ap": base.safe_average_precision(labels, equal_pred),
                            "roc": base.safe_roc_auc(labels, equal_pred),
                        }
                    )
                agg = self._aggregate([item["ap"] for item in values])
                b1_rows.append(
                    {
                        "selection_role": "branch1",
                        "candidate_b1_train_days": train_days,
                        "candidate_b1_half_life_days": half_life,
                        "candidate_L_obs_hours": "all_1_6_24",
                        "val_pr_auc_mean": agg["mean"],
                        "val_pr_auc_median": agg["median"],
                        "val_pr_auc_q25": agg["q25"],
                        "val_pr_auc_std": agg["std"],
                        "val_pr_auc_min": agg["min"],
                        "val_roc_auc_mean": float(np.mean([item["roc"] for item in values])),
                    }
                )

        b2_rows: list[dict[str, Any]] = []
        for train_days in TRAIN_DAYS_GRID:
            for half_life in HALF_LIFE_GRID:
                values = b2_metrics.get((train_days, half_life), [])
                if len(values) != VALIDATION_BLOCKS:
                    continue
                agg = self._aggregate([item["ap"] for item in values])
                b2_rows.append(
                    {
                        "selection_role": "branch2",
                        "candidate_b2_train_days": train_days,
                        "candidate_b2_half_life_days": half_life,
                        "candidate_L_obs_hours": "history_only",
                        "val_pr_auc_mean": agg["mean"],
                        "val_pr_auc_median": agg["median"],
                        "val_pr_auc_q25": agg["q25"],
                        "val_pr_auc_std": agg["std"],
                        "val_pr_auc_min": agg["min"],
                        "val_roc_auc_mean": float(np.mean([item["roc"] for item in values])),
                    }
                )
        if not b1_rows or not b2_rows:
            raise ValueError(f"No valid branch candidates at origin {origin_bin}.")

        previous_drift = bool(previous and previous.get("adwin_triggered", False))
        b1_pool = self._apply_adwin_policy(
            b1_rows,
            previous,
            previous_drift,
            "candidate_b1_train_days",
            "candidate_b1_half_life_days",
        )
        b2_pool = self._apply_adwin_policy(
            b2_rows,
            previous,
            previous_drift,
            "candidate_b2_train_days",
            "candidate_b2_half_life_days",
        )
        b1_best = max(
            b1_pool,
            key=lambda row: self._candidate_key(
                row,
                previous_drift,
                "candidate_b1_train_days",
                "candidate_b1_half_life_days",
            ),
        ).copy()
        b2_best = max(
            b2_pool,
            key=lambda row: self._candidate_key(
                row,
                previous_drift,
                "candidate_b2_train_days",
                "candidate_b2_half_life_days",
            ),
        ).copy()

        # The selected Branch 1 setting keeps all three input windows active.
        b1_train_days = int(b1_best["candidate_b1_train_days"])
        b1_half_life = int(b1_best["candidate_b1_half_life_days"])
        b1_input_aps: dict[str, float] = {}
        for obs in OBS_HOURS_GRID:
            per_block = []
            for block_idx in range(VALIDATION_BLOCKS):
                pred = block_b1[(block_idx, b1_train_days, b1_half_life, obs)]
                per_block.append(
                    base.safe_average_precision(blocks[block_idx]["val_labels"], pred)
                )
            b1_input_aps[str(obs)] = float(np.mean(per_block))
        b1_weight_raw = {key: 0.20 + max(value, 0.0) for key, value in b1_input_aps.items()}
        b1_weight_total = sum(b1_weight_raw.values())
        b1_weights = {
            key: float(value / b1_weight_total) for key, value in b1_weight_raw.items()
        }

        b1_selected_preds: list[np.ndarray] = []
        b2_selected_preds: list[np.ndarray] = []
        b1_selected_aps: list[float] = []
        b2_selected_aps: list[float] = []
        fused_candidates: list[dict[str, Any]] = []
        for block_idx in range(VALIDATION_BLOCKS):
            b1_pred = sum(
                b1_weights[str(obs)]
                * block_b1[(block_idx, b1_train_days, b1_half_life, obs)]
                for obs in OBS_HOURS_GRID
            )
            b2_pred = block_b2[
                (block_idx, int(b2_best["candidate_b2_train_days"]), int(b2_best["candidate_b2_half_life_days"]))
            ]
            b1_selected_preds.append(b1_pred)
            b2_selected_preds.append(b2_pred)
            labels = blocks[block_idx]["val_labels"]
            b1_selected_aps.append(base.safe_average_precision(labels, b1_pred))
            b2_selected_aps.append(base.safe_average_precision(labels, b2_pred))

        for lambda_value in LAMBDA_GRID:
            aps: list[float] = []
            rocs: list[float] = []
            for block_idx in range(VALIDATION_BLOCKS):
                labels = blocks[block_idx]["val_labels"]
                fused = lambda_value * b1_selected_preds[block_idx] + (1.0 - lambda_value) * b2_selected_preds[block_idx]
                aps.append(base.safe_average_precision(labels, fused))
                rocs.append(base.safe_roc_auc(labels, fused))
            agg = self._aggregate(aps)
            fused_candidates.append(
                {
                    "selection_role": "fusion",
                    "candidate_lambda": float(lambda_value),
                    "val_pr_auc_mean": agg["mean"],
                    "val_pr_auc_median": agg["median"],
                    "val_pr_auc_q25": agg["q25"],
                    "val_pr_auc_std": agg["std"],
                    "val_pr_auc_min": agg["min"],
                    "val_roc_auc_mean": float(np.mean(rocs)),
                }
            )
        fusion_best = max(
            fused_candidates,
            key=lambda row: (
                float(row["val_pr_auc_mean"]),
                float(row["val_pr_auc_median"]),
                float(row["val_pr_auc_q25"]),
                -float(row["val_pr_auc_std"]),
                -abs(float(row["candidate_lambda"]) - 0.5),
            ),
        )
        selected_lambda = float(fusion_best["candidate_lambda"])
        latest_idx = 0
        latest_labels = blocks[latest_idx]["val_labels"]
        latest_fused = (
            selected_lambda * b1_selected_preds[latest_idx]
            + (1.0 - selected_lambda) * b2_selected_preds[latest_idx]
        )
        latest_ap = base.safe_average_precision(latest_labels, latest_fused)
        latest_disagreement = float(
            np.mean(np.abs(b1_selected_preds[latest_idx] - b2_selected_preds[latest_idx]))
        )
        latest_prevalence = float(np.mean(latest_labels))

        loss_detector = ADWINLite.from_dict(previous.get("adwin_loss_state") if previous else None)
        disagreement_detector = ADWINLite.from_dict(
            previous.get("adwin_disagreement_state") if previous else None
        )
        prevalence_detector = ADWINLite.from_dict(
            previous.get("adwin_prevalence_state") if previous else None
        )
        loss_trigger = loss_detector.update(1.0 - latest_ap)
        disagreement_trigger = disagreement_detector.update(latest_disagreement)
        prevalence_trigger = prevalence_detector.update(latest_prevalence)
        adwin_triggered = bool(loss_trigger or disagreement_trigger or prevalence_trigger)
        previous_latest_ap = None if previous is None else previous.get("latest_val_pr_auc")
        improving = previous_latest_ap is not None and latest_ap >= float(previous_latest_ap)
        stable_count = int(previous.get("adwin_stable_count", 0)) if previous else 0
        stable_count = stable_count + 1 if (not adwin_triggered and improving) else 0

        if previous is None:
            action = "initial"
        elif adwin_triggered:
            action = "adwin_shrink_recommendation"
        elif stable_count >= 3:
            action = "adwin_expand_recommendation"
        else:
            action = "hold"

        selected = {
            "selection_stage": stage,
            "selection_origin_idx": int(origin_bin),
            "selection_origin_time": _time_text(self.engine, origin_bin),
            "train_end_time": _time_text(self.engine, blocks[0]["train_end"]),
            "validation_start_time": _time_text(self.engine, blocks[-1]["val_start"]),
            "validation_end_time": _time_text(self.engine, blocks[0]["val_end"]),
            "validation_block_count": VALIDATION_BLOCKS,
            "validation_block_days": VALIDATION_BINS * base.STEP_MINUTES / (24 * 60),
            "purge_hours": int(base.PURGE_NS // HOUR_NS),
            "selected_b1_train_days": b1_train_days,
            "selected_b1_half_life_days": b1_half_life,
            "selected_b1_input_hours": list(OBS_HOURS_GRID),
            "selected_b1_input_weights": b1_weights,
            "selected_b2_train_days": int(b2_best["candidate_b2_train_days"]),
            "selected_b2_half_life_days": int(b2_best["candidate_b2_half_life_days"]),
            "selected_L_train_days": b1_train_days,
            "selected_L_obs_hours": -1,
            "selected_lambda": selected_lambda,
            "val_pr_auc": float(fusion_best["val_pr_auc_mean"]),
            "val_roc_auc": float(fusion_best["val_roc_auc_mean"]),
            "val_pr_auc_b1": float(np.mean(b1_selected_aps)),
            "val_pr_auc_b2": float(np.mean(b2_selected_aps)),
            "val_pr_auc_b1_median": float(np.median(b1_selected_aps)),
            "val_pr_auc_b2_median": float(np.median(b2_selected_aps)),
            "prediction_disagreement": latest_disagreement,
            "validation_positive_rate": latest_prevalence,
            "latest_val_pr_auc": float(latest_ap),
            "adst_action": action,
            "adwin_triggered": adwin_triggered,
            "adwin_loss_triggered": bool(loss_trigger),
            "adwin_disagreement_triggered": bool(disagreement_trigger),
            "adwin_prevalence_triggered": bool(prevalence_trigger),
            "adwin_stable": stable_count >= 3,
            "adwin_stable_count": stable_count,
            "adwin_loss_state": loss_detector.to_dict(),
            "adwin_disagreement_state": disagreement_detector.to_dict(),
            "adwin_prevalence_state": prevalence_detector.to_dict(),
            "previous_val_pr_auc": previous_latest_ap,
            "val_pr_auc_delta": None if previous_latest_ap is None else float(latest_ap - float(previous_latest_ap)),
            "candidate_count": len(b1_rows) + len(b2_rows) + len(fused_candidates),
            "branch1_input_validation_ap": b1_input_aps,
        }

        common = {
            "selection_stage": stage,
            "selection_origin_idx": int(origin_bin),
            "selection_origin_time": _time_text(self.engine, origin_bin),
            "purge_hours": int(base.PURGE_NS // HOUR_NS),
        }
        audit_rows: list[dict[str, Any]] = []
        for row in b1_rows + b2_rows + fused_candidates:
            audit_row = dict(common)
            audit_row.update(row)
            audit_rows.append(audit_row)
        return selected, audit_rows

    def _fit_b1_pair(
        self,
        train_df: pd.DataFrame,
        train_tensor: np.ndarray,
        train_labels: np.ndarray,
        train_bins: np.ndarray,
        train_end: int,
        half_life_days: int,
        seed: int,
    ) -> tuple[Any, Any]:
        weights = _recency_weights(train_bins, train_end, half_life_days)
        tree = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=int(seed),
        )
        tree.fit(train_df, train_labels, sample_weight=weights)

        cnn = base.TemporalCNN1D(in_channels=7, hidden_channels=32, dropout=0.2)
        optimizer = torch.optim.AdamW(cnn.parameters(), lr=0.005, weight_decay=1e-4)
        positive_weight = max(
            1.0,
            (train_labels == 0).sum() / max(1, (train_labels == 1).sum()),
        )
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight], dtype=torch.float32),
            reduction="none",
        )
        x_train = torch.tensor(train_tensor, dtype=torch.float32)
        y_train = torch.tensor(train_labels, dtype=torch.float32)
        w_train = torch.tensor(weights, dtype=torch.float32)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train, w_train),
            batch_size=4096,
            shuffle=True,
            generator=generator,
        )
        base.seed_everything(int(seed))
        cnn.train()
        for _ in range(5):
            for batch_x, batch_y, batch_w in loader:
                optimizer.zero_grad()
                raw_loss = criterion(cnn(batch_x), batch_y)
                loss = (raw_loss * batch_w).mean()
                loss.backward()
                optimizer.step()
        cnn.eval()
        return tree, cnn

    def _fit_final_models(
        self,
        selected: dict[str, Any],
        test_start: int,
        model_seed: int,
    ) -> dict[str, Any]:
        b1_train_end = int(test_start - PURGE_BINS)
        b2_train_end = int(test_start - PURGE_BINS)
        b1_train_start = max(
            0,
            b1_train_end - int(int(selected["selected_b1_train_days"]) * DAY_NS // STEP_NS),
        )
        b2_train_start = max(
            0,
            b2_train_end - int(int(selected["selected_b2_train_days"]) * DAY_NS // STEP_NS),
        )
        b1_bins, b1_gpus, b1_labels = self.sample_indices(
            b1_train_start,
            b1_train_end,
            np.random.default_rng(self.seed + model_seed + 1),
        )
        b2_bins, b2_gpus, b2_labels = self.sample_indices(
            b2_train_start,
            b2_train_end,
            np.random.default_rng(self.seed + model_seed + 2),
        )
        if np.unique(b1_labels).size < 2 or np.unique(b2_labels).size < 2:
            raise ValueError("Final branch training block has insufficient classes.")

        b1_models: list[dict[str, Any]] = []
        for idx, obs in enumerate(OBS_HOURS_GRID):
            b1_df, b1_tensor = self.engine.extract_branch1_features(b1_bins, b1_gpus, obs)
            tree, cnn = self._fit_b1_pair(
                b1_df,
                b1_tensor,
                b1_labels,
                b1_bins,
                b1_train_end,
                int(selected["selected_b1_half_life_days"]),
                self.seed + model_seed + 101 * (idx + 1),
            )
            b1_models.append({"obs_hours": obs, "tree": tree, "cnn": cnn})

        b2_df = self.engine.extract_branch2_features(b2_bins, b2_gpus, self.history_map)
        if self.branch2_mode == "history_0910":
            b2_weights = np.ones(len(b2_labels), dtype=np.float64)
        else:
            b2_weights = _recency_weights(
                b2_bins,
                b2_train_end,
                int(selected["selected_b2_half_life_days"]),
            )
        b2_lr = _fit_weighted_logistic(
            b2_df,
            b2_labels,
            b2_weights,
            self.seed + model_seed + 201,
        )
        b2_gbdt = None
        if self.branch2_mode != "history_0910":
            b2_gbdt = HistGradientBoostingClassifier(
                max_iter=120,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=self.seed + model_seed + 202,
            )
            b2_gbdt.fit(b2_df, b2_labels, sample_weight=b2_weights)

        return {
            "b1_models": b1_models,
            "b2_lr": b2_lr,
            "b2_gbdt": b2_gbdt,
            "b1_weights": {str(k): float(v) for k, v in selected["selected_b1_input_weights"].items()},
            "b1_train_start": b1_train_start,
            "b1_train_end": b1_train_end,
            "b2_train_start": b2_train_start,
            "b2_train_end": b2_train_end,
            "b1_true_prior": float(self.gt_matrix[b1_train_start:b1_train_end].mean()),
            "b2_true_prior": float(self.gt_matrix[b2_train_start:b2_train_end].mean()),
            "b1_sample_prior": float(b1_labels.mean()),
            "b2_sample_prior": float(b2_labels.mean()),
        }

    def _predict_cycle(
        self,
        models: dict[str, Any],
        selected: dict[str, Any],
        cycle_idx: int,
        cycle_start: int,
        cycle_end: int,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        test_bins = np.arange(cycle_start, cycle_end, self.test_stride_bins, dtype=np.int32)
        if len(test_bins) == 0:
            raise ValueError(f"Empty held-out test cycle {cycle_idx}.")
        n_time = len(test_bins)
        n_gpu = self.engine.num_gpus
        b1_scores = np.zeros((n_time, n_gpu), dtype=np.float32)
        b2_scores = np.zeros((n_time, n_gpu), dtype=np.float32)
        fused_scores = np.zeros((n_time, n_gpu), dtype=np.float32)
        lambda_value = float(selected["selected_lambda"])

        for rel_idx, current_bin in enumerate(test_bins):
            all_gpus = np.arange(n_gpu, dtype=np.int32)
            current_bins = np.full(n_gpu, int(current_bin), dtype=np.int32)
            b1_prob = np.zeros(n_gpu, dtype=np.float64)
            for model_info in models["b1_models"]:
                obs = int(model_info["obs_hours"])
                b1_df, b1_tensor = self.engine.extract_branch1_features(current_bins, all_gpus, obs)
                p_tree = model_info["tree"].predict_proba(b1_df)[:, 1]
                with torch.no_grad():
                    p_cnn = torch.sigmoid(
                        model_info["cnn"](torch.tensor(b1_tensor, dtype=torch.float32))
                    ).cpu().numpy()
                p_model = 0.5 * p_tree + 0.5 * p_cnn
                p_model = base.adjusted_probability(
                    p_model,
                    models["b1_true_prior"],
                    models["b1_sample_prior"],
                )
                b1_prob += models["b1_weights"][str(obs)] * p_model

            b2_df = self.engine.extract_branch2_features(current_bins, all_gpus, self.history_map)
            if self.branch2_mode == "history_0910":
                p_b2 = models["b2_lr"].predict_proba(b2_df)[:, 1]
            else:
                p_b2 = 0.5 * models["b2_lr"].predict_proba(b2_df)[:, 1] + 0.5 * models["b2_gbdt"].predict_proba(b2_df)[:, 1]
            b2_prob = base.adjusted_probability(
                p_b2,
                models["b2_true_prior"],
                models["b2_sample_prior"],
            )
            b1_scores[rel_idx] = b1_prob
            b2_scores[rel_idx] = b2_prob
            fused_scores[rel_idx] = lambda_value * b1_prob + (1.0 - lambda_value) * b2_prob

        gt_test = self.gt_matrix[test_bins]
        y_test = gt_test.ravel()
        order = np.argsort(-fused_scores, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=np.uint16)
        np.put_along_axis(
            ranks,
            order,
            np.arange(1, n_gpu + 1, dtype=np.uint16)[None, :],
            axis=1,
        )
        top_k = min(100, n_gpu)
        positives = int(gt_test.sum())
        prevalence = float(gt_test.mean())
        hits = int((gt_test & (ranks <= top_k)).sum())
        ap_fused = base.safe_average_precision(y_test, fused_scores.ravel())
        ap_b1 = base.safe_average_precision(y_test, b1_scores.ravel())
        ap_b2 = base.safe_average_precision(y_test, b2_scores.ravel())
        metrics = {
            "origin_idx": int(cycle_idx),
            "model_origin_time": _time_text(self.engine, cycle_start),
            "test_start_time": _time_text(self.engine, cycle_start),
            "test_end_time": _time_text(self.engine, cycle_end - 1),
            "evaluation_scope": "common_terminal_heldout",
            "selected_b1_train_days": int(selected["selected_b1_train_days"]),
            "selected_b1_half_life_days": int(selected["selected_b1_half_life_days"]),
            "selected_b2_train_days": int(selected["selected_b2_train_days"]),
            "selected_b2_half_life_days": int(selected["selected_b2_half_life_days"]),
            "selected_b1_input_hours": "1,6,24",
            "selected_lambda": lambda_value,
            "adst_action": selected.get("adst_action"),
            "adwin_triggered": bool(selected.get("adwin_triggered", False)),
            "pr_auc_fused": ap_fused,
            "pr_auc_b1": ap_b1,
            "pr_auc_b2": ap_b2,
            "pr_auc_fused_normalized": _normalized_pr_auc(ap_fused, prevalence),
            "pr_auc_b1_normalized": _normalized_pr_auc(ap_b1, prevalence),
            "pr_auc_b2_normalized": _normalized_pr_auc(ap_b2, prevalence),
            "roc_auc_fused": base.safe_roc_auc(y_test, fused_scores.ravel()),
            "recall_at_100": hits / max(positives, 1),
            "lift_at_100": (hits / (len(test_bins) * top_k)) / max(prevalence, 1e-12),
            "positives": positives,
            "test_prevalence": prevalence,
        }
        top_mask = ranks <= top_k
        rel_idx, gpu_idx = np.where(top_mask)
        tape = pd.DataFrame(
            {
                "decision_time": pd.to_datetime(
                    self.engine.bin_start_ns[test_bins][rel_idx], unit="ns", utc=True
                ),
                "gpu_id": self.engine.gpu_ids[gpu_idx],
                "fused_risk": fused_scores[rel_idx, gpu_idx],
                "b1_risk": b1_scores[rel_idx, gpu_idx],
                "b2_risk": b2_scores[rel_idx, gpu_idx],
                "risk_rank": ranks[rel_idx, gpu_idx],
                "b1_train_days": int(selected["selected_b1_train_days"]),
                "b1_half_life_days": int(selected["selected_b1_half_life_days"]),
                "b2_train_days": int(selected["selected_b2_train_days"]),
                "b2_half_life_days": int(selected["selected_b2_half_life_days"]),
                "lambda_fusion": lambda_value,
                "target_24h": gt_test[rel_idx, gpu_idx].astype(np.uint8),
            }
        )
        return metrics, tape

    def _generate_summary_report(
        self,
        metrics_df: pd.DataFrame,
        selection_df: pd.DataFrame,
        split_info: dict[str, Any],
    ) -> None:
        def weighted_ap(column: str) -> float:
            weights = metrics_df["positives"].to_numpy(dtype=float)
            values = metrics_df[column].to_numpy(dtype=float)
            return float(np.average(values, weights=np.maximum(weights, 1.0)))

        mean_fused = float(metrics_df["pr_auc_fused"].mean())
        mean_b1 = float(metrics_df["pr_auc_b1"].mean())
        mean_b2 = float(metrics_df["pr_auc_b2"].mean())
        report_lines = [
            "# [ADST v2] All-XID Branch-specific Recency + ADWIN 보고서",
            "",
            "## Material Passport",
            "- **Material type**: Code experiment result and reproducibility record",
            "- **Status**: COMPLETED",
            f"- **Runner**: `{Path(__file__).name}`",
            f"- **Target**: All-XID unified onset, 24-hour horizon",
            f"- **Seed**: `{self.seed}`",
            f"- **Terminal held-out fraction**: `{TEST_FRACTION:.0%}`",
            "",
            "## 1. 실험 계약",
            "- Branch 1: 1h·6h·24h telemetry 입력을 모두 유지하는 parallel ensemble",
            "- Branch 2: `xid_count_30d`, `days_since_xid`만 사용하는 strict History-only",
            "- Train·Validation·Test: 시간순, Train–Validation 및 Validation–Test 사이 36시간 purge",
            "- Validation: 3개의 rolling 3일 block, PR-AUC 평균을 1순위로 하고 median·Q25·변동성을 tie-break",
            "- Recency: Branch별 half-life 후보 3일·7일·14일",
            "- ADWIN: validation loss, prediction disagreement, positive prevalence를 개발 구간에서만 감시",
            "- Fusion: `p = lambda*p_B1 + (1-lambda)*p_B2`, Branch별 선택 완료 후 Lambda 선택",
            "- Test: terminal held-out label은 선택·ADWIN 상태 갱신에 사용하지 않음",
            "",
            "## 2. Terminal held-out 평균 지표",
            f"- **Fused PR-AUC**: `{mean_fused:.6f}`",
            f"- **Branch 1 PR-AUC**: `{mean_b1:.6f}`",
            f"- **Branch 2 PR-AUC**: `{mean_b2:.6f}`",
            f"- **Fused normalized PR-AUC skill**: `{float(metrics_df['pr_auc_fused_normalized'].mean()):.6f}`",
            f"- **B2 positive-weighted PR-AUC**: `{weighted_ap('pr_auc_b2'):.6f}`",
            f"- **Fused ROC-AUC**: `{float(metrics_df['roc_auc_fused'].mean()):.6f}`",
            f"- **Recall@100**: `{float(metrics_df['recall_at_100'].mean()):.2%}`",
            f"- **Lift@100**: `{float(metrics_df['lift_at_100'].mean()):.3f}x`",
            "",
            "## 3. Branch 2 성능 하락 진단",
            "- 현재 결과는 이전 GitHub 원형 결과와 feature·재학습·split 계약이 달라 직접적인 성능 하락으로 단정하지 않는다.",
            "- 본 실험에서는 Branch 2 입력을 History-only로 고정해 calendar/telemetry 신호가 섞이지 않도록 했다.",
            "- 성능 해석은 macro PR-AUC, positive-weighted PR-AUC, 양성률별 결과를 함께 사용한다.",
            "- ADWIN은 terminal test 결과를 보고 window를 사후 조정하지 않는다.",
            "",
            "## 4. 양성률 구간별 Branch 2",
        ]
        q1 = float(metrics_df["test_prevalence"].quantile(1 / 3))
        q2 = float(metrics_df["test_prevalence"].quantile(2 / 3))
        groups = [
            ("low_prevalence", metrics_df["test_prevalence"] <= q1),
            ("mid_prevalence", (metrics_df["test_prevalence"] > q1) & (metrics_df["test_prevalence"] <= q2)),
            ("high_prevalence", metrics_df["test_prevalence"] > q2),
        ]
        for name, mask in groups:
            subset = metrics_df.loc[mask]
            if subset.empty:
                continue
            report_lines.append(
                f"- `{name}`: cycles={len(subset)}, prevalence={subset['test_prevalence'].mean():.4f}, "
                f"B2 PR-AUC={subset['pr_auc_b2'].mean():.6f}, "
                f"B2 normalized={subset['pr_auc_b2_normalized'].mean():.6f}"
            )
        report_lines.extend(
            [
                "",
                "## 5. ADWIN 및 선택 이력",
                f"- ADWIN trigger 횟수: `{int(metrics_df['adwin_triggered'].sum())}` cycle에 해당하는 고정 선택 상태 기록",
                f"- 선택 stage 수: `{selection_df['selection_stage'].nunique() if 'selection_stage' in selection_df else 0}`",
                "- `selection_history.csv`에는 Branch 1·Branch 2 후보, Lambda 후보, block별 집계 기준을 남긴다.",
                "",
                "## 6. 해석 제한",
                "- ADST v2의 원인별 효과는 현재 v2 결과 하나만으로 분리되지 않으므로, 기존 커밋 버전과 동일 held-out test에서 ablation 비교가 필요하다.",
                "- fixed terminal benchmark와 향후 adaptive prequential 평가를 혼합하지 않는다.",
                f"- **Split**: `{split_info['test_start_time']}` ~ `{split_info['test_end_time']}`",
            ]
        )
        (self.output_dir / "adst_v2_report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
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
        selection_rows: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None

        print("[1/5] All-XID ADST v2 branch-specific development selection", flush=True)
        print(
            f"  development origins={len(dev_origins):,}, validation blocks={VALIDATION_BLOCKS}, "
            f"terminal test fraction={TEST_FRACTION:.0%}, purge={int(base.PURGE_NS // HOUR_NS)}h",
            flush=True,
        )
        for selection_idx, origin_bin in enumerate(dev_origins):
            checkpoint_name = f"dev_{selection_idx:04d}"
            loaded = self._load_selection_checkpoint(checkpoint_name)
            if loaded is None:
                base.seed_everything(self.seed + selection_idx)
                best, candidates = self.select_adst_config(
                    int(origin_bin),
                    np.random.default_rng(self.seed + selection_idx),
                    previous,
                    "development",
                )
                self._save_selection_checkpoint(checkpoint_name, best, candidates)
            else:
                best, candidates = loaded
            selection_rows.extend(candidates)
            previous = best
            print(
                f"  [{selection_idx + 1}/{len(dev_origins)}] {best['selection_origin_time']} "
                f"-> B1={best['selected_b1_train_days']}d/{best['selected_b1_half_life_days']}d, "
                f"B2={best['selected_b2_train_days']}d/{best['selected_b2_half_life_days']}d, "
                f"lambda={best['selected_lambda']:.1f}, val PR-AUC={float(best['val_pr_auc']):.4f}, "
                f"action={best['adst_action']}",
                flush=True,
            )

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
            self.output_dir / "adst_v2_selection_history.csv", index=False
        )
        _write_json_atomic(
            self.output_dir / "experiment_manifest.json",
            {
                "runner": Path(__file__).name,
                "base_runner": "ML/run_bidirectional_adst_fusion.py",
                "reference_runner": "ML/run_sliding_adst_lambda_heldout.py",
                "target": "all_xids",
                "branch2_contract": "history_only",
                "branch2_features": ["xid_count_30d", "days_since_xid"],
                "branch1_input_hours": list(OBS_HOURS_GRID),
                "branch1_parallel_ensemble": True,
                "recency_half_life_days": list(HALF_LIFE_GRID),
                "adwin_delta": ADWIN_DELTA,
                "validation_blocks": VALIDATION_BLOCKS,
                "validation_block_days": 3,
                "retrain_cadence_hours": self.retrain_cadence_hours,
                "negative_ratio": self.negative_ratio,
                "test_stride_bins": self.test_stride_bins,
                "seed": self.seed,
                "purge_hours": int(base.PURGE_NS // HOUR_NS),
                "heldout_test_fraction": TEST_FRACTION,
                "lambda_grid": [float(value) for value in LAMBDA_GRID],
                "warmup_start_time": _time_text(self.engine, warmup_bins),
                "test_start_time": _time_text(self.engine, test_start),
                "test_end_time": _time_text(self.engine, test_end - 1),
                "selection_origin_count": int(len(dev_origins)),
                "final_selection": final_selected,
                "command": " ".join(sys.argv),
            },
        )

        print("[2/5] Fitting branch-specific recency-weighted final models", flush=True)
        models = self._fit_final_models(final_selected, test_start, model_seed=900_001)
        print(
            f"  B1 train={_time_text(self.engine, models['b1_train_start'])}..{_time_text(self.engine, models['b1_train_end'])}, "
            f"L={final_selected['selected_b1_train_days']}d, half-life={final_selected['selected_b1_half_life_days']}d; "
            f"B2 L={final_selected['selected_b2_train_days']}d, half-life={final_selected['selected_b2_half_life_days']}d; "
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
        metrics_df.to_csv(self.output_dir / "adst_v2_metrics.csv", index=False)
        tape_df.to_parquet(
            self.output_dir / "adst_v2_risk_tape.parquet",
            index=False,
            compression="zstd",
        )
        split_info = {
            "warmup_start_time": _time_text(self.engine, warmup_bins),
            "test_start_time": _time_text(self.engine, test_start),
            "test_end_time": _time_text(self.engine, test_end - 1),
        }
        self._generate_summary_report(metrics_df, selection_df, split_info)
        print(f"[4/5] Saved outputs in {self.output_dir}", flush=True)
        print(f"[5/5] Elapsed: {(time.time() - started) / 60.0:.1f} minutes", flush=True)
        return metrics_df, tape_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-XID ADST v2 with branch-specific windows, recency, ADWIN, and held-out test"
    )
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/2026-09-13_All-XID_ADST_v2_BranchSpecific_Recency_RobustLambda",
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
    pipeline = BranchSpecificADST(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        resume=args.resume,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

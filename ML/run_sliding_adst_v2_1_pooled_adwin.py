"""All-XID ADST v2.1: pooled validation and matured-batch ADWIN.

This is a separate experiment runner.  It reuses the GitHub-base data and
model contracts through the v2 runner, but changes the selection layer after
the v2 diagnosis:

* Branch 1 keeps the 1h/6h/24h parallel ensemble.
* Branch 2 remains strict History-only.
* Branch settings are selected on pooled, time-ordered validation feedback
  from the latest six development origins.
* ADWIN is updated with chronological 30-minute, label-matured validation
  batches.  A trigger requires two consecutive raw detections and observes a
  three-cycle cooldown.
* The terminal held-out split remains fixed and is never used for adaptation.
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

import run_bidirectional_adst_fusion as base
import run_sliding_adst_v2_branch_specific as v2


STEP_NS = v2.STEP_NS
DAY_NS = v2.DAY_NS
HOUR_NS = v2.HOUR_NS
PURGE_BINS = v2.PURGE_BINS
VALIDATION_BINS = v2.VALIDATION_BINS
VALIDATION_BLOCKS = v2.VALIDATION_BLOCKS
TRAIN_DAYS_GRID = v2.TRAIN_DAYS_GRID
OBS_HOURS_GRID = v2.OBS_HOURS_GRID
HALF_LIFE_GRID = v2.HALF_LIFE_GRID
B2_HISTORY_0910_HALF_LIFE_GRID = (0,)
LAMBDA_GRID = v2.LAMBDA_GRID
TEST_FRACTION = v2.TEST_FRACTION
POOL_ORIGINS = 6
FEEDBACK_BATCH_BINS = max(1, int(30 * 60 // base.STEP_MINUTES))
ADWIN_MIN_BATCHES = 8
ADWIN_CONSECUTIVE_TRIGGERS = 2
ADWIN_COOLDOWN_CYCLES = 3


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


class PooledBatchADST(v2.BranchSpecificADST):
    """v2 model contract with a pooled selector and batch-level drift state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pool_origin_bins: list[int] = []
        self.pool_rows: list[dict[str, Any]] = []

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
        """Same v2 fit, with the seed set before CNN construction."""
        weights = v2._recency_weights(train_bins, train_end, half_life_days)
        tree = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=int(seed),
        )
        tree.fit(train_df, train_labels, sample_weight=weights)
        base.seed_everything(int(seed))
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

    def _fit_selector(
        self,
        branch: str,
        train_df: pd.DataFrame,
        train_labels: np.ndarray,
        train_bins: np.ndarray,
        train_end: int,
        half_life: int,
        seed: int,
    ) -> Any:
        return self._fit_selection_model(
            branch,
            train_df,
            None,
            train_labels,
            train_bins,
            train_end,
            half_life,
            seed,
        )

    def _b2_half_life_grid(self) -> tuple[int, ...]:
        """Return the B2 recency candidates for the selected protocol."""
        if self.branch2_mode == "history_0910":
            return B2_HISTORY_0910_HALF_LIFE_GRID
        return HALF_LIFE_GRID

    def _evaluate_origin_full(self, origin_bin: int) -> dict[str, Any]:
        """Evaluate all v2 selector candidates at one purged origin."""
        blocks = self._selection_blocks(origin_bin)
        for block in blocks:
            block_idx = int(block["block_idx"])
            train_end = int(block["train_end"])
            val_bins = block["val_bins"]
            val_gpus = block["val_gpus"]
            val_labels = block["val_labels"]
            val_b2_df = self.engine.extract_branch2_features(
                val_bins, val_gpus, self.history_map
            )
            val_b1_df = {
                obs: self.engine.extract_branch1_features(val_bins, val_gpus, obs)[0]
                for obs in OBS_HOURS_GRID
            }
            train_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[int, pd.DataFrame]]] = {}
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
                train_b1_df = {
                    obs: self.engine.extract_branch1_features(
                        train_bins, train_gpus, obs
                    )[0]
                    for obs in OBS_HOURS_GRID
                }
                train_cache[train_days] = (
                    train_bins,
                    train_gpus,
                    train_labels,
                    train_b2_df,
                    train_b1_df,
                )

            block_b1: dict[tuple[int, int, int], np.ndarray] = {}
            block_b2: dict[tuple[int, int], np.ndarray] = {}
            for train_days, cache in train_cache.items():
                train_bins, _, train_labels, train_b2_df, train_b1_df = cache
                for half_life in HALF_LIFE_GRID:
                    for obs in OBS_HOURS_GRID:
                        b1_model = self._fit_selector(
                            "b1",
                            train_b1_df[obs],
                            train_labels,
                            train_bins,
                            train_end,
                            half_life,
                            self.seed
                            + origin_bin
                            + block_idx
                            + train_days
                            + half_life
                            + obs,
                        )
                        block_b1[(train_days, half_life, obs)] = b1_model.predict_proba(
                            val_b1_df[obs]
                        )[:, 1]
                for half_life in self._b2_half_life_grid():
                    b2_model = self._fit_selector(
                        "b2",
                        train_b2_df,
                        train_labels,
                        train_bins,
                        train_end,
                        half_life,
                        self.seed + origin_bin + block_idx + train_days + half_life,
                    )
                    block_b2[(train_days, half_life)] = b2_model.predict_proba(val_b2_df)[:, 1]
            block["pred_b1"] = block_b1
            block["pred_b2"] = block_b2
        return {"origin_bin": int(origin_bin), "blocks": blocks}

    def _score_selected_origin(
        self,
        origin_bin: int,
        b1_train_days: int,
        b1_half_life: int,
        b2_train_days: int,
        b2_half_life: int,
    ) -> dict[str, Any]:
        """Recompute only one selected branch pair at an origin.

        This is used for the final pooled lambda/input-weight selection so the
        selected pair is evaluated on the same six origins as the branch rows.
        """
        blocks = self._selection_blocks(origin_bin)
        for block in blocks:
            block_idx = int(block["block_idx"])
            train_end = int(block["train_end"])
            train_specs = {
                "b1": int(b1_train_days),
                "b2": int(b2_train_days),
            }
            train_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[int, pd.DataFrame]]] = {}
            for branch, train_days in train_specs.items():
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
                    raise ValueError(
                        f"Selected {branch} training block has insufficient classes at {origin_bin}."
                    )
                if branch == "b2":
                    train_df = self.engine.extract_branch2_features(
                        train_bins, train_gpus, self.history_map
                    )
                    train_cache[branch] = (train_bins, train_gpus, train_labels, train_df, {})
                else:
                    train_df = {
                        obs: self.engine.extract_branch1_features(
                            train_bins, train_gpus, obs
                        )[0]
                        for obs in OBS_HOURS_GRID
                    }
                    train_cache[branch] = (train_bins, train_gpus, train_labels, pd.DataFrame(), train_df)

            val_bins = block["val_bins"]
            val_gpus = block["val_gpus"]
            val_b2_df = self.engine.extract_branch2_features(
                val_bins, val_gpus, self.history_map
            )
            val_b1_df = {
                obs: self.engine.extract_branch1_features(val_bins, val_gpus, obs)[0]
                for obs in OBS_HOURS_GRID
            }
            b1_preds: dict[int, np.ndarray] = {}
            b1_bins, _, b1_labels, _, b1_train_df = train_cache["b1"]
            for obs in OBS_HOURS_GRID:
                model = self._fit_selector(
                    "b1",
                    b1_train_df[obs],
                    b1_labels,
                    b1_bins,
                    train_end,
                    b1_half_life,
                    self.seed + origin_bin + block_idx + b1_train_days + b1_half_life + obs,
                )
                b1_preds[obs] = model.predict_proba(val_b1_df[obs])[:, 1]
            b2_bins, _, b2_labels, b2_train_df, _ = train_cache["b2"]
            b2_model = self._fit_selector(
                "b2",
                b2_train_df,
                b2_labels,
                b2_bins,
                train_end,
                b2_half_life,
                self.seed + origin_bin + block_idx + b2_train_days + b2_half_life,
            )
            block["b1_preds"] = b1_preds
            block["b2_pred"] = b2_model.predict_proba(val_b2_df)[:, 1]
        return {"origin_bin": int(origin_bin), "blocks": blocks}

    def _branch_rows(self, context: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        b1_rows: list[dict[str, Any]] = []
        b2_rows: list[dict[str, Any]] = []
        blocks = context["blocks"]
        for train_days in TRAIN_DAYS_GRID:
            for half_life in HALF_LIFE_GRID:
                b1_values: list[float] = []
                b1_rocs: list[float] = []
                input_values: dict[str, list[float]] = {str(obs): [] for obs in OBS_HOURS_GRID}
                valid_b1 = True
                for block in blocks:
                    labels = block["val_labels"]
                    b1_preds = []
                    for obs in OBS_HOURS_GRID:
                        pred = block.get("pred_b1", {}).get((train_days, half_life, obs))
                        if pred is None:
                            valid_b1 = False
                            break
                        b1_preds.append(pred)
                        input_values[str(obs)].append(
                            base.safe_average_precision(labels, pred)
                        )
                    if valid_b1:
                        equal_pred = np.mean(b1_preds, axis=0)
                        b1_values.append(base.safe_average_precision(labels, equal_pred))
                        b1_rocs.append(base.safe_roc_auc(labels, equal_pred))
                if valid_b1 and len(b1_values) == len(blocks):
                    s = _stats(b1_values)
                    b1_rows.append(
                        {
                            "selection_role": "branch1",
                            "candidate_b1_train_days": int(train_days),
                            "candidate_b1_half_life_days": int(half_life),
                            "candidate_L_obs_hours": "all_1_6_24",
                            "val_pr_auc_mean": s["mean"],
                            "val_pr_auc_median": s["median"],
                            "val_pr_auc_q25": s["q25"],
                            "val_pr_auc_std": s["std"],
                            "val_pr_auc_min": s["min"],
                            "val_roc_auc_mean": float(np.mean(b1_rocs)),
                            "branch1_input_validation_ap": {
                                key: float(np.mean(values)) if values else float("nan")
                                for key, values in input_values.items()
                            },
                        }
                    )
            for half_life in self._b2_half_life_grid():
                b2_values: list[float] = []
                b2_rocs: list[float] = []
                valid_b2 = True
                for block in blocks:
                    labels = block["val_labels"]
                    b2_pred = block.get("pred_b2", {}).get((train_days, half_life))
                    if b2_pred is None:
                        valid_b2 = False
                        break
                    b2_values.append(base.safe_average_precision(labels, b2_pred))
                    b2_rocs.append(base.safe_roc_auc(labels, b2_pred))
                if valid_b2 and len(b2_values) == len(blocks):
                    s = _stats(b2_values)
                    b2_rows.append(
                        {
                            "selection_role": "branch2",
                            "candidate_b2_train_days": int(train_days),
                            "candidate_b2_half_life_days": int(half_life),
                            "candidate_L_obs_hours": "history_only",
                            "val_pr_auc_mean": s["mean"],
                            "val_pr_auc_median": s["median"],
                            "val_pr_auc_q25": s["q25"],
                            "val_pr_auc_std": s["std"],
                            "val_pr_auc_min": s["min"],
                            "val_roc_auc_mean": float(np.mean(b2_rocs)),
                        }
                    )
        if not b1_rows or not b2_rows:
            raise ValueError(f"No valid branch candidates at {context['origin_bin']}.")
        return b1_rows, b2_rows

    def _pool_branch_rows(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        def pool(role: str, keys: tuple[str, ...]) -> list[dict[str, Any]]:
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for row in rows:
                if row.get("selection_role") != role:
                    continue
                grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
            result: list[dict[str, Any]] = []
            for key, group in grouped.items():
                means = [float(item["val_pr_auc_mean"]) for item in group]
                s = _stats(means)
                row = {name: value for name, value in zip(keys, key)}
                row.update(
                    {
                        "selection_role": role,
                        "val_pr_auc_mean": s["mean"],
                        "val_pr_auc_median": s["median"],
                        "val_pr_auc_q25": s["q25"],
                        "val_pr_auc_std": s["std"],
                        "val_pr_auc_min": s["min"],
                        "val_roc_auc_mean": float(np.mean([item["val_roc_auc_mean"] for item in group])),
                        "pool_origin_count": len(group),
                    }
                )
                if role == "branch1":
                    input_values: dict[str, list[float]] = {str(obs): [] for obs in OBS_HOURS_GRID}
                    for item in group:
                        for obs in OBS_HOURS_GRID:
                            value = item.get("branch1_input_validation_ap", {}).get(str(obs))
                            if value is not None and np.isfinite(float(value)):
                                input_values[str(obs)].append(float(value))
                    row["branch1_input_validation_ap"] = {
                        obs: float(np.mean(values)) if values else float("nan")
                        for obs, values in input_values.items()
                    }
                    row["candidate_L_obs_hours"] = "all_1_6_24"
                else:
                    row["candidate_L_obs_hours"] = "history_only"
                result.append(row)
            return result

        return (
            pool("branch1", ("candidate_b1_train_days", "candidate_b1_half_life_days")),
            pool("branch2", ("candidate_b2_train_days", "candidate_b2_half_life_days")),
        )

    def _choose_branch(
        self,
        rows: list[dict[str, Any]],
        previous: dict[str, Any] | None,
        train_key: str,
        half_key: str,
    ) -> dict[str, Any]:
        drift = bool(previous and previous.get("adwin_triggered", False))
        pool = self._apply_adwin_policy(rows, previous, drift, train_key, half_key)
        return max(
            pool,
            key=lambda row: self._candidate_key(row, drift, train_key, half_key),
        ).copy()

    @staticmethod
    def _advance_detector(
        detector: v2.ADWINLite,
        previous_state: dict[str, Any] | None,
        value: float,
    ) -> tuple[bool, bool, dict[str, Any]]:
        state = previous_state or {}
        cooldown = max(0, int(state.get("cooldown_remaining", 0)))
        if cooldown:
            cooldown -= 1
        raw = bool(detector.update(value))
        streak = int(state.get("trigger_streak", 0)) + 1 if raw else 0
        effective = bool(raw and streak >= ADWIN_CONSECUTIVE_TRIGGERS and cooldown == 0)
        if effective:
            cooldown = ADWIN_COOLDOWN_CYCLES
            streak = 0
        payload = detector.to_dict()
        payload.update(
            {
                "trigger_streak": streak,
                "cooldown_remaining": cooldown,
            }
        )
        return raw, effective, payload

    def _feedback_batches(
        self,
        contexts: list[dict[str, Any]],
        b1_weights: dict[str, float],
        selected_lambda: float,
        previous: dict[str, Any] | None,
    ) -> tuple[list[dict[str, float]], int]:
        last_bin = int(previous.get("adwin_last_batch_bin", -1)) if previous else -1
        batches: list[dict[str, float]] = []
        for context in contexts:
            for block in context["blocks"]:
                b1 = sum(
                    b1_weights[str(obs)] * block["b1_preds"][obs]
                    for obs in OBS_HOURS_GRID
                )
                b2 = block["b2_pred"]
                fused = selected_lambda * b1 + (1.0 - selected_lambda) * b2
                bins = np.asarray(block["val_bins"], dtype=np.int64)
                labels = np.asarray(block["val_labels"], dtype=np.float64)
                for batch_bin in np.unique(bins // FEEDBACK_BATCH_BINS):
                    batch_bin = int(batch_bin)
                    if batch_bin <= last_bin:
                        continue
                    mask = (bins // FEEDBACK_BATCH_BINS) == batch_bin
                    p = np.clip(fused[mask].astype(np.float64), 1e-6, 1.0 - 1e-6)
                    y = labels[mask]
                    log_loss = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
                    batches.append(
                        {
                            "batch_bin": float(batch_bin),
                            "loss": float(np.clip(log_loss, 0.0, 1.0)),
                            "disagreement": float(np.mean(np.abs(b1[mask] - b2[mask]))),
                            "prevalence": float(np.mean(y)),
                        }
                    )
        batches.sort(key=lambda row: row["batch_bin"])
        return batches, (int(batches[-1]["batch_bin"]) if batches else last_bin)

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
        b1_train = int(b1_best["candidate_b1_train_days"])
        b1_half = int(b1_best["candidate_b1_half_life_days"])
        b2_train = int(b2_best["candidate_b2_train_days"])
        b2_half = int(b2_best["candidate_b2_half_life_days"])
        input_aps = {str(obs): [] for obs in OBS_HOURS_GRID}
        block_refs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for context in contexts:
            for block in context["blocks"]:
                labels = block["val_labels"]
                for obs in OBS_HOURS_GRID:
                    input_aps[str(obs)].append(
                        base.safe_average_precision(labels, block["b1_preds"][obs])
                    )
                block_refs.append((context, block))
        input_mean = {key: float(np.mean(values)) for key, values in input_aps.items()}
        raw_weights = {key: 0.20 + max(value, 0.0) for key, value in input_mean.items()}
        total = sum(raw_weights.values())
        b1_weights = {key: float(value / total) for key, value in raw_weights.items()}

        b1_values: list[float] = []
        b2_values: list[float] = []
        b1_rocs: list[float] = []
        b2_rocs: list[float] = []
        b1_selected: list[np.ndarray] = []
        b2_selected: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        disagreement_values: list[float] = []
        for _, block in block_refs:
            labels = block["val_labels"]
            b1 = sum(
                b1_weights[str(obs)] * block["b1_preds"][obs]
                for obs in OBS_HOURS_GRID
            )
            b2 = block["b2_pred"]
            b1_selected.append(b1)
            b2_selected.append(b2)
            labels_list.append(labels)
            b1_values.append(base.safe_average_precision(labels, b1))
            b2_values.append(base.safe_average_precision(labels, b2))
            b1_rocs.append(base.safe_roc_auc(labels, b1))
            b2_rocs.append(base.safe_roc_auc(labels, b2))
            disagreement_values.append(float(np.mean(np.abs(b1 - b2))))

        fusion_rows: list[dict[str, Any]] = []
        for lambda_value in LAMBDA_GRID:
            aps: list[float] = []
            rocs: list[float] = []
            for labels, b1, b2 in zip(labels_list, b1_selected, b2_selected):
                fused = lambda_value * b1 + (1.0 - lambda_value) * b2
                aps.append(base.safe_average_precision(labels, fused))
                rocs.append(base.safe_roc_auc(labels, fused))
            s = _stats(aps)
            fusion_rows.append(
                {
                    "selection_role": "fusion",
                    "candidate_lambda": float(lambda_value),
                    "val_pr_auc_mean": s["mean"],
                    "val_pr_auc_median": s["median"],
                    "val_pr_auc_q25": s["q25"],
                    "val_pr_auc_std": s["std"],
                    "val_pr_auc_min": s["min"],
                    "val_roc_auc_mean": float(np.mean(rocs)),
                    "pool_origin_count": len(contexts),
                    "pool_validation_block_count": len(block_refs),
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

        detectors = {
            "loss": v2.ADWINLite.from_dict(previous.get("adwin_loss_state") if previous else None),
            "disagreement": v2.ADWINLite.from_dict(previous.get("adwin_disagreement_state") if previous else None),
            "prevalence": v2.ADWINLite.from_dict(previous.get("adwin_prevalence_state") if previous else None),
        }
        feedback, last_batch_bin = self._feedback_batches(
            contexts, b1_weights, selected_lambda, previous
        )
        states = {
            "loss": previous.get("adwin_loss_state") if previous else None,
            "disagreement": previous.get("adwin_disagreement_state") if previous else None,
            "prevalence": previous.get("adwin_prevalence_state") if previous else None,
        }
        raw_hits = {"loss": False, "disagreement": False, "prevalence": False}
        effective_hits = {"loss": False, "disagreement": False, "prevalence": False}
        for batch in feedback:
            for name, value in (
                ("loss", batch["loss"]),
                ("disagreement", batch["disagreement"]),
                ("prevalence", batch["prevalence"]),
            ):
                raw, effective, state = self._advance_detector(
                    detectors[name], states[name], value
                )
                raw_hits[name] = raw_hits[name] or raw
                effective_hits[name] = effective_hits[name] or effective
                states[name] = state
        adwin_triggered = bool(any(effective_hits.values()))
        previous_latest = None if previous is None else previous.get("latest_val_pr_auc")
        improving = previous_latest is not None and latest_ap >= float(previous_latest)
        stable_count = int(previous.get("adwin_stable_count", 0)) if previous else 0
        stable_count = stable_count + 1 if (not any(raw_hits.values()) and improving) else 0
        if previous is None:
            action = "initial"
        elif adwin_triggered:
            action = "adwin_shrink_recommendation"
        elif stable_count >= 3:
            action = "adwin_expand_recommendation"
        else:
            action = "hold"

        common = {
            "selection_stage": stage,
            "selection_origin_idx": int(origin_bin),
            "selection_origin_time": _time_text(self.engine, origin_bin),
            "purge_hours": int(base.PURGE_NS // HOUR_NS),
            "validation_mode": validation_mode,
            "pool_origin_count": len(contexts),
            "pool_validation_block_count": len(block_refs),
        }
        audit_rows: list[dict[str, Any]] = []
        for row in branch_rows + fusion_rows:
            audit_row = dict(common)
            audit_row.update(row)
            audit_rows.append(audit_row)

        selected = {
            **common,
            "train_end_time": _time_text(self.engine, latest_block["train_end"]),
            "validation_start_time": _time_text(self.engine, latest_context["blocks"][-1]["val_start"]),
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
            "prediction_disagreement": latest_disagreement,
            "validation_positive_rate": latest_prevalence,
            "latest_val_pr_auc": float(latest_ap),
            "adst_action": action,
            "adwin_triggered": adwin_triggered,
            "adwin_loss_triggered": bool(effective_hits["loss"]),
            "adwin_disagreement_triggered": bool(effective_hits["disagreement"]),
            "adwin_prevalence_triggered": bool(effective_hits["prevalence"]),
            "adwin_loss_raw_triggered": bool(raw_hits["loss"]),
            "adwin_disagreement_raw_triggered": bool(raw_hits["disagreement"]),
            "adwin_prevalence_raw_triggered": bool(raw_hits["prevalence"]),
            "adwin_stable": stable_count >= 3,
            "adwin_stable_count": stable_count,
            "adwin_loss_state": states["loss"] or detectors["loss"].to_dict(),
            "adwin_disagreement_state": states["disagreement"] or detectors["disagreement"].to_dict(),
            "adwin_prevalence_state": states["prevalence"] or detectors["prevalence"].to_dict(),
            "adwin_last_batch_bin": int(last_batch_bin),
            "adwin_feedback_batch_count": len(feedback),
            "adwin_min_batches": ADWIN_MIN_BATCHES,
            "adwin_consecutive_trigger_requirement": ADWIN_CONSECUTIVE_TRIGGERS,
            "adwin_cooldown_cycles": ADWIN_COOLDOWN_CYCLES,
            "previous_val_pr_auc": previous_latest,
            "val_pr_auc_delta": None if previous_latest is None else float(latest_ap - float(previous_latest)),
            "candidate_count": len(audit_rows),
            "branch1_input_validation_ap": input_mean,
        }
        return selected, audit_rows

    def select_adst_config(
        self,
        origin_bin: int,
        rng: np.random.Generator,
        previous: dict[str, Any] | None,
        stage: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        del rng
        if stage == "heldout_preselection" and self.pool_rows and self.pool_origin_bins:
            pooled_b1, pooled_b2 = self._pool_branch_rows(self.pool_rows)
            if not pooled_b1 or not pooled_b2:
                raise ValueError("Pooled development rows do not contain both branches.")
            b1_best = self._choose_branch(
                pooled_b1, previous, "candidate_b1_train_days", "candidate_b1_half_life_days"
            )
            b2_best = self._choose_branch(
                pooled_b2, previous, "candidate_b2_train_days", "candidate_b2_half_life_days"
            )
            contexts = [
                self._score_selected_origin(
                    int(pool_origin),
                    int(b1_best["candidate_b1_train_days"]),
                    int(b1_best["candidate_b1_half_life_days"]),
                    int(b2_best["candidate_b2_train_days"]),
                    int(b2_best["candidate_b2_half_life_days"]),
                )
                for pool_origin in self.pool_origin_bins
            ]
            branch_rows = pooled_b1 + pooled_b2
            return self._compose_selection(
                contexts,
                b1_best,
                b2_best,
                branch_rows,
                previous,
                stage,
                origin_bin,
                "pooled_latest_6_development_origins",
            )

        context = self._evaluate_origin_full(origin_bin)
        b1_rows, b2_rows = self._branch_rows(context)
        b1_best = self._choose_branch(
            b1_rows, previous, "candidate_b1_train_days", "candidate_b1_half_life_days"
        )
        b2_best = self._choose_branch(
            b2_rows, previous, "candidate_b2_train_days", "candidate_b2_half_life_days"
        )
        selected, audit_rows = self._compose_selection(
            [{
                "origin_bin": int(origin_bin),
                "blocks": [
                    {
                        **block,
                        "b1_preds": {
                            obs: block["pred_b1"][(
                                int(b1_best["candidate_b1_train_days"]),
                                int(b1_best["candidate_b1_half_life_days"]),
                                obs,
                            )]
                            for obs in OBS_HOURS_GRID
                        },
                        "b2_pred": block["pred_b2"][(
                            int(b2_best["candidate_b2_train_days"]),
                            int(b2_best["candidate_b2_half_life_days"]),
                        )],
                    }
                    for block in context["blocks"]
                ],
            }],
            b1_best,
            b2_best,
            b1_rows + b2_rows,
            previous,
            stage,
            origin_bin,
            "local_3_rolling_validation_blocks",
        )
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
                np.average(metrics_df[column].to_numpy(dtype=float), weights=np.maximum(weights, 1.0))
            )

        lines = [
            "# [ADST v2.1] All-XID Pooled Validation + Batch ADWIN 보고서",
            "",
            "## Material Passport",
            "- **Status**: COMPLETED",
            f"- **Runner**: `{Path(__file__).name}`",
            "- **Target**: All-XID unified onset, 24-hour horizon",
            "- **Branch 1**: 1h·6h·24h telemetry parallel ensemble",
            f"- **Branch 2**: 2026-09-10 History-only Historical Logistic (xid_count_30d, days_since_xid, no recency)" if self.branch2_mode == "history_0910" else "- **Branch 2**: strict History-only (xid_count_30d, days_since_xid)",
            "- **Split**: chronological Sliding Training, 36-hour purge, fixed terminal held-out test",
            "",
            "## Terminal held-out mean",
            f"- Fused PR-AUC: `{metrics_df['pr_auc_fused'].mean():.6f}`",
            f"- Branch 1 PR-AUC: `{metrics_df['pr_auc_b1'].mean():.6f}`",
            f"- Branch 2 PR-AUC: `{metrics_df['pr_auc_b2'].mean():.6f}`",
            f"- Fused normalized PR-AUC: `{metrics_df['pr_auc_fused_normalized'].mean():.6f}`",
            f"- B2 positive-weighted PR-AUC: `{weighted('pr_auc_b2'):.6f}`",
            f"- Fused ROC-AUC: `{metrics_df['roc_auc_fused'].mean():.6f}`",
            f"- Recall@100: `{metrics_df['recall_at_100'].mean():.2%}`",
            f"- Lift@100: `{metrics_df['lift_at_100'].mean():.3f}x`",
            "",
            "## Selector and ADWIN",
            "- Final branch settings are selected from pooled validation feedback over the latest six development origins.",
            "- Lambda is selected only after the pooled selected Branch 1 and Branch 2 predictions are fixed.",
            "- ADWIN input is label-matured 30-minute validation loss, prediction disagreement, and sampled prevalence.",
            f"- Effective ADWIN trigger cycles: `{int(metrics_df['adwin_triggered'].sum())}`",
            f"- Feedback batches recorded in final state: `{int(selection_df.tail(1)['adwin_feedback_batch_count'].iloc[0]) if 'adwin_feedback_batch_count' in selection_df and not selection_df.empty else 0}`",
            "- Terminal held-out labels are used only for final reporting, never for selection or drift state.",
            "",
            "## Scope and limitations",
            "- v2.1 is a controlled methodology experiment; improvement is judged on the same held-out population as v1/v2.",
            "- Pooled validation reduces dependence on one recent validation slice but may respond more slowly to abrupt drift.",
            f"- Held-out interval: `{split_info['test_start_time']}` ~ `{split_info['test_end_time']}`",
        ]
        (self.output_dir / "adst_v2_1_report.md").write_text("\n".join(lines), encoding="utf-8")

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

        print("[1/5] All-XID ADST v2.1 pooled development selection", flush=True)
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
                f"B2={best['selected_b2_train_days']}d/{best['selected_b2_half_life_days']}d, "
                f"lambda={best['selected_lambda']:.1f}, val PR-AUC={float(best['val_pr_auc']):.4f}, "
                f"batches={best.get('adwin_feedback_batch_count', 0)}, action={best['adst_action']}",
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
        selection_df.to_csv(self.output_dir / "adst_v2_1_selection_history.csv", index=False)

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
                "reference_runner": "ML/run_sliding_adst_v2_branch_specific.py",
                "target": "all_xids",
                "branch2_contract": "history_only_2026_09_10" if self.branch2_mode == "history_0910" else "history_only",
                "branch2_features": ["xid_count_30d", "days_since_xid"],
                "branch2_model": "historical_logistic" if self.branch2_mode == "history_0910" else "historical_logistic_plus_gbdt",
                "branch2_recency": "none" if self.branch2_mode == "history_0910" else "validation_selected_half_life",
                "branch1_input_hours": list(OBS_HOURS_GRID),
                "branch1_parallel_ensemble": True,
                "recency_half_life_days": list(HALF_LIFE_GRID),
                "branch2_half_life_days": list(self._b2_half_life_grid()),
                "validation_mode": "pooled_latest_6_development_origins",
                "pooled_origin_count": POOL_ORIGINS,
                "validation_blocks_per_origin": VALIDATION_BLOCKS,
                "adwin_feedback_batch_minutes": 30,
                "adwin_min_batches": ADWIN_MIN_BATCHES,
                "adwin_consecutive_trigger_requirement": ADWIN_CONSECUTIVE_TRIGGERS,
                "adwin_cooldown_cycles": ADWIN_COOLDOWN_CYCLES,
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
                "pool_origin_times": [_time_text(self.engine, value) for value in self.pool_origin_bins],
                "final_selection": final_selected,
                "command": " ".join(sys.argv),
            },
        )

        print("[2/5] Fitting final branch-specific models", flush=True)
        models = self._fit_final_models(final_selected, test_start, model_seed=900_001)
        print(
            f"  B1 L={final_selected['selected_b1_train_days']}d/{final_selected['selected_b1_half_life_days']}d, "
            f"B2 L={final_selected['selected_b2_train_days']}d/{final_selected['selected_b2_half_life_days']}d, "
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
        metrics_df.to_csv(self.output_dir / "adst_v2_1_metrics.csv", index=False)
        tape_df.to_parquet(
            self.output_dir / "adst_v2_1_risk_tape.parquet",
            index=False,
            compression="zstd",
        )
        self._generate_summary_report(metrics_df, selection_df, split_info)
        print(f"[4/5] Saved outputs in {self.output_dir}", flush=True)
        print(f"[5/5] Elapsed: {(time.time() - started) / 60.0:.1f} minutes", flush=True)
        return metrics_df, tape_df


def main() -> None:
    parser = argparse.ArgumentParser(description="All-XID ADST v2.1 pooled validation and batch ADWIN")
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/2026-09-14_All-XID_ADST_v2_1_PooledValidation_BatchADWIN",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--branch2-mode",
        choices=("current", "history_0910"),
        default="current",
        help="Branch 2 protocol; history_0910 restores the 2026-09-10 History-only Logistic path.",
    )
    args = parser.parse_args()

    data_dir = base.find_data_dir()
    cache_dir = base.PROJECT_ROOT / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    output_dir = base.PROJECT_ROOT / args.output_dir
    engine = base.UnifiedDataEngine(data_dir=data_dir, cache_dir=cache_dir)
    _, history_map, gt_matrix = engine.load_all_xid_ledger()
    pipeline = PooledBatchADST(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        branch2_mode=args.branch2_mode,
        resume=args.resume,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

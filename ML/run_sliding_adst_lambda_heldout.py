"""
[All-XID Sliding ADST + Validation Lambda Fusion]

Controlled experiment variant of the GitHub runner.  It preserves the
GitHub data engine, label construction, Branch 1 input and History-only
Branch 2 input, while adding purged Sliding Training, terminal held-out
evaluation, validation-selected Lambda fusion, and resumable checkpoints.
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


STEP_NS = base.STEP_NS
DAY_NS = base.DAY_NS
HOUR_NS = base.HOUR_NS
PURGE_BINS = max(1, int(base.PURGE_NS // STEP_NS))
VALIDATION_BINS = int(3 * DAY_NS // STEP_NS)
TEST_FRACTION = 0.20
LAMBDA_GRID = base.LAMBDA_GRID


def _time_text(engine: base.UnifiedDataEngine, bin_index: int) -> str:
    return pd.Timestamp(engine.bin_start_ns[int(bin_index)], unit="ns", tz="UTC").isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temp_path.replace(path)


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


class SlidingADSTLambdaHeldout:
    def __init__(
        self,
        engine: base.UnifiedDataEngine,
        history_map: dict[int, np.ndarray],
        gt_matrix: np.ndarray,
        output_dir: Path,
        retrain_cadence_hours: int = 24,
        negative_ratio: int = 10,
        test_stride_bins: int = 6,
        seed: int = 20260905,
        resume: bool = False,
    ) -> None:
        self.engine = engine
        self.history_map = history_map
        self.gt_matrix = gt_matrix
        self.output_dir = output_dir
        self.retrain_cadence_hours = int(retrain_cadence_hours)
        self.negative_ratio = int(negative_ratio)
        self.test_stride_bins = max(1, int(test_stride_bins))
        self.seed = int(seed)
        self.resume = bool(resume)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sample_indices(
        self,
        start_bin: int,
        end_bin: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reuse the GitHub sampler contract for positive/negative points."""
        gt_sub = self.gt_matrix[start_bin:end_bin]
        pos_rel_bins, pos_gpus = np.where(gt_sub)
        pos_bins = pos_rel_bins + start_bin
        pos_flat = pos_bins.astype(np.int64) * self.engine.num_gpus + pos_gpus
        n_pos = len(pos_flat)
        target_neg = min(n_pos * self.negative_ratio, 200_000)
        pos_set = set(pos_flat.tolist())
        negatives: set[int] = set()
        while len(negatives) < target_neg:
            batch_size = max(5000, 2 * (target_neg - len(negatives)))
            sample_bins = rng.integers(start_bin, end_bin, size=batch_size, dtype=np.int32)
            sample_gpus = rng.integers(0, self.engine.num_gpus, size=batch_size, dtype=np.int32)
            flats = sample_bins.astype(np.int64) * self.engine.num_gpus + sample_gpus
            for flat in flats:
                flat_int = int(flat)
                if flat_int not in pos_set and flat_int not in negatives:
                    negatives.add(flat_int)
                if len(negatives) >= target_neg:
                    break
        all_flat = np.r_[pos_flat, np.fromiter(negatives, dtype=np.int64)]
        labels = np.r_[
            np.ones(n_pos, dtype=np.uint8),
            np.zeros(len(negatives), dtype=np.uint8),
        ]
        all_bins = (all_flat // self.engine.num_gpus).astype(np.int32)
        all_gpus = (all_flat % self.engine.num_gpus).astype(np.int32)
        order = np.lexsort((all_gpus, all_bins))
        return all_bins[order], all_gpus[order], labels[order]

    def split_bounds(self) -> tuple[int, int, int]:
        warmup_bins = int(30 * DAY_NS // STEP_NS)
        usable_bins = self.engine.num_bins - warmup_bins
        test_start = warmup_bins + int(np.floor(usable_bins * (1.0 - TEST_FRACTION)))
        test_start = min(max(test_start, warmup_bins + 1), self.engine.num_bins - 1)
        return warmup_bins, test_start, self.engine.num_bins

    def _selection_bounds(self, origin_bin: int) -> tuple[int, int, int]:
        val_end = int(origin_bin - PURGE_BINS)
        val_start = int(val_end - VALIDATION_BINS)
        train_end = int(val_start - PURGE_BINS)
        return train_end, val_start, val_end

    def select_adst_config(
        self,
        origin_bin: int,
        rng: np.random.Generator,
        previous: dict[str, Any] | None,
        stage: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Select L_train, L_obs and lambda using only a purged validation block."""
        train_end, val_start, val_end = self._selection_bounds(origin_bin)
        if train_end <= 0 or val_start <= 0 or val_end <= val_start:
            raise ValueError(f"Insufficient purged history at origin {origin_bin}.")
        val_bins, val_gpus, val_labels = self.sample_indices(val_start, val_end, rng)
        if len(val_labels) == 0 or np.unique(val_labels).size < 2:
            raise ValueError(f"Validation block at origin {origin_bin} has insufficient classes.")
        val_b2_df = self.engine.extract_branch2_features(val_bins, val_gpus, self.history_map)
        candidates: list[dict[str, Any]] = []

        for train_days in base.CANDIDATE_L_TRAIN_DAYS:
            train_start = max(0, train_end - int(train_days * DAY_NS // STEP_NS))
            train_bins, train_gpus, train_labels = self.sample_indices(train_start, train_end, rng)
            if len(train_labels) == 0 or np.unique(train_labels).size < 2:
                continue
            train_b2_df = self.engine.extract_branch2_features(
                train_bins, train_gpus, self.history_map
            )
            b2_selector = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=200, class_weight="balanced", random_state=self.seed),
            )
            b2_selector.fit(train_b2_df, train_labels)
            val_b2_probs = b2_selector.predict_proba(val_b2_df)[:, 1]
            b2_ap = base.safe_average_precision(val_labels, val_b2_probs)

            for obs_hours in base.CANDIDATE_L_OBS_HOURS:
                train_b1_df, _ = self.engine.extract_branch1_features(
                    train_bins, train_gpus, obs_hours
                )
                val_b1_df, _ = self.engine.extract_branch1_features(
                    val_bins, val_gpus, obs_hours
                )
                b1_selector = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=200, class_weight="balanced", random_state=self.seed),
                )
                b1_selector.fit(train_b1_df, train_labels)
                val_b1_probs = b1_selector.predict_proba(val_b1_df)[:, 1]
                b1_ap = base.safe_average_precision(val_labels, val_b1_probs)
                disagreement = float(np.mean(np.abs(val_b1_probs - val_b2_probs)))
                prevalence = float(np.mean(val_labels))

                for lambda_value in LAMBDA_GRID:
                    fused = lambda_value * val_b1_probs + (1.0 - lambda_value) * val_b2_probs
                    candidates.append({
                        "selection_stage": stage,
                        "selection_origin_idx": int(origin_bin),
                        "selection_origin_time": _time_text(self.engine, origin_bin),
                        "train_start_time": _time_text(self.engine, train_start),
                        "train_end_time": _time_text(self.engine, train_end),
                        "validation_start_time": _time_text(self.engine, val_start),
                        "validation_end_time": _time_text(self.engine, val_end),
                        "purge_hours": int(base.PURGE_NS // HOUR_NS),
                        "candidate_L_train_days": int(train_days),
                        "candidate_L_obs_hours": int(obs_hours),
                        "candidate_lambda": float(lambda_value),
                        "val_pr_auc": base.safe_average_precision(val_labels, fused),
                        "val_roc_auc": base.safe_roc_auc(val_labels, fused),
                        "val_pr_auc_b1": b1_ap,
                        "val_pr_auc_b2": b2_ap,
                        "prediction_disagreement": disagreement,
                        "validation_positive_rate": prevalence,
                    })

        if not candidates:
            raise ValueError(f"No valid ADST candidates at origin {origin_bin}.")
        best = max(
            candidates,
            key=lambda row: (
                float(row["val_pr_auc"]),
                -abs(float(row["candidate_lambda"]) - 0.5),
                -int(row["candidate_L_train_days"]),
            ),
        ).copy()
        best["selected_L_train_days"] = int(best.pop("candidate_L_train_days"))
        best["selected_L_obs_hours"] = int(best.pop("candidate_L_obs_hours"))
        best["selected_lambda"] = float(best.pop("candidate_lambda"))
        previous_score = None if previous is None else float(previous["val_pr_auc"])
        previous_days = None if previous is None else int(previous["selected_L_train_days"])
        current_days = int(best["selected_L_train_days"])
        if previous is None:
            action = "initial"
        elif current_days < previous_days:
            action = "shrink_train_window"
        elif current_days > previous_days:
            action = "expand_train_window"
        elif int(best["selected_L_obs_hours"]) != int(previous["selected_L_obs_hours"]):
            action = "change_input_window"
        else:
            action = "hold"
        best["adst_action"] = action
        best["previous_val_pr_auc"] = previous_score
        best["val_pr_auc_delta"] = None if previous_score is None else float(best["val_pr_auc"] - previous_score)
        best["candidate_count"] = len(candidates)
        return best, candidates

    def _selection_checkpoint_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"selection_{name}.json"

    def _load_selection_checkpoint(self, name: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        path = self._selection_checkpoint_path(name)
        if not (self.resume and path.exists()):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["best"], payload["candidates"]

    def _save_selection_checkpoint(
        self,
        name: str,
        best: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> None:
        _write_json_atomic(
            self._selection_checkpoint_path(name),
            {"best": best, "candidates": candidates},
        )

    def _fit_final_models(
        self,
        selected: dict[str, Any],
        test_start: int,
        model_seed: int,
    ) -> dict[str, Any]:
        train_end = int(test_start - PURGE_BINS)
        train_start = max(
            0,
            train_end - int(int(selected["selected_L_train_days"]) * DAY_NS // STEP_NS),
        )
        train_bins, train_gpus, train_labels = self.sample_indices(
            train_start,
            train_end,
            np.random.default_rng(self.seed + model_seed),
        )
        if len(train_labels) == 0 or np.unique(train_labels).size < 2:
            raise ValueError("Final training block has insufficient classes.")
        base.seed_everything(self.seed + model_seed)
        obs_hours = int(selected["selected_L_obs_hours"])
        b1_train_df, b1_train_tensor = self.engine.extract_branch1_features(
            train_bins, train_gpus, obs_hours
        )
        b2_train_df = self.engine.extract_branch2_features(
            train_bins, train_gpus, self.history_map
        )

        b1_tree = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.seed,
        )
        b1_tree.fit(b1_train_df, train_labels)

        b1_cnn = base.TemporalCNN1D(in_channels=7, hidden_channels=32, dropout=0.2)
        optimizer = torch.optim.AdamW(b1_cnn.parameters(), lr=0.005, weight_decay=1e-4)
        positive_weight = max(
            1.0,
            (train_labels == 0).sum() / max(1, (train_labels == 1).sum()),
        )
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
        )
        x_train = torch.tensor(b1_train_tensor, dtype=torch.float32)
        y_train = torch.tensor(train_labels, dtype=torch.float32)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.seed + model_seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=4096,
            shuffle=True,
            generator=loader_generator,
        )
        b1_cnn.train()
        for _ in range(5):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = b1_cnn(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        b1_cnn.eval()

        b2_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=250, class_weight="balanced", random_state=self.seed),
        )
        b2_lr.fit(b2_train_df, train_labels)
        b2_gbdt = HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=self.seed,
        )
        b2_gbdt.fit(b2_train_df, train_labels)
        true_positive = int(self.gt_matrix[train_start:train_end].sum())
        true_total = (train_end - train_start) * self.engine.num_gpus
        return {
            "b1_tree": b1_tree,
            "b1_cnn": b1_cnn,
            "b2_lr": b2_lr,
            "b2_gbdt": b2_gbdt,
            "train_start": train_start,
            "train_end": train_end,
            "true_prior": true_positive / max(true_total, 1),
            "sample_prior": int(train_labels.sum()) / max(len(train_labels), 1),
        }

    def _cycle_checkpoint_paths(self, cycle_idx: int) -> tuple[Path, Path]:
        stem = f"cycle_{cycle_idx:04d}"
        return (
            self.checkpoint_dir / f"{stem}_metrics.json",
            self.checkpoint_dir / f"{stem}_risk_tape.parquet",
        )

    def _load_cycle_checkpoint(self, cycle_idx: int) -> tuple[dict[str, Any], pd.DataFrame] | None:
        metrics_path, tape_path = self._cycle_checkpoint_paths(cycle_idx)
        if not (self.resume and metrics_path.exists() and tape_path.exists()):
            return None
        return json.loads(metrics_path.read_text(encoding="utf-8")), pd.read_parquet(tape_path)

    def _save_cycle_checkpoint(
        self,
        cycle_idx: int,
        metrics: dict[str, Any],
        tape: pd.DataFrame,
    ) -> None:
        metrics_path, tape_path = self._cycle_checkpoint_paths(cycle_idx)
        temp_tape = tape_path.with_name(tape_path.name + ".tmp")
        tape.to_parquet(temp_tape, index=False, compression="zstd")
        temp_tape.replace(tape_path)
        _write_json_atomic(metrics_path, metrics)

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
        obs_hours = int(selected["selected_L_obs_hours"])
        lambda_value = float(selected["selected_lambda"])

        for rel_idx, current_bin in enumerate(test_bins):
            all_gpus = np.arange(n_gpu, dtype=np.int32)
            current_bins = np.full(n_gpu, int(current_bin), dtype=np.int32)
            b1_df, b1_tensor = self.engine.extract_branch1_features(
                current_bins, all_gpus, obs_hours
            )
            b2_df = self.engine.extract_branch2_features(
                current_bins, all_gpus, self.history_map
            )
            p_b1_tree = models["b1_tree"].predict_proba(b1_df)[:, 1]
            with torch.no_grad():
                p_b1_cnn = torch.sigmoid(
                    models["b1_cnn"](torch.tensor(b1_tensor, dtype=torch.float32))
                ).cpu().numpy()
            p_b1 = 0.5 * p_b1_tree + 0.5 * p_b1_cnn
            p_b2_lr = models["b2_lr"].predict_proba(b2_df)[:, 1]
            p_b2_gbdt = models["b2_gbdt"].predict_proba(b2_df)[:, 1]
            p_b2 = 0.5 * p_b2_lr + 0.5 * p_b2_gbdt
            cal_b1 = base.adjusted_probability(
                p_b1, models["true_prior"], models["sample_prior"]
            )
            cal_b2 = base.adjusted_probability(
                p_b2, models["true_prior"], models["sample_prior"]
            )
            b1_scores[rel_idx] = cal_b1
            b2_scores[rel_idx] = cal_b2
            fused_scores[rel_idx] = lambda_value * cal_b1 + (1.0 - lambda_value) * cal_b2

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
        hits = int((gt_test & (ranks <= top_k)).sum())
        prevalence = float(gt_test.mean())
        metrics = {
            "origin_idx": int(cycle_idx),
            "model_origin_time": _time_text(self.engine, cycle_start),
            "test_start_time": _time_text(self.engine, cycle_start),
            "test_end_time": _time_text(self.engine, cycle_end - 1),
            "evaluation_scope": "common_terminal_heldout",
            "selected_L_train_days": int(selected["selected_L_train_days"]),
            "selected_L_obs_hours": int(selected["selected_L_obs_hours"]),
            "selected_lambda": lambda_value,
            "pr_auc_fused": base.safe_average_precision(y_test, fused_scores.ravel()),
            "pr_auc_b1": base.safe_average_precision(y_test, b1_scores.ravel()),
            "pr_auc_b2": base.safe_average_precision(y_test, b2_scores.ravel()),
            "roc_auc_fused": base.safe_roc_auc(y_test, fused_scores.ravel()),
            "recall_at_100": hits / max(positives, 1),
            "lift_at_100": (hits / (len(test_bins) * top_k)) / max(prevalence, 1e-12),
            "positives": positives,
            "test_prevalence": prevalence,
        }
        top_mask = ranks <= top_k
        rel_idx, gpu_idx = np.where(top_mask)
        tape = pd.DataFrame({
            "decision_time": pd.to_datetime(self.engine.bin_start_ns[test_bins][rel_idx], unit="ns", utc=True),
            "gpu_id": self.engine.gpu_ids[gpu_idx],
            "fused_risk": fused_scores[rel_idx, gpu_idx],
            "b1_risk": b1_scores[rel_idx, gpu_idx],
            "b2_risk": b2_scores[rel_idx, gpu_idx],
            "risk_rank": ranks[rel_idx, gpu_idx],
            "L_train_days": int(selected["selected_L_train_days"]),
            "L_obs_hours": int(selected["selected_L_obs_hours"]),
            "lambda_fusion": lambda_value,
            "target_24h": gt_test[rel_idx, gpu_idx].astype(np.uint8),
        })
        return metrics, tape

    def _generate_summary_report(
        self,
        metrics_df: pd.DataFrame,
        selection_df: pd.DataFrame,
        split_info: dict[str, Any],
    ) -> None:
        mean_fused = float(metrics_df["pr_auc_fused"].mean())
        mean_b1 = float(metrics_df["pr_auc_b1"].mean())
        mean_b2 = float(metrics_df["pr_auc_b2"].mean())
        mean_roc = float(metrics_df["roc_auc_fused"].mean())
        mean_recall = float(metrics_df["recall_at_100"].mean())
        mean_lift = float(metrics_df["lift_at_100"].mean())
        better_b1 = int((metrics_df["pr_auc_fused"] > metrics_df["pr_auc_b1"]).sum())
        better_b2 = int((metrics_df["pr_auc_fused"] > metrics_df["pr_auc_b2"]).sum())
        best_branch = metrics_df[["pr_auc_b1", "pr_auc_b2"]].max(axis=1)
        better_best = int((metrics_df["pr_auc_fused"] > best_branch).sum())
        final_rows = selection_df[
            (selection_df["selection_stage"] == "heldout_preselection")
            & selection_df["selected_lambda"].notna()
        ]
        if final_rows.empty:
            final_rows = selection_df.tail(1)
        lambda_dist = final_rows["selected_lambda"].value_counts().to_dict() if not final_rows.empty else {}
        window_dist = (
            final_rows[["selected_L_train_days", "selected_L_obs_hours"]].value_counts().to_dict()
            if not final_rows.empty else {}
        )
        report_lines = [
            "# [Sliding ADST + Lambda Fusion] All-XID 실험 보고서",
            "",
            "## 1. 실험 계약",
            "- **타깃**: All-XID 통합 onset, 향후 24시간 horizon",
            "- **Branch 1**: GitHub 이식본 telemetry 입력과 병렬 앙상블 유지",
            "- **Branch 2**: `xid_count_30d`, `days_since_xid`만 사용하는 History-only",
            "- **전처리**: 오류 직전 10분 buffer 제외",
            f"- **Purge**: Train–Validation 및 Validation–held-out Test 사이 {int(base.PURGE_NS // HOUR_NS)}시간",
            f"- **Held-out Test**: 시간순 terminal {TEST_FRACTION:.0%} 공통 구간 ({split_info['test_start_time']} 이후)",
            f"- **ADST 후보**: L_train={base.CANDIDATE_L_TRAIN_DAYS}일, L_obs={base.CANDIDATE_L_OBS_HOURS}시간",
            "- **Fusion**: `p = lambda * p_B1 + (1-lambda) * p_B2`, lambda를 Validation PR-AUC로 선택",
            "",
            "## 2. Held-out Test 평균 지표",
            f"- **Fused PR-AUC**: `{mean_fused:.6f}` (B1 `{mean_b1:.6f}`, B2 `{mean_b2:.6f}`)",
            f"- **Fused ROC-AUC**: `{mean_roc:.6f}`",
            f"- **Recall@100**: `{mean_recall:.2%}`",
            f"- **Lift@100**: `{mean_lift:.3f}x`",
            "",
            "## 3. Validation 기반 적응 선택 기록",
            f"- **최종 held-out preselection Lambda**: {lambda_dist}",
            f"- **최종 선택 window 조합**: {window_dist}",
            f"- **Fused > B1인 cycle**: {better_b1}/{len(metrics_df)}",
            f"- **Fused > B2인 cycle**: {better_b2}/{len(metrics_df)}",
            f"- **Fused > 두 Branch 중 최선인 cycle**: {better_best}/{len(metrics_df)}",
            "",
            "## 4. 해석 주의사항",
            "- 모든 cycle은 동일한 terminal held-out 구간 안에서 평가하며, Test 정답은 window·Lambda 선택에 사용하지 않는다.",
            "- Fused가 평균적으로 높더라도 모든 cycle에서 최선 Branch를 이긴다고 가정하지 않는다.",
            "- checkpoint에는 selection과 cycle별 metrics/risk tape가 저장되어 중단 후 완료 구간을 건너뛸 수 있다.",
        ]
        (self.output_dir / "bidirectional_adst_report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        started = time.time()
        base.seed_everything(self.seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        warmup_bins, test_start, test_end = self.split_bounds()
        cadence_bins = int(self.retrain_cadence_hours * 60 // base.STEP_MINUTES)
        min_origin = warmup_bins + int(21 * DAY_NS // STEP_NS) + 2 * PURGE_BINS + VALIDATION_BINS
        dev_origins = np.arange(min_origin, test_start, cadence_bins, dtype=np.int32)
        selection_rows: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None

        print("[1/5] All-XID Sliding ADST selection on development history", flush=True)
        print(
            f"  development origins={len(dev_origins):,}, terminal test fraction={TEST_FRACTION:.0%}, "
            f"purge={int(base.PURGE_NS // HOUR_NS)}h",
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
                f"-> L_train={best['selected_L_train_days']}d, L_obs={best['selected_L_obs_hours']}h, "
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
        selection_df.to_csv(self.output_dir / "bidirectional_adst_selection_history.csv", index=False)
        _write_json_atomic(
            self.output_dir / "experiment_manifest.json",
            {
                "runner": Path(__file__).name,
                "base_runner": "ML/run_bidirectional_adst_fusion.py",
                "target": "all_xids",
                "branch2_contract": "history_only",
                "retrain_cadence_hours": self.retrain_cadence_hours,
                "negative_ratio": self.negative_ratio,
                "test_stride_bins": self.test_stride_bins,
                "seed": self.seed,
                "purge_hours": int(base.PURGE_NS // HOUR_NS),
                "validation_days": VALIDATION_BINS * base.STEP_MINUTES / (24 * 60),
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

        print("[2/5] Fitting final models before purged terminal held-out test", flush=True)
        models = self._fit_final_models(final_selected, test_start, model_seed=900_001)
        print(
            f"  train={_time_text(self.engine, models['train_start'])}..{_time_text(self.engine, models['train_end'])}, "
            f"L_train={final_selected['selected_L_train_days']}d, "
            f"L_obs={final_selected['selected_L_obs_hours']}h, "
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
        metrics_df.to_csv(self.output_dir / "bidirectional_adst_metrics.csv", index=False)
        tape_df.to_parquet(
            self.output_dir / "bidirectional_adst_risk_tape.parquet",
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
        description="All-XID Sliding ADST with validation Lambda fusion and terminal held-out test"
    )
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/2026-09-13_All-XID_Sliding_ADST_Lambda_Heldout_Rerun_01",
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
    pipeline = SlidingADSTLambdaHeldout(
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

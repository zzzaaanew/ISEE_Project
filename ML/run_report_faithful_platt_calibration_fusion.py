"""Report-faithful Pareto Fusion with development-only Platt calibration.

This runner is intentionally separate from the existing report-faithful
Pareto/Momentum runner.  It preserves the current project contract and adds
one controlled change: B1 Cascade and B2 History-only scores are calibrated
with branch-specific Platt logistic calibrators fitted only on pooled,
time-ordered development validation predictions.  The common terminal
held-out test is never used to fit or select a calibrator.

The held-out cycle output is the calibrated result, while
``fusion_calibration_metrics.csv`` keeps an aligned raw-vs-Platt comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

import run_bidirectional_adst_fusion as base
import run_fusion_diversified_b1_history_b2 as integrated
from branch1_enhanced_features import EnhancedTelemetryEngine
from run_report_faithful_pareto_momentum_fusion import (
    REPORT_ALPHA,
    REPORT_W_CLEAN,
    REPORT_W_REP,
    ReportFaithfulParetoMomentumFusion,
    _variant_name,
    report_pareto_lambda,
)


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(len(labels), 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (scores >= left) & (scores <= right if right == 1.0 else scores < right)
        if not np.any(mask):
            continue
        error += float(mask.sum()) / total * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return float(error)


def _calibration_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return {
        "brier": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "ece_10": _expected_calibration_error(labels, scores, 10),
        "pr_auc": float(base.safe_average_precision(labels, scores)),
        "roc_auc": float(base.safe_roc_auc(labels, scores)),
        "prevalence": float(labels.mean()),
        "samples": int(len(labels)),
        "positives": int(labels.sum()),
    }


class _PlattCalibrator:
    """A small, serializable Platt scaling wrapper."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.model = LogisticRegression(
            C=1_000_000.0,
            solver="lbfgs",
            max_iter=2_000,
            random_state=self.seed,
        )
        self.fit_count = 0
        self.positive_count = 0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "_PlattCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.uint8)
        if np.unique(labels).size < 2:
            raise ValueError("Platt calibration requires both classes in development OOF data.")
        self.model.fit(_logit(scores).reshape(-1, 1), labels)
        self.fit_count = int(len(labels))
        self.positive_count = int(labels.sum())
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        flat = self.model.predict_proba(_logit(values).reshape(-1, 1))[:, 1]
        return flat.reshape(values.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "platt_logistic_sigmoid",
            "feature": "logit(raw_probability)",
            "C": 1_000_000.0,
            "solver": "lbfgs",
            "max_iter": 2_000,
            "seed": self.seed,
            "fit_count": self.fit_count,
            "positive_count": self.positive_count,
            "coef": self.model.coef_.ravel().tolist(),
            "intercept": self.model.intercept_.ravel().tolist(),
        }


class PlattReportFaithfulParetoFusion(ReportFaithfulParetoMomentumFusion):
    """Report-faithful Fusion with branch-specific development-only Platt scaling."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.b1_calibrator: _PlattCalibrator | None = None
        self.b2_calibrator: _PlattCalibrator | None = None
        self.calibration_rows: list[dict[str, Any]] = []
        self.calibration_manifest: dict[str, Any] = {}

    def _b2_adjusted_oof_scores(
        self,
        records: list[dict[str, Any]],
        selected_train_days: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the final B2 score scale used at test time.

        The parent validation records retain model probabilities before the
        sampling-prior correction.  Final B2 predictions apply that correction,
        so calibration must use the same adjusted scale to avoid a train/test
        score mismatch.
        """

        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for record in records:
            origin_bin = int(record["origin_bin"])
            block_idx = int(record["block_idx"])
            block = next(
                block
                for block in self._selection_blocks(origin_bin)
                if int(block["block_idx"]) == block_idx
            )
            train_end = int(block["train_end"])
            train_start = max(
                0,
                train_end
                - int(selected_train_days * integrated.DAY_NS // integrated.STEP_NS),
            )
            rng = np.random.default_rng(
                self.seed
                + origin_bin
                + 100_003 * (block_idx + 1)
                + 1_009 * int(selected_train_days)
            )
            train_bins, train_gpus, train_labels = self.sample_indices(
                train_start, train_end, rng
            )
            true_prior = float(self.gt_matrix[train_start:train_end].mean())
            sample_prior = float(train_labels.mean())
            raw = np.asarray(
                record["predictions"][int(selected_train_days)], dtype=np.float64
            )
            scores.append(base.adjusted_probability(raw, true_prior, sample_prior))
            labels.append(np.asarray(record["labels"], dtype=np.uint8))
        return np.concatenate(scores), np.concatenate(labels)

    def _fit_calibrators(
        self,
        b1_records: list[dict[str, Any]],
        b2_records: list[dict[str, Any]],
        selected_b2_train_days: int,
    ) -> None:
        b1_scores = np.concatenate(
            [np.asarray(record["cascade"], dtype=np.float64) for record in b1_records]
        )
        b1_labels = np.concatenate(
            [np.asarray(record["labels"], dtype=np.uint8) for record in b1_records]
        )
        b2_scores, b2_labels = self._b2_adjusted_oof_scores(
            b2_records, int(selected_b2_train_days)
        )

        self.b1_calibrator = _PlattCalibrator(self.seed + 1_100_001).fit(
            b1_scores, b1_labels
        )
        self.b2_calibrator = _PlattCalibrator(self.seed + 1_100_002).fit(
            b2_scores, b2_labels
        )
        b1_platt = self.b1_calibrator.predict(b1_scores)
        b2_platt = self.b2_calibrator.predict(b2_scores)
        self.calibration_manifest = {
            "method": "platt_logistic_sigmoid",
            "fit_scope": "pooled_development_validation_oof_only",
            "heldout_used_for_fit": False,
            "branch1_score": "cascade_probability",
            "branch2_score": "prior_adjusted_history_only_probability",
            "selected_b2_train_days": int(selected_b2_train_days),
            "branch1_raw": _calibration_summary(b1_labels, b1_scores),
            "branch1_platt": _calibration_summary(b1_labels, b1_platt),
            "branch2_raw": _calibration_summary(b2_labels, b2_scores),
            "branch2_platt": _calibration_summary(b2_labels, b2_platt),
            "branch1_calibrator": self.b1_calibrator.to_dict(),
            "branch2_calibrator": self.b2_calibrator.to_dict(),
        }
        (self.output_dir / "fusion_calibrator.json").write_text(
            json.dumps(self.calibration_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _select_lambda(
        self,
        b1_records: list[dict[str, Any]],
        b2_records: list[dict[str, Any]],
        selected_b2_train_days: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        result = super()._select_lambda(
            b1_records, b2_records, selected_b2_train_days
        )
        self._fit_calibrators(
            b1_records, b2_records, int(selected_b2_train_days)
        )
        return result

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
        if self.b1_calibrator is None or self.b2_calibrator is None:
            raise RuntimeError("Platt calibrators were not fitted before held-out prediction.")

        test_bins = np.arange(
            cycle_start, cycle_end, self.test_stride_bins, dtype=np.int32
        )
        if len(test_bins) == 0:
            raise ValueError(f"Empty held-out test cycle {cycle_idx}.")
        n_gpu = self.engine.num_gpus
        b1_scores = np.zeros((len(test_bins), n_gpu), dtype=np.float64)
        b2_scores = np.zeros_like(b1_scores)
        lambda_matrices = {
            spec["variant"]: np.zeros_like(b1_scores) for spec in self.variant_specs
        }
        raw_fused_matrices = {
            spec["variant"]: np.zeros_like(b1_scores) for spec in self.variant_specs
        }
        platt_fused_matrices = {
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
                raw_b2,
                float(b2_model["true_prior"]),
                float(b2_model["sample_prior"]),
            )
            b1_scores[rel_idx] = b1
            b2_scores[rel_idx] = b2
            counts = _pareto_counts(
                self.history_map,
                int(self.engine.bin_start_ns[current_bin]),
                n_gpu,
            )
            b1_platt = self.b1_calibrator.predict(b1)
            b2_platt = self.b2_calibrator.predict(b2)
            for spec in self.variant_specs:
                lambdas = report_pareto_lambda(
                    counts, spec["w_clean"], spec["w_rep"], spec["alpha"]
                )
                name = spec["variant"]
                lambda_matrices[name][rel_idx] = lambdas
                raw_fused_matrices[name][rel_idx] = lambdas * b1 + (1.0 - lambdas) * b2
                platt_fused_matrices[name][rel_idx] = (
                    lambdas * b1_platt + (1.0 - lambdas) * b2_platt
                )

        gt = self.gt_matrix[test_bins]
        primary = self.variant_specs[0]
        primary_name = primary["variant"]
        raw_metrics, _ = self._score_metrics(
            raw_fused_matrices[primary_name],
            b1_scores,
            b2_scores,
            gt,
            cycle_idx,
            cycle_start,
            cycle_end,
            selected,
            cascade_meta,
            primary,
            lambda_matrices[primary_name],
        )
        b1_platt_matrix = self.b1_calibrator.predict(b1_scores)
        b2_platt_matrix = self.b2_calibrator.predict(b2_scores)
        platt_metrics, platt_tape = self._score_metrics(
            platt_fused_matrices[primary_name],
            b1_platt_matrix,
            b2_platt_matrix,
            gt,
            cycle_idx,
            cycle_start,
            cycle_end,
            selected,
            cascade_meta,
            primary,
            lambda_matrices[primary_name],
        )
        platt_metrics["calibration"] = "platt"
        platt_tape["calibration"] = "platt"
        self.calibration_rows.append(
            {
                "origin_idx": int(cycle_idx),
                "calibration": "raw_vs_platt",
                "raw_pr_auc_fused": float(raw_metrics["pr_auc_fused"]),
                "platt_pr_auc_fused": float(platt_metrics["pr_auc_fused"]),
                "raw_pr_auc_b1": float(raw_metrics["pr_auc_b1"]),
                "platt_pr_auc_b1": float(platt_metrics["pr_auc_b1"]),
                "raw_pr_auc_b2": float(raw_metrics["pr_auc_b2"]),
                "platt_pr_auc_b2": float(platt_metrics["pr_auc_b2"]),
                "raw_roc_auc_fused": float(raw_metrics["roc_auc_fused"]),
                "platt_roc_auc_fused": float(platt_metrics["roc_auc_fused"]),
                "raw_recall_at_100": float(raw_metrics["recall_at_100"]),
                "platt_recall_at_100": float(platt_metrics["recall_at_100"]),
                "raw_lift_at_100": float(raw_metrics["lift_at_100"]),
                "platt_lift_at_100": float(platt_metrics["lift_at_100"]),
            }
        )

        for spec in self.variant_specs:
            name = spec["variant"]
            variant_metrics, _ = self._score_metrics(
                platt_fused_matrices[name],
                b1_platt_matrix,
                b2_platt_matrix,
                gt,
                cycle_idx,
                cycle_start,
                cycle_end,
                selected,
                cascade_meta,
                spec,
                lambda_matrices[name],
            )
            variant_metrics["calibration"] = "platt"
            self._variant_rows = [
                row
                for row in self._variant_rows
                if not (
                    int(row.get("origin_idx", -1)) == int(cycle_idx)
                    and row.get("pareto_variant") == name
                )
            ]
            self._variant_rows.append(variant_metrics)
        return platt_metrics, platt_tape

    def _write_manifest(self, *args: Any, **kwargs: Any) -> None:
        super()._write_manifest(*args, **kwargs)
        path = self.output_dir / "experiment_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        payload["runner"] = Path(__file__).name
        payload["calibration"] = {
            **self.calibration_manifest,
            "primary_heldout_output": "platt",
            "raw_comparison_file": "fusion_calibration_metrics.csv",
        }
        integrated._write_json_atomic(path, payload)

    def run(self):
        metrics, tape = super().run()
        if self.calibration_rows:
            pd.DataFrame(self.calibration_rows).sort_values("origin_idx").to_csv(
                self.output_dir / "fusion_calibration_metrics.csv", index=False
            )
        path = self.output_dir / "experiment_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        payload["calibration"] = {
            **self.calibration_manifest,
            "primary_heldout_output": "platt",
            "raw_comparison_file": "fusion_calibration_metrics.csv",
        }
        integrated._write_json_atomic(path, payload)
        report_path = self.output_dir / "fusion_report.md"
        report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        comparison = pd.DataFrame(self.calibration_rows)
        report += "\n\n## Platt Calibration\n"
        report += "- Calibrator: branch-specific Platt logistic sigmoid.\n"
        report += "- Fit scope: pooled development Validation/OOF only; Held-Out Test was not used.\n"
        report += f"- Held-Out raw Fusion PR-AUC: `{comparison['raw_pr_auc_fused'].mean():.6f}`\n"
        report += f"- Held-Out Platt Fusion PR-AUC: `{comparison['platt_pr_auc_fused'].mean():.6f}`\n"
        report += f"- Held-Out raw Fusion ROC-AUC: `{comparison['raw_roc_auc_fused'].mean():.6f}`\n"
        report += f"- Held-Out Platt Fusion ROC-AUC: `{comparison['platt_roc_auc_fused'].mean():.6f}`\n"
        report += "- `fusion_calibration_metrics.csv` contains the aligned raw-vs-Platt cycle comparison.\n"
        report_path.write_text(report, encoding="utf-8")
        return metrics, tape


def _self_test() -> None:
    labels = np.array([0, 0, 0, 1, 0, 1, 0, 0], dtype=np.uint8)
    scores = np.array([0.02, 0.04, 0.10, 0.20, 0.25, 0.30, 0.60, 0.80])
    calibrator = _PlattCalibrator(20260921).fit(scores, labels)
    calibrated = calibrator.predict(scores)
    assert calibrated.shape == scores.shape
    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
    summary = _calibration_summary(labels, calibrated)
    assert summary["samples"] == len(labels)
    print("[Self-test] Platt calibrator and calibration metrics passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report-faithful Pareto Fusion with development-only Platt calibration"
    )
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
        default="experiments/2026-09-21_All-XID_ReportFaithful_ParetoMomentumCascade_Platt_01",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return

    data_dir = integrated.base.find_data_dir()
    cache_dir = integrated.base.PROJECT_ROOT / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = integrated.base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    engine = EnhancedTelemetryEngine(
        data_dir=data_dir,
        cache_dir=cache_dir,
        include_topology=args.include_topology,
    )
    _, history_map, gt_matrix = engine.load_all_xid_ledger()
    pipeline = PlattReportFaithfulParetoFusion(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=integrated.base.PROJECT_ROOT / args.output_dir,
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

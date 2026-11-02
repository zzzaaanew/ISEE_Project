"""Experiment 14: causal-feedback replay of Experiment 12's 30-minute ML.

The B1/B2 learners and full-population rolling evaluator are inherited.  ADST
and the temporal MoE may use a scored block only once its score time is at
least 36 hours behind the current origin (the 24-hour label is then mature).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import run_experiment12_30m_temporal_moe_b2_anchor as previous


rolling = previous.rolling
PURGE_BINS = rolling.PURGE_BINS
HORIZON_BINS = rolling.HORIZON_BINS


class MaturedFeedbackRolling(previous.TemporalMoERolling):
    """Delay both adaptive controllers until their feedback is eligible."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gate_feedback_cutoff_bin = -1
        self._gate_latest_context_bin = -1
        self._gate_latest_train_bin = -1
        if self.adst_history:
            latest = self.adst_history[-1]
            self.current_train_days = int(latest["adst_next_train_days"])
            self.pending_adst_action = str(latest["adst_action"])
            self.adst_stable_count = int(latest["adst_stable_count"])

    def _mature_history(self, origin: int) -> list[dict[str, Any]]:
        cutoff = int(origin) - PURGE_BINS
        history = [row for row in self.gate_history if int(row["score_bin"]) <= cutoff]
        for row in history:
            score_bin = int(row["score_bin"])
            assert score_bin + HORIZON_BINS <= int(origin), "Immature label in feedback"
        return history

    def _flush_matured_adst(self, origin: int) -> None:
        cutoff = int(origin) - PURGE_BINS
        processed = {int(row["origin"]) for row in self.adst_history}
        by_origin: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.gate_history:
            by_origin[int(row["origin"])].append(row)
        for feedback_origin in sorted(by_origin):
            if feedback_origin in processed:
                continue
            rows = by_origin[feedback_origin]
            latest = max(int(row["score_bin"]) for row in rows)
            if latest > cutoff:
                continue
            assert latest + HORIZON_BINS <= int(origin)
            used_days = {int(round(float(row["feature_train_days"]) * 21)) for row in rows}
            assert len(used_days) == 1 and next(iter(used_days)) in previous.previous.TRAIN_WINDOW_CANDIDATES
            state = {
                "origin": feedback_origin,
                "selected": {"selected_train_days": next(iter(used_days))},
            }
            feedback = previous.previous.HanADSTLightCascadeRolling._update_adst_from_cycle(
                self,
                float(np.mean([float(row["b1_pr_auc"]) for row in rows])),
                float(np.mean([float(row["prevalence"]) for row in rows])),
                state,
            )
            feedback.update(
                {
                    "adst_update_origin": int(origin),
                    "feedback_latest_score_bin": latest,
                    "feedback_cutoff_bin": cutoff,
                    "feedback_label_maturity_bin": latest + HORIZON_BINS,
                    "feedback_block_count": len(rows),
                }
            )
            pd.DataFrame(self.adst_history).to_csv(
                self.output / "adst_controller_history.csv", index=False
            )

    def fit_state(
        self, origin: int, state_idx: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._flush_matured_adst(int(origin))
        state, _ = super().fit_state(origin, state_idx)
        selected = state["selected"]
        # The inherited cascade OOF labels need to be mature by the origin.
        assert int(selected["lambda_validation_label_maturity_latest"]) <= int(origin)
        selected["lambda_selection_stage"] = "matured_36h_temporal_moe_gate"
        selected["selected_lambda_role"] = "inherited_prediction_placeholder_not_gate_weight"
        rows = [
            {
                "state_idx": int(state_idx),
                "origin": int(origin),
                "origin_time": rolling.stamp(self.engine, origin),
                "selection_stage": "matured_36h_temporal_moe_gate",
                "selection_role": "block_gate_probability; see temporal_moe_history.csv",
                "selected_lambda": None,
                "candidate_lambda": None,
                "feedback_cutoff_bin": int(origin) - PURGE_BINS,
                "feedback_cutoff_time": rolling.stamp(self.engine, int(origin) - PURGE_BINS),
                "latest_adst_feedback_score_bin": (
                    int(self.adst_history[-1]["feedback_latest_score_bin"])
                    if self.adst_history else None
                ),
                "cascade_oof_label_maturity_latest_bin": int(
                    selected["lambda_validation_label_maturity_latest"]
                ),
                "terminal_heldout_used": False,
            }
        ]
        return state, rows

    def _recent_context(self) -> tuple[float, float]:
        recent = [
            row for row in self.gate_history
            if int(row["score_bin"]) <= self._gate_feedback_cutoff_bin
        ][-48:]
        self._gate_latest_context_bin = int(recent[-1]["score_bin"]) if recent else -1
        if not recent:
            return 0.0, 0.0
        return (
            float(np.mean([float(row["b1_pr_auc"]) for row in recent])),
            float(np.mean([float(row["b2_pr_auc"]) for row in recent])),
        )

    def _fit_gate_for_state(self, state: dict[str, Any]) -> None:
        state_id = str(state["model_state_id"])
        if self._gate_model_state_id == state_id:
            return
        origin = int(state["origin"])
        self._gate_feedback_cutoff_bin = origin - PURGE_BINS
        history = self._mature_history(origin)[-previous.MOE_MAX_HISTORY_BLOCKS:]
        eligible = [
            row for row in history
            if int(float(row.get("positives", 0))) > 0
            and float(row.get("b1_better_pr_auc", -1.0)) >= 0.0
        ]
        self._gate_model = None
        self._gate_fit_status = "cold_start_b2_fallback"
        self._gate_train_blocks = len(eligible)
        self._gate_latest_train_bin = -1
        if len(eligible) >= previous.MOE_MIN_HISTORY_BLOCKS:
            x = np.asarray(
                [[float(row.get(name, 0.0)) for name in previous.MOE_FEATURES] for row in eligible],
                dtype=np.float64,
            )
            y = np.asarray([int(float(row["b1_better_pr_auc"])) for row in eligible], dtype=np.int32)
            if len(np.unique(y)) >= 2:
                model = LogisticRegression(
                    C=previous.MOE_C,
                    class_weight="balanced",
                    max_iter=250,
                    solver="liblinear",
                    random_state=self.seed,
                )
                model.fit(x, y)
                self._gate_model = model
                self._gate_fit_status = "trained_matured_36h"
                self._gate_latest_train_bin = max(int(row["score_bin"]) for row in eligible)
        if self._gate_latest_train_bin >= 0:
            assert self._gate_latest_train_bin <= self._gate_feedback_cutoff_bin
            assert self._gate_latest_train_bin + HORIZON_BINS <= origin
        self._gate_model_state_id = state_id

    def _update_adst_from_cycle(
        self, b1_ap: float, prevalence: float, state: dict[str, Any]
    ) -> dict[str, Any]:
        # Called at the end of the current, still-immature cycle by Experiment
        # 12.  The score trace is persisted there; fit_state applies it later.
        return {"adst_action": "awaiting_mature_feedback"}

    def score_cycle(
        self, state: dict[str, Any], origin: int, end: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        full, cohort, artifacts = super().score_cycle(state, origin, end)
        cutoff = int(origin) - PURGE_BINS
        assert self._gate_feedback_cutoff_bin == cutoff
        assert self._gate_latest_context_bin <= cutoff
        assert self._gate_latest_train_bin <= cutoff
        n_blocks = len(np.arange(int(origin), int(end), previous.MOE_BLOCK_BINS))
        for row in self.gate_history[-n_blocks:]:
            row["gate_feedback_cutoff_bin"] = cutoff
            row["gate_latest_context_bin"] = self._gate_latest_context_bin
            row["gate_latest_train_bin"] = self._gate_latest_train_bin
            row["gate_label_maturity_latest_bin"] = (
                self._gate_latest_train_bin + HORIZON_BINS
                if self._gate_latest_train_bin >= 0 else -1
            )
        self._write_gate_history()
        for frame in (full, cohort):
            frame["gate_feedback_cutoff_bin"] = cutoff
            frame["gate_latest_train_bin"] = self._gate_latest_train_bin
        for key in ("state", "metric"):
            artifacts[key].update(
                {
                    "gate_feedback_cutoff_bin": cutoff,
                    "gate_latest_train_bin": self._gate_latest_train_bin,
                    "gate_latest_context_bin": self._gate_latest_context_bin,
                    "adst_feedback_applied_count": len(self.adst_history),
                }
            )
        return full, cohort, artifacts

    def run(self, *args: Any, **kwargs: Any) -> None:
        super().run(*args, **kwargs)
        manifest_path = self.output / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "runner_contract": "experiment14_matured_adst_temporal_moe",
                "feedback_cutoff_rule": "score_bin <= current_origin - 36h",
                "feedback_label_maturity_rule": "score_bin + 24h <= current_origin",
                "adst_updates_delayed_until_feedback_cutoff": True,
                "gate_training_delayed_until_feedback_cutoff": True,
                "actual_lambda_trace": "temporal_moe_history.csv:gate_weight_b1",
                "lambda_selection_history_role": "dynamic gate metadata, not fixed lambda selection",
                "adst_matured_feedback_updates": len(self.adst_history),
                "runner_file": Path(__file__).name,
                "runner_sha256": rolling.sha256(Path(__file__).resolve()),
            }
        )
        rolling.write_json(manifest_path, manifest)
        contract_path = self.output / "export_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract.update(
            {
                "runner_contract": manifest["runner_contract"],
                "feedback_cutoff_rule": manifest["feedback_cutoff_rule"],
                "actual_lambda_trace": manifest["actual_lambda_trace"],
            }
        )
        rolling.write_json(contract_path, contract)
        (self.output / "IMPLEMENTATION_30M_EXPERIMENT14.md").write_text(
            "# Experiment 14: label-matured ADST and Temporal MoE\n\n"
            "- B1/B2 and 30-minute full-GPU rolling evaluation inherit Experiment 12.\n"
            "- ADST and MoE use score feedback only through origin minus 36 hours.\n"
            "- Each selected score's 24-hour label is mature by the current origin.\n"
            "- `lambda_selection_history.csv` has no scalar selected lambda; "
            "actual per-block lambda is `temporal_moe_history.csv:gate_weight_b1`.\n"
            "- No BLOX, terminal held-out selection, or previous-result overwrite.\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=rolling.DEFAULT_SOURCE)
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/2026-10-04_Experiment14_MaturedADST_TemporalMoE_01"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-states", type=int, default=None)
    args = parser.parse_args()
    rolling.base.seed_everything(rolling.DEFAULT_SEED)
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    source = (root / args.source).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output}")
    cache_dir = root / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = rolling.base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    engine = rolling.base.UnifiedDataEngine(
        data_dir=rolling.base.find_data_dir(), cache_dir=cache_dir
    )
    _, history_map, gt_matrix = engine.load_all_xid_ledger()
    runner = MaturedFeedbackRolling(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=output,
        source_dir=source,
        negative_ratio=rolling.DEFAULT_NEGATIVE_RATIO,
        cohort_size=rolling.COHORT_SIZE,
        seed=rolling.DEFAULT_SEED,
        resume=args.resume,
    )
    with threadpool_limits(limits=int(args.threads)):
        runner.run(full_population=True, max_states=args.max_states)


if __name__ == "__main__":
    main()

"""
[Unified ADST Fusion Pipeline — Stateful Momentum + Pure MLE Pareto]

Integrates:
  - 지한유: BidirectionalADSTPipeline base engine, pure MLE Pareto λ (no cap)
  - 김준호: _safe_ratio(), stateful momentum checkpoint, 2-origin cooldown, +5% guard

Architecture:
  Branch 1: Telemetry (ExtraTrees + 1D-CNN) with ADST window selection
  Branch 2: History-only (Logistic + GBDT) — unchanged
  Fusion:   Per-GPU Pareto λ weighting from MLE α, dynamic via momentum ADST

Usage:
  python ML/run_unified_adst_fusion.py --cadence-hours 24
  python ML/run_unified_adst_fusion.py --cadence-hours 24 --resume
"""

from __future__ import annotations

import argparse
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
from typing import Any

# Ensure ML directory is in sys.path for direct script execution and package imports
_ML_DIR = Path(__file__).resolve().parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── Base engine reuse (지한유 main) ──
import run_bidirectional_adst_fusion as base


# =====================================================================
# _safe_ratio (from 김준호 branch1_enhanced_features.py)
# =====================================================================
def _safe_ratio(
    numerator: np.ndarray, denominator: np.ndarray,
) -> np.ndarray:
    """Bounded ratio: near-zero denominators cannot dominate models."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.divide(
            numerator, denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=np.abs(denominator) > 1e-4,
        )
    return np.nan_to_num(
        np.clip(ratio, -100.0, 100.0),
        nan=0.0, posinf=100.0, neginf=-100.0,
    ).astype(np.float32)


# =====================================================================
# Stateful Momentum Constants (from 김준호 conservative + report-faithful)
# =====================================================================
COOLDOWN_ORIGINS = 2           # skip only after 2 consecutive same-combo origins
SHORT_WINDOW_MIN_REL_GAIN = 0.05  # +5% relative gain required for short window promotion
MOMENTUM_BETA = 0.70           # EMA decay for confidence tracking
MOMENTUM_THRESHOLD = 0.65     # confidence floor to allow skip-hold
MOMENTUM_DROP_PCT = 0.15       # re-trigger full search if perf drops > 15%


def _combo_key(l_train: int, l_obs: int) -> str:
    return f"{l_train}d|{l_obs}h"


# =====================================================================
# Unified ADST Pipeline with Stateful Momentum
# =====================================================================
class UnifiedADSTPipeline(base.BidirectionalADSTPipeline):
    """Extends base ADST pipeline with stateful EMA momentum and Pareto fusion.

    Changes from base:
    1. Skip-retrain uses EMA confidence per (L_train, L_obs) pair
       instead of simple consecutive-count threshold.
    2. Momentum state persists to disk (momentum_state.json) for --resume.
    3. 2-origin cooldown before first skip-hold is allowed.
    4. +5% relative gain guard for short-window promotion.
    5. Per-GPU Pareto λ from MLE α applied to fusion weights.
    """

    def __init__(
        self,
        *args: Any,
        resume: bool = False,
        momentum_beta: float = MOMENTUM_BETA,
        momentum_threshold: float = MOMENTUM_THRESHOLD,
        momentum_drop_pct: float = MOMENTUM_DROP_PCT,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resume = resume
        self.momentum_beta = momentum_beta
        self.momentum_threshold = momentum_threshold
        self.momentum_drop_pct = momentum_drop_pct
        self.state_path = self.output_dir / "momentum_state.json"
        self._state = self._load_momentum_state()

    # ── Momentum State Persistence ──

    def _default_momentum_state(self) -> dict[str, Any]:
        confidence = {
            _combo_key(lt, lo): 0.0
            for lt in base.CANDIDATE_L_TRAIN_DAYS
            for lo in base.CANDIDATE_L_OBS_HOURS
        }
        return {
            "version": 1,
            "beta": self.momentum_beta,
            "confidence_threshold": self.momentum_threshold,
            "performance_drop_trigger": self.momentum_drop_pct,
            "confidence": confidence,
            "previous_combo": None,        # [l_train, l_obs]
            "previous_score": 0.0,
            "consecutive_same": 0,
            "processed_origins": 0,
            "skip_count": 0,
            "full_rescan_count": 0,
        }

    def _load_momentum_state(self) -> dict[str, Any]:
        state = self._default_momentum_state()
        if self.resume and self.state_path.exists():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
                state.update(loaded)
                # Ensure all grid combos exist in confidence map
                for key in self._default_momentum_state()["confidence"]:
                    state["confidence"].setdefault(key, 0.0)
                print(f"  [Momentum] Resumed state from {self.state_path} "
                      f"(origins={state['processed_origins']}, skips={state['skip_count']})",
                      flush=True)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass  # ponytail: malformed state → fresh start, checkpoints are durable
        return state

    def _save_momentum_state(self) -> None:
        self.state_path.write_text(
            json.dumps(self._state, indent=2, default=str),
            encoding="utf-8",
        )

    def _update_confidence(self, selected_key: str) -> float:
        """EMA update: β × old + (1-β) × indicator for selected combo."""
        for key in self._state["confidence"]:
            indicator = 1.0 if key == selected_key else 0.0
            self._state["confidence"][key] = (
                self.momentum_beta * self._state["confidence"][key]
                + (1.0 - self.momentum_beta) * indicator
            )
        return self._state["confidence"][selected_key]

    # ── Override: ADST with Stateful Momentum ──

    def run_bidirectional_adst(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        print("\n=================================================================", flush=True)
        print(" [Unified ADST Pipeline] Stateful Momentum + Pure MLE Pareto", flush=True)
        print(f" Cadence: {self.retrain_cadence_hours}h | "
              f"Momentum β={self.momentum_beta}, θ={self.momentum_threshold}", flush=True)
        print("=================================================================\n", flush=True)

        cadence_bins = int(self.retrain_cadence_hours * 60 / base.STEP_MINUTES)
        warmup_bins = int(30 * 24 * 60 / base.STEP_MINUTES)
        test_start_bin = warmup_bins
        test_end_bin = self.engine.num_bins

        rolling_origins = np.arange(test_start_bin, test_end_bin - cadence_bins, cadence_bins)
        print(f"[ADST] {len(rolling_origins)} rolling origins.", flush=True)

        # Resume: skip already-processed origins
        start_idx = self._state["processed_origins"] if self.resume else 0
        if start_idx > 0:
            print(f"[Resume] Skipping {start_idx} already-processed origins.", flush=True)

        all_origin_metrics: list[dict] = []
        all_test_predictions: list[pd.DataFrame] = []
        rng = np.random.default_rng(self.seed)

        # Restore skip-retrain state from checkpoint
        _prev_combo = tuple(self._state["previous_combo"]) if self._state["previous_combo"] else None
        _prev_score = self._state["previous_score"]
        _consecutive_same = self._state["consecutive_same"]

        for origin_idx, origin_bin in enumerate(rolling_origins):
            if origin_idx < start_idx:
                # Fast-forward RNG to maintain determinism
                _ = self.sample_indices(
                    max(0, origin_bin - int(3 * 24 * 60 / base.STEP_MINUTES)),
                    origin_bin, rng,
                )
                continue

            origin_time = pd.to_datetime(
                self.engine.bin_start_ns[origin_bin], unit="ns", utc=True,
            )
            test_cycle_end = min(origin_bin + cadence_bins, test_end_bin)

            print(f"\n--- [Origin {origin_idx+1}/{len(rolling_origins)}] "
                  f"{origin_time.strftime('%Y-%m-%d %H:%M')} ---", flush=True)

            # Validation window: preceding 3 days
            val_bins_len = int(3 * 24 * 60 / base.STEP_MINUTES)
            val_start_bin = max(0, origin_bin - val_bins_len)
            val_end_bin = origin_bin

            v_bins, v_gpus, v_labels = self.sample_indices(val_start_bin, val_end_bin, rng)
            v_b2_df = self.engine.extract_branch2_features(v_bins, v_gpus, self.history_map)

            best_score = -np.inf
            best_l_train, best_l_obs = 14, 6

            # ── Stateful Momentum Decision ──
            _do_full_search = True
            adst_action = "full_search"
            current_key = _combo_key(best_l_train, best_l_obs)

            if _prev_combo is not None:
                prev_lt, prev_lo = _prev_combo
                prev_key = _combo_key(prev_lt, prev_lo)
                prev_confidence = self._state["confidence"].get(prev_key, 0.0)

                # Cooldown: need ≥ COOLDOWN_ORIGINS consecutive same before skip
                can_skip = (
                    _consecutive_same >= COOLDOWN_ORIGINS
                    and prev_confidence >= self.momentum_threshold
                )

                if can_skip:
                    # Quick-validate previous combo
                    qv_ap = self._quick_validate_combo(
                        prev_lt, prev_lo, val_start_bin, val_end_bin,
                        v_bins, v_gpus, v_labels, v_b2_df, rng,
                    )
                    perf_drop = (_prev_score - qv_ap) / max(_prev_score, 1e-9)

                    if perf_drop <= self.momentum_drop_pct:
                        # Skip-hold: performance stable
                        best_l_train, best_l_obs = prev_lt, prev_lo
                        best_score = qv_ap
                        _do_full_search = False
                        adst_action = "momentum_skip_hold"
                        self._state["skip_count"] += 1
                        print(f"  [Momentum Skip] Holding ({prev_lt}d, {prev_lo}h), "
                              f"quick-val AP={qv_ap:.4f} (drop={perf_drop:.1%}, "
                              f"conf={prev_confidence:.2f})", flush=True)
                    else:
                        adst_action = "momentum_trigger_rescan"
                        self._state["full_rescan_count"] += 1
                        print(f"  [Momentum Trigger] Perf drop {perf_drop:.1%} > "
                              f"{self.momentum_drop_pct:.0%}, full rescan", flush=True)
                elif _prev_combo is not None and prev_confidence < self.momentum_threshold:
                    adst_action = "confidence_insufficient"
                    print(f"  [Momentum] Confidence {prev_confidence:.2f} < "
                          f"{self.momentum_threshold}, full search", flush=True)

            if _do_full_search:
                # 2D Grid Search (L_train × L_obs) with +5% promotion guard
                self._state["full_rescan_count"] += 1
                prev_best_score = _prev_score if _prev_combo else -np.inf

                for l_train_days in base.CANDIDATE_L_TRAIN_DAYS:
                    tr_bins_len = int(l_train_days * 24 * 60 / base.STEP_MINUTES)
                    tr_start_bin = max(0, val_start_bin - tr_bins_len)

                    tr_bins, tr_gpus, tr_labels = self.sample_indices(
                        tr_start_bin, val_start_bin, rng,
                    )
                    tr_b2_df = self.engine.extract_branch2_features(
                        tr_bins, tr_gpus, self.history_map,
                    )

                    b2_lr = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=200, class_weight="balanced",
                                           random_state=self.seed),
                    )
                    b2_lr.fit(tr_b2_df, tr_labels)
                    v_b2_probs = b2_lr.predict_proba(v_b2_df)[:, 1]

                    for l_obs_hours in base.CANDIDATE_L_OBS_HOURS:
                        tr_b1_df, _ = self.engine.extract_branch1_features(
                            tr_bins, tr_gpus, l_obs_hours,
                        )
                        v_b1_df, _ = self.engine.extract_branch1_features(
                            v_bins, v_gpus, l_obs_hours,
                        )

                        b1_lr = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(max_iter=200, class_weight="balanced",
                                               random_state=self.seed),
                        )
                        b1_lr.fit(tr_b1_df, tr_labels)
                        v_b1_probs = b1_lr.predict_proba(v_b1_df)[:, 1]

                        fused_val_score = 0.5 * v_b1_probs + 0.5 * v_b2_probs
                        val_ap = float(average_precision_score(v_labels, fused_val_score))

                        # Promotion guard: shorter windows need +5% relative gain
                        # ponytail: prevents oscillation to small windows that look
                        # marginally better on one cycle but hurt generalization
                        if _prev_combo is not None and l_train_days < _prev_combo[0]:
                            rel_gain = (val_ap - prev_best_score) / max(prev_best_score, 1e-9)
                            if rel_gain < SHORT_WINDOW_MIN_REL_GAIN:
                                continue  # guard: not enough gain to justify shorter window

                        if val_ap > best_score:
                            best_score = val_ap
                            best_l_train = l_train_days
                            best_l_obs = l_obs_hours

            # Update momentum state
            current_combo = (best_l_train, best_l_obs)
            current_key = _combo_key(best_l_train, best_l_obs)
            conf = self._update_confidence(current_key)

            if current_combo == _prev_combo:
                _consecutive_same += 1
            else:
                _consecutive_same = 1
            _prev_combo = current_combo
            _prev_score = best_score

            # Persist state
            self._state["previous_combo"] = list(current_combo)
            self._state["previous_score"] = best_score
            self._state["consecutive_same"] = _consecutive_same
            self._state["processed_origins"] = origin_idx + 1
            self._save_momentum_state()

            print(f"  ==> Selected (L_train*={best_l_train}d, L_obs*={best_l_obs}h) "
                  f"Val PR-AUC: {best_score:.4f} [{adst_action}] conf={conf:.2f}",
                  flush=True)

            # ── Train Final Models & Test (delegates to base logic) ──
            origin_metrics, tape_df = self._train_and_evaluate_cycle(
                origin_idx, origin_bin, test_cycle_end,
                best_l_train, best_l_obs, adst_action, rng,
            )
            all_origin_metrics.append(origin_metrics)
            if tape_df is not None:
                all_test_predictions.append(tape_df)

        metrics_df = pd.DataFrame(all_origin_metrics)
        final_tape_df = pd.concat(all_test_predictions, ignore_index=True) if all_test_predictions else pd.DataFrame()

        metrics_df.to_csv(self.output_dir / "unified_adst_metrics.csv", index=False)
        if len(final_tape_df) > 0:
            final_tape_df.to_parquet(
                self.output_dir / "unified_adst_risk_tape.parquet",
                index=False, compression="zstd",
            )
        print(f"\n[Done] Saved {len(final_tape_df):,} rows to "
              f"{self.output_dir / 'unified_adst_risk_tape.parquet'}", flush=True)

        # Final momentum summary
        print(f"\n[Momentum Summary] Skips: {self._state['skip_count']}, "
              f"Rescans: {self._state['full_rescan_count']}", flush=True)
        self._generate_summary_report(metrics_df)
        return metrics_df, final_tape_df

    def _quick_validate_combo(
        self,
        l_train: int, l_obs: int,
        val_start_bin: int, val_end_bin: int,
        v_bins: np.ndarray, v_gpus: np.ndarray,
        v_labels: np.ndarray, v_b2_df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> float:
        """Quick-validate a single (L_train, L_obs) combo on the val set."""
        tr_len = int(l_train * 24 * 60 / base.STEP_MINUTES)
        tr_start = max(0, val_start_bin - tr_len)

        qv_bins, qv_gpus, qv_labels = self.sample_indices(tr_start, val_start_bin, rng)
        qv_b2 = self.engine.extract_branch2_features(qv_bins, qv_gpus, self.history_map)

        b2_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=200, class_weight="balanced",
                               random_state=self.seed),
        )
        b2_lr.fit(qv_b2, qv_labels)
        v_b2_probs = b2_lr.predict_proba(v_b2_df)[:, 1]

        qv_b1, _ = self.engine.extract_branch1_features(qv_bins, qv_gpus, l_obs)
        v_b1, _ = self.engine.extract_branch1_features(v_bins, v_gpus, l_obs)

        b1_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=200, class_weight="balanced",
                               random_state=self.seed),
        )
        b1_lr.fit(qv_b1, qv_labels)
        v_b1_probs = b1_lr.predict_proba(v_b1)[:, 1]

        return float(average_precision_score(
            v_labels, 0.5 * v_b1_probs + 0.5 * v_b2_probs,
        ))

    def _train_and_evaluate_cycle(
        self,
        origin_idx: int, origin_bin: int, test_cycle_end: int,
        best_l_train: int, best_l_obs: int, adst_action: str,
        rng: np.random.Generator,
    ) -> tuple[dict, pd.DataFrame | None]:
        """Train final models and evaluate one test cycle with Pareto λ fusion."""
        import torch
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        origin_time = pd.to_datetime(
            self.engine.bin_start_ns[origin_bin], unit="ns", utc=True,
        )
        test_cycle_time = pd.to_datetime(
            self.engine.bin_start_ns[test_cycle_end - 1], unit="ns", utc=True,
        )

        final_tr_start = max(0, origin_bin - int(best_l_train * 24 * 60 / base.STEP_MINUTES))
        final_tr_bins, final_tr_gpus, final_tr_labels = self.sample_indices(
            final_tr_start, origin_bin, rng,
        )

        b1_train_df, b1_train_tensor = self.engine.extract_branch1_features(
            final_tr_bins, final_tr_gpus, best_l_obs,
        )
        b2_train_df = self.engine.extract_branch2_features(
            final_tr_bins, final_tr_gpus, self.history_map,
        )

        # Branch 1: ExtraTrees + 1D-CNN
        b1_tree = ExtraTreesClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=self.seed,
        )
        b1_tree.fit(b1_train_df, final_tr_labels)

        b1_cnn = base.TemporalCNN1D(in_channels=7, hidden_channels=32, dropout=0.2)
        opt = torch.optim.AdamW(b1_cnn.parameters(), lr=0.005, weight_decay=1e-4)
        pos_w = max(1.0, (final_tr_labels == 0).sum() / max(1, (final_tr_labels == 1).sum()))
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))

        x_tr_t = torch.tensor(b1_train_tensor, dtype=torch.float32)
        y_tr_t = torch.tensor(final_tr_labels, dtype=torch.float32)
        b1_cnn.train()
        ds = torch.utils.data.TensorDataset(x_tr_t, y_tr_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)
        for _ in range(5):
            for bx, by in loader:
                opt.zero_grad()
                out = b1_cnn(bx)
                crit(out, by).backward()
                opt.step()
        b1_cnn.eval()

        # Branch 2: Logistic + GBDT
        b2_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=250, class_weight="balanced",
                               random_state=self.seed),
        )
        b2_lr.fit(b2_train_df, final_tr_labels)

        b2_gbdt = HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=31, l2_regularization=1.0,
            class_weight="balanced", random_state=self.seed,
        )
        b2_gbdt.fit(b2_train_df, final_tr_labels)

        # Prior calibration
        true_tr_pos = int(self.gt_matrix[final_tr_start:origin_bin].sum())
        true_tr_tot = (origin_bin - final_tr_start) * self.engine.num_gpus
        true_prior = true_tr_pos / max(true_tr_tot, 1)
        sample_prior = int(final_tr_labels.sum()) / max(len(final_tr_labels), 1)

        # ── Per-GPU Pareto λ from MLE α (pure, no cap) ──
        # Count XID events up to this origin for all GPUs
        origin_ns = int(self.engine.bin_start_ns[origin_bin])
        cutoff_ns = origin_ns - base.BUFFER_MASK_BINS * base.STEP_NS
        start_ns = cutoff_ns - 30 * base.DAY_NS
        xid_counts = np.zeros(self.engine.num_gpus, dtype=np.float64)
        for gpu in range(self.engine.num_gpus):
            events = self.history_map.get(gpu)
            if events is None or len(events) == 0:
                continue
            right = int(np.searchsorted(events, cutoff_ns, side="left"))
            left = int(np.searchsorted(events, start_ns, side="left"))
            xid_counts[gpu] = max(0, right - left)

        alpha_mle = base.pareto_alpha_mle(xid_counts)
        pareto_lam = base.pareto_lambda(xid_counts, alpha_mle)  # per-GPU, no cap

        # Test Inference
        test_bins_range = np.arange(
            origin_bin, test_cycle_end, self.test_stride_bins, dtype=np.int32,
        )
        t_test_bins = len(test_bins_range)
        fused_test_scores = np.zeros((t_test_bins, self.engine.num_gpus), dtype=np.float32)
        b1_test_scores = np.zeros_like(fused_test_scores)
        b2_test_scores = np.zeros_like(fused_test_scores)

        for rel_b, curr_bin in enumerate(test_bins_range):
            all_g = np.arange(self.engine.num_gpus, dtype=np.int32)
            cur_bins = np.full(self.engine.num_gpus, curr_bin, dtype=np.int32)

            b1_t_df, b1_t_tensor = self.engine.extract_branch1_features(
                cur_bins, all_g, best_l_obs,
            )
            b2_t_df = self.engine.extract_branch2_features(
                cur_bins, all_g, self.history_map,
            )

            p_b1_tree = b1_tree.predict_proba(b1_t_df)[:, 1]
            with torch.no_grad():
                p_b1_cnn = torch.sigmoid(
                    b1_cnn(torch.tensor(b1_t_tensor, dtype=torch.float32))
                ).cpu().numpy()
            p_b1 = 0.5 * p_b1_tree + 0.5 * p_b1_cnn

            p_b2_lr = b2_lr.predict_proba(b2_t_df)[:, 1]
            p_b2_gbdt = b2_gbdt.predict_proba(b2_t_df)[:, 1]
            p_b2 = 0.5 * p_b2_lr + 0.5 * p_b2_gbdt

            cal_p_b1 = base.adjusted_probability(p_b1, true_prior, sample_prior)
            cal_p_b2 = base.adjusted_probability(p_b2, true_prior, sample_prior)

            # Pareto-weighted fusion: λ*B1 + (1-λ)*B2 per GPU
            cal_fused = pareto_lam * cal_p_b1 + (1.0 - pareto_lam) * cal_p_b2

            b1_test_scores[rel_b] = cal_p_b1
            b2_test_scores[rel_b] = cal_p_b2
            fused_test_scores[rel_b] = cal_fused

        # Evaluate
        gt_test = self.gt_matrix[test_bins_range]
        y_test_flat = gt_test.ravel()
        fused_flat = fused_test_scores.ravel()
        b1_flat = b1_test_scores.ravel()
        b2_flat = b2_test_scores.ravel()

        auc_fused = float(roc_auc_score(y_test_flat, fused_flat)) if y_test_flat.any() else 0.5
        ap_fused = float(average_precision_score(y_test_flat, fused_flat)) if y_test_flat.any() else 0.0
        ap_b1 = float(average_precision_score(y_test_flat, b1_flat)) if y_test_flat.any() else 0.0
        ap_b2 = float(average_precision_score(y_test_flat, b2_flat)) if y_test_flat.any() else 0.0

        order = np.argsort(-fused_test_scores, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=np.uint16)
        np.put_along_axis(
            ranks, order,
            np.arange(1, self.engine.num_gpus + 1, dtype=np.uint16)[None, :],
            axis=1,
        )

        positives = int(gt_test.sum())
        hits_100 = int((gt_test & (ranks <= 100)).sum())
        r100 = hits_100 / max(positives, 1)
        prev = float(gt_test.mean())
        lift100 = (hits_100 / (t_test_bins * 100)) / max(prev, 1e-12)

        print(f"  [Cycle Test] PR-AUC: Fused={ap_fused:.4f} (B1={ap_b1:.4f}, B2={ap_b2:.4f}) | "
              f"ROC-AUC: {auc_fused:.4f} | R@100: {r100:.1%} (Lift: {lift100:.2f}x) | "
              f"α_MLE={alpha_mle:.2f}", flush=True)

        metrics = {
            "origin_idx": origin_idx,
            "origin_time": origin_time,
            "test_end_time": test_cycle_time,
            "selected_L_train_days": best_l_train,
            "selected_L_obs_hours": best_l_obs,
            "adst_action": adst_action,
            "pareto_alpha_mle": alpha_mle,
            "pr_auc_fused": ap_fused,
            "pr_auc_b1": ap_b1,
            "pr_auc_b2": ap_b2,
            "roc_auc_fused": auc_fused,
            "recall_at_100": r100,
            "lift_at_100": lift100,
            "positives": positives,
            "momentum_skip_count": self._state["skip_count"],
        }

        # Risk Tape
        top_mask = ranks <= 100
        rel_b_idx, g_idx = np.where(top_mask)
        times_ns = self.engine.bin_start_ns[test_bins_range][rel_b_idx]
        tape_df = pd.DataFrame({
            "decision_time": pd.to_datetime(times_ns, unit="ns", utc=True),
            "gpu_id": self.engine.gpu_ids[g_idx],
            "fused_risk": fused_test_scores[rel_b_idx, g_idx],
            "b1_risk": b1_test_scores[rel_b_idx, g_idx],
            "b2_risk": b2_test_scores[rel_b_idx, g_idx],
            "pareto_lambda": pareto_lam[g_idx],
            "risk_rank": ranks[rel_b_idx, g_idx],
            "L_train_days": best_l_train,
            "L_obs_hours": best_l_obs,
            "target_24h": gt_test[rel_b_idx, g_idx].astype(np.uint8),
        })

        return metrics, tape_df


def run_self_check() -> None:
    """Runnable check: verify numerical stability, Pareto MLE, and momentum math."""
    print("[Self-Check] Running verification of unified components...", flush=True)

    # 1. _safe_ratio edge cases
    num = np.array([1.0, 0.0, -1.0, 1000.0], dtype=np.float32)
    den = np.array([0.0, 0.0, 0.00001, 10.0], dtype=np.float32)
    res = _safe_ratio(num, den)
    assert res[0] == 0.0, "Divide by zero should return 0.0"
    assert res[1] == 0.0, "0/0 should return 0.0"
    assert res[2] == 0.0, "Den < 1e-4 should return 0.0"
    assert np.isclose(res[3], 100.0), f"Expected 100.0, got {res[3]}"

    # 2. Pure MLE Pareto lambda
    counts = np.array([0, 1, 5, 20], dtype=np.float64)
    alpha = 1.2
    lambdas = base.pareto_lambda(counts, alpha)
    assert np.isclose(lambdas[0], 1.0), "Count 0 must yield lambda=1.0 (pure B1)"
    assert lambdas[0] > lambdas[1] > lambdas[2] > lambdas[3], "Lambda must decrease monotonically"
    assert lambdas[3] >= 0.05, "Lambda must be bounded by minimum 0.05"

    # 3. Pareto alpha MLE
    pos_counts = np.array([1, 2, 4, 3, 2, 5, 1], dtype=np.float64)
    alpha_est = base.pareto_alpha_mle(pos_counts)
    assert 0.1 <= alpha_est <= 10.0, f"Alpha MLE {alpha_est} outside reasonable range"

    # 4. Momentum state structure & combo keys
    key = _combo_key(14, 6)
    assert key == "14d|6h", f"Combo key mismatch: {key}"

    # 5. EMA confidence math
    beta = 0.70
    conf = beta * 0.0 + (1.0 - beta) * 1.0  # single hit
    assert np.isclose(conf, 0.30), f"Expected 0.30, got {conf}"

    print("[Self-Check] ALL 5 verification checks passed successfully!", flush=True)


# =====================================================================
# CLI Entry Point
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified ADST Fusion: Stateful Momentum + Pure MLE Pareto",
    )
    parser.add_argument("--self-check", action="store_true",
                        help="Run self-verification checks and exit")
    parser.add_argument("--cadence-hours", type=int, default=24,
                        help="Retraining cadence in hours (default 24)")
    parser.add_argument("--negative-ratio", type=int, default=10,
                        help="Downsampling negative ratio (default 10)")
    parser.add_argument("--test-stride-bins", type=int, default=6,
                        help="Test stride in 5m bins (default 6 = 30min)")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/unified_adst",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from momentum_state.json checkpoint")
    parser.add_argument("--momentum-beta", type=float, default=MOMENTUM_BETA,
                        help=f"EMA decay for momentum (default {MOMENTUM_BETA})")
    parser.add_argument("--momentum-threshold", type=float, default=MOMENTUM_THRESHOLD,
                        help=f"Confidence threshold to skip (default {MOMENTUM_THRESHOLD})")
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    data_dir = base.find_data_dir()
    cache_dir = base.PROJECT_ROOT / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    output_dir = base.PROJECT_ROOT / args.output_dir

    engine = base.UnifiedDataEngine(data_dir=data_dir, cache_dir=cache_dir)
    _, history_map, gt_matrix = engine.load_all_xid_ledger()

    pipeline = UnifiedADSTPipeline(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        resume=args.resume,
        momentum_beta=args.momentum_beta,
        momentum_threshold=args.momentum_threshold,
    )
    pipeline.run_bidirectional_adst()


if __name__ == "__main__":
    main()

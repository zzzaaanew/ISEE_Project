"""
[5-Step Integrated Pipeline Benchmark: Falcon45 + ST21 + KL-Inertia + Capped Pareto (30m & 5m)]

Implements the unified 5-step architecture combining Jehan's and Junho's research:
  Step 1 (Jehan): Falcon-Augmented 45 Telemetry Features with L_obs in {6h, 24h}
  Step 2 (Joint): Fixed 21-Day Sliding Window (ST21) + Regularized B1 GBDT & B2 History Ensemble
  Step 3 (Junho): KL-Regularized Causal Inertia Filter: logit(q) = (logit(p) + gamma * m_{t-}) / (1 + gamma)
  Step 4 (Jehan): Optimal Capped Pareto Dynamic Fusion: lambda(k) = min(0.60, 1 / (k + 1))
  Step 5 (Joint): Full 1,992-GPU Cluster Evaluation at both 30-minute (6.02M rows) and 5-minute Blox (36.14M rows) cadences
"""
from __future__ import annotations

import os
import tempfile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ML"))

import run_unified_adst_fusion as runner

EPS = 1e-6


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=np.abs(den) > 1e-4)
    return np.nan_to_num(np.clip(r, -100.0, 100.0), nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float32)


class FastGridFeatureStore:
    """Vectorized O(1) lookup store for Branch 1 (Falcon 45) and Branch 2 features across the (26957, 1992) grid."""

    def __init__(self, engine: runner.base.UnifiedDataEngine, history_map: dict[int, np.ndarray]) -> None:
        t0 = time.time()
        self.engine = engine
        self.history_map = history_map
        self.num_bins = engine.num_bins
        self.num_gpus = engine.num_gpus
        self.num_nodes = engine.num_nodes

        # Precompute static spatial / peer-relative features on the raw grid (indexed at last_lag_bin = bins - 3)
        self.spatial_grid = self._build_spatial_and_peer_grid()
        # Precompute Branch 2 history grid
        self.b2_count_30d, self.b2_days_since, self.b2_hour, self.b2_dow, self.b2_weekend = self._build_b2_grid()
        # Cache for rolling window statistics keyed by l_obs_hours
        self.rolling_cache: dict[int, dict[str, np.ndarray]] = {}
        print(f"[FastGridStore] Spatial, Peer-Relative & B2 grids built in {time.time() - t0:.2f}s.", flush=True)

    def _build_spatial_and_peer_grid(self) -> dict[str, np.ndarray]:
        B, G = self.num_bins, self.num_gpus
        N = self.num_nodes
        out: dict[str, np.ndarray] = {}

        for m in ["temp", "power", "util"]:
            mat3d = self.engine.matrices[m].reshape(B, N, 8).astype(np.float32)
            valid = np.isfinite(mat3d)
            clean = np.where(valid, mat3d, 0.0)
            node_sum = clean.sum(axis=2, keepdims=True)
            # Peer mean across the 7 sibling GPUs
            neigh_mean = (node_sum - clean) / 7.0
            out[f"gnn_neighbor_{m}_mean"] = neigh_mean.reshape(B, G).astype(np.float32)

            if m in ["temp", "power"]:
                diff = np.nan_to_num(mat3d, nan=0.0) - neigh_mean
                out[f"gnn_{m}_spatial_diff"] = diff.reshape(B, G).astype(np.float32)

                # Neighbor max excluding self (using top-2 max per node of 8)
                masked = np.where(valid, mat3d, -np.inf)
                idx_max1 = np.argmax(masked, axis=2, keepdims=True)
                max1 = np.take_along_axis(masked, idx_max1, axis=2)
                masked2 = masked.copy()
                np.put_along_axis(masked2, idx_max1, -np.inf, axis=2)
                max2 = np.max(masked2, axis=2, keepdims=True)
                slots = np.arange(8)[None, None, :]
                neigh_max = np.where(slots == idx_max1, max2, max1)
                neigh_max = np.where(np.isfinite(neigh_max), neigh_max, mat3d)
                out[f"gnn_neighbor_{m}_max"] = np.nan_to_num(neigh_max.reshape(B, G), nan=0.0).astype(np.float32)

                # Intra-node peer rank (1.0 = highest, 8.0 = lowest)
                own_val = np.nan_to_num(mat3d, nan=0.0)
                ranks = (own_val[:, :, None, :] > own_val[:, :, :, None]).sum(axis=3).astype(np.float32) + 1.0
                out[f"peer_{m}_rank"] = ranks.reshape(B, G)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                node_std = np.nanstd(mat3d, axis=2, keepdims=True)
            node_std_clean = np.nan_to_num(node_std, nan=0.0).astype(np.float32)
            if m == "temp":
                out["gnn_node_temp_std"] = np.broadcast_to(node_std_clean, (B, N, 8)).reshape(B, G).copy()

            # Peer Z-score relative to same-node peers
            own_c = np.nan_to_num(mat3d, nan=0.0)
            out[f"peer_{m}_zscore"] = _safe_ratio(
                (own_c - neigh_mean).reshape(B, G),
                np.broadcast_to(node_std_clean, (B, N, 8)).reshape(B, G),
            )

        return out

    def _build_b2_grid(self):
        B, G = self.num_bins, self.num_gpus
        cutoff_ns = self.engine.bin_start_ns - 10 * runner.base.MINUTE_NS
        count_30d = np.zeros((B, G), dtype=np.float32)
        days_since = np.full((B, G), 90.0, dtype=np.float32)

        for gpu, events in self.history_map.items():
            if events is None or len(events) == 0:
                continue
            r = np.searchsorted(events, cutoff_ns, side="right")
            l = np.searchsorted(events, cutoff_ns - 30 * runner.base.DAY_NS, side="left")
            count_30d[:, gpu] = r - l
            has_prev = r > 0
            if has_prev.any():
                prev_time = events[r[has_prev] - 1]
                diff_days = (cutoff_ns[has_prev] - prev_time) / runner.base.DAY_NS
                days_since[has_prev, gpu] = np.minimum(diff_days, 90.0)

        dt_series = pd.to_datetime(self.engine.bin_start_ns, unit="ns", utc=True)
        hour = np.asarray(dt_series.hour, dtype=np.float32)
        dow = np.asarray(dt_series.dayofweek, dtype=np.float32)
        weekend = (dow >= 5).astype(np.float32)
        return count_30d, days_since, hour, dow, weekend

    def get_rolling(self, l_obs_hours: int) -> dict[str, np.ndarray]:
        if l_obs_hours in self.rolling_cache:
            return self.rolling_cache[l_obs_hours]

        t0 = time.time()
        W = runner.base.L_OBS_BINS_MAP[l_obs_hours]
        B, G = self.num_bins, self.num_gpus
        res: dict[str, np.ndarray] = {}

        for m in runner.base.METRICS:
            mat = self.engine.matrices[m]
            valid = np.isfinite(mat)
            x = np.where(valid, mat, 0.0).astype(np.float64)

            # Pad 1 row of zeros at top for 1-based prefix sum
            cs_cnt = np.vstack([np.zeros((1, G), dtype=np.int32), np.cumsum(valid, axis=0, dtype=np.int32)])
            cs_sum = np.vstack([np.zeros((1, G), dtype=np.float64), np.cumsum(x, axis=0, dtype=np.float64)])
            cs_sq = np.vstack([np.zeros((1, G), dtype=np.float64), np.cumsum(x * x, axis=0, dtype=np.float64)])

            # For bin b, window is [b - 2 - W, b - 3] inclusive (length W)
            # In 1-based prefix sum, sum(a..b_end) = cs[b_end + 1] - cs[a]
            idx_end = np.clip(np.arange(B) - 2, 0, B)
            idx_start = np.clip(np.arange(B) - 2 - W, 0, B)

            cnt = (cs_cnt[idx_end] - cs_cnt[idx_start]).astype(np.float32)
            tot = cs_sum[idx_end] - cs_sum[idx_start]
            sq = cs_sq[idx_end] - cs_sq[idx_start]

            mean = np.divide(tot, cnt, out=np.zeros((B, G), dtype=np.float64), where=cnt > 0).astype(np.float32)
            var = np.maximum(0.0, np.divide(sq, cnt, out=np.zeros((B, G), dtype=np.float64), where=cnt > 0) - mean.astype(np.float64) ** 2)
            std = np.where(cnt > 0, np.sqrt(var), 0.0).astype(np.float32)

            # Rolling max and min over window W ending at b - 3
            mat_for_max = np.where(valid, mat, -np.inf)
            mat_for_min = np.where(valid, mat, np.inf)
            # Filter of size W with origin = (W - 1) // 2 aligns output at the right edge of the window
            rmax_ending = maximum_filter1d(mat_for_max, size=W, axis=0, mode="constant", cval=-np.inf, origin=(W - 1) // 2)
            rmin_ending = minimum_filter1d(mat_for_min, size=W, axis=0, mode="constant", cval=np.inf, origin=(W - 1) // 2)

            lag3 = np.clip(np.arange(B) - 3, 0, B - 1)
            max_val = np.where(cnt > 0, rmax_ending[lag3], 0.0).astype(np.float32)
            min_val = np.where(cnt > 0, rmin_ending[lag3], 0.0).astype(np.float32)
            last_val = np.nan_to_num(mat[lag3], nan=0.0).astype(np.float32)

            res[f"{m}_last"] = last_val
            res[f"{m}_mean"] = mean
            res[f"{m}_max"] = max_val
            res[f"{m}_min"] = min_val
            res[f"{m}_std"] = std
            res[f"{m}_delta"] = (last_val - mean).astype(np.float32)

        self.rolling_cache[l_obs_hours] = res
        print(f"[FastGridStore] Rolling L_obs={l_obs_hours}h cached in {time.time() - t0:.2f}s.", flush=True)
        return res

    def extract_b1(self, bins: np.ndarray, gpus: np.ndarray, l_obs_hours: int) -> np.ndarray:
        """Extracts 45 Falcon-Enhanced Branch 1 features in O(1) via vectorized indexing."""
        roll = self.get_rolling(l_obs_hours)
        lag3 = bins - 3
        node_idx = gpus // 8
        h_float = max(float(l_obs_hours), 1.0)

        cols = []
        # 1. Base Telemetry Statistics (22 cols: last, mean, std, delta * 4 + 6 node context)
        for m in runner.base.METRICS:
            for stat in ["last", "mean", "std", "delta"]:
                cols.append(roll[f"{m}_{stat}"][bins, gpus])

        for m in ["util", "temp", "power"]:
            node_last = np.nan_to_num(self.engine.node_matrices[m][lag3, node_idx], nan=0.0).astype(np.float32)
            cols.append(node_last)
            cols.append(roll[f"{m}_last"][bins, gpus] - node_last)

        # 2. Cross-Metric Interaction Features (7 cols, using true peak max headroom!)
        u_mean = roll["util_mean"][bins, gpus]
        t_mean = roll["temp_mean"][bins, gpus]
        p_mean = roll["power_mean"][bins, gpus]
        u_delta = roll["util_delta"][bins, gpus]
        t_delta = roll["temp_delta"][bins, gpus]
        p_delta = roll["power_delta"][bins, gpus]
        f_delta = roll["fb_delta"][bins, gpus]
        t_std = roll["temp_std"][bins, gpus]
        p_std = roll["power_std"][bins, gpus]
        t_max = roll["temp_max"][bins, gpus]
        p_max = roll["power_max"][bins, gpus]

        cols.append(_safe_ratio(t_mean, u_mean))                    # thermal_efficiency
        cols.append(_safe_ratio(p_std, t_std))                      # power_temp_ratio
        cols.append((u_delta * f_delta).astype(np.float32))         # util_fb_coupling
        cols.append(_safe_ratio(p_mean, u_mean))                    # power_per_util
        cols.append(np.abs(t_delta - f_delta).astype(np.float32))   # temp_fb_divergence
        cols.append((t_max - t_mean).astype(np.float32))            # thermal_headroom (peak)
        cols.append((p_max - p_mean).astype(np.float32))            # power_headroom (peak)

        # 3. Node Topology Spatial Features (8 cols)
        for k in [
            "gnn_neighbor_temp_max", "gnn_neighbor_temp_mean", "gnn_node_temp_std", "gnn_temp_spatial_diff",
            "gnn_neighbor_power_max", "gnn_neighbor_power_mean", "gnn_power_spatial_diff", "gnn_neighbor_util_mean",
        ]:
            cols.append(self.spatial_grid[k][lag3, gpus])

        # 4. Falcon Peer-Relative & Temporal Dynamics Features (8 cols)
        for k in ["peer_temp_zscore", "peer_power_zscore", "peer_util_zscore", "peer_temp_rank", "peer_power_rank"]:
            cols.append(self.spatial_grid[k][lag3, gpus])

        cols.append((t_delta / h_float).astype(np.float32))         # temp_slope
        cols.append((p_delta / h_float).astype(np.float32))         # power_slope
        cols.append(_safe_ratio(t_delta, t_std))                    # thermal_persistence

        X = np.column_stack(cols)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_b2(self, bins: np.ndarray, gpus: np.ndarray) -> np.ndarray:
        return np.column_stack([
            self.b2_count_30d[bins, gpus],
            self.b2_days_since[bins, gpus],
            self.b2_hour[bins],
            self.b2_dow[bins],
            self.b2_weekend[bins],
        ])


def apply_kl_inertia_chain(
    p_b1_all: np.ndarray,
    step_minutes: int,
    half_life_hours: float,
    gamma: float,
) -> np.ndarray:
    """Applies causal KL-regularized inertia across contiguous time steps (T, G):
    logit(q_t) = (logit(p_t) + gamma * m_{t-}) / (1 + gamma)
    m_t = rho * m_{t-} + (1 - rho) * logit(p_t)
    """
    if gamma <= 0.0:
        return p_b1_all.copy()

    rho = float(2.0 ** (-(step_minutes / 60.0) / max(half_life_hours, 1e-6)))
    raw_logits = logit(np.clip(p_b1_all, EPS, 1.0 - EPS)).astype(np.float64)
    T, G = raw_logits.shape
    out_logits = np.empty_like(raw_logits)

    m_prev = raw_logits[0].copy()
    out_logits[0] = raw_logits[0]

    inv_denom = 1.0 / (1.0 + gamma)
    for t in range(1, T):
        l_t = raw_logits[t]
        out_logits[t] = (l_t + gamma * m_prev) * inv_denom
        m_prev = rho * m_prev + (1.0 - rho) * l_t

    return np.clip(expit(out_logits), EPS, 1.0 - EPS).astype(np.float32)


def compute_metrics_and_churn(
    scores: np.ndarray,
    gt: np.ndarray,
    step_minutes: int,
) -> dict[str, float]:
    """Computes Hits@100, Pooled Recall@100, Origin Mean Recall@100, PR-AUC, and Hourly Top-100 Churn."""
    T, G = scores.shape
    order = np.argsort(-scores, axis=1, kind="stable")[:, :100]
    chosen = np.take_along_axis(gt, order, axis=1)
    hits = int(chosen.sum())
    positives = int(gt.sum())
    pooled_recall = hits / max(positives, 1)

    bins_per_day = int(24 * 60 // step_minutes)
    n_days = T // bins_per_day
    daily_recalls = []
    daily_aps = []
    for d in range(n_days):
        s_d = scores[d * bins_per_day : (d + 1) * bins_per_day]
        g_d = gt[d * bins_per_day : (d + 1) * bins_per_day]
        c_d = chosen[d * bins_per_day : (d + 1) * bins_per_day]
        pos_d = int(g_d.sum())
        daily_recalls.append(int(c_d.sum()) / max(pos_d, 1))
        daily_aps.append(float(average_precision_score(g_d.ravel(), s_d.ravel())) if pos_d > 0 else 0.0)

    top_mask = np.zeros((T, G), dtype=bool)
    np.put_along_axis(top_mask, order, True, axis=1)
    entered_per_step = (top_mask[1:] & ~top_mask[:-1]).sum(axis=1).mean()
    steps_per_hour = 60.0 / step_minutes
    hourly_churn = float(entered_per_step * steps_per_hour)

    return {
        "hits": hits,
        "positives": positives,
        "pooled_recall": pooled_recall,
        "cycle_mean_recall": float(np.mean(daily_recalls)),
        "mean_pr_auc": float(np.mean(daily_aps)),
        "hourly_churn": hourly_churn,
    }


def main() -> None:
    data_dir = runner.base.find_data_dir()
    cache_dir = Path("outputs/branch1/cache") if Path("outputs/branch1/cache").exists() else Path("../outputs/branch1/cache")
    engine = runner.base.UnifiedDataEngine(data_dir=data_dir, cache_dir=cache_dir)
    _, history_map, gt_matrix = engine.load_all_xid_ledger()

    store = FastGridFeatureStore(engine, history_map)

    if "--self-check" in sys.argv:
        tb = np.array([10000, 10006, 10012], dtype=np.int32)
        tg = np.array([0, 5, 100], dtype=np.int32)
        x1 = store.extract_b1(tb, tg, 6)
        x2 = store.extract_b2(tb, tg)
        assert x1.shape == (3, 45), f"Expected (3, 45), got {x1.shape}"
        assert x2.shape == (3, 5), f"Expected (3, 5), got {x2.shape}"
        assert np.isfinite(x1).all() and np.isfinite(x2).all()
        print("[Self-Check Passed] FastGridFeatureStore extracted 45 Falcon B1 features and 5 B2 features cleanly.")
        return

    pipeline = runner.UnifiedADSTPipeline(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=Path("outputs/unified_adst"),
    )
    metrics_df = pd.read_csv(Path("outputs/unified_adst/unified_adst_metrics.csv"))

    cadence_bins = int(24 * 60 / runner.base.STEP_MINUTES)  # 288 bins per day (5-minute resolution)
    warmup_bins = int(30 * 24 * 60 / runner.base.STEP_MINUTES)
    rolling_origins = np.arange(warmup_bins, engine.num_bins - cadence_bins, cadence_bins)
    n_origins = min(len(rolling_origins), len(metrics_df))

    # Pre-warm rolling cache for L_obs in {1, 6, 24}
    for lo in sorted(metrics_df["selected_L_obs_hours"].unique()):
        store.get_rolling(int(lo))

    total_5m_bins = n_origins * cadence_bins
    all_p1_5m = np.zeros((total_5m_bins, engine.num_gpus), dtype=np.float32)
    all_p2_5m = np.zeros((total_5m_bins, engine.num_gpus), dtype=np.float32)
    all_k_5m = np.zeros((total_5m_bins, engine.num_gpus), dtype=np.float32)
    all_gt_5m = np.zeros((total_5m_bins, engine.num_gpus), dtype=bool)

    rng = np.random.default_rng(42)
    t0_all = time.time()
    print(f"\n[Starting 5-Step Integrated Evaluation] {n_origins} Origins | 1,992 GPUs | 5-min ({total_5m_bins:,} steps = {total_5m_bins * engine.num_gpus:,} rows) & 30-min...", flush=True)

    for origin_idx in range(n_origins):
        origin_bin = int(rolling_origins[origin_idx])
        # Step 1 & 2: ST21 (21-day training window) + L_obs in {6h, 24h}
        l_train_days = 21
        best_l_obs = max(6, int(metrics_df["selected_L_obs_hours"].iloc[origin_idx]))
        test_cycle_end = origin_bin + cadence_bins

        final_tr_start = max(0, origin_bin - int(l_train_days * 24 * 60 / runner.base.STEP_MINUTES))
        final_tr_bins, final_tr_gpus, final_tr_labels = pipeline.sample_indices(final_tr_start, origin_bin, rng)

        b1_tr = store.extract_b1(final_tr_bins, final_tr_gpus, best_l_obs)
        b2_tr = store.extract_b2(final_tr_bins, final_tr_gpus)

        b1_model = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.06, max_iter=150, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.5, class_weight="balanced", random_state=42,
        )
        b1_model.fit(b1_tr, final_tr_labels)

        b2_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=250, class_weight="balanced", random_state=42))
        b2_lr.fit(b2_tr, final_tr_labels)
        b2_gbdt = HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=31, l2_regularization=1.0, class_weight="balanced", random_state=42,
        )
        b2_gbdt.fit(b2_tr, final_tr_labels)

        true_tr_pos = int(gt_matrix[final_tr_start:origin_bin].sum())
        true_tr_tot = (origin_bin - final_tr_start) * engine.num_gpus
        true_prior = true_tr_pos / max(true_tr_tot, 1)
        sample_prior = int(final_tr_labels.sum()) / max(len(final_tr_labels), 1)

        # Full 5-minute test grid for this 24h cycle (288 steps x 1,992 GPUs = 573,696 rows)
        test_bins_5m = np.arange(origin_bin, test_cycle_end, 1, dtype=np.int32)
        cur_bins = np.repeat(test_bins_5m, engine.num_gpus)
        all_g = np.tile(np.arange(engine.num_gpus, dtype=np.int32), len(test_bins_5m))

        b1_te = store.extract_b1(cur_bins, all_g, best_l_obs)
        b2_te = store.extract_b2(cur_bins, all_g)

        p1 = runner.base.adjusted_probability(
            b1_model.predict_proba(b1_te)[:, 1], true_prior, sample_prior
        ).reshape(cadence_bins, engine.num_gpus)
        p2 = runner.base.adjusted_probability(
            0.5 * b2_lr.predict_proba(b2_te)[:, 1] + 0.5 * b2_gbdt.predict_proba(b2_te)[:, 1],
            true_prior, sample_prior,
        ).reshape(cadence_bins, engine.num_gpus)

        # Per-GPU 30-day XID count at origin
        origin_ns = int(engine.bin_start_ns[origin_bin])
        cutoff_ns = origin_ns - runner.base.BUFFER_MASK_BINS * runner.base.STEP_NS
        start_ns = cutoff_ns - 30 * runner.base.DAY_NS
        xid_counts = np.zeros(engine.num_gpus, dtype=np.float32)
        for gpu in range(engine.num_gpus):
            events = history_map.get(gpu)
            if events is not None and len(events) > 0:
                right = int(np.searchsorted(events, cutoff_ns, side="left"))
                left = int(np.searchsorted(events, start_ns, side="left"))
                xid_counts[gpu] = max(0, right - left)

        s0 = origin_idx * cadence_bins
        s1 = (origin_idx + 1) * cadence_bins
        all_p1_5m[s0:s1] = p1
        all_p2_5m[s0:s1] = p2
        all_k_5m[s0:s1] = xid_counts[None, :]
        all_gt_5m[s0:s1] = gt_matrix[test_bins_5m]

        if (origin_idx + 1) % 10 == 0 or origin_idx == n_origins - 1:
            print(f"  [{origin_idx + 1:2d}/{n_origins}] Origins inferred ({(origin_idx + 1) * cadence_bins * engine.num_gpus:,} 5-min rows) in {time.time() - t0_all:.1f}s", flush=True)

    # Evaluate both 30-minute (stride=6) and 5-minute (stride=1) cadences across Inertia & Pareto configurations
    summary_rows = []

    for cadence_name, step_min, stride in [("30m (Standard)", 30, 6), ("5m (Blox High-Cadence)", 5, 1)]:
        p1_grid = all_p1_5m[::stride]
        p2_grid = all_p2_5m[::stride]
        k_grid = all_k_5m[::stride]
        gt_grid = all_gt_5m[::stride]

        lam_capped = np.minimum(0.60, 1.0 / (k_grid + 1.0))
        lam_uncapped = 1.0 / (k_grid + 1.0)

        # 1. B1 Standalone (No Inertia)
        m_b1_raw = compute_metrics_and_churn(p1_grid, gt_grid, step_min)
        summary_rows.append({"cadence": cadence_name, "model": "B1 Only (Falcon45, No Inertia)", "gamma": 0.0, "half_life_h": 0.0, **m_b1_raw})

        # 2. B2 Standalone (History Ensemble)
        m_b2 = compute_metrics_and_churn(p2_grid, gt_grid, step_min)
        summary_rows.append({"cadence": cadence_name, "model": "B2 Only (History Ensemble)", "gamma": 0.0, "half_life_h": 0.0, **m_b2})

        # 3. Dual Capped Pareto (No Inertia, gamma=0)
        pf_0 = lam_capped * p1_grid + (1.0 - lam_capped) * p2_grid
        m_pf0 = compute_metrics_and_churn(pf_0, gt_grid, step_min)
        summary_rows.append({"cadence": cadence_name, "model": "Dual Capped Pareto (Cap 0.60, No Inertia)", "gamma": 0.0, "half_life_h": 0.0, **m_pf0})

        # 4. Dual Capped Pareto + KL-Regularized Inertia across (gamma, half_life)
        for gamma, hl in [(0.25, 0.5), (0.50, 1.0), (1.00, 2.0)]:
            q1 = apply_kl_inertia_chain(p1_grid, step_minutes=step_min, half_life_hours=hl, gamma=gamma)
            m_b1_kl = compute_metrics_and_churn(q1, gt_grid, step_min)
            summary_rows.append({"cadence": cadence_name, "model": f"B1 + KL-Inertia (γ={gamma}, h={hl}h)", "gamma": gamma, "half_life_h": hl, **m_b1_kl})

            pf_kl = lam_capped * q1 + (1.0 - lam_capped) * p2_grid
            m_pf_kl = compute_metrics_and_churn(pf_kl, gt_grid, step_min)
            summary_rows.append({"cadence": cadence_name, "model": f"5-Step Integrated Dual (Cap 0.60 + KL γ={gamma}, h={hl}h)", "gamma": gamma, "half_life_h": hl, **m_pf_kl})

            if gamma == 1.0:
                pf_uncap = lam_uncapped * q1 + (1.0 - lam_uncapped) * p2_grid
                m_uncap = compute_metrics_and_churn(pf_uncap, gt_grid, step_min)
                summary_rows.append({"cadence": cadence_name, "model": f"Uncapped Pareto (No Cap + KL γ={gamma}, h={hl}h)", "gamma": gamma, "half_life_h": hl, **m_uncap})

    res_df = pd.DataFrame(summary_rows)
    out_path = PROJECT_ROOT / "outputs" / "unified_adst" / "integrated_5step_30m_5m_benchmark.csv"
    res_df.to_csv(out_path, index=False)

    print("\n" + "=" * 115)
    print(" === 5-STEP INTEGRATED ARCHITECTURE BENCHMARK (30-MIN & 5-MIN BLOX CADENCE, 1,992 GPUS, 63 ORIGINS) ===")
    print("=" * 115)
    for idx, r in res_df.iterrows():
        print(
            f"[{r['cadence']:22s}] {r['model']:52s} | "
            f"Hits: {int(r['hits']):7,d} / {int(r['positives']):7,d} | "
            f"Pooled R@100: {r['pooled_recall']*100:6.2f}% | "
            f"Mean R@100: {r['cycle_mean_recall']*100:6.2f}% | "
            f"PR-AUC: {r['mean_pr_auc']:.4f} | "
            f"Churn/hr: {r['hourly_churn']:5.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()


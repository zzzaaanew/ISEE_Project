"""Past-only B2-anchored candidate-union reranking on Experiment 12 saved scores.

No branch is fitted here. This is an exploratory replay because the source B1
ADST controller may have used labels before their 24-hour maturity time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_experiment12_top100 import (
    DECISION_NS,
    GRID_META,
    IDEAL_DCG,
    K,
    ONSET_LEDGER,
    PROJECT,
    SCORE_COLUMNS,
    SOURCE,
    check,
    read_ground_truth,
    sha256,
)


OUTPUT = PROJECT / "experiments" / "2026-10-01_Experiment12_B2Anchored_Union_Rerank_01"
REFERENCE = PROJECT / "experiments" / "2026-10-01_Experiment12_Top100_Validation_01"
M_GRID = (150, 250, 400)
N_GRID = (50, 100, 200)
SWAP_GRID = (0, 5, 10)
RATIO_GRID = (1.0, 1.2)
SEED = 20260905
HOUR_NS = 60 * 60 * 1_000_000_000
DAY_NS = 24 * HOUR_NS
GPU_COUNT = 1992
TRAIN_DAYS = 7
VALIDATION_DAYS = 7
MIN_SPLIT_QUERIES = 250
DISCOUNT = 1.0 / np.log2(np.arange(2, K + 2, dtype=np.float64))


@dataclass(slots=True)
class Query:
    timestamp_ns: int
    origin_idx: int
    b1: np.ndarray
    b2: np.ndarray
    rank1: np.ndarray
    rank2: np.ndarray
    order2: np.ndarray
    label: np.ndarray


def ranks_and_order(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores), dtype=np.int16)
    rank[order] = np.arange(1, len(scores) + 1, dtype=np.int16)
    return rank, order.astype(np.int16)


def features(query: Query) -> np.ndarray:
    n = len(query.b1)
    p1 = np.clip(query.b1, 1e-7, 1 - 1e-7)
    p2 = np.clip(query.b2, 1e-7, 1 - 1e-7)
    r1 = 1.0 - query.rank1.astype(np.float64) / n
    r2 = 1.0 - query.rank2.astype(np.float64) / n
    return np.column_stack((r1, r2, np.log(p1 / (1 - p1)), np.log(p2 / (1 - p2)), r1 - r2))


def candidate_mask(query: Query, m: int, n: int) -> np.ndarray:
    return (query.rank2 <= m) | (query.rank1 <= n)


def proposed_top100(query: Query, meta_risk: np.ndarray, m: int, n: int, max_swaps: int, ratio: float) -> tuple[np.ndarray, int]:
    top = query.order2[:K].copy()
    if max_swaps == 0:
        return top, 0
    eligible = candidate_mask(query, m, n) & (query.rank2 > K) & (query.rank1 < query.rank2)
    challenger = np.flatnonzero(eligible)
    if not len(challenger):
        return top, 0
    challenger = challenger[np.argsort(-meta_risk[challenger], kind="stable")]
    swaps = 0
    for slot in range(K - 1, K - 1 - max_swaps, -1):
        if swaps == len(challenger):
            break
        incoming = int(challenger[swaps])
        incumbent = int(top[slot])
        if meta_risk[incoming] < ratio * meta_risk[incumbent]:
            break
        top[slot] = incoming
        swaps += 1
    check(len(set(top.tolist())) == K, "Reranker duplicated a Top-100 GPU")
    check(set(top.tolist()).issubset(set(np.flatnonzero(candidate_mask(query, m, n)).tolist())), "Top-100 escaped candidate union")
    return top, swaps


def full_order(query: Query, top: np.ndarray) -> np.ndarray:
    in_top = np.zeros(len(query.b2), dtype=bool)
    in_top[top] = True
    order = np.concatenate((top, query.order2[~in_top[query.order2]]))
    check(len(order) == len(query.b2) and len(np.unique(order)) == len(order), "Full rerank is not a permutation")
    return order


def query_ndcg(query: Query, top: np.ndarray) -> float:
    positives = int(query.label.sum())
    if not positives:
        return float("nan")
    return float(query.label[top] @ DISCOUNT / IDEAL_DCG[min(positives, K)])


def synthetic_test() -> None:
    n = 120
    b1_order = np.r_[110, np.arange(110), np.arange(111, n)]
    b1 = np.empty(n, dtype=float)
    b1[b1_order] = np.linspace(1.0, 0.01, n)
    b2 = np.linspace(1.0, 0.01, n)
    r1, _ = ranks_and_order(b1)
    r2, o2 = ranks_and_order(b2)
    query = Query(0, 0, b1, b2, r1, r2, o2, np.zeros(n, dtype=bool))
    meta = np.full(n, 0.1)
    meta[110] = 0.9
    unchanged, count0 = proposed_top100(query, meta, 100, 1, 0, 1.0)
    changed, count1 = proposed_top100(query, meta, 100, 1, 1, 1.2)
    check(count0 == 0 and np.array_equal(unchanged, o2[:K]), "K=0 must exactly recover B2")
    check(count1 == 1 and int(changed[-1]) == 110, "B1-only challenger was not promoted")
    check(len(np.unique(full_order(query, changed))) == n, "Full order test failed")


def load_source(max_origins: int | None) -> tuple[list[list[Query]], np.ndarray, list[Path], dict, pd.DataFrame]:
    manifest_path = SOURCE / "experiment_manifest.json"
    parts_manifest = SOURCE / "full_population_manifest.json"
    original_metrics = SOURCE / "full_period_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    part_contract = json.loads(parts_manifest.read_text(encoding="utf-8"))
    check(manifest["status"] == "COMPLETED" and manifest["target"] == "all_xids", "Source experiment incomplete or wrong target")
    check((manifest["gpu_count"], manifest["origin_count"], manifest["decision_interval_seconds"], manifest["horizon_hours"], manifest["purge_hours"], manifest["feature_cutoff_minutes"], manifest["warmup_days"]) == (1992, 64, 1800, 24, 36, 10, 30), "Source experiment contract changed")
    parts = [SOURCE / path for path in part_contract["full_parts"]]
    check(len(parts) == 64 and all(path.is_file() for path in parts), "Expected all 64 score parts")
    reference = pd.read_csv(original_metrics).sort_values("origin_idx").reset_index(drop=True)
    check(len(reference) == 64 and reference["origin_idx"].tolist() == list(range(64)), "Source origin metrics changed")
    gpu_ids, bins_ns, ground_truth, _ = read_ground_truth()
    origin_groups: list[list[Query]] = []
    used_parts = parts if max_origins is None else parts[:max_origins]
    for idx, part in enumerate(used_parts):
        ref = reference.iloc[idx]
        cols = ["timestamp", "gpu_uid", "b1_probability", "b2_probability", "model_state_id"]
        frame = pq.read_table(part, columns=cols).to_pandas()
        n_times = int(ref["decision_time_count"])
        check(len(frame) == n_times * GPU_COUNT, f"origin {idx}: row count mismatch")
        check(frame["model_state_id"].eq(str(ref["model_state_id"])).all(), f"origin {idx}: model state mismatch")
        observed_gpus = frame["gpu_uid"].to_numpy().reshape(n_times, GPU_COUNT)
        check(np.array_equal(observed_gpus, np.broadcast_to(gpu_ids, observed_gpus.shape)), f"origin {idx}: GPU order mismatch")
        time_ns = pd.to_datetime(frame["timestamp"], utc=True).astype("int64").to_numpy().reshape(n_times, GPU_COUNT)
        check(np.all(time_ns == time_ns[:, :1]), f"origin {idx}: mixed timestamp")
        query_times = time_ns[:, 0]
        check(np.all(np.diff(query_times) == DECISION_NS), f"origin {idx}: cadence mismatch")
        check(pd.Timestamp(query_times[0], unit="ns", tz="UTC") == pd.Timestamp(ref["origin_time"]), f"origin {idx}: start mismatch")
        bin_idx = (query_times - bins_ns[0]) // (5 * 60 * 1_000_000_000)
        check(np.all((bin_idx >= 0) & (bin_idx < len(bins_ns))), f"origin {idx}: timestamp out of grid")
        check(np.array_equal(bins_ns[bin_idx], query_times), f"origin {idx}: grid mismatch")
        labels = ground_truth[bin_idx]
        check(int(labels.sum()) == int(ref["positives"]), f"origin {idx}: positive labels mismatch")
        check(np.isclose(float(labels.mean()), float(ref["prevalence"]), rtol=0, atol=1e-14), f"origin {idx}: prevalence mismatch")
        b1 = frame["b1_probability"].to_numpy(dtype=np.float64).reshape(n_times, GPU_COUNT)
        b2 = frame["b2_probability"].to_numpy(dtype=np.float64).reshape(n_times, GPU_COUNT)
        check(np.isfinite(b1).all() and np.isfinite(b2).all(), f"origin {idx}: nonfinite score")
        check(((b1 >= 0) & (b1 <= 1)).all() and ((b2 >= 0) & (b2 <= 1)).all(), f"origin {idx}: invalid score range")
        group = []
        for j in range(n_times):
            r1, _ = ranks_and_order(b1[j])
            r2, o2 = ranks_and_order(b2[j])
            group.append(Query(int(query_times[j]), idx, b1[j].copy(), b2[j].copy(), r1, r2, o2, labels[j].copy()))
        origin_groups.append(group)
        if idx == 0 or (idx + 1) % 8 == 0 or idx + 1 == len(used_parts):
            print(f"loaded/checked {idx + 1}/{len(used_parts)} origins", flush=True)
    return origin_groups, gpu_ids, used_parts, manifest, reference


def ceiling_rows(group: list[Query], origin_idx: int) -> list[dict]:
    positives = sum(int(q.label.sum()) for q in group)
    rows = []
    for m in M_GRID:
        for n in N_GRID:
            in_union = 0
            oracle = 0
            total_candidates = 0
            for q in group:
                mask = candidate_mask(q, m, n)
                p = int(q.label[mask].sum())
                in_union += p
                oracle += min(K, p)
                total_candidates += int(mask.sum())
            rows.append({"origin_idx": origin_idx, "M": m, "N": n, "positives": positives, "union_positive_gpu_times": in_union, "union_capture": in_union / max(positives, 1), "oracle_top100_hits": oracle, "oracle_top100_recall": oracle / max(positives, 1), "mean_union_size": total_candidates / len(group)})
    return rows


def fit_meta(train_queries: list[Query]) -> object:
    x = np.vstack([features(q) for q in train_queries])
    y = np.concatenate([q.label for q in train_queries]).astype(np.uint8)
    check(len(np.unique(y)) == 2, "Meta training has only one class")
    model = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=100, random_state=SEED))
    model.fit(x, y)
    return model


def predict_meta(model: object, queries: list[Query]) -> list[np.ndarray]:
    x = np.vstack([features(q) for q in queries])
    p = model.predict_proba(x)[:, 1]
    return [row.copy() for row in p.reshape(len(queries), GPU_COUNT)]


def select_parameters(validation: list[Query], risks: list[np.ndarray]) -> tuple[dict, dict]:
    positives = sum(int(q.label.sum()) for q in validation)
    b2_hits = sum(int(q.label[q.order2[:K]].sum()) for q in validation)
    b2_ndcg = [query_ndcg(q, q.order2[:K]) for q in validation]
    b2_ndcg_sum = float(np.nansum(b2_ndcg))
    candidates = []
    for m in M_GRID:
        for n in N_GRID:
            for swaps in SWAP_GRID:
                for ratio in RATIO_GRID:
                    if swaps == 0 and ratio != RATIO_GRID[0]:
                        continue
                    hits = 0
                    ndcg_sum = 0.0
                    changed = 0
                    for q, risk in zip(validation, risks):
                        top, count = proposed_top100(q, risk, m, n, swaps, ratio)
                        hits += int(q.label[top].sum())
                        ndcg_sum += float(np.nan_to_num(query_ndcg(q, top), nan=0.0))
                        changed += count
                    candidates.append({"M": m, "N": n, "max_swaps": swaps, "risk_ratio": ratio, "validation_hits": hits, "validation_ndcg_sum": ndcg_sum, "validation_swaps": changed})
    eligible = [row for row in candidates if row["validation_hits"] > b2_hits and row["validation_ndcg_sum"] >= b2_ndcg_sum - 1e-12]
    if not eligible:
        selected = {"M": 150, "N": 50, "max_swaps": 0, "risk_ratio": 1.0, "validation_hits": b2_hits, "validation_ndcg_sum": b2_ndcg_sum, "validation_swaps": 0}
        reason = "no_past_oof_improvement_with_ndcg_guard"
    else:
        selected = max(eligible, key=lambda row: (row["validation_hits"], row["validation_ndcg_sum"], -row["max_swaps"], -row["M"] - row["N"], row["risk_ratio"]))
        reason = "past_oof_selected"
    audit = {"selection_reason": reason, "validation_positives": positives, "validation_b2_hits": b2_hits, "validation_b2_ndcg_sum": b2_ndcg_sum, "validation_grid_count": len(candidates)}
    return selected, audit


def decide_for_origin(prior: list[Query], current_origin_ns: int, current: list[Query]) -> tuple[dict, list[np.ndarray] | None, dict]:
    cutoff = current_origin_ns - 36 * HOUR_NS
    eligible = [q for q in prior if q.timestamp_ns <= cutoff]
    check(all(q.timestamp_ns + 24 * HOUR_NS <= current_origin_ns for q in eligible), "Immature label entered OOF")
    default = {"M": 150, "N": 50, "max_swaps": 0, "risk_ratio": 1.0}
    base_audit = {"current_origin_ns": current_origin_ns, "feedback_cutoff_ns": cutoff, "eligible_query_count": len(eligible), "last_eligible_ns": max((q.timestamp_ns for q in eligible), default=None)}
    if not eligible:
        return default, None, base_audit | {"selection_reason": "insufficient_past_oof", "train_query_count": 0, "validation_query_count": 0}
    last = eligible[-1].timestamp_ns
    validation = [q for q in eligible if last - VALIDATION_DAYS * DAY_NS < q.timestamp_ns <= last]
    train = [q for q in eligible if last - (TRAIN_DAYS + VALIDATION_DAYS) * DAY_NS < q.timestamp_ns <= last - VALIDATION_DAYS * DAY_NS]
    audit = base_audit | {"train_query_count": len(train), "validation_query_count": len(validation), "train_last_ns": max((q.timestamp_ns for q in train), default=None), "validation_first_ns": min((q.timestamp_ns for q in validation), default=None), "validation_last_ns": max((q.timestamp_ns for q in validation), default=None)}
    if len(train) < MIN_SPLIT_QUERIES or len(validation) < MIN_SPLIT_QUERIES or not any(q.label.any() for q in train):
        return default, None, audit | {"selection_reason": "insufficient_past_oof"}
    check(train[-1].timestamp_ns < validation[0].timestamp_ns <= cutoff, "OOF temporal split violated")
    model = fit_meta(train)
    validation_risks = predict_meta(model, validation)
    selected, selection_audit = select_parameters(validation, validation_risks)
    current_risks = predict_meta(model, current) if selected["max_swaps"] else None
    return selected, current_risks, audit | selection_audit


def evaluate_new(group: list[Query], risks: list[np.ndarray] | None, selection: dict) -> tuple[dict, list[np.ndarray], list[np.ndarray], list[int]]:
    positives = sum(int(q.label.sum()) for q in group)
    hits = 0
    ndcg_values = []
    all_y = []
    all_score = []
    orders = []
    scores = []
    swaps = []
    for j, q in enumerate(group):
        risk = q.b2 if risks is None else risks[j]
        top, count = proposed_top100(q, risk, selection["M"], selection["N"], selection["max_swaps"], selection["risk_ratio"])
        order = full_order(q, top)
        score = np.empty(GPU_COUNT, dtype=np.float64)
        score[order] = q.b2[q.order2]
        hits += int(q.label[top].sum())
        ndcg_values.append(query_ndcg(q, top))
        all_y.append(q.label)
        all_score.append(score)
        orders.append(order)
        scores.append(score)
        swaps.append(count)
    y = np.concatenate(all_y).astype(np.uint8)
    score_flat = np.concatenate(all_score)
    precision = hits / (len(group) * K)
    prevalence = positives / (len(group) * GPU_COUNT)
    metrics = {"model": "CandidateUnion", "decision_time_count": len(group), "gpu_count": GPU_COUNT, "top_k": K, "positive_queries": int(sum(int(q.label.any()) for q in group)), "zero_positive_queries": int(sum(int(not q.label.any()) for q in group)), "positives": positives, "prevalence": prevalence, "hits_at_100": hits, "recall_at_100": hits / max(positives, 1), "precision_at_100": precision, "lift_at_100": precision / max(prevalence, 1e-12), "ndcg_at_100_positive_queries": float(np.nanmean(ndcg_values)) if np.isfinite(ndcg_values).any() else None, "ndcg_sum_positive_queries": float(np.nansum(ndcg_values)), "pr_auc": float(average_precision_score(y, score_flat)) if positives else 0.0, "total_swaps": int(sum(swaps))}
    return metrics, orders, scores, swaps


def write_rank_part(path: Path, group: list[Query], gpu_ids: np.ndarray, state_id: str, orders: list[np.ndarray], scores: list[np.ndarray], swaps: list[int], selection: dict) -> None:
    count = len(group) * GPU_COUNT
    ranks = np.empty((len(group), GPU_COUNT), dtype=np.uint16)
    for j, order in enumerate(orders):
        ranks[j, order] = np.arange(1, GPU_COUNT + 1, dtype=np.uint16)
    frame = pd.DataFrame({"timestamp": pd.to_datetime(np.repeat([q.timestamp_ns for q in group], GPU_COUNT), utc=True), "gpu_uid": np.tile(gpu_ids, len(group)), "model_state_id": [state_id] * count, "b1_probability": np.concatenate([q.b1 for q in group]), "b2_probability": np.concatenate([q.b2 for q in group]), "rerank_score": np.concatenate(scores), "rerank_rank": ranks.ravel(), "in_candidate_union": np.concatenate([candidate_mask(q, selection["M"], selection["N"]) for q in group]), "swaps_at_timestamp": np.repeat(swaps, GPU_COUNT)})
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--smoke-origins", type=int, default=None, help="Read-only 4-origin contract check; writes no results")
    args = parser.parse_args()
    synthetic_test()
    if args.smoke_origins is not None:
        check(args.smoke_origins == 4, "Only the approved 4-origin smoke run is supported")
    output = args.output.resolve()
    if args.smoke_origins is None:
        check(not output.exists(), f"Output already exists: {output}")
    groups, gpu_ids, parts, manifest, original = load_source(args.smoke_origins)
    if args.smoke_origins is not None:
        for idx, group in enumerate(groups):
            ceiling_rows(group, idx)
        print(json.dumps({"status": "SMOKE_OK", "origins": len(groups), "queries": sum(map(len, groups)), "rows": sum(map(len, groups)) * GPU_COUNT}), flush=True)
        return
    check(sum(map(len, groups)) == 3053 and sum(map(len, groups)) * GPU_COUNT == 6081576, "Full score coverage mismatch")
    prior: list[Query] = []
    ceiling: list[dict] = []
    selected_rows: list[dict] = []
    new_rows: list[dict] = []
    output.mkdir(parents=True, exist_ok=False)
    rank_dir = output / "fusion_rank_parts"
    rank_dir.mkdir()
    for idx, group in enumerate(groups):
        origin_ns = group[0].timestamp_ns
        selection, risks, audit = decide_for_origin(prior, origin_ns, group)
        metrics, orders, scores, swaps = evaluate_new(group, risks, selection)
        metrics.update({"origin_idx": idx, "model_state_id": str(original.iloc[idx]["model_state_id"]), "origin_time": str(original.iloc[idx]["origin_time"])})
        new_rows.append(metrics)
        ceiling.extend(ceiling_rows(group, idx))
        selected_rows.append({"origin_idx": idx, "origin_time": str(original.iloc[idx]["origin_time"]), **selection, **audit, "actual_swaps": int(sum(swaps))})
        write_rank_part(rank_dir / f"rank_part_{idx:04d}.parquet", group, gpu_ids, str(original.iloc[idx]["model_state_id"]), orders, scores, swaps, selection)
        prior.extend(group)
        if idx == 0 or (idx + 1) % 8 == 0 or idx == 63:
            print(f"reranked {idx + 1}/64 origins", flush=True)
    existing = pd.read_csv(REFERENCE / "top100_origin_metrics.csv")
    check(len(existing) == 64 * 3, "Reference metrics changed")
    for idx, row in enumerate(new_rows):
        b2 = existing.loc[(existing["origin_idx"] == idx) & (existing["model"] == "B2")].iloc[0]
        check(row["positives"] == int(b2["positives"]) and np.isclose(row["prevalence"], float(b2["prevalence"]), atol=1e-14, rtol=0), f"origin {idx}: reference labels diverged")
        if selected_rows[idx]["max_swaps"] == 0:
            check(row["hits_at_100"] == int(b2["hits_at_100"]), f"origin {idx}: B2 fallback mismatch")
    new = pd.DataFrame(new_rows)
    all_metrics = pd.concat((existing, new), ignore_index=True)
    ceiling_df = pd.DataFrame(ceiling)
    selections = pd.DataFrame(selected_rows)
    b2 = existing.loc[existing["model"] == "B2"].sort_values("origin_idx").reset_index(drop=True)
    pooled_oracle = ceiling_df.groupby(["M", "N"], as_index=False).agg({"positives": "sum", "union_positive_gpu_times": "sum", "oracle_top100_hits": "sum"})
    pooled_oracle["union_capture"] = pooled_oracle["union_positive_gpu_times"] / pooled_oracle["positives"]
    pooled_oracle["oracle_top100_recall"] = pooled_oracle["oracle_top100_hits"] / pooled_oracle["positives"]
    positive_queries = int(new["positive_queries"].sum())
    report = {"status": "EXPLORATORY_COMPLETED", "source_experiment": SOURCE.name, "origin_count": 64, "score_rows": 6081576, "decision_time_count": 3053, "gpu_count": GPU_COUNT, "gpu_time_positives": int(new["positives"].sum()), "positive_queries": positive_queries, "new_model": {"macro_origin_recall_at_100": float(new["recall_at_100"].mean()), "pooled_gpu_time_recall_at_100": float(new["hits_at_100"].sum() / new["positives"].sum()), "macro_origin_ndcg_at_100": float(new["ndcg_at_100_positive_queries"].mean()), "pooled_positive_query_ndcg_at_100": float(new["ndcg_sum_positive_queries"].sum() / positive_queries), "macro_origin_pr_auc": float(new["pr_auc"].mean()), "hits_at_100": int(new["hits_at_100"].sum()), "total_swaps": int(new["total_swaps"].sum()), "active_origins": int((selections["max_swaps"] > 0).sum())}, "b2": {"hits_at_100": int(b2["hits_at_100"].sum()), "pooled_gpu_time_recall_at_100": float(b2["hits_at_100"].sum() / b2["positives"].sum()), "macro_origin_recall_at_100": float(b2["recall_at_100"].mean()), "pooled_positive_query_ndcg_at_100": float(b2["ndcg_sum_positive_queries"].sum() / b2["positive_queries"].sum())}, "pooled_oracle_grid": pooled_oracle.to_dict(orient="records"), "caveat": "Source B1 ADST feedback may be non-causal; this saved-score replay is exploratory, not final prospective evidence."}
    report["new_minus_b2"] = {"pooled_hit_delta": report["new_model"]["hits_at_100"] - report["b2"]["hits_at_100"], "macro_origin_recall_delta": report["new_model"]["macro_origin_recall_at_100"] - report["b2"]["macro_origin_recall_at_100"], "pooled_recall_delta": report["new_model"]["pooled_gpu_time_recall_at_100"] - report["b2"]["pooled_gpu_time_recall_at_100"], "origin_wins": int((new["hits_at_100"].to_numpy() > b2["hits_at_100"].to_numpy()).sum()), "origin_ties": int((new["hits_at_100"].to_numpy() == b2["hits_at_100"].to_numpy()).sum()), "origin_losses": int((new["hits_at_100"].to_numpy() < b2["hits_at_100"].to_numpy()).sum())}
    contract = {"goal_ids": ["G-007", "G-008"], "seed": SEED, "source_contract": {key: manifest[key] for key in ("target", "warmup_days", "purge_hours", "horizon_hours", "feature_cutoff_minutes", "decision_interval_seconds", "gpu_count", "origin_count")}, "selection": {"feedback_cutoff": "score timestamp <= current origin - 36 hours; 24-hour label mature", "split": "last 14 eligible days, earlier 7 train and later 7 validation; each at least 250 queries", "grid": {"M": M_GRID, "N": N_GRID, "max_swaps": SWAP_GRID, "risk_ratio": RATIO_GRID}, "objective": "validation hits@100 strictly above B2 and NDCG sum non-inferior; otherwise K=0", "meta_features": ["B1 rank percentile", "B2 rank percentile", "B1 logit score", "B2 logit score", "rank percentile gap"], "meta_model": "standardized L2 logistic regression", "fallback": "B2 exact when insufficient mature OOF or no validated gain"}, "rank_score": "B2 probability values reassigned by reranked within-query ordering; ranking score, not calibrated probability", "oracle": "union positive coverage and best possible Top-100 hits within union are ex-post diagnostics only", "label_caveat": "GPU-time 24-hour onset labels repeat episodes; not episode recall", "base_score_caveat": report["caveat"], "sha256": {"source_manifest": sha256(SOURCE / "experiment_manifest.json"), "part_manifest": sha256(SOURCE / "full_population_manifest.json"), "reference_metrics": sha256(REFERENCE / "top100_origin_metrics.csv"), "grid_meta": sha256(GRID_META), "onset_ledger": sha256(ONSET_LEDGER), "code": sha256(Path(__file__).resolve()), "score_parts": {path.name: sha256(path) for path in parts}}}
    ceiling_df.to_csv(output / "candidate_ceiling.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(output / "selection_history.csv", index=False, encoding="utf-8-sig")
    all_metrics.to_csv(output / "origin_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "run_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    check(len(list(rank_dir.glob("rank_part_*.parquet"))) == 64, "Not exactly 64 rank parts")
    print(json.dumps({"status": report["status"], "output": str(output), "new_model": report["new_model"], "new_minus_b2": report["new_minus_b2"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

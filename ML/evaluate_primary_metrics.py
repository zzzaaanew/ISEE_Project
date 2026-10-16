"""
Primary Metrics Evaluator (Junho 1st Priority Standard)
======================================================
준호가 정의한 1차 핵심 지표 평가 표준 모듈:
1. Top-100 Hits (hits_at_100)
2. Pooled Recall@100 (pooled_recall_at_100)
3. Pooled Positive-Query NDCG@100 (pooled_positive_query_ndcg_at_100)
4. PR-AUC (Average Precision: Macro-Origin PR-AUC / Pooled PR-AUC)
5. Precision@100 & Lift@100 (Secondary diagnostics)

이 모듈은 로컬 전용이며, git에 커밋되지 않습니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

K = 100
DISCOUNT = 1.0 / np.log2(np.arange(2, K + 2, dtype=np.float64))
IDEAL_DCG = np.r_[0.0, np.cumsum(DISCOUNT)]


def compute_top100_primary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    origin_ids: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """
    준호의 1차 핵심 지표를 정확하게 계산합니다.

    Parameters
    ----------
    scores : np.ndarray
        형태 (n_times, n_gpus)의 예측 확률 또는 위험도 점수.
    labels : np.ndarray
        형태 (n_times, n_gpus)의 이진 정답 라벨 (0 또는 1).
    origin_ids : np.ndarray, optional
        각 시점(time query)의 origin 식별자 (길이 n_times).
        제공될 경우 origin별 Macro PR-AUC 및 Macro 지표를 추가 산출합니다.
    verbose : bool, default=True
        True일 경우 결과를 정돈된 테이블 형태로 콘솔에 출력합니다.

    Returns
    -------
    dict
        준호 기준의 1차 핵심 지표 딕셔너리.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)

    if scores.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"scores and labels must be 2D (n_times, n_gpus), got {scores.shape} and {labels.shape}")
    if scores.shape != labels.shape:
        raise ValueError(f"Shape mismatch: scores {scores.shape} vs labels {labels.shape}")

    n_times, n_gpus = scores.shape
    positives = int(labels.sum())
    total_evals = n_times * n_gpus
    prevalence = positives / max(total_evals, 1)

    # 1. Stable descending ranking
    order = np.argsort(-scores, axis=1, kind="stable")
    top_indices = order[:, :K]
    top_labels = np.take_along_axis(labels, top_indices, axis=1)

    # 2. Hits & Precision & Recall
    hits = int(top_labels.sum())
    precision = hits / (n_times * K)
    recall = hits / max(positives, 1)
    lift = precision / max(prevalence, 1e-12)

    # 3. NDCG@100 (Positive query only)
    positive_per_query = labels.sum(axis=1).astype(np.int32)
    positive_queries = positive_per_query > 0
    total_pos_queries = int(positive_queries.sum())

    dcg = top_labels @ DISCOUNT
    ndcg = np.full(n_times, np.nan, dtype=np.float64)
    if total_pos_queries > 0:
        ideal_caps = np.minimum(positive_per_query[positive_queries], K)
        ndcg[positive_queries] = dcg[positive_queries] / IDEAL_DCG[ideal_caps]
        pooled_ndcg = float(np.nansum(ndcg[positive_queries]) / total_pos_queries)
    else:
        pooled_ndcg = 0.0

    # 4. PR-AUC (Average Precision)
    y_flat = labels.ravel().astype(np.uint8)
    s_flat = scores.ravel()
    pooled_pr_auc = float(average_precision_score(y_flat, s_flat)) if positives > 0 else 0.0

    # 5. Origin-level Macro metrics (if provided)
    macro_pr_auc = pooled_pr_auc
    macro_recall = recall
    macro_ndcg = pooled_ndcg
    if origin_ids is not None:
        origin_ids = np.asarray(origin_ids)
        if len(origin_ids) == n_times:
            unique_origins = np.unique(origin_ids)
            origin_prs = []
            origin_recalls = []
            origin_ndcgs = []
            for org in unique_origins:
                mask = (origin_ids == org)
                org_labels = labels[mask]
                org_scores = scores[mask]
                org_pos = int(org_labels.sum())
                if org_pos > 0:
                    org_pr = float(average_precision_score(org_labels.ravel().astype(np.uint8), org_scores.ravel()))
                    origin_prs.append(org_pr)
                    
                    org_order = np.argsort(-org_scores, axis=1, kind="stable")
                    org_top_labels = np.take_along_axis(org_labels, org_order[:, :K], axis=1)
                    org_hits = int(org_top_labels.sum())
                    origin_recalls.append(org_hits / org_pos)

                    org_pos_q = org_labels.sum(axis=1) > 0
                    if np.any(org_pos_q):
                        org_dcg = org_top_labels @ DISCOUNT
                        org_ideal = IDEAL_DCG[np.minimum(org_labels.sum(axis=1)[org_pos_q], K)]
                        origin_ndcgs.append(float(np.mean(org_dcg[org_pos_q] / org_ideal)))

            if origin_prs:
                macro_pr_auc = float(np.mean(origin_prs))
            if origin_recalls:
                macro_recall = float(np.mean(origin_recalls))
            if origin_ndcgs:
                macro_ndcg = float(np.mean(origin_ndcgs))

    results = {
        "hits_at_100": hits,
        "pooled_recall_at_100": float(recall),
        "pooled_positive_query_ndcg_at_100": float(pooled_ndcg),
        "origin_mean_pr_auc": float(macro_pr_auc),
        "pooled_pr_auc": float(pooled_pr_auc),
        "precision_at_100": float(precision),
        "lift_at_100": float(lift),
        "evaluated_positives": positives,
        "evaluated_decision_times": n_times,
        "positive_queries": total_pos_queries,
        "prevalence": float(prevalence),
    }

    if verbose:
        print_primary_metrics_card(results)

    return results


def print_primary_metrics_card(m: dict, title: str = "Model Primary Evaluation (Junho Standard)") -> None:
    """터미널에 깔끔한 성능 리포트 카드를 출력합니다 (cp949 호환)."""
    border = "=" * 65
    print(f"\n{border}")
    print(f"  [REPORT] {title}")
    print(border)
    print(f"  [1차 핵심 지표 (Primary Metrics)]")
    print(f"  * Top-100 Hits         : {m['hits_at_100']:,} 건 (전체 {m['evaluated_positives']:,}건 중)")
    print(f"  * Pooled Recall@100    : {m['pooled_recall_at_100']:.6f} ({m['pooled_recall_at_100']*100:.2f}%)")
    print(f"  * Positive NDCG@100    : {m['pooled_positive_query_ndcg_at_100']:.6f}")
    print(f"  * Origin Mean PR-AUC   : {m['origin_mean_pr_auc']:.6f}")
    print(f"  ---------------------------------------------------------------")
    print(f"  [운영 진단 보조 지표 (Diagnostics)]")
    print(f"  * Precision@100        : {m['precision_at_100']:.6f}")
    print(f"  * Lift@100             : {m['lift_at_100']:.3f}x (자연 유병률 대비)")
    print(f"  * Pooled PR-AUC        : {m['pooled_pr_auc']:.6f}")
    print(f"  * 평가 규모            : {m['evaluated_decision_times']:,} 시점, 양성 시점 {m['positive_queries']:,}개")
    print(f"{border}\n")


def evaluate_dataframe_top100(
    df: pd.DataFrame,
    score_col: str = "probability",
    label_col: str = "target_24h",
    time_col: str = "timestamp",
    gpu_col: str = "gpu_id",
    origin_col: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Pandas DataFrame 형태의 예측 결과에서 준호 핵심 지표를 계산합니다.
    """
    times = df[time_col].unique()
    times.sort()
    gpus = df[gpu_col].unique()
    n_times, n_gpus = len(times), len(gpus)

    # Pivot to (n_times, n_gpus)
    pivot_score = df.pivot(index=time_col, columns=gpu_col, values=score_col).reindex(index=times, columns=gpus)
    pivot_label = df.pivot(index=time_col, columns=gpu_col, values=label_col).reindex(index=times, columns=gpus).fillna(0)

    origin_ids = None
    if origin_col and origin_col in df.columns:
        origin_map = df.drop_duplicates(time_col).set_index(time_col)[origin_col]
        origin_ids = origin_map.reindex(times).to_numpy()

    return compute_top100_primary_metrics(
        scores=pivot_score.to_numpy(),
        labels=pivot_label.to_numpy(),
        origin_ids=origin_ids,
        verbose=verbose,
    )

# All-XID diversified Branch 1 + History-only Branch 2 Fusion 보고서

## Material Passport
- **Status**: COMPLETED
- **Runner**: `run_fusion_diversified_b1_history_b2.py`
- **Target**: All-XID unified onset, 24-hour horizon
- **Evaluation**: full GPU-time score matrices, then Fusion, then Top-100 tape
- **Seed**: `20260905`

## 실험 계약
- 시간순 Sliding Training, 36시간 purge, 공통 terminal held-out 20%
- Branch 1: 1h·6h·24h telemetry, 4-family diversified ensemble, soft cascade
- Branch 2: History-only `xid_count_30d`, `days_since_xid`, historical logistic
- Lambda: paired pooled Validation PR-AUC 선택, held-out label 미사용
- Top-100: Fusion score로 전체 GPU를 정렬한 후 저장

## Terminal held-out 평균
- **Fusion PR-AUC**: `0.025036`
- **Branch 1 cascade PR-AUC**: `0.025036`
- **Branch 2 History-only PR-AUC**: `0.061371`
- **Fusion ROC-AUC**: `0.689027`
- **Fusion Recall@100**: `12.13%`
- **Fusion Lift@100**: `2.417x`
- **Selected Lambda**: `1.0`
- **Evaluated GPUs per decision time**: `1992`
- **Held-out cycles**: `13`

## 해석 경계
- PR-AUC·ROC-AUC는 Top-100으로 자르기 전 전체 GPU-time score를 사용했다.
- Risk tape는 Fusion rank 기준 Top-100만 저장한 운영 후보 목록이다.
- Lambda가 0 또는 1이면 해당 결과는 사실상 한 Branch 중심이며, Fusion synergy가 확인된 것으로 해석하지 않는다.

## 산출물
- fusion_metrics.csv
- fusion_risk_tape.parquet
- fusion_selection_history.csv
- experiment_manifest.json
- checkpoints/
# [ADST v2] All-XID Branch-specific Recency + ADWIN 보고서

## Material Passport
- **Material type**: Code experiment result and reproducibility record
- **Status**: COMPLETED
- **Runner**: `run_sliding_adst_v2_branch_specific.py`
- **Target**: All-XID unified onset, 24-hour horizon
- **Seed**: `20260905`
- **Terminal held-out fraction**: `20%`

## 1. 실험 계약
- Branch 1: 1h·6h·24h telemetry 입력을 모두 유지하는 parallel ensemble
- Branch 2: `xid_count_30d`, `days_since_xid`만 사용하는 strict History-only
- Train·Validation·Test: 시간순, Train–Validation 및 Validation–Test 사이 36시간 purge
- Validation: 3개의 rolling 3일 block, PR-AUC 평균을 1순위로 하고 median·Q25·변동성을 tie-break
- Recency: Branch별 half-life 후보 3일·7일·14일
- ADWIN: validation loss, prediction disagreement, positive prevalence를 개발 구간에서만 감시
- Fusion: `p = lambda*p_B1 + (1-lambda)*p_B2`, Branch별 선택 완료 후 Lambda 선택
- Test: terminal held-out label은 선택·ADWIN 상태 갱신에 사용하지 않음

## 2. Terminal held-out 평균 지표
- **Fused PR-AUC**: `0.015348`
- **Branch 1 PR-AUC**: `0.022795`
- **Branch 2 PR-AUC**: `0.013600`
- **Fused normalized PR-AUC skill**: `0.004525`
- **B2 positive-weighted PR-AUC**: `0.027009`
- **Fused ROC-AUC**: `0.490635`
- **Recall@100**: `10.46%`
- **Lift@100**: `2.084x`

## 3. Branch 2 성능 하락 진단
- 현재 결과는 이전 GitHub 원형 결과와 feature·재학습·split 계약이 달라 직접적인 성능 하락으로 단정하지 않는다.
- 본 실험에서는 Branch 2 입력을 History-only로 고정해 calendar/telemetry 신호가 섞이지 않도록 했다.
- 성능 해석은 macro PR-AUC, positive-weighted PR-AUC, 양성률별 결과를 함께 사용한다.
- ADWIN은 terminal test 결과를 보고 window를 사후 조정하지 않는다.

## 4. 양성률 구간별 Branch 2
- `low_prevalence`: cycles=5, prevalence=0.0024, B2 PR-AUC=0.007954, B2 normalized=0.005598
- `mid_prevalence`: cycles=4, prevalence=0.0068, B2 PR-AUC=0.005903, B2 normalized=-0.000880
- `high_prevalence`: cycles=4, prevalence=0.0258, B2 PR-AUC=0.028356, B2 normalized=0.002678

## 5. ADWIN 및 선택 이력
- ADWIN trigger 횟수: `0` cycle에 해당하는 고정 선택 상태 기록
- 선택 stage 수: `2`
- `selection_history.csv`에는 Branch 1·Branch 2 후보, Lambda 후보, block별 집계 기준을 남긴다.

## 6. 해석 제한
- ADST v2의 원인별 효과는 현재 v2 결과 하나만으로 분리되지 않으므로, 기존 커밋 버전과 동일 held-out test에서 ablation 비교가 필요하다.
- fixed terminal benchmark와 향후 adaptive prequential 평가를 혼합하지 않는다.
- **Split**: `2023-08-05T06:45:00+00:00` ~ `2023-08-18T00:00:00+00:00`
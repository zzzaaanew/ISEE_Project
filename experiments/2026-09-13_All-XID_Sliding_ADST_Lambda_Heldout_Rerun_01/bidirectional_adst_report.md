# [Sliding ADST + Lambda Fusion] All-XID 실험 보고서

## 1. 실험 계약
- **타깃**: All-XID 통합 onset, 향후 24시간 horizon
- **Branch 1**: GitHub 이식본 telemetry 입력과 병렬 앙상블 유지
- **Branch 2**: `xid_count_30d`, `days_since_xid`만 사용하는 History-only
- **전처리**: 오류 직전 10분 buffer 제외
- **Purge**: Train–Validation 및 Validation–held-out Test 사이 36시간
- **Held-out Test**: 시간순 terminal 20% 공통 구간 (2023-08-05T06:45:00+00:00 이후)
- **ADST 후보**: L_train=[7, 14, 21]일, L_obs=[1, 6, 24]시간
- **Fusion**: `p = lambda * p_B1 + (1-lambda) * p_B2`, lambda를 Validation PR-AUC로 선택

## 2. Held-out Test 평균 지표
- **Fused PR-AUC**: `0.036928` (B1 `0.023381`, B2 `0.033162`)
- **Fused ROC-AUC**: `0.679305`
- **Recall@100**: `12.56%`
- **Lift@100**: `2.503x`

## 3. Validation 기반 적응 선택 기록
- **최종 held-out preselection Lambda**: {0.6: 1}
- **최종 선택 window 조합**: {(21.0, 6.0): 1}
- **Fused > B1인 cycle**: 6/13
- **Fused > B2인 cycle**: 10/13
- **Fused > 두 Branch 중 최선인 cycle**: 3/13

## 4. 해석 주의사항
- 모든 cycle은 동일한 terminal held-out 구간 안에서 평가하며, Test 정답은 window·Lambda 선택에 사용하지 않는다.
- Fused가 평균적으로 높더라도 모든 cycle에서 최선 Branch를 이긴다고 가정하지 않는다.
- checkpoint에는 selection과 cycle별 metrics/risk tape가 저장되어 중단 후 완료 구간을 건너뛸 수 있다.
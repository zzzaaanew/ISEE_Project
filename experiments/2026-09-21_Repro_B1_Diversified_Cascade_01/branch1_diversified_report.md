# Branch 1 diversified parallel + cascade 실험 보고서

## 실험 계약
- All-XID 통합 onset, 24시간 horizon, 오류 직전 10분 제외
- 시간순 Sliding Training, 36시간 purge, 최근 development origin pooled validation
- Branch 1 telemetry 입력 1시간·6시간·24시간 모두 사용
- Logistic, Extra Trees, HistGradientBoosting, TemporalCNN1D 병렬 앙상블
- prediction disagreement·uncertainty 기반 soft residual cascade
- Branch 2와 Lambda fusion은 실행하지 않음

## Terminal held-out 평균
- Parallel PR-AUC: 0.022790
- Cascade PR-AUC: 0.025036
- Parallel median PR-AUC: 0.006147
- Cascade median PR-AUC: 0.010958
- Parallel normalized PR-AUC: 0.012204
- Cascade normalized PR-AUC: 0.014407
- Parallel ROC-AUC: 0.673679
- Cascade ROC-AUC: 0.689027
- Cascade가 Parallel보다 높은 cycle: 9/13

## 기준선
- 동일 terminal held-out Branch 1 v1 PR-AUC: 0.023381
- GitHub-base v2.1 Branch 1 PR-AUC: 0.022086
- Cascade가 v1보다 높은지: True

## 산출물
- branch1_diversified_metrics.csv
- branch1_diversified_risk_tape.parquet
- branch1_diversified_selection_history.csv
- experiment_manifest.json
- checkpoints/
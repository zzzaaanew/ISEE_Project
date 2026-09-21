# All-XID Report-faithful Pareto Fusion + Momentum Skip-Retrain

## 실험 계약
- All-XID `[13, 31, 43, 45, 94]`, 24시간 horizon, 오류 직전 10분 제외
- 시간순 Sliding Training, 36시간 purge, 공통 terminal held-out test
- Branch 1: Parallel Ensemble + soft residual Cascade 유지
- Branch 2: 1시간 telemetry 요약값을 제외한 strict History-only
- 전체 GPU-time 점수 → Pareto Fusion → Top-100 risk tape

## 적용한 보고서 방식
- Pareto: `w_rep + (w_clean-w_rep)*(1/(1+count_30d))^alpha`; primary=`wc0.50_wr0.02_a1.0`
- clean GPU는 B1 비중이 높고, 반복 XID GPU는 History-only B2 비중이 높아진다.
- Momentum: beta=.70, pair confidence>=.65이면 9-grid 대신 selected pair 1회 quick validation
- quick validation PR-AUC가 직전 대비 15% 이상 하락하면 full 9-grid rescan
- confidence/momentum/skip count는 checkpoints/momentum_state.json에 저장되어 resume 후 복원된다.

## Terminal held-out 평균 (primary)
- Fusion PR-AUC: `0.020614`
- Branch 1 Cascade PR-AUC: `0.020299`
- Branch 2 History-only PR-AUC: `0.061371`
- Fusion ROC-AUC: `0.693419`
- Recall@100: `6.95%`
- Lift@100: `1.384x`
- Mean Pareto B1 weight: `0.4524`

## Pareto sensitivity grid
- `wc0.30_wr0.05_a1.0`: PR-AUC `0.021346`, ROC-AUC `0.695810`, mean B1 weight `0.2752`
- `wc0.30_wr0.02_a1.0`: PR-AUC `0.021307`, ROC-AUC `0.695887`, mean B1 weight `0.2723`
- `wc0.30_wr0.05_a1.5`: PR-AUC `0.021255`, ROC-AUC `0.696285`, mean B1 weight `0.2695`
- `wc0.30_wr0.02_a1.5`: PR-AUC `0.021193`, ROC-AUC `0.696018`, mean B1 weight `0.2659`
- `wc0.50_wr0.05_a1.0`: PR-AUC `0.020620`, ROC-AUC `0.693320`, mean B1 weight `0.4554`
- `wc0.50_wr0.02_a1.0`: PR-AUC `0.020614`, ROC-AUC `0.693419`, mean B1 weight `0.4524`
- `wc0.50_wr0.05_a1.5`: PR-AUC `0.020555`, ROC-AUC `0.693762`, mean B1 weight `0.4451`
- `wc0.50_wr0.02_a1.5`: PR-AUC `0.020529`, ROC-AUC `0.693638`, mean B1 weight `0.4415`

## 산출물
- fusion_metrics.csv (primary variant)
- fusion_variant_metrics.csv (approved sensitivity grid)
- fusion_risk_tape.parquet (primary variant, full GPU before Top-100)
- momentum_selection_trace.csv
- checkpoints/momentum_state.json
- experiment_manifest.json

정확한 보고서 원본 commit의 w_clean/w_rep 값은 현재 원격 main에서 복구되지 않아, 승인된 2×2×2 sensitivity grid로 대체 검증했다.

## Platt Calibration
- Calibrator: branch-specific Platt logistic sigmoid.
- Fit scope: pooled development Validation/OOF only; Held-Out Test was not used.
- Held-Out raw Fusion PR-AUC: `0.022981`
- Held-Out Platt Fusion PR-AUC: `0.020614`
- Held-Out raw Fusion ROC-AUC: `0.699862`
- Held-Out Platt Fusion ROC-AUC: `0.693419`
- `fusion_calibration_metrics.csv` contains the aligned raw-vs-Platt cycle comparison.

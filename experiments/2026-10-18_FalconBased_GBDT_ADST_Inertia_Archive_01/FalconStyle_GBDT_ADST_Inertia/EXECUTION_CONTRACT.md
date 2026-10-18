# 명세 v1.1 실행 계약

사용자 승인: 명세 수정 후 “실행해보자”. 관련 목표 G-004·G-007·G-008.

- 원본 Parquet 5개를 SHA256으로 확인하고, 기존 cache를 verified Parquet cache와 대조한다. 기존 모델 점수는 학습에 쓰지 않는다.
- GitHub 신규 EnhancedTelemetryEngine의 실제 45열 호출, CUDA XGBoost150trees, Logistic ST21, B1 ST7/14/21과 성숙피드백 ADST를 새 학습한다.
- 36h purge, All-XID24h inclusive GT, 실제 최신 telemetry lag15min/명목 차단10min, 매일 재학습, 전체1992GPU/30분/64origins를 적용한다.
- B1 logit 지연 EWMA와 KL 추론 결합의 유무, 상한 없는 Pareto와 E19 기반 residual Fusion을 교차한16개 조건이다. 원본 E19의 별도 관성 항을 재추가하지 않는다.
- 정확히 같은 raw 모델은 학습 정책·Inertia·Fusion 간 공유한다. 최대 B1 192회 및 B2 64회 fit이며 E19 calibration/residual은 별도 학습한다.
- 실행 경로는 ML/run_experiment.py. 이전 단일 모델 Falcon 실험이 종료된 뒤 GPU 순차 실행한다. 기존 실행을 중단하지 않는다.
- 1-origin 실제 학습 smoke 및 합성 과거 이력으로 E19 trained 경로를 검증한 뒤64origin을 지속한다. smoke 결과를 전체결과로 보고하지 않는다.
- 실행 상한12시간(신규 학습 시작부터), 프로세스 생존/출력 상태/CPU·GPU backend 증거를 남긴다. timeout 시 소유한 신규 실험 프로세스만 종료한다. 실패를 자동으로 재시도하지 않는다.
- 산출물: config/vendor/hash, fit_trace, branches,16arm score/rank,ADST history,checkpoint,metrics,summary,internal/independent audit,누적 연대기MD/HTML.
- 기존 E14/E19 재학습·BLOX·환경 패키지 추가·원본 수정·Git commit/push는 포함하지 않는다.

실행은 기존 Falcon Python3.12 환경을 재사용한다. 특징 추출을 위해 original base UnifiedDataEngine 클래스만 AST로 로드하며 신규 EnhancedTelemetryEngine의 특징 메서드는 원문 그대로 실행한다. E19의4개 수학 함수도 원문 AST로 로드한다. CUDA GBDT는 GPU, LR/특징/지연상태/소규모 SciPy Fusion은 CPU2threads이며 실제 구간별 시간으로 보고한다.

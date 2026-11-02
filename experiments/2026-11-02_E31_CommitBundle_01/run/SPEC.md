# E31 실행 계약 — 2026-11-02 승인

목표 G-004·G-007·G-008. 사용자의 「그럼 진행하자」는 기존 5항목 계획 및 후속 관성/Fusion/5분 설명을 승인했다. 초록에는 ML 수치를 사용하고 E30 운영 최적화는 이번 실행에 포함하지 않는다.

- A 기본30/ST21, B Falcon-inspired45/ST21, C 기본30/ADST, D Falcon-inspired45/ADST. B2는 네 조건 공통 history/context LR+GBDT ensemble/ST21. 원형 Falcon 재현 및 동일 task peer 특징으로 주장하지 않는다.
- 원본 5분 telemetry Parquet 4개와 All-XID onset metadata에서 새 학습. 입력·소스 SHA256 기록. 예전 저장 점수·이전 L_obs 선택 CSV는 입력으로 사용하지 않는다.
- B1 HGB 및 B2 LR/HGB의 10/26 파라미터 유지. 6시간 특징, lag3 마지막 5분 bin(10분 차단), 24시간 예측 `(t,t+24h]`. 구 코드의 t=onset 양성 포함을 제거하며 exact duplicate onset을 제거한다. 역사적 수치와 동조건 재현이 아니다.
- 학습 구간은 `[origin−36h−L_days, origin−36h)`, L=21 또는7/14/21. 실제 달력 학습창 길이를 유지한다. 초기14, AP 하락15%, 안정3회인 E11 상태전이를 원본 AST로 호출한다. E14와 같이 사이클 마지막 점수가36h 성숙한 뒤에만 피드백. 피드백 통계는 원형과 동일하게 30분 full-GPU AP의 평균이며 모델 점수는5분마다 생성한다.
- 24h마다 재학습, 5분마다 전 GPU 실제 추론. Branch별 원점수와 origin·시각·GPU·30일 XID 횟수를 먼저 Parquet에 저장한다. B2는 매 origin 1번(두 estimator)만 학습해 공유한다. 점수 선별 후 Fusion하지 않는다.
- 원본 XID 기록 Time 열 전체를 확인하여 종료 시점을 정한다. 발견된11개 전역 기록 공백과 미래24h가 겹치거나 관측 범위 밖인 라벨은 모든 arm의 학습·피드백·평가에서 공통 제외한다. 해당 시점 점수 산출 자체는 유지한다. 개별 GPU의 nonmissing XID 관측 완전성은 아직 인증되지 않았으며 기존 ledger의 한계로 남긴다.
- 마지막 유효 종료 시점에서14일 달력 구간을 terminal로 분리한다. 공백 때문에 14일 모든 점수가 평가 가능한 것은 아니다. 과거 열람 이력이 있는 retrospective terminal prequential 평가이며 새 미관측 데이터가 아니다. 정해진 학습/ADST 규칙의 성숙 후 온라인 업데이트는 허용하되 terminal 성능으로 설정을 선택하지 않는다.
- 주 비교 관성 B1만 gamma1/반감기2h; B2 관성0. 이전 raw logit EMA를 현재 보정에 사용한 뒤 갱신한다. 사이클·terminal 경계에서 상태를 초기화하지 않는다. Fusion lambda=min(.6,1/(k+1)), alpha1, k는 origin−10분 이전 최근30일의 고유 GPU-onset 시각 개수. k는 일별 유지한다. cap은 연구 가정이며 최적해라고 부르지 않는다.
- 관성없는 B1/B2/관성B1/Fusion관성없음/Fusion주설정을 비교한다. 개발 민감도는 D에서 cap .5/.6/.7/.8/1 및 gamma/반감기 (.25,.5),(.5,1),(1,2)와관성없음을 보고한다. 주 비교 설정은 사전 고정하며 민감도 결과로 terminal 설정을 바꾸지 않는다.
- 30분 평가는 동일 Branch 원점수를6개중1개 추출한 다음30분 감쇠로 관성·Fusion을 별도 적용한다. 5분 Fusion 단순추출/30분 점수 유지 보간과 구별한다.
- 주지표는 eligible origin block별 average_precision_score 평균(AP). pooled R@100, origin block평균 ROC-AUC, 연속 유효시점의 시간당 Top100 신규진입, 완전한24h~10min 경고창을 갖는 고유GPU/onset-time 포착률을 분리한다. 부분 origin block도 별도 행과 길이를 남긴다. onset은 session episode가 아니다. 전체기간 집계는 terminal을 포함하므로 두번째 독립 test가 아니다.
- balanced class weight를 유지하므로 odds prior 복원의 sample prior는 유효0.5로 수정한다(기존 비가중 표본 prior 적용 오류). 이는 경험적 확률 calibration 인증이 아니며 raw/risk score로 해석한다. 네 arm에 동일 적용한다.
- 메모리 제한으로 원본 FastGridFeatureStore를 작은 시간 chunk에 그대로 호출해 disk-backed53개 합집합 특징과5개B2 특징을 만든다. 특징정의 변경 없이 원형 전체grid와 chunk결과 일치검사를 한다. 원본 source는 수정하지 않는다.
- seed42, 기존 e19-reproduction venv, CPU2threads. 기존 결과·입력·원본코드 보존, 설치·E30·커밋·푸시 제외. 자원 부족·누수·입력 계약 실패시 중단하고 오류/부분산출물을 보존한다. 기초검사 후1origin pilot, 이어 전체기간 실행과 사후평가까지 수행한다.

진입점: `run_experiment.py prepare`, `pilot`, `branches`, `evaluate`. `all`은 준비 후 전체 학습·평가를 연결한다. 체크포인트에 완료 origin과 controller/pending feedback을 저장한다. 미완료 파일이 있으면 자동 덮어쓰기하지 않는다.

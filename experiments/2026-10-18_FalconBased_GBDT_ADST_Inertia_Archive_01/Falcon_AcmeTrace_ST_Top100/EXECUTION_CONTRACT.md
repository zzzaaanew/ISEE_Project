# Falcon-AcmeTrace-ST-Top100 실행 계약

사용자 승인: 2026-10-17, 앞선 설계에 대한 "실행해봐". 관련 G-004/G-007/G-008.

## 범위

격리 환경 준비, 실제 GPU backend 확인, frozen 5분 Parquet에서 새 특징 생성,
과거 validation의 후보 선택 및 첫 origin 새 ST 학습/전체 GPU scoring을 실행한다.
64-origin 실행은 첫 origin의 시간·자원·결과를 확인한 후 판단한다.
기존 E14/E19 실행은 재개하지 않으며 기존 소스/데이터/환경은 보존한다.
Git commit/push, 외부 데이터 업로드, 다른 프로그램 종료는 하지 않는다.

## 원문과 이식

원문: https://arxiv.org/html/2609.34473v1 (III.B–D, IV).
이식 구조: DenseTelemetryEncoder(132) + XIDHazardEncoder(11) → concatenate(143)
→ 과거 validation에서 선정한 트리 앙상블 1개 → 21일 Sliding Training
→ 모든 1,992 GPU의 점수 → stable Top100.
두 encoder는 특징 생성 모듈이며 독립 위험 모델 또는 score Fusion이 아니다.

원문과 다른 조건:

- 고장별 ticket 대신 All-XID 24시간 inclusive-bin 라벨.
- 1분 원자료 대신 frozen 5분 평균 시계열. std/min/max는 해당 5분 평균들의 시간창 통계다.
- 21일 ST·24시간 재학습·30분 scoring·36시간 purge·직전10분 제외.
- 처음 평가하기 전 성숙한 3일 validation의 AP로 후보 선택 후 모델 family/설정 고정.
- 동일 host/task peer, ECC/remap, typed-fault learner, Multi-Horizon 결합 없음.
- 임계값이 불명확한 persistence 특징을 임의 proxy로 대체하지 않음.
- EventPolicy kernel만 구현. 경고 정책 튜닝·사건 평가를 이번 ML pilot에서 실행하지 않음.
- 논문 정확 hyperparameter는 미확정이며 이번 preset을 원문 값이라고 표시하지 않음.

## 입력과 시간 계약

config.json의 5개 Parquet를 data_inventory.json SHA256과 대조한다.
grid_meta.npz의 GPU IDs/5분 grid를 기준으로 새 mean/obs 캐시를 생성한다.
GT는 기존 E19 평가기의 onset_bin-288..onset_bin inclusive 규칙을 유지한다.
telemetry 최신 bin은 decision_bin-3, XID cutoff는 decision_time-10분 inclusive다.
보간은 각 허용된 과거 창 내부에서만 수행한다. 끝의 결측은 과거 가장 가까운 관측으로 처리한다.
나머지 NaN의 median imputer는 각 fit의 학습 데이터에만 fit한다.
*_obs는 제공 값의 평균이며 원시 관측 횟수로 환산하지 않는다.

원본 E19 score 파일은 평가 일정(timestamp)과 새 Falcon 결과 이후의 대조에만 사용한다.
그 확률/관성 열은 학습·특징·모델 선택 입력에 포함하지 않는다.

## 학습 preset

- XGBoost: CUDA hist, 300 trees, depth6, learning_rate0.05, max_bin64, subsample0.8.
- CatBoost: GPU, 300 iterations, depth6, learning_rate0.05, Logloss.
- LightGBM: OpenCL GPU, 300 trees,31 leaves,learning_rate0.05,max_bin63,min_child_samples50.
- Random Forest: CPU,200 trees,max_depth16,min_samples_leaf5.
- Isolation Forest: CPU,200 trees,max_samples256,healthy sampled rows only.
- 각 family는 seed20260905,CPU threads2. RF/IF를 다른 GPU 알고리즘으로 교체하지 않는다.
- 모든 positive 5분 rows + min(positive×10,200000) negative samples.
  분류기에는 negative inverse-sampling weights를 적용하고 별도 balanced weight를 중복 적용하지 않는다.
- 후보 validation 및 최종 scoring은 전체 population이다.

## 증거와 중단

versions/environment_preflight.json: synthetic backend 확인이며 원데이터 pilot 결과가 아니다.
runs/pilot_01: execution_contract,feature_manifest,trace,status,selected_learner,
전체 score part,model checkpoint,metrics,summary를 보존한다.
trace의 각 candidate fit→validation→선택된 모델 최종 fit→전체 scoring을 확인한다.
GPU fit은 XGBoost save_config 등으로 확인하며 desktop GPU 사용률을 ML 사용률로 귀속하지 않는다.
GPU99–100%·약2.7GB/3GB 경쟁이 관찰되어 시간은 공유 GPU 환경의 실측으로 보고한다.
pilot hard timeout1800초, 입력/시간 계약 오류나 OOM은 실패 증거를 보존한다.

## 결과 판정

첫 origin 성공은 STANDALONE_PILOT이며 전체 Falcon 재현 완료가 아니다.
전체 실행은64 origins/3,053시점/6,081,576 rows 및 같은 label/keys를 확인해야 한다.
AP는 origin 전체 GPU-time rows,Recall@100은 GPU-time 라벨 기준이다.
새 기간 일반화나 원본 Falcon 우월성은 이번 개발 기간 비교에서 주장하지 않는다.

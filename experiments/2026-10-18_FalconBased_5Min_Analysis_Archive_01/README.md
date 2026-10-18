# Falcon 기반 5분 전체 결과 및 30분 대조 분석

2026-10-18에 완료한 **B1 ST21 + Inertia**의 64-origin 전체 결과다. 일별 GBDT/LR 모델은 고정하고, 5분마다 특징·과거 상태·점수·Top-100을 새로 계산했다. 최종 순위는 B1 점수 기준이며, 5분 Dual-Branch/Fusion 전체 성능을 주장하는 자료가 아니다.

기존 [실행 코드·검증 아카이브](../2026-10-18_FalconBased_GBDT_ADST_Inertia_Archive_01/)는 그대로 보존한다. 이 폴더는 그 이후의 **공통 시점·점수 유지 대조·onset·추천 교체량·Inertia 제거 분석**을 새로 기록한다.

## 1. 검증 범위와 정답

- `FIVE_MINUTE_COMPLETED_AND_AUDITED`; 독립 감사 `FIVE_MINUTE_SCORE_ARTIFACTS_VERIFIED`.
- 64/64 origins, 1,992 GPU, 18,313 시점, 36,479,496 GPU-time 점수.
- 양성 317,209 GPU-time; All-XID 향후 24시간 inclusive-bin 정답, 양성률 약 0.870%.
- 36h purge, 명목 10분 feature cutoff/실제 telemetry 15분 lag, 일별 모델 재사용, Inertia 반감기 2h 유지.
- 이번 분석의 모델 학습 횟수는 0이다. 원시 점수와 모델을 수정하지 않고 저장 결과를 재계산했다.
- AP는 origin별 average precision의 평균(양성 없는 origin은 0), Recall@100은 전체 양성 GPU-time을 합친 pooled 지표다. GPU-time 적중은 고유 onset/하드웨어 고장 수와 다르다.

## 2. 기본 결과: 평가 주기를 명시

| 평가 | 평균 AP | Recall@100 | Precision@100 | 양성 적중 | 시점 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 30분 | 0.022381456 | 11.6914% | 2.0275% | 6,190 | 3,053 |
| 새 5분 전체 | 0.022004021 | 11.8316% | 2.0494% | 37,531 | 18,313 |

평가 시점·양성 분모가 달라 적중 수 약 6배를 예측 성능 6배 향상으로 해석하지 않는다. 평균 AP는 상대 1.69% 낮고 Recall은 0.14%p 높다.

## 3. 동일 시점 대조

### 3.1 같은 5분 평가 시점에서 비교

`30m_held_on5`는 가장 최근에 사용 가능해진 30분 점수를 다음 갱신까지 유지한다. 미래 점수를 가져오지 않는다. 새 5분 점수와 같은 GPU·시점·정답·Top-100 예산으로 평가한다.

| 방식 | 평균 AP | Recall@100 | 양성 적중 |
| --- | ---: | ---: | ---: |
| 30분 점수 유지 | 0.022213278 | 11.6059% | 36,815 |
| 새 5분 예측 | 0.022004021 | 11.8316% | 37,531 |

5분 갱신은 **716개 양성 GPU-time 추가 포착**, Recall +0.2257%p, AP -0.94%다. 7-origin moving-block bootstrap AP 차이 95% 구간은 [-0.001063, +0.000486], Recall 차이는 [-0.0811, +0.4276]%p로 모두 0을 포함한다. 보편적 우위/동등성 검정으로 해석하지 않는다.

### 3.2 원래 30분 시점만 추출

- 새 5분 실행에서 공통 3,053개 시점만 추출하면 AP 0.022298597, Recall 11.8330%, hits 6,265다.
- 기존 30분 대비 +75 hits, Top-100 평균 겹침 95.43/100이다.
- 공통 시점의 Inertia 전 B1 확률 최대 차이는 **0**이다. 학습 모델 차이가 아니라, 중간 5분 점수로 갱신된 과거 상태 경로의 차이를 본다.

## 4. onset 포착과 추천 교체

동일한 1,402개 고유 `(gpu_id, onset_time)`을 대상으로 한다. 발생 전 24시간이 평가 기간 안에 완전히 들어와야 하며, `[onset-24h, onset)` 중 한 번이라도 Top-100이면 포착으로 센다. 검증된 하드웨어 고장 episode나 예방된 고장 수가 아니다.

| 방식 | 포착 onset | 포착률 | 시간당 추천 교체량 |
| --- | ---: | ---: | ---: |
| 기존 30분 예측 | 625 | 44.58% | 49.19개 |
| 30분 점수를 5분 시점에 유지 | 637 | 45.44% | 49.19개 |
| 새 5분 예측 | 735 | 52.43% | 198.39개 |

같은 5분 시점 기준 +98 onsets/+6.99%p를 관찰했으며 교체량은 약 4.03배다. 1회 진입 정의는 자주 바뀌는 추천에 유리할 수 있다. 포착한 event 집합이 달라 조건부 선행시간의 평균/중앙값 차이를 인과적 조기 탐지 효과로 단정하지 않는다.

## 5. 5분에서 Inertia 제거 대조

같은 학습 모델의 저장된 원시 B1 확률을 직접 순위화한 prediction-only ablation이다. 재학습하지 않았다.

| 지표 | Inertia 없음 | Inertia 적용 |
| --- | ---: | ---: |
| 평균 AP | 0.018598749 | 0.022004021 |
| Recall@100 | 11.1898% | 11.8316% |
| 양성 적중 | 35,495 | 37,531 |
| onset 포착률 | 76.75% | 52.43% |
| 시간당 추천 교체량 | 365.44개 | 198.39개 |

Inertia의 AP는 상대 18.31% 높고 +2,036 GPU-time을 포착한다. 7-origin bootstrap AP 차이 구간 [+0.000980, +0.006924]는 양수다. 반면 1회 onset 포착은 1,076→735로 줄었다. 지속 위험의 상위 유지와 일시적 신규 신호 진입 사이의 trade-off다. AP 개선과 onset 포착 개선을 같은 주장으로 취급하지 않는다.

## 6. 시간과 한계

- 구간별 기록 시간 합계 993.33초(약 16분33초), 중앙값 15.48초, 최대 17.04초.
- 선정·파일럿·감사를 포함한 후속 체인 1,228.85초(약 20분29초).
- 64 origins 모두 CUDA 추론 trace 확인, 신규 학습 0회.
- 날짜 묶음 offline replay다. 단일 온라인 요청 지연이나 동일 작업 CPU 대비 GPU 가속배수는 측정하지 않았다.
- 기존 개발기간에서 선택한 ST21 + Inertia의 후속 분석이다. 독립 terminal test가 아니며, 선택 편향 및 다중비교를 보정한 우월성 주장을 하지 않는다.
- 다음 검토 후보는 연속 경고/최소 선행시간 조건, validation에서 고정한 Inertia 세기·반감기, 동일 5분 시점의 B1+B2+Fusion 비교다. 이 폴더에서 해당 추가 실험은 수행하지 않았다.

## 7. 산출물과 재계산

- [analysis.json](results/analysis.json): 여섯 평가 조건의 집계, 운영 지표, paired bootstrap, 확률 지표, 시간, 원본 score SHA256.
- [comparisons_by_origin.csv](results/comparisons_by_origin.csv): 64 origins × 6 조건, 총 384행.
- `results/source_*`: 원래 30분/5분 독립 감사와 5분 최종 상태의 사본.
- [analyze_5m.py](ML/analyze_5m.py): 원본 점수 SHA·공통 모집단/정답·최종 집계를 검사하고 결과를 생성하는 분석 진입점.
- `vendor/`: 기존 실행에 사용한 metric/paired-bootstrap helper의 원문 스냅샷. 이 분석은 해당 모듈의 순수 계산 함수만 호출한다.
- `ARCHIVE_MANIFEST.json`: 파일별 SHA256 및 분석/입력 provenance.

Python 3.12 환경과 의존성은 이전 아카이브 `Falcon_AcmeTrace_ST_Top100/versions/requirements-lock.txt`를 참조한다. 원시 Parquet, full score tapes, 모델/캐시는 본 compact archive에 포함하지 않았다. source audit 파일의 score hash와 이전 아카이브의 omitted-artifact manifest로 식별한다.

```text
python ML/analyze_5m.py --source-5m /path/to/conditional_5m_recovery_01 --source-30m /path/to/full_01 --data-directory /path/to/acmetrace --output /path/to/new-analysis
```

분석용 출력 폴더는 원본 실행 폴더와 분리해야 한다. 이미 검증된 두 score tape와 원시 onset Parquet가 있어야 재계산할 수 있다.

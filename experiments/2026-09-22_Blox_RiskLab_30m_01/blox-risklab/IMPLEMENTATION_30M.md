# Blox RiskLab 30분 구현·실제 실행 결과

2026-09-22 · 사용자 승인 범위 · 목표 G-007, G-008, G-009

## 결론

구현, ML 고정 cohort 예측 재생성, Blox 연결, 실제 데이터 전체 replay, 대시보드 실행 및 재현성 검증을 완료했다. 다만 **현재 99% validation recall 임계값은 전체 평가기간의 100 GPU를 모두 제외한다.** 위험 제외 정책의 완료 작업은 0건이며, 위험도 미사용 LAS는 16,194/16,560건을 완료했다. 현재 기준을 유효한 운영 정책이나 성능 개선으로 채택하지 않는다. 임계값을 test 결과에 맞춰 변경하지 않았다.

- 대시보드: http://127.0.0.1:8115
- 확정 결과: `runs_actual/comparison_20260922_30m_verified/`
- 실행 안내: [RUN_ACTUAL_30M.md](RUN_ACTUAL_30M.md)
- 비교 표: [comparison.html](runs_actual/comparison_20260922_30m_verified/comparison.html)
- 실제 데이터 감사: [actual_validation.json](runs_actual/comparison_20260922_30m_verified/actual_validation.json)

## 구현 범위

| 기능 | 구현 |
|---|---|
| 원본 설정에서 고정 cohort 예측 생성 | `ISEE_Project_GitHubBase/ML/export_risklab_30m.py` |
| 입력 checksum·스키마·시간·100 GPU 완전성 | `risklab/feeds.py` |
| 실제 XID episode·작업 trace·서버 매핑 | `risklab/prepare_replay_inputs.py` |
| 최소 서버 수 배치, random·nonsticky·확률 tie-break | `risklab/placement_policy.py` |
| 30분 판단과 개별 event 시각, Gang 복구·정비 | `risklab/replay_engine.py` |
| 장애·예방·스케줄링 비용 분리 | `risklab/loss_ledger.py` |
| 배치 runner·중단·실패 기록·원자적 상태 저장 | `risklab/replay_runner.py` |
| worker subprocess, cursor API, GPU 화면·결과 | `risklab/replay_web.py`, `risklab/web/replay.html` |
| 정확한 job/GPU 상태 구간과 round별 지표 | `risklab/timelines.py`, `round_metrics.jsonl`, `metrics.parquet` |
| 네 정책 비교와 반복 실행 checksum | `risklab/replay_compare.py` |

FIFO/LAS는 인접 Blox checkout의 실제 `schedulers.Fifo/Las.schedule()`을 호출한다. 스케줄러의 job 순서와 placement는 분리했다. Tiresias2Q는 기존 RiskLab의 2-queue 의미를 유지한 adapter이며 native upstream 구현으로 표시하지 않는다. 실제 GPU 제어·학습 framework 내부 checkpoint는 이 simulator의 범위가 아니다.

기존 `engine.py`, synthetic launcher와 원본 실험은 보존했다. actual launcher는 새 provider/engine을 사용한다. 원본 실험 폴더의 Git 변경은 없고, 별도 ML exporter를 추가했다. 커밋·푸시는 하지 않았다.

## 입력과 모델 상태

- 원본: `2026-09-21_All-XID_ReportFaithful_ParetoMomentumCascade_01`.
- 원본 저장 tape는 시점별 Top-100으로 총 1,828 GPU가 등장한다. 이를 고정 100 GPU로 오인하지 않고 별도 예측을 생성했다.
- 원본 telemetry grid의 lexical 첫 100 GPU, 실제 서버 13개. `10.140.0.131-0`부터 `10.140.0.143-3`까지이며 원본 node/GPU header와 매핑을 확인했다.
- UTC 2023-08-05 06:45부터 2023-08-18 00:00 미만, 611시점 × 100 GPU = 61,100행. 의사결정 간격 1,800초, 마지막 구간은 15분이다.
- 원본 모델 가중치가 저장되어 있지 않아 frozen 설정에서 재학습했다. 저장된 selection, 36시간 purge, raw Pareto(.5, .02, alpha=1), held-out 분할을 유지했다. Platt 적용이나 새 hyperparameter 탐색은 하지 않았다.
- hardness quantile은 원본처럼 전체 1,992 GPU로 계산한 뒤 고정 100개를 추출했다.
- 원본 Top-100과 겹치는 7,678행의 예측 차이는 MAE 0.0166035, 최대 절대 차이 0.2330145다. **원래 학습 객체/예측을 비트 단위로 복원한 결과가 아니다.** 재생성된 최종 가중치는 `model_cache/final_*.pkl`에 저장하고 SHA를 남겼다.
- `export_provenance.json`은 Python·torch·sklearn 등 학습 환경, 현재 exporter와 참조 ML 코드, 최종 모델 SHA를 기록한다. 실행 후 캡처이며 threshold 공통 함수 추출은 기존 inline 계산과 동등하다.
- raw XID 15초 관측에서 5분 bucket의 nonzero gap ≤600초를 병합했다. 실제 최초 관측시각을 보존한 31개 episode, 28개 GPU다. 초기 queue와 GPU 상태는 각각 empty/healthy 가정이다.
- 기간 내 도착 55,104개 중 COMPLETED·GPU 수 1..100·양의 duration 조건의 16,560개를 사용했다. 38,544개는 제외했다. 실패·취소 duration은 정상 useful work로 쓰지 않았다.
- 전체 클러스터 작업 도착을 100 GPU 용량에서 replay했다. **실제 해당 100 GPU의 측정된 작업 부하로 일반화할 수 없다.**

## Validation 기준과 확인된 한계

Validation 기간: UTC 2023-07-30 21:40~2023-08-02 21:35. 표본 122,606개, 양성 11,146개. 원본 negative sampling을 유지했고 모든 양성을 포함했다. Threshold용 cascade는 시간상 앞선 OOF만 사용하고 36시간 purge를 적용했다. Cascade 학습 최대 시각은 2023-07-28 21:35다. Held-out은 threshold 선택에 사용하지 않았다.

| 목표 recall | Validation 실측 recall | threshold |
|---|---:|---:|
| 0.9 | 0.90005383 | 0.0264372393 |
| 0.95 | 0.95002692 | 0.0238262349 |
| 0.99 | 0.99004127 | 0.0221697840 |

Primary threshold는 0.022169783968252598인데 실제 cohort 예측의 최솟값은 0.023385562732999247이다. 따라서 611회 모두 eligible GPU=0이다. Validation recall은 미래 운영 가능성이나 held-out recall의 보장이 아니다. 이 분포 차이의 원인을 특정 모델 요소로 확정하지 않는다.

90%·95% threshold 값은 validation 민감도 정보로 보존했으며 이번 확정 비교는 99% 기준이다. 모든 masked 정책이 배치를 시작하지 못했으므로 이 실제 데이터 결과만으로 packing/random/nonsticky의 상대 성능을 판단할 수 없다. 해당 분기 동작은 회귀 fixture에서 검증했다.

## 실제 결과

동일 입력 manifest·LAS·seed=17에서 비교했다. 평균/95% 대기는 미완료 작업의 관측 종료까지 대기도 포함한다. JCT는 완료 작업만의 값이며 makespan은 미완료 작업이 있어 censored다.

| 정책 | 완료 | 미완료 | 평균 대기(h) | p95 대기(h) | 장애 손실 집계(초) | 예방 비용 집계(초) |
|---|---:|---:|---:|---:|---:|---:|
| risk_blind_packed | 16,194 | 366 | 17.73 | 57.21 | 179,475 | 0 |
| risk_mask_packed | 0 | 16,560 | 159.47 | 296.54 | 187,110 | 108,475,440 |
| risk_mask_random | 0 | 16,560 | 159.47 | 296.54 | 187,110 | 108,475,440 |
| risk_mask_packed_nonsticky | 0 | 16,560 | 159.47 | 296.54 | 187,110 | 108,475,440 |

장애 손실/예방 비용 합계에는 작업 초와 GPU 초가 섞여 있으므로 경과시간처럼 해석하지 않는다. `failure_checkpoint_loss_seconds`, `failure_recovery_delay_seconds`, `failure_gpu_unavailable_seconds`를 별도로 제공한다. `maintenance_unavailable_seconds`는 drain 대기·정비·cooldown의 GPU 가용성 손실이다. LAS에 따른 `scheduling_checkpoint_seconds`, `scheduling_relaunch_seconds`도 별도 항목이며 큐 대기는 장애 손실에 포함하지 않는다.

위험도 미사용 정책의 checkpoint 손실은 105 작업 초, 장애 복구는 1,200 작업 초, GPU unavailable은 178,170 GPU 초다. masked 정책의 작업 장애 손실 0은 처리한 작업이 없기 때문이며 개선 근거가 아니다. masked 정책도 실제 XID의 reactive unavailable 187,110 GPU 초가 발생한다.

Checkpoint interval=300초, 예방/스케줄링 checkpoint cost=60초, relaunch=300초, maintenance capacity=1·duration=900초·cooldown=300초는 고정 가정이다. 정기 checkpoint는 공통 기준의 순간 상태 저장으로 모델링했다. 서버 상관 실패는 primary에서 제외했다.

## 검증과 성능

- 단위·통합·회귀 테스트 **22/22 통과**: cadence, schema/checksum, validation threshold, 정확한 서버 수 oracle, no-fit, atomic Gang, migration, 재현성, 이벤트·손실·timeline 보존, worker 중단/오류, Windows read/replace 충돌.
- 실제 네 정책 모두 611개 round 지표, 31개 XID episode, 109,890,000 GPU 초의 timeline 보존, queue integral, 입력 checksum을 확인했다.
- 동일 masked 정책 반복 실행의 `decisions.jsonl`, `events.jsonl`, `summary.json` SHA-256이 모두 일치했다. CLI와 별도 dashboard worker도 세 파일 모두 일치했다.
- excluded GPU 신규 배치=0. 실제 masked 결과는 배치 자체가 0이므로, 배치가 가능한 조건의 no-fit/Gang/최소 서버 동작 증거는 별도 회귀 테스트다.
- Chrome 대시보드에서 실제 실행 버튼→완료→보고서 흐름을 검증했다. HTTP 202 응답 270ms, 100 GPU tile, 보고서 HTTP 200, JavaScript 오류 0, 390px 모바일 가로 overflow 없음.
- 대시보드 실행 ID: `run_20260921_174239_e43f486c`.
- 테스트 기록: `tests/validation_20260922.{json,txt}`, `tests/browser_actual.json`. 화면: `tests/dashboard_actual_desktop.png`, `tests/dashboard_actual_mobile.png`.

| 정책 | 전체 실행(s) | 입력 준비(s) | 결과 저장(s) | 최대 판단(s) |
|---|---:|---:|---:|---:|
| risk_blind_packed | 25.496 | 0.559 | 1.049 | 0.0634 |
| risk_mask_packed | 27.150 | 0.672 | 0.452 | 0.1658 |
| risk_mask_random | 23.023 | 0.425 | 0.297 | 0.1414 |
| risk_mask_packed_nonsticky | 22.926 | 0.340 | 0.346 | 0.1238 |

Batch process peak working set은 약 232.4 MiB였다. 이는 같은 프로세스의 누적 high-water mark로 각 정책의 독립 peak 측정치는 아니다. 측정은 로컬 환경의 실행 증거이며 일반적인 성능 보장은 아니다. Probability tape는 Parquet batch로 검증/순차 소비하고, 이벤트·round 지표는 append 로그로 기록하며 dashboard는 bounded cursor를 사용한다.

## 장애 수정 기록과 재현 시 주의점

첫 실제 dashboard 실행 `run_20260921_173828_25d66e75`는 Windows에서 status.json 읽기 핸들이 열린 동안 atomic replace가 WinError 5로 실패했다. 기존 코드에서 열린 읽기 핸들로 같은 오류를 재현했다. 고유 temp 파일과 최대 약 1초의 제한된 PermissionError 재시도를 적용하고, 원래 traceback을 status 저장보다 먼저 남기도록 수정했다. 실제 열린 핸들이 닫힌 후 저장 성공 및 영구 오류를 숨기지 않는 두 회귀 테스트를 통과했다. 수정 후 full replay와 CLI parity를 확인했다. 실패 run과 수정 전 비교는 진단 기록으로 보존하고, 확정 수치는 `_verified` 폴더만 사용한다. Debug status: DONE.

Windows 기본 cp949와 UTF-8 산출물을 혼용하지 않도록 모든 결과 읽기에서 UTF-8을 명시한다. Blox checkout에 `.git` metadata가 없어 source commit은 확인 불가이며 실제 scheduler 파일 SHA를 기록한다. 원본 학습 객체 부재와 시점별 Top-100 artifact는 향후 같은 통합 작업에서 먼저 확인할 사항이다.

## 남은 연구 액션

G-007·G-008·G-009는 연구 목표로 계속 활성이다. 담당 이준호. 다음 확인은 후속 실험 범위 승인 시다.

1. Validation과 최종 inference의 score 분포·prior·threshold 전달 가능성을 development-only로 점검한다.
2. 미리 정한 90%/95% recall 기준 및 capacity-aware 기준의 별도 평가 설계를 검토한다. Test 성능을 보고 primary threshold를 역으로 최적화하지 않는다.
3. 실제 cohort와 대응되는 작업 부하 및 carry-in 상태를 확보하고, checkpoint/maintenance 시간 민감도를 검토한다.
4. 배치가 실제로 발생하는 조건에서 packing/random/nonsticky의 효과와 서버 상관 실패 민감도를 평가한다.

위 후속 실험은 이번 구현 완료와 구분하며 임의로 실행하지 않았다.

# Blox RiskLab 실제 데이터 replay

목표: G-007·G-008·G-009. 2026-09-22 사용자 승인, 의사결정 간격 30분.

## 실행

프로젝트 루트에서 기존 ML 환경으로 예측을 생성한다. 설정 탐색이나 Platt calibration은 하지 않는다.

```powershell
& 'C:\Users\Public\Documents\ESTsoft\CreatorTemp\iisee_b1_adst_venv\Scripts\python.exe' -B -u ISEE_Project_GitHubBase\ML\export_risklab_30m.py --source ISEE_Project_GitHubBase\experiments\2026-09-21_All-XID_ReportFaithful_ParetoMomentumCascade_01 --output blox_repo_actual\blox-risklab\inputs\reportfaithful_30m_20260922
```

이후 `blox_repo_actual/blox-risklab` 폴더에서 실행한다. Python 3.13, pandas, numpy, pyarrow가 필요하다.

```powershell
py -3.13 -B -m risklab.prepare_replay_inputs --project ..\.. --inputs inputs\reportfaithful_30m_20260922
py -3.13 -B start_risklab_actual.py --port 8115 --no-browser
py -3.13 -B -m risklab.replay_runner --manifest inputs\reportfaithful_30m_20260922\run_manifest.json --output runs_actual\run_example --policy risk_mask_packed --scheduler Las
py -3.13 -B -m unittest discover -s tests -v
```

브라우저: http://127.0.0.1:8115 . 새로운 실행마다 별도 run ID를 사용한다. 기존 출력 폴더 덮어쓰기는 거부한다.

## 데이터 계약

- 원본 tape는 611시점 × 시점별 위험 Top-100이다. 고정 cohort 입력으로 직접 사용하지 않는다.
- 원본 telemetry grid를 lexical 정렬한 첫 100 GPU를 고정한다. 실제 GPU ID의 server prefix와 local GPU suffix를 보존하며 node telemetry header와 검증한다.
- 원본 설정·가중치·Cascade alpha를 고정하여 학습 모델을 재생성한다. 원본 학습 가중치가 보존되어 있지 않아 원래 학습 객체를 그대로 복구했다고 주장하지 않는다.
- inference hardness의 quantile은 원본처럼 전체 GPU를 대상으로 계산한 뒤 고정 cohort를 추출한다.
- validation threshold는 시간상 분리된 cascade로 생성한다. Cascade fit 표본에 대한 in-sample prediction을 threshold 근거로 사용하지 않는다. 원본 validation의 negative sampling을 유지하며 validation recall은 미래 운영 recall 보장이 아니다.
- 30분 위험도 갱신·배치 판단 사이에 XID, 작업 완료, recovery, maintenance 완료는 실제 event 시각에 처리한다.
- XID는 원본 15초 관측에서 episode를 만든다. 5분 bucket의 nonzero gap <=10분을 병합하며 timestamp는 최초 실제 관측 시각이다. 일시적인 clean 관측보다 명시적인 gap 병합 규칙을 우선한다.
- 실패·취소된 작업의 duration은 정상 완료 소요시간으로 사용할 수 없어 COMPLETED·GPU 1..100·양의 duration만 포함한다. 전체 클러스터의 도착 부하를 100 GPU에서 replay하므로 실제 100 GPU 운영 부하로 일반화하지 않는다.
- 초기 queue는 비어 있다. 실제 trace에 checkpoint 정보가 없어 300초 checkpoint interval, 60초 예방/스케줄링 checkpoint 비용, 300초 relaunch를 가정한다. 정기 checkpoint 자체는 공통 기준의 순간 상태 저장으로 모델링한다.
- Maintenance capacity=1, duration=900초, cooldown=300초. 모두 측정된 물리 시간 대신 고정 가정이며 manifest에서 변경 가능하다.
- 실패 GPU unavailable과 예방 정비 unavailable은 서로 다른 원인으로 분리한다. 작업 초와 GPU 초가 섞인 합계를 elapsed time처럼 해석하지 않는다. 처리량·JCT·queue wait도 함께 보고한다.

## Blox 경계

FIFO/LAS는 인접 Blox checkout의 실제 `schedulers.Fifo/Las.schedule()`을 호출한다. deployment/grpc 초기화 없이 동일 state interface를 전달한다. Tiresias2Q는 기존 RiskLab의 Blox-compatible 2-queue 정렬 의미를 유지한다. upstream에 완성된 native Tiresias2Q가 있다는 주장은 하지 않는다.

새 경로는 `ActualRiskLabSimulation`과 명시적 provider를 사용한다. 기존 `engine.py`, `external_risk.py` monkey patch 경로와 synthetic launcher는 보존하고 actual launcher에서는 import하지 않는다. synthetic 데이터는 테스트에만 사용한다.

## 결과

`manifest.json`, `status.json`, `events.jsonl`, `decisions.jsonl`, `snapshots.jsonl`, `round_metrics.jsonl`, `summary.json`, `metrics.parquet`, `job_summary.parquet`, `gpu_loss.parquet`, `job_timeline.parquet`, `gpu_timeline.parquet`, `performance.json`, `report.html`을 보존한다. 중단과 worker 오류에서도 부분 event log와 원인을 보존한다.

## 요구사항 추적

| ID | 근거 | 구현 | 검증 | 상태 |
|---|---|---|---|---|
| R-01 | 사용자 30분 결정 | feeds, replay_engine | cadence·event timing tests | confirmed (fixture + actual replay) |
| R-02 | 계획 §5 고정100·checksum | export_risklab_30m, feeds | invalid input + 100 GPU determinism | confirmed (actual 61,100 rows; fixed 100 GPU) |
| R-03 | §5 validation recall99 | thresholds, exporter | threshold maximality, no-test selection | confirmed (validation threshold 0.02216978397; actual eligible=0) |
| R-04 | §8 최소서버·no-fit | placement_policy | exact oracle comparison, no-fit | confirmed |
| R-05 | §9 Gang·정비·복구 | replay_engine | failure, migration, cooldown, censoring | confirmed (fixture + actual replay) |
| R-06 | §10 loss 분리 | loss_ledger | duplicate/queue/cause tests | confirmed (fixture + actual replay) |
| R-07 | §11 dashboard worker | replay_web, replay_runner | HTTP202, worker/CLI parity, crash/stop | confirmed (fixture + actual replay) |
| R-08 | §17 stream | feeds, ArtifactWriter | Parquet batches, bounded cursor | confirmed (actual 23~27s/replay, 611 round metrics) |
| R-09 | §20 실제 full replay | inputs/run_manifest, runs_actual | input hashes, policy summaries | confirmed (4 policies + repeat + dashboard parity) |

실제 실행 결과는 별도 `IMPLEMENTATION_30M.md`에 갱신한다. 커밋·푸시는 이 작업에 포함하지 않는다.

## 확인된 결과 (2026-09-22)

전체 22개 테스트와 실제 네 정책·반복 실행·dashboard/CLI SHA 일치 검증을 통과했다. 확정 비교는 `runs_actual/comparison_20260922_30m_verified`다. 현재 99% recall threshold는 모든 GPU를 제외하여 masked 정책의 완료 작업이 0건이다. 위험도 미사용 LAS는 16,194/16,560건 완료했다. **실행 완료와 정책 효과 검증 성공은 다르다.** 상세 해석은 [IMPLEMENTATION_30M.md](IMPLEMENTATION_30M.md)를 따른다.

```powershell
py -3.13 -B -m risklab.replay_compare --manifest inputs\reportfaithful_30m_20260922\run_manifest.json --output runs_actual\comparison_new
py -3.13 -B -m tests.audit_actual
```

감사 스크립트는 이번 확정 비교와 `tests/browser_actual.json`의 실행을 확인한다. 새 비교 폴더는 별도 지정한다.

## 위험 선호 정책 적용

2026-09-22 후속 사용자 승인으로 dashboard/API 기본 정책을 `risk_prefer_packed`로 변경했다. 화면은 데이터 발생 시각만 표시한다. 최신 비교와 검증은 [POLICY_REVISION_2026-09-22.md](POLICY_REVISION_2026-09-22.md)를 참조한다. 기존 99% manifest와 비교 결과는 보존한다.

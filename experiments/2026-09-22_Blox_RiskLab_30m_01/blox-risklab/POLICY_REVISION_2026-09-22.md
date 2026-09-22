# 위험 선호 정책 적용 결과 · 2026-09-22

목표 G-009. 사용자는 정책 적용을 승인하고 시간 표시는 **데이터가 발생한 시간만**으로 확정했다.

## 적용

- 새 dashboard/API 기본 정책: `risk_prefer_packed`.
- 실제 고장·정비·cooldown 상태는 배치 불가. 예측 고위험은 경고이며 새 정책에서는 배치 금지가 아니다.
- 실제 배치 가능한 GPU 중 **서버 수 최소 → 위험 점수 합 최소 → 결정적 동률 처리**. 서버별 필요한 GPU 수를 선택하는 exact DP를 사용하며 전수 조합 oracle과 대조했다. 점수 합은 비교 기준이며 Gang 고장 확률이 아니다.
- 점수 상승에 따른 자동 drain·정비·Gang 이동 없음. 실제 XID 복구 및 기존 LAS 선점은 유지했다.
- 원본 모델·threshold·입력·과거 결과는 보존했다. CLI에서 원본 manifest를 명시하면 기존 정책을 유지하므로 새 정책에는 `--policy risk_prefer_packed`를 사용한다.

## 화면

- 데이터 날짜·시각만 표시하며 UTC/KST를 선택할 수 있다. 실행 시작 wall clock과 실제 실행 경과시간은 화면에 추가하지 않았다.
- 실제 서버 13개로 GPU 100개를 묶었다. 서버 ID, local GPU 번호, 작업 ID, 상태·위험 점수를 표시한다.
- 이벤트는 원본 timestamp와 관련 서버/GPU/작업을 표시한다. 완료 이벤트에도 할당됐던 서버/GPU를 보존한다. 미배치/복구 대기 이벤트는 미배치로 표시한다.
- 최신 100개 이벤트를 bounded tail로 읽어 빠른 replay 뒤에도 오래된 로그에 화면이 머물지 않는다. 기존 순차 cursor API도 유지했다.
- 고위험 경고는 점선, 실제 정책 제외는 붉은 테두리로 구분한다.

## 실제 비교

동일 입력·LAS·seed=17, 611회 의사결정. 총 작업 16,560개.

| 지표 | 위험도 미사용 | 기존 99% 제외 | 새 위험 선호 |
|---|---:|---:|---:|
| 완료 작업 | 16,194 | 0 | 16,194 |
| 미완료 작업 | 366 | 16,560 | 366 |
| 장애 checkpoint 손실(작업 초) | 105 | 0 | 120 |
| 장애 복구 지연(작업 초) | 1,200 | 0 | 900 |
| 위 두 장애 항목 합(작업 초) | 1,305 | 0 | 1,020 |
| 실제 장애 unavailable(GPU 초) | 178,170 | 187,110 | 178,170 |
| 선점 checkpoint/relaunch(작업 초) | 2,878,920 | 0 | 2,880,000 |
| 평균 대기(초) | 63,832.02 | 574,104.34 | 63,844.11 |
| p95 대기(초) | 205,961.95 | 1,067,526 | 205,961.95 |
| 예방 unavailable(GPU 초) | 0 | 108,475,440 | 0 |

전면 차단은 해소됐다. 미사용 정책 대비 장애 작업 손실은 285초 감소했지만 선점 비용은 1,080초 증가했고 평균 대기는 약 12.09초 증가했다. 두 작업 시간 비용의 합은 오히려 795초 증가한다. **종합 개선이나 위험 예측의 운영 효과를 확정하지 않는다.** 이미 확인한 동일 평가 구간의 탐색적 결과다. 원래 workload·checkpoint·정비 가정과 가중치 재생성의 한계도 그대로 적용된다.

## 검증

- 기존 및 새 정책 회귀 테스트 26/26 통과.
- 고위험 전체에서도 작업 진행, 실제 장애 복구 유지, 최소 서버/최소 score 전수 조합 비교, timestamp·server 매핑, bounded tail 검증.
- 세 정책 모두 XID 31개, 611 round, GPU 시간 보존, job queue 적분, 원본 입력 checksum 확인.
- 새 정책의 반복 실행 및 별도 dashboard worker와 CLI의 events·decisions·summary SHA-256 모두 일치.
- Chrome: 13서버·100 GPU, UTC `2023-08-18 00:00:00` ↔ KST `2023-08-18 09:00:00`, 보고서 HTTP 200, JS 오류 0, 모바일 overflow 없음.
- 증거: `runs_actual/policy_revision_20260922/verification.json`, `reproducibility.json`, `comparison.csv`, `comparison.html`, `tests/browser_policy_revision.json`.

## 실행

```powershell
py -3.13 -B -m risklab.replay_runner --manifest inputs\reportfaithful_30m_20260922\run_manifest.json --output runs_actual\run_prefer_new --policy risk_prefer_packed --scheduler Las
py -3.13 -B start_risklab_actual.py --port 8115 --no-browser
```

G-009의 정책 변경·데이터 시각/서버 표시 액션은 완료. 연구 목표는 활성 상태이며, 다음 확인은 별도 평가 구간·부하 가정 및 선점 비용을 포함한 후속 연구 범위 확정 시다. 커밋·푸시 없음.

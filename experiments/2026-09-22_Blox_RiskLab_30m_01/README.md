# Blox RiskLab · 30분 replay 아카이브

G-009 구현 및 2026-09-22 사용자 요청에 따른 Blox 브랜치 보존본.

## 실행

Python 3.13에서 archive root 기준:

```powershell
python -m pip install -r requirements.txt
cd blox-risklab
python -B -m unittest discover -s tests -v
python -B -m risklab.replay_runner --manifest inputs/reportfaithful_30m_20260922/run_manifest.json --output runs_actual/run_example --policy risk_prefer_packed --scheduler Las
python -B start_risklab_actual.py --port 8115 --no-browser
```

화면: http://127.0.0.1:8115 . 기본 UI 정책은 위험 선호, 데이터 발생 시각(UTC/KST)·실제 서버 13개·100 GPU를 표시한다. 기존 manifest의 정책 설정은 보존했으므로 CLI에는 위처럼 `--policy`를 명시한다.

## 포함과 경로

- 현재 actual replay Python/HTML, 회귀 테스트, Blox native scheduler 소스와 MIT LICENSE.
- checksum으로 고정된 최종 재현 입력: probability·failure·job·topology와 validation threshold. 원본 telemetry가 아닌 해당 평가를 재현하는 약 2.4 MB의 최종 파생 입력이다.
- `results/`: 실제 비교 요약·성능·원본 provenance와 재현성 감사. 과거 보고서의 `runs_actual/...` 참조는 이 아카이브의 `results/...` 요약에 대응하며 전체 이벤트 로그는 로컬에 보존했다.
- `evidence/`: 브라우저 검증·스크린샷과 기존 테스트 기록. 아카이브 재검증은 별도 `archive_validation` 기록을 따른다.
- `ML/export_risklab_30m.py`: exporter 소스 스냅샷. ML 재학습은 기존 저장소 ML 모듈과 별도 원본 데이터·학습 환경이 필요하며 이 파일 단독 실행을 의미하지 않는다. 제공된 최종 tape replay에는 torch나 원본 telemetry가 필요 없다.
- portable input manifest는 Blox 경로만 상대경로로 수정했다. `results`의 절대 경로는 당시 실행 provenance로 보존한 값이며 새 실행 경로가 아니다.

## 확인된 결과와 남은 문제

동일 16,560개 작업에서 기존 99% 제외 완료 0건, 위험도 미사용/새 위험 선호 모두 16,194건. 새 정책의 장애 작업 손실은 1,305→1,020초이나 선점 비용은 1,080초 증가했다. 종합 개선으로 단정하지 않는다.

- replay 범위는 원본 전체 5월~8월 중 held-out인 UTC 2023-08-05 06:45~08-18 00:00 미만이다.
- 실제 XID 31 episode/28 GPU/4서버, 해당 cohort 코드는 모두43. 물리 GPU 고장 확정 수와 구분한다.
- 실제 데이터 기반 이벤트 시뮬레이션이며 실제 GPU에서 학습을 실행하지 않는다. workload는 전체 클러스터 도착을 100 GPU에 재생한 가정이다.
- **현재 손실·예방 비용 카드는 종료 summary에서만 갱신한다. 실행 중 누적 집계와 새 실행 시 카드 초기화는 미구현이다.**
- **현재 합계는 작업 초와 GPU 초를 혼합하므로 경과시간으로 해석할 수 없다.** 개별 loss component를 확인한다.
- 정비/복구 시간은 가정이며 원본 모델 가중치가 없어 frozen 설정에서 재생성한 예측이다.

원본 telemetry, 모델 캐시, 분할 예측 중간파일, 대용량 events/decisions/timelines는 제외했다. 원본 실험과 작업 디렉터리는 변경하지 않았다.

---

## 2026-10-17 업데이트 · 다중 작업 배치 공동 최적화(Co-Optimization) 및 인터랙티브 시뮬레이션 대시보드

Falcon-45 (Branch 1 급성 센서 텔레메트리) + 만성 정비 이력 (Branch 2) + Capped Pareto 결합 확률(`probability`, `b1_probability`, `b2_probability`)을 모두 활용한 **다중 작업 배치 공동 최적화 시뮬레이터(`full_co_optimized`)** 및 **단독 실행형 HTML 플레이어/대시보드**가 추가되었다.

### 1. 새로 추가된 핵심 HTML 대시보드 및 문서 (`docs/`)
서버 구동 없이 브라우저에서 즉시 열어볼 수 있는 단독(Standalone) HTML 파일 및 계획서:
- `docs/BLOX_실시간_시뮬레이션_플레이어_2026-10-17.html`: **611개 전체 의사결정 스텝(30분 단위, 12.7일)이 내장된 인터랙티브 시뮬레이션 플레이어** (정책 전환, 재생/일시정지/배속, 13개 서버 × 100대 GPU 실시간 상태·확률·XID 충돌 비교).
- `docs/BLOX_아키텍처_및_최적화_시각화_대시보드_2026-10-17.html`: 시뮬레이터 아키텍처 다이어그램, 수리모형, Track 1·2·3 벤치마크 결과 및 차트 대시보드.
- `docs/BLOX_최적화_시뮬레이터_마스터_계획서_2026-10-17.md` (`.html`): 4대 구조적 병목 진단 및 Solution A (배치 최적화) + Solution B (XID 31/43 차등 페널티) 마스터 계획서.
- `blox-risklab/risklab/web/replay.html`: 로컬 웹 서버(`http://127.0.0.1:8115`) 연동 실시간 리플레이 및 새 시뮬레이션 런처 UI.

### 2. 빠른 실행 가이드 (`blox-risklab/`)

```powershell
cd experiments/2026-09-22_Blox_RiskLab_30m_01/blox-risklab

# 1) 단위 및 최적화 정책 회귀 테스트 실행
python -B -m unittest discover -s tests -v

# 2) 로컬 인터랙티브 웹 시뮬레이터 서버 실행 (http://127.0.0.1:8115)
python -B -m risklab.replay_web --port 8115

# 3) 17개 시나리오(Track 1: 정책 Ablation, Track 2: 스케줄러 호환성, Track 3: XID 31/43 차등 모델링) 전체 벤치마크 재현
python -B run_comprehensive_optimization_benchmark.py

# 4) 벤치마크 시각화 차트 및 단독 HTML 플레이어 재생성
python -B visualize_optimization_results.py
python -B build_standalone_player_html.py
```

### 3. 주요 최적화 성과 요약 (`results/optimization_benchmark_20261017/`)
- **Track 1 (동일 하드웨어 조건, Las 스케줄러, 16,560개 작업)**:
  - `risk_blind_packed` 대비 **`full_co_optimized`** 적용 시:
    - **실행 중 XID 작업 피격 횟수**: `20회 → 14회` (**-30.0% 감소**)
    - **유휴 상태 XID 회피 성공**: `11회 → 17회` (**+54.5% 증가**)
    - **작업 측 총 유효 손실(`job_side_loss_seconds`)**: `35,685s → 7,680s` (**-78.5% 감소**)
    - **선점 오버헤드(`scheduler_preemption_seconds`)**: `33,240s → 5,640s` (**-83.0% 감소**)
- **Track 3 (XID 31 치명적 HW 결함 8건 vs XID 43 소프트 채널 리셋 23건 차등 모델링)**:
  - GPU 비가용 손실: `178,170 GPU-s → 34,590 GPU-s` (**-80.6% 감소**)
  - 총 시스템 손실(`total_loss_seconds`): `213,855s → 42,270s` (**-80.2% 감소**)


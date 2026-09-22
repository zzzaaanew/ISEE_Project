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

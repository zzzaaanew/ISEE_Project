# 5분 단위 위험도 산출 결과 전체

2026-10-18에 완료·독립 검증된 `conditional_5m_recovery_01` 실행 디렉터리의 **모든 272개 파일**을 2026-10-25에 원문 그대로 보관했다. 집계와 분석만 포함했던 이전 아카이브에 이어, 이번에는 전체 위험도 점수와 Top-100 Parquet를 실제로 포함한다.

## 범위

| 파일 | 개수 | 내용 |
| --- | ---: | --- |
| `run/scores/origin_XXXX.parquet` | 64 | 1,992 GPU 전체 위험도·순위·정답·Branch 확률·Inertia 상태 |
| `run/top100/origin_XXXX.parquet` | 64 | 각 5분 시점의 상위 100개 GPU 전체 열 |
| `run/scores/origin_XXXX.identity.json` | 64 | 구간별 모델/점수 SHA256·평가 지표·공통 30분 시점 검증 |
| `run/fusion_models/origin_XXXX.json` | 64 | 해당 실행의 `not_selected` 상태 기록 |
| `run/state/checkpoint.joblib` | 1 | 완료된 추론 상태 체크포인트 |
| `run/*.json`, `run/*.csv`, `run/*.jsonl`, `run/*.log` | 15 | 실행 계약·명세·선정·최종 감사·집계·추론 로그 |

원시 telemetry, 원본 XID 입력, 학습 캐시 및 30분 학습 모델은 이 결과 디렉터리 밖의 별도 자료다. 본 업로드는 기존 5분 결과 파일을 복사한 것으로 재학습이나 점수 변경을 수행하지 않았다.

## 결과 정의

- 최종 순위: **B1 ST21 + Inertia**, 일별 학습 모델 고정, 5분마다 새 특징·상태·점수 산출.
- 64 origins, 18,313 시점, 36,479,496 GPU-time 점수.
- Top-100 총 1,831,300행. 양성 GPU-time 317,209행, Top-100 양성 적중 37,531행.
- 평균 origin AP 0.022004021132449474, pooled Recall@100 0.11831631511085751.
- All-XID 향후 24h inclusive-bin 타깃, 36h purge, nominal feature cutoff 10min/latest telemetry lag 15min.
- 독립 감사 `FIVE_MINUTE_SCORE_ARTIFACTS_VERIFIED`; 최종 감독 상태 `FIVE_MINUTE_COMPLETED_AND_AUDITED`.
- `risk_score`가 최종 위험도, `rank`가 해당 timestamp 내 1~1,992 순위다. `timestamp`, `gpu_uid`가 점수 키이며 `label`은 평가용 정답이다.
- B2 확률 등의 열도 보존하지만 이번 5분 최종 순위는 B1 기준이다. 5분 Dual-Branch/Fusion 전체 성능 결과로 해석하지 않는다.

## Git LFS로 원본 다운로드

Parquet 128개와 checkpoint 1개는 **Git LFS**로 저장한다. 일반 Git 객체에는 pointer가 들어가고 원본 바이트는 원격 LFS 저장소에 있다. LFS가 활성화된 clone/pull로 실제 결과 파일을 내려받는다.

```text
git lfs install
git lfs pull --include="experiments/2026-10-25_FalconBased_5Min_FullScores_Archive_01/run/**"
```

파일별 원본 바이트 수와 SHA256은 [MANIFEST.json](MANIFEST.json)에 있다. LFS pointer의 `oid sha256`은 이 원본 SHA256과 일치해야 한다. `run/independent_audit.json`은 전체 점수 파일의 기존 검증 SHA256도 포함한다.

## 관련 아카이브

- [실행 코드·30분/5분 집계·실험 명세](../2026-10-18_FalconBased_GBDT_ADST_Inertia_Archive_01/)
- [5분 공통 시점·onset·교체량·Inertia 분석 및 재계산 코드](../2026-10-18_FalconBased_5Min_Analysis_Archive_01/)

이전 아카이브는 그대로 보존했다. 원래 `summary.json`의 pending-audit 표기는 계산 완료 시점의 기록이며, 이후 완료된 `independent_audit.json`과 `watcher_status.json`의 최종 상태를 함께 확인한다.

# Experiment 30 — 전체기간 ON 공동 Queue–Placement 실행 아카이브

- 목표: G-009. 보관 승인: 2026-10-24. 원 실행: 2026-10-23.
- 대상 원격/브랜치: `zzzaaanew/ISEE_Project:research/xid-recovery-30m-validation`.
- 상태: 3개 전체기간 실행 exit0, 비교 집계·전수 회계/입력/보존 검증 PASS. **최종 정책 채택·독립 재현·일반화는 미완료**.
- 기존 실험은 이동/덮어쓰지 않고 이 새 폴더에 승인된 스냅샷만 복사했다. 원본 실행 모델/비용/점수는 변경하지 않았다.

## 먼저 볼 파일

- `snapshot/experiments/2026-10-23_FullQueuePlacement_01/SPEC.md`: 실행 구조·수식·결과·한계.
- 같은 폴더의 `reports/comparison.json`, `comparison.csv`: 전체 작업·공통 완료 작업 비교.
- `reports/final_verification.json`: 152개 frozen 파일 및 입력/코드/회계 검증.
- `reports/limitations_evidence.json`: GPU 수요별 완료·미시작·비용 재집계.
- `runs/{fifo,tiresias,joint}/jobs.parquet`, `gpu_state_time.parquet`: 결과 재검산용 출력이며 원시 입력 데이터가 아니다.
- `docs/실험 기록/실험_연대기.md` 및 `.html`: 기존 E29 다음에 E30을 추가한 동일 내용의 누적 연구 노트.
- `ARCHIVE_MANIFEST.json`: 모든 보관 파일의 SHA-256, 제외 파일, 원본 입력 hash, 환경·검사 요약.

## 같은 조건의 결과

UTC 2023-06-15 09:40–08-18 00:05 끝 제외, 100 GPU·140,786 Job·140 XID episode·18,317회 5분 판단·모든 arm 위험도 ON·seed17.

| 지표 | FIFO | Tiresias-style | joint |
|---|---:|---:|---:|
| 완료 Job |74,822|75,340|88,697|
| 미완료 Job |65,964|65,446|52,089|
| 전체 Job restricted 평균체류(초) |1,829,169.001|1,765,216.522|1,312,150.215|
| 32 GPU 이상 완료 /631개 |231|234|14|
| 스케줄러 선점(건) |0|113,398|612|
| 활성 XID 작업 중단(건) |70|69|82|
| 오류 rollback(GPU초) |202,620|150,765|85,350|

미완료도 종료 시각까지 포함한 평균체류다. joint는 Tiresias-style보다 완료 +13,357개·평균체류 −25.666%지만 대형 작업을 크게 희생했다. 전체 작업 최종 완료시간이나 모든 작업의 개선을 뜻하지 않는다.

## 주요 한계와 후속 우선순위

1. **공정성이 가장 큰 결함**: 32GPU 완료179→11개,64GPU51→3개. joint의32GPU이상631개 중615개가 미시작이고 전체52,057개도 미시작이다. `F=300×미선택 수+360×선점 수+60×해제대기 신규 수`는 작업 수를 우선하는 단일결정 대리목적이다. 최대대기·대형 Gang 서비스 하한이 없다.
2. **ML 위험의 추가 기여 미분리**: 모든 군이 같은 위험도를 쓴다. Queue 선택·선점 억제·위치가 함께 달라져 개선을 ML의 성과로 단정할 수 없다. raw24h score는 1차 F의 동률 기준이고, 회피 가능한 오류손실과 선점비용의 경제적 trade-off가 아니다.
3. **근사해와 제한된 배정**: 오래된 queued256+수요별 대표+running에서 정/역 DP와 FIFO의 최대3개 proposal만 비교한다. 전역 bound/gap·다기간 H/terminal이 없고 기존 lease GPU집합은 고정한다. 적극적인 예방이주·격리는 없다.
4. **기준선·비용의 한정**: strict FIFO는 backfill이 없고 Tiresias는2큐3600GPU초/승급2의 style이며 원형 재현이 아니다. CP300유효초/정지60초·선점60+300·R0 재시도gap660초는 고정 시나리오다. XID별 실측 복구분포로 검증되지 않았다. E27은 periodic 저장정지가 없어 직접 동조건 대조가 아니다.
5. **현실성·외적 검증 부족**: 1,992 GPU 중 고정100cohort에 클러스터 완료trace 도착을 투입한 부하198.616%, 빈 초기상태·완료trace 선택·고정속도다. Job/PID 매핑 없이 GPU-local 앱 오류 민감도이며, 이번 hardware recovery/GPU downtime은0이다. 소프트웨어 오류의 GPU 이동 회피효과·진단/정비 효과를 입증하지 않는다. 네트워크/공유I/O 비용도 모델링하지 않았다.
6. **평가/재현 경계**: 같은 개발기간·seed1회·종료 시 미완료 절단이다. 30분 점수의 과거유지로5분 운영한 것이지 새5분 ML이 아니다. PYTHONHASHSEED 미고정/미기록과 active set 동률순서 영향이 있어 독립 bitwise 재현은 미검증이다.

후속은 대형 Gang/장기대기 보호를 우선 명세한 뒤, 모든군 ON인 Queue-only/Placement-only/joint ablation과 회피가능 위험·다기간 목적을 별도 승인해 검증한다. 결과를 보고 penalty를 늘리거나 이번 보관 작업에서 새 실험을 돌리지 않는다.

## 코드·데이터 복원 경계

- `snapshot/experiments/`는 실행 당시 형제 폴더 관계를 보존한다. E30 실행기와 frozen E25 `src/risklab`, E27 config/threshold, E29 입력감사 증거를 포함한다.
- **원시 risk/provenance/job/failure/topology Parquet은 업로드하지 않았다.** `input_inventory.json`의 정확한 SHA-256과 위치를 따로 복원해야 한다. 이 아카이브만으로 새 전체 replay를 실행할 수 있다고 주장하지 않는다.
- 역사적 config/manifest의 절대 경로를 보존했다. 다른 컴퓨터에서 실행하려면 별도 작업 사본에만 입력 경로와 `blox_source_root`를 재지정해야 한다. base commit `e082e3fbc26fe159f384cb2f6ade6a1fb5030314`의 `experiments/2026-09-22_Blox_RiskLab_30m_01/blox-main`도 필요할 수 있다.
- 기존 결과 폴더는 재실행이 덮어쓰지 못하도록 보호돼 있다. `supervise.py`/`connection.py`를 아카이브 결과 위에서 실행하지 않는다. `analysis/extend_guard.py`는 종료된 당시 PID 전용 제어 이력이므로 재사용하지 않는다.
- 결과표 재검산은 복사된 `runs/*/jobs.parquet`을 읽는 것으로 가능하다. `analysis/verify.py`는 원 작업환경의 원시 입력/152개 파일 경로까지 검사한 코드이므로 자료 없이 원격에서 통과하지 않는다. 저장된 PASS와 독립 재현을 구분한다.
- 큰 `events.jsonl.gz`/`decisions.jsonl.gz`는 로컬 원본에 보존하고 manifest에 해시만 기록했다. 따라서 공개 보관분만으로 사건별 재검증까지 완료했다고 주장하지 않는다.

## 검증·보존·변경 금지

- 실행 본체 root Python hash는 원 manifest와 일치, frozen152파일 보존검사 PASS. 보관 사본도 SHA-256 일치 및 Python AST 구문검사한다. 새 ML/시뮬레이션 실행은 없다.
- 보관기는 표준 라이브러리로 복사·해시·AST 검사만 수행하며, 원본/과거 archive를 삭제하거나 덮어쓰지 않는다. 데이터와 결과를 외부 서비스에 따로 업로드하지 않는다.
- 일반 ship 스킬의 필수 `review/checklist.md`가 설치본에서 없어 해당 workflow는 중단했다. 프로젝트 승인 규칙과 연구 아카이브 스킬로 전환해 실행 소스 수동 검토·전수 회계/해시·아카이브 검사로 대체했다. 그 체크리스트 검토/자동 릴리스가 완료됐다고 기록하지 않는다.
- main/Blox 병합·PR·강제푸시·기존 dirty 변경 스테이징·설치·버전/스킬 변경은 이번 범위가 아니다.

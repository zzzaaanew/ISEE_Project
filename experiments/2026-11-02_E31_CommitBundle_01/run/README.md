# E31 — 실행 및 산출물 안내

## 2026-11-02 재개 resume_01

사용자 「재실행」 승인으로39개완료origin/22,374,144행/234classifier fit을재사용해40번째부터진행한다. 기존코드·체크포인트·설정·실패로그는 `attempts/resume_01/before`에보존하고기존39개점수의SHA256은 `preserved_scores.json`에고정했다. 모델·seed·분할·purge·점수주기·관성/Fusion설정은동일하다.

재개 진입점은 `supervise.py --attempt resume_01`이다. 시작한attempt를다시호출하면중단한다. 각origin을새프로세스에서1회실행하고매핑을해제한다. 첫재개origin검증후남은origin→평가→전수검증→완료기록을연결한다. 상태JSON접근오류는재시도하며비핵심monitor저장실패로감독기를종료하지않는다. 연구학습의실패는자동재시도하지않는다.

현재권위상태는 `attempts/resume_01/monitor.json` 및종료시 `terminal_status.json`이다. root monitor에도동일상태를기록한다. root `progress.json`은완료진행률이며체크포인트/실제저장파일이우선이다. 새로그는 `attempts/resume_01/logs`에따로남긴다. 이전감독기종료원인은실제traceback의Windows WinError5 monitor교체실패였으며,대화종료에의한것으로단정하지않는다. 회귀검사11개PASS의근거는 `regression_tests.log`다.

사용자 승인: 2026-11-02 「그럼 진행하자」. 목표 G-004/G-007/G-008.

실행기는 `supervise.py`이며 prepare → pilot → pilot검증 → branches(완료origin 재학습없음) → evaluate → 최종검증 순서다. 현재 상태는 `monitor.json`, 실제 Branch 완료 수는 `progress.json`으로 확인한다. 현재 실행 중인 worker를 중복 실행하지 않는다. 기존 stage log가 있으면 supervisor의 자동 재시도는 차단된다.

`finish_handoff.py`는 별도대기프로세스로 감독기의종료상태를확인한뒤 누적실험연대기(Markdown/HTML)·목표/진행문서와요구사항매트릭스를갱신한다. 재학습·설정변경·외부알림은없다. 이기록작업의상태는 `completion_handoff_status.json`, 로그는 `logs/completion_handoff.log`다. `COMPLETED`일때도 `experiment_success`가true인지함께확인한다. 기록오류는실제모델성공과별도보고하며자동재시도하지않는다.

기존 환경: 프로젝트 루트의 `migrate/migration/e19-reproduction/.venv/Scripts/python.exe` (Python3.12.14, numpy2.5.3, pandas3.0.1, scipy1.18.1, sklearn1.9.0, pyarrow25.0.1). 설치 없음. 코드 복사본은 sources/에 고정되어 있고 실제 호출 심볼은 source_bridge.py가 선택한다. 원본모듈의 cache 생성·Torch import를 피하기 위한 AST 선택이며 원본 함수본문을 수정하지 않는다.

## 산출물

- `branch_scores/origin_*.parquet`: 전 GPU의 실제5분 B1 A/B/C/D 및 공통B2 원점수. `gpu_idx` 대응은 `gpu_mapping.csv`. `origin_bin`, timestamp, 30일 XID count 포함. 모델의 경험적 calibration을 인증한 확률로 해석하지 않는다.
- `fusion_scores/5m`, `fusion_scores/30m`: 저장 Branch점수에서 사후계산한 B1관성, 관성없는Fusion, 주Fusion. 30분은 Branch점수를 부분추출하고30분 감쇠를 별도사용한다.
- `metrics_summary.csv`, `metrics_by_origin.csv`: development/terminal/full,5분/30분 결과. terminal14일은calendar구간이고 공통라벨마스크로 일부시점이 제외된다. full은terminal포함이다. `dev_` 민감도는D 개발구간만 평가한다.
- `input_manifest.json`, `feature_manifest.json`, `grid_manifest.json`, `split_manifest.json`, `frozen_code.json`, `fit_trace.csv`, `adst_C_trace.csv`, `adst_D_trace.csv`, stage별 environment/status/log: 재현·방법론근거.
- `pilot_validation.json`, `final_validation.json`: 실제실행 검증. 파일이 없으면 해당 검증 완료가 아니다.
- `cache/`: 디스크매핑 원본행렬·53개특징합집합·5개history특징·label/eligibility. 원본Parquet를수정하지않으며재실행 입력은원본데이터다. 오래된cache를자동참조하지않는다.

주 비교는 A basic30/ST21, B Falcon45/ST21, C basic30/ADST, D Falcon45/ADST; 모두6h, B2/ST21이다. 매origin학습창은36h전끝나는달력일수다. 원본XID 전역기록11개공백에영향받는미래라벨은학습/feedback/평가에서제외한다. 마지막24h의불완전라벨을음성으로처리하지않는다. 개별GPU nonmissing관측은미인증이다.

실험계약 전체는 SPEC.md와config.json, 누적결과·해석은 프로젝트의 `실험 기록/실험_연대기.md` 및 동일HTML에서 확인한다. 별도실험재시도·설정변경·설치·E30·커밋/푸시는자동수행하지않는다.

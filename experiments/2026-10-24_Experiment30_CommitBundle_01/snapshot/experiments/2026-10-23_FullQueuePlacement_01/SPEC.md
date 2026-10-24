# Experiment 30 — 전체기간 Queue–Placement 실행

- 목표 G-009. 사용자 최신 승인: 실제전체연결 구현 → 연결검사 → 동일대화에서3개전기간실행 → 비교·기록. 기초검사에서종료하지않는다.
- 상태: 세 arm 전체기간 exit0·비교 PASS, 2026-10-24 전수 회계/보존검증 PASS. 기존 E25/E27/E29 원본 보존. ML재학습/점수변경/패키지설치 없음. 본실험당시 Git쓰기는 없었고,10/24 별도 사용자승인으로 아카이브 커밋/푸시를 진행한다.
- Material Passport: experiment-agent run; 2026-10-23; UNVERIFIED(독립재현/일반화없음).

## 고정 입력·비교 범위

- E27 risk1.json의입력hash·Parquet/provenance·cohort·Job/XIDtrace그대로 사용.
- UTC 2023-06-15 09:40–2023-08-18 00:05 끝제외,100GPU·140,786Job·18,317회300초판단.
- E19 B1관성/B2비관성Fusion의원정보1800초과거유지. All-XID24h/warmup30일/purge36h/cutoff10분은원점수계보. 새학습/분할/튜닝없음.
- 세arm 전부ON: fifo / tiresias / joint. 같은GPU-local XID민감도·R0재시도660초·동일seed17. 소프트웨어오류이동회피효과를입증한것은아님.

## 공통 실행·비용 변경

- frozen E25 ActualRiskLabSimulation의실제도착/완료/XID분류/정비/복구를상속한다. decision/advance/checkpoint예약을새격리adapter에서교체.
- periodic checkpoint300초마다60초정지(그동안GPU점유),스케줄러변경은기존GPU점유checkpoint60초후해제+재실행300초. 동일lease는추가선점비용0.
- 고장시저장된checkpoint로rollback. checkpoint중고장도이전저장본을사용한다. R0retry_gap660에별도restart300을더하지않음.
- periodic60초는기존예방CP비용60을공통정지시나리오로적용한가정이지실측측정치가아니다. E27무정지저장과달라세새arm끼리비교한다.
- 도착·고장·완료는event정확시각,최적화는300초마다. 선점GPU해제후60초시점의planned commit은이전결정의실행이지추가최적화가아니다.

## 기준선·전체규모 탐색

- FIFO: 제출순strictHOL,이미실행중인lease유지. backfill없음.
- Tiresias-style:2큐,서비스3600GPU초에서강등,큐내최초실행순FIFO/미실행작업후삽입,대기≥2×정책epoch실행시간이면승급/카운터reset. 3600·2는사전에고정한첫시나리오설정이며원문완전재현/최적설정이아님.
- 공통pack:선택된기존lease유지,신규작업은관측가능checkpointage+660과GPU수로노출순서를정하고낮은rawscore GPU를배정한다. 실제duration/미래사건은policy JobView에없다.
- joint:현재running모두+queued오래된256개+각Gang수요의가장오래된대표를후보로한다. 나머지작업도실제Queue에유지되며잘라서평가에서빼지않는다. 후보제한횟수로그.
- 용량DP(100GPU)는Job선택proposal을만든다. incumbent/FIFO·정방향DP·역방향DP의 **완전한GPU목표배정**을만들고실제선점·해제대기·위험노출을다시평가해선택한다. 입장Job집합은LAS로고정하지않음.

`F = 300×미선택Job수 + 360×기존실행Job중단수 + 60×선점GPU해제를기다리는신규Job수`

- F최소→`Σ_j(checkpoint_age_j+660)Σ_g raw_score_g x_jg`최소→변경수순. 첫F는E29단일결정surrogate에서공통잔여시간상수를제외하고실제해제대기를추가한것이다.
- **다기간기대flow time이나보정된기대오류손실이아니다.** raw위험은동률배정·후보선택기준이다. DP제안부분정확성과전체근사탐색을구분하고global gap/bound는null,FEASIBLE_APPROXIMATE로보고한다. longterm논문수준목표완성으로표현하지않음.
- 첫전기간검증의목적은이구조의실제운영지표를확인하는것이며결과를보고penalty/설정을유리하게바꾸지않는다. H/terminal/확률보정/현실적duration추정은후속이다.

## 실행·중단·산출물

- `connection.py`:작은phase/누수/Gang/DPoracle검사+실제첫6시간을세arm에서실행. PASS면계속 `supervise.py --gate FROZEN/GATEPASS`로3개전기간실행한다. 승인재요청이나기초검사완료로종료하지않는다.
- 전기간3arm 병렬,당초hardtimeout7200초·30초processalive/progress감시. 사용자명시승인으로최초시작기준21600초로연장했다. `analysis/extend_guard.py`가오래된감독프로세스만잠시정지해7200초중단을막고worker는그대로유지했으며,전부종료후감독기를재개했다(resume status0). `time_budget_extension.json`·`monitor_extension.json`이효력있는연장기록이고원래monitor의7200은초기설정이다. 로그정체는알림만,자동재시도/결과덮어쓰기없음.
- runs/arm: manifest·summary·progress JSON,전체jobs Parquet,GPU상태시간Parquet,events/decisions압축JSONL.
- reports:connection·monitor·comparison JSON/CSV·stdout/stderr. 누적연대기MD/HTML·프로젝트목표/진행갱신.
- 주평가:전체도착Job기준restricted flow(실패/미완료는T에서censor),완료·미완료·실패,전체큐대기,공통완료Job pairedJCT,완료/보존/잔여GPU초,오류/선점/periodicCP비용을단위분리.
- 반드시full140,786Job/100GPU/18,317판단·hash·진행/rollback·GPU용량시간·체류시간적분을검사한다. 단일seed·고정기간개발평가이며통계적우월성/일반화주장안함.

## 실행 경로 확인

- `reports/connection.json`: PASS,8개검사그룹(40개DP작은문제포함)과실제첫6시간863Job/72판단을세arm에서통과. 내부45.281초. 성능튜닝에사용하지않음.
- `reports/connection_executor.json`: 실제자식실행exit0·50.319초·stdout/stderr·시간기록. 연결검사1회.
- PASS직후 `supervise.py --gate FROZEN/GATEPASS`가세run을시작했다. 초기launcher PID와실제worker PID는Windowsvenv리다이렉터때문에다르며actual worker는각progress/manifest의pid를따른다.
- 감독세션은담당code_runner agent가감시하고,메인은reports/monitor.json·runs/*/progress.json을확인한다. 메인에서에이전트전용세션ID에직접접속한시도는Unknown process id로거부됐지만실험자체의오류/재시도는아니다.
- 본실행중E30코드hash와frozen152파일hash일치를재확인했다. 본실행시작이후정책/비용/코드를수정하지않음. 최종완료후다시검사한다.

## 완료 결과·검증 (2026-10-24 갱신)

| 지표 | FIFO ON | Tiresias-style ON | joint ON |
|---|---:|---:|---:|
| 완료Job |74,822|75,340|88,697|
| 미완료Job |65,964|65,446|52,089|
| 미시작Job |65,934|65,329|52,057|
| 전체Job restricted 평균체류(초) |1,829,169.001|1,765,216.522|1,312,150.215|
| 완료유효GPU초 |379,810,856|377,337,839|409,094,968|
| 스케줄러선점(건) |0|113,398|612|
| 활성XID작업중단(건) |70|69|82|
| 오류rollback GPU초 |202,620|150,765|85,350|
| 32GPU이상완료/631개 |231|234|14|
| 실행wall초 |5,979.234|6,807.750|5,390.047|

- `comparison.json`: PASS_DESCRIPTIVE_APPROXIMATE_SURROGATE. joint/Tiresias 체류−25.666%·완료+13,357개. 완료작업모집단이달라완료평균JCT만으로판정하지않는다. pairwise공통완료75,119개에서joint JCT평균−691,901.480초,느려진작업934개도있다.
- `final_verification.json`: PASS. 152frozen파일·코드/입력/provenance·전체모집단·큐/CP/진행/rollback·GPU상태시간·체류적분검증. 독립재실행검증은아니다.
- `limitations_evidence.json`: 기존작업표를수요별재집계한기술통계. 새시뮬레이션/튜닝없음. 수요32GPU이상631개중joint14개완료·615개미시작;32GPU450개중11개,64GPU167개중3개완료. joint최대누적큐대기5,297,726초다.

## 주요 한계 및 연구 주장 경계

1. **공정성 미보장**: F는정확한기대flow time이아니라300초미선택비용·360초선점비용·60초해제대기비용이다. GPU수대비많은작업을선택할유인이있고최대대기/대형Gang서비스하한이없다. 대형작업완료악화가실제관측됐으므로평균개선을최종정책우월성으로승격하지않는다.
2. **ML 기여 미분리**: 모든arm이같은위험을쓴다. joint개선에는Queue선택·선점억제·배정이함께달라진다. joint활성XID중단82건은Tiresias69건보다많고rollbackGPU초는더적다. rawscore노출·오류횟수·유실작업량의개선을동일시하지않는다.
3. **위험과 비용의 경제적 결합 부재**: raw24hscore는F동률때만작용한다. 기대고장확률·5분회피가능오류손실과전환비용을같은척도로최적화하지않는다. 원정보1800초유지이며새300초ML추론이아니다.
4. **탐색/배정 제한**: 오래된queued256+Gang대표+running,정/역DP+FIFO3개proposal을평가한다. 후보제한16,806/18,317회,전역bound/gap없음. 기존선택lease는GPU집합고정,신규greedy위치이며적극적예방이주/격리없음. DP부분정확성은전체정책최적성을뜻하지않는다.
5. **기준선/비용 시나리오**: FIFO는strictHOL/backfill없음,Tiresias는2큐3600GPU초/승급2의style로원형전체재현아님. CP300유효연산마다60정지·선점60+300·R0gap660은실측종합분포가아니다. periodic정지는E27과달라직접성능대조불가.
6. **사건·워크로드 현실성**: 앱/Job PID매핑없이GPU-local앱민감도,이번하드웨어복구0·GPU비가용0이다. 이동으로소프트웨어오류가해결되거나진단/정비가이득을낸증거가아니다. 원전체1,992GPU중고정100cohort에클러스터도착작업을투입해offered load198.616%;완료trace만선택·빈초기상태·고정서비스속도·네트워크/공유I/O비용미모형화가남는다.
7. **평가/재현 경계**: 동일개발기간·seed17의1회기술비교,기간종료에미완료절단. 통계적유의성·독립기간일반화없음. PYTHONHASHSEED미고정/미기록과active set동률순서영향으로독립bitwise재현도미검증이다. manifest코드hash는root*.py이며frozen의존소스는152파일보존검사와함께해석한다.

판정: 실제전체실행·회계/입력충실도통과,논문수준최종최적화/정책채택보류. 후속은공정성보호·ON내Queue/Placement ablation·회피가능위험보정/다기간목적의별도계획승인이다.

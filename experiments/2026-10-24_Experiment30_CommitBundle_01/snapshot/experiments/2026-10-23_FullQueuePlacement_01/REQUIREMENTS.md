# E30 실제 전체 실행 요구사항

사용자 최신 승인: G-009 실제 trace 연결·3개 전체기간 ON 실행·비교까지 완료. E29의기초검증종료를반복하지않음.

| ID | 출처/의도 | 실제 구현/호출 | 설정 | 수집할 증거 | 현재 판정 | 차이·후속 |
|---|---|---|---|---|---|---|
| R01 | 사용자 필수Parquet·ON | run.build→E25 load_inputs→FullSimulation.refresh_risk/decision→policies.pack | E27risk/provenance hash, 모든arm ON | 각manifest·18,317decision·risk exposure | full PASS·hash전수PASS | raw24hscore이며5분확률아님 |
| R02 | 전체데이터·5분 | frozen Job/XID/topology+fullE27risk | 140,786Job·100GPU·18,317ticks | summary·jobs·동일input감사 | 세arm full PASS | 원모집단1,992중기존100cohort·클러스터전체도착부하 stress |
| R03 | FIFO/Tiresias 기준선 | policies.choose/FullSimulation._views | strictFIFO,2큐3600GPU초/승급2 | decision타깃·service/promotion | 연결/full완료 | Tiresias-style,원형전체재현아님 |
| R04 | Queue+Placement 공동변수 | capacity_dp→pack→plan_value→완전mapping비교 | oldqueue256+Gang대표+running,정/역DP+incumbent | selected_differs_from_fifo·후보수·F/risk·latency | 작은40oracle·full완료,선택변경17,046회 | 단일결정surrogate근사,globalbound/gap없음 |
| R05 | 같은lease 비용0 | 선택incumbent GPU유지/interrupt실제변경 | scheduling60+300 | 실제시각event·CP예약·중단초 | fixture confirmed | 위험변경만으로의무재배치안함 |
| R06 | 공통정확비용 | engine.advance/process_events/interrupt | periodic300/60pause·실제선점CP60 GPU점유·R0retry660 | 진행/유실·GPU상태시간·CP/복구·체류적분 | fixture/full회계전수PASS | periodic60는공통가정;E27무정지와직접대조안함 |
| R07 | 과거정보만 | JobView에duration/label/미래XID없음,실제환경만접근 | horizon24h점수그대로 | dataclass필드검사·고정hash·runtimepolicy경로 | 필드검사 confirmed | 현실적duration추정/확률보정미구현 |
| R08 | 끝까지실행/감시 | connection PASS→supervise→3run→evaluate | 初7200s→사용자승인21600s·30s감시·병렬·재시도0 | monitor/extension/로그/exit/comparison | 전부exit0·COMPLETED | worker유지·감독기연장만적용 |
| R09 | 비교정직성 | evaluate.comparison + analysis.verify/limitations | 전체Job restrictedflow·공통완료pairedJCT·GPU초/작업초분리 | comparison·final_verification·limitations_evidence | 기술비교/회계PASS·정책채택보류 | 단일기간/seed,공정성악화·ML기여미분리 |

2026-10-24 갱신: 전체run·comparison·회계/보존검사완료. 152frozen파일정확보존. 범위를한정한PASS이며독립재현·전역최적성·최종정책우월성은아니다. 32GPU이상완료234→14개악화로공정성검토필수. 입력어댑터/전체엔진은E30이고ML을재학습한것이아니다.

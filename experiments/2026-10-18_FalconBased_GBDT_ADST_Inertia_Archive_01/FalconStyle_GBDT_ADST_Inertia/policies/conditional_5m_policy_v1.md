# 조건부 5분 위험 점수 산출 계약 v1

사용자는 현재 30분 실험을 완료한 뒤 B1 또는 Fusion의 성능 향상이 확인되면 추가 승인 없이 5분 간격의 후속 실행을 진행하도록 승인했다. 이 문서는 그 조건을 결과 확인 전에 고정한다. 기존 30분 실행의 main·config·vendor·학습·점수는 변경하지 않는다.

현재 supervisor PID 22224의 생성시각과 명령줄 SHA256을 정책 JSON에 고정했다. 이 프로세스를 종료하지 않고 최대 12시간 기다린다. `COMPLETED_AND_AUDITED`, 독립 감사 64 origins/6,081,576행/3,053시점/52,945양성, purge 및 라벨 성숙 위반 0을 모두 확인한 뒤에만 후속 후보를 평가한다. 현재 실행이 실패·시간초과·부분 완료이면 후속 학습/추론을 시작하지 않고 watcher 상태에 이유를 남긴다.

기준은 완료된 기존 GPU E19의 동일 64개 origin 점수다. B1의 기준 열은 `b1_probability`, Fusion의 기준 열은 `fusion_dual_inertia`다. 기준 파일의 `(timestamp,gpu_uid)`를 새 실험의 동일 키에 맞추고, 새 실험의 검증된 동일 키 라벨을 사용한다. 기준과 후보의 전체 GPU 모집단·기간·라벨이 다르면 선정을 중단한다.

- B1 후보 8개: 각 ST7/ST14/ST21/ADST 정책의 Inertia off/on 점수 `q_<base>`.
- Fusion 후보 16개: 같은 8개 조건의 상한 없는 Pareto/E19 기반 점수 `risk_<arm>`.
- 통과 조건: origin별 전체 행 AP의 64일 평균이 해당 기준보다 `1e-12` 초과 증가하며, pooled Recall@100은 기준보다 `1e-12` 이상 감소하지 않는다.
- 통과한 후보 중 평균 AP가 가장 높은 조건 하나만 선정한다. AP 차이가 `1e-12` 이내면 pooled Recall@100이 높은 조건, 이후 열 이름 사전순으로 정한다. B1·Fusion 종류에 우선권을 주지 않는다.
- 통과 조건이 없으면 `NO_POSITIVE_IMPROVEMENT`로 끝낸다. 24개 후보와 기준의 수치·차이·통과 여부를 모두 `selection.json`에 공개한다. 선정에는 유의성 검정 통과를 추가로 요구하지 않는다.

이는 이미 분석 중인 개발기간에서의 선택이다. 새 held-out 성능이나 다른 구조 대비 인과적 향상으로 해석하지 않는다. 원본 E19와 모델·특징이 다르므로 성능 차이를 Fusion이나 Inertia 하나의 효과로 귀속하지 않는다. source execution contract/main/vendor와 저장 모델·점수·기준 파일 SHA256을 남긴다.

후속은 선정한 한 조건의 기존 일일 모델과 정책 이력을 유지한 5분 추론이다. 36시간 purge와 특징 cutoff를 유지하고, Inertia 감쇠는 실제 5분 경과시간에 맞춘다. 기존 모델을 다시 학습하거나 현재 30분 코드의 hash를 바꾸지 않는다. 새 5분 진입점에서 첫 origin → 독립 감사 → 전체 64 origins resume → 독립 감사 및 누적 기록 순서로 진행한다. 첫 origin 또는 감사가 실패하면 전체 실행을 시작하지 않는다.

watcher CLI:

```text
python tools/conditional_5m.py --source runs/full_01 --target runs/conditional_5m_01 --wait-supervisor-pid 22224 --max-wait-seconds 43200 --max-run-seconds 43200
```

새 추론·감사 프로세스만 자식으로 실행하고, 창을 숨기며, 표준출력·오류·UTC/KST 상태와 프로세스 정체성을 기록한다. 후속 실행 시간 상한은 실제 후속 프로세스 시작부터 12시간이다. 시간초과 때 이 watcher가 만든 자식만 종료하며 자동 재시도는 없다. 외부 메시지·데이터 업로드·Git 작업·cron/heartbeat는 실행하지 않는다.

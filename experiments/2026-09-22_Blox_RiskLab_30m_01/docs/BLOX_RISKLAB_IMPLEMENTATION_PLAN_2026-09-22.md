# Blox + RiskLab 실제 데이터 기반 구현계획서

작성일: 2026-09-22  
상태: 승인된 구현계획  
관련 목표: G-007, G-008, G-009  
범위: Branch 1·2 ML 결과를 Blox/RiskLab 운영 시뮬레이터로 연결

> 이 문서는 구현계획서다. 작성 시점에는 코드 수정, 데이터 변환, ML 재학습, simulator 실행을 수행하지 않는다.

## 1. 결정사항 요약

질문을 통해 다음 사항을 확정했다.

1. ML probability는 RiskLab에서 binary placement mask를 만드는 데 사용한다.
2. 실제 오류 손실의 primary 평가는 관측된 XID event tape replay로 수행한다.
3. probability 기반 stochastic failure 생성은 sensitivity 분석으로만 둔다.
4. probability threshold는 simulator 결과가 아니라 ML validation에서 선택한다.
5. validation threshold는 99% recall을 기본 목표로 한다.
6. 90%·95% recall threshold는 sensitivity 분석으로 보고한다.
7. high/critical GPU 때문에 Gang-Job을 구성할 수 없으면 fallback 없이 queue에 남긴다.
8. 실제 GPU manifest에서 고정된 연속 100 GPU cohort를 사용한다.
9. 실제 GPU-to-server mapping을 유지한다.
10. server correlated failure는 core simulator에서 제외하고 별도 sensitivity로 둔다.
11. fixed-size synchronous Gang-Job을 기본 실행 모델로 둔다.
12. Gang-Job의 한 GPU에서 failure가 발생하면 전체 Gang-Job을 checkpoint/recovery/relaunch한다.
13. active migration도 한 GPU만 교체하지 않고 전체 Gang-Job 단위 checkpoint/relaunch로 모델링한다.
14. failure loss와 preventive overhead를 분리한다.
15. normal queue wait는 failure lost time에 포함하지 않는다.
16. checkpoint 정보가 있으면 실제값을 사용하고, 없으면 고정값을 기본으로 하며 sensitivity를 추가한다.
17. 실제 XID는 All-XID 기준 GPU별 onset episode로 만든다.
18. 5분 bucket 기준 최대 10분 gap까지 같은 XID episode로 병합한다.
19. Blox scheduler는 FIFO/LAS/Tiresias2Q를 선택 가능하게 유지한다.
20. 핵심 Placement 비교에서는 scheduler를 고정한다.
21. Placement core objective는 binary eligibility 안에서 사용하는 server 수 최소화다.
22. probability가 낮은 eligible GPU를 추가로 정렬하는 방식은 core가 아니라 sensitivity다.
23. dashboard와 batch runner 모두 실제 100 GPU 전체 replay를 지원한다.
24. dashboard replay는 background worker로 실행하고 UI는 progress/result를 polling한다.
25. 실행 경로에서는 synthetic preset을 제거하고, synthetic은 unit/integration test fixture로만 보존한다.

## 2. 목적과 비목적

### 2.1 목적

이 구현은 다음 pipeline을 재현 가능하게 만드는 것을 목적으로 한다.

```text
Branch 1·2 ML output
        │
        ▼
GPU별 probability tape
        │
        ▼
Validation에서 고정한 threshold 적용
        │
        ▼
Binary eligibility mask
        │
        ├──────────────► Blox scheduler: Job order
        │
        └──────────────► PlacementPolicy: eligible GPU + 최소 server 수
                                           │
                                           ▼
                              RiskLab event/recovery engine
                              XID replay / maintenance / migration
                                           │
                                           ▼
                                      Loss Ledger
```

핵심 연구 시스템의 역할은 다음과 같이 나눈다.

| 구성요소 | 책임 | 책임지지 않는 것 |
|---|---|---|
| ML pipeline | probability 생성, validation threshold 선택, manifest 생성 | simulator 결과를 이용한 threshold 조정 |
| Blox scheduler | Job priority/order, scheduler별 preemption 결정 | GPU failure, XID, recovery loss |
| PlacementPolicy | eligible GPU 안에서 Gang-Job GPU 집합 결정 | ML 재학습, failure 확률 생성 |
| RiskLab engine | GPU/Job 상태, XID replay, checkpoint, maintenance, recovery | scheduler priority 자체의 재구현 |
| LossLedger | failure loss와 preventive overhead 분리 집계 | 단일 임의 composite score만 출력 |
| Dashboard | 전체 replay 실행 요청, 진행상황·이벤트·결과 표시 | 별도 simulation 로직 보유 |

### 2.2 비목적

- Blox 원본 scheduler를 새로 구현하지 않는다.
- Themis auction을 구현하지 않는다.
- Gandiva time-slicing, grow/shrink를 core에 넣지 않는다.
- continuous confidence ensemble을 추가하지 않는다.
- 실제 cluster에 명령을 보내는 online controller를 만들지 않는다.
- probability를 5분 failure 확률로 임의 변환해 primary 결과를 만들지 않는다.
- server-level correlated failure를 실제 관측 없이 core 가정으로 넣지 않는다.

## 3. 현재 코드 진단과 재사용 범위

현재 RiskLab은 Blox와 reliability event layer가 연결된 PoC 상태다.

| 파일 | 현재 역할 | 구현에서의 처리 |
|---|---|---|
| `risklab/domain.py` | GPU, Job, Event dataclass와 상태 | GPU eligibility/topology, Job loss component를 확장 |
| `risklab/engine.py` | 시간 loop, Job 진행, failure, drain, maintenance, 결과 | provider와 placement를 명시적으로 주입하고 loss 계산을 분리 |
| `risklab/blox_adapter.py` | 실제 Blox scheduler/placement import와 snapshot 변환 | scheduler order와 custom placement를 분리 |
| `risklab/external_risk.py` | dashboard risk feed와 monkey-patching | legacy compatibility는 유지하되 core 경로는 explicit interface로 전환 |
| `risklab/config.py` | dashboard preset과 64 GPU 제한 설정 | 실제 data manifest 기반 config를 추가하고 실행 경로에서 synthetic preset 제거 |
| `risklab/controller.py` | 현재 thread 기반 run 시작·상태 전달 | 100 GPU replay background worker 관리자로 확장 |
| `risklab/storage.py` | run/history 저장 | manifest, JSONL event, summary, report artifact를 확장 |
| `risklab/web*.py` | dashboard API | run 시작·상태 조회·event tail·summary 제공 |

현재 `external_risk.py`는 `RiskLabSimulation.__init__`, `update_risks`, `apply_preventive_drain`, `schedule`, `snapshot`, `BloxAdapter.decide`를 monkey-patching한다. 이 구조는 빠른 PoC에는 유용하지만, 실제 연구 결과를 재현하려면 호출 경로가 명시적인 provider/policy dependency여야 한다. 따라서 monkey-patching은 즉시 삭제하지 않고 legacy compatibility로 격리하며, 새 replay 경로는 explicit interface만 사용한다.

현재 `config.py`는 `nodes`와 `gpus_per_node`를 각각 최대 8로 제한해 최대 64 GPU만 표현한다. 100 GPU 연구에서는 `nodes × gpus_per_node`를 인위적으로 늘리는 대신 `gpu_manifest`의 실제 row 수를 cluster capacity로 사용한다.

현재 `engine.py`의 `operational_loss_index`는 checkpoint GPU-hours, maintenance GPU-hours, 평균 queue wait를 하나의 가중합으로 계산한다. 이 값은 PoC 운영 지표로는 보존할 수 있지만, 본 연구의 primary metric으로 사용하지 않는다.

## 4. 최종 아키텍처

### 4.1 모듈 구조

```text
                    ┌──────────────────────────────┐
                    │ ML output artifacts          │
                    │ probability tape             │
                    │ threshold/model manifests    │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ Feed layer                    │
                    │ RiskTapeProvider              │
                    │ FailureTapeProvider           │
                    │ JobTraceProvider              │
                    │ TopologyProvider              │
                    └──────────────┬───────────────┘
                                   │
           ┌───────────────────────▼──────────────────────┐
           │ RiskLabSimulation                              │
           │                                                │
           │ current risk → mask → state transition         │
           │ current XID → gang failure/recovery            │
           │ current jobs → Blox scheduler order            │
           │ order + eligible GPUs → PlacementPolicy        │
           │ events → LossLedger                             │
           └───────────┬───────────────────────┬────────────┘
                       │                       │
          ┌────────────▼──────────┐  ┌─────────▼───────────┐
          │ CLI replay runner     │  │ Dashboard worker    │
          │ deterministic batch   │  │ background process  │
          └────────────┬──────────┘  └─────────┬───────────┘
                       │                       │
                       └───────────┬───────────┘
                                   ▼
                     run manifest/events/summary/report
```

### 4.2 시간 진행 순서

모든 runner는 같은 engine loop를 사용한다. 한 decision interval은 기본 300초다.

```text
for t in decision_times:
    1. 이전 round의 running Job progress 반영
    2. maintenance 완료와 cooldown 상태 갱신
    3. RiskTapeProvider에서 p(i,t) 읽기
    4. 현재 시각까지 도착한 Job admission
    5. FailureTapeProvider에서 XID episode event(t) 읽기
    6. event가 해당 GPU의 running Gang-Job에 영향을 주면 group recovery
    7. threshold로 binary eligibility 갱신
    8. active migration policy면 excluded GPU를 포함한 Gang-Job 재배치 예약
    9. Blox scheduler가 Job order와 suspend 대상을 결정
   10. PlacementPolicy가 eligible GPU에서 allocation 생성
   11. Gang-Job launch/relaunch
   12. EventLedger와 LossLedger 기록
   13. dashboard/batch snapshot 기록
```

현재 시각의 XID event는 관측된 상태 변화로 처리할 수 있지만, 미래 XID event는 decision에 사용하지 않는다. `probability_tape`와 `failure_tape`를 별도 provider로 두는 이유가 이 leakage를 방지하기 위해서다.

## 5. 입력·출력 계약

### 5.1 Probability tape

파일: `probability_tape.parquet`

필수 컬럼:

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `timestamp` | datetime | decision time, timezone 명시 |
| `gpu_uid` | string | 실제 telemetry GPU ID |
| `probability` | float32 | ML이 출력한 probability |
| `model_version` | string | ML artifact version |

검증 규칙:

- `probability`는 `[0, 1]`이어야 한다.
- 동일한 `(timestamp, gpu_uid)`는 하나만 허용한다.
- 100 GPU cohort의 모든 decision time이 존재해야 한다.
- timestamp는 5분 decision grid에 정렬되어야 한다.
- threshold는 이 파일의 값으로 다시 선택하지 않는다.

### 5.2 Threshold manifest

파일: `threshold_manifest.json`

```json
{
  "model_version": "branch1_branch2_final_v1",
  "prediction_horizon": "24h",
  "decision_interval_seconds": 300,
  "selection_split": "validation_only",
  "target_recall": 0.99,
  "threshold": 0.72,
  "threshold_rule": "highest_threshold_with_validation_recall_at_least_target",
  "gpu_mapping_version": "gpu_map_v1"
}
```

threshold 선택 규칙:

1. validation probability와 onset label로 threshold 후보를 계산한다.
2. validation recall이 0.99 이상인 후보만 남긴다.
3. 그중 가장 높은 threshold를 선택한다.
4. validation에서 목표 recall을 달성할 수 없으면 성공한 것으로 처리하지 않고 ML 단계의 infeasible 결과로 기록한다.
5. test label이나 simulator lost time은 threshold 선택에 사용하지 않는다.

90%와 95% recall operating point는 threshold sensitivity로 저장하지만, primary threshold는 99% recall manifest를 사용한다.

### 5.3 Failure tape

파일: `failure_tape.parquet`

필수 컬럼:

| 컬럼 | 의미 |
|---|---|
| `timestamp` | XID onset timestamp |
| `gpu_uid` | XID가 관측된 GPU |
| `xid_codes` | episode에서 관측된 XID code 목록 |
| `episode_id` | GPU별 episode 식별자 |
| `episode_start` | episode의 첫 onset |
| `episode_end` | episode의 마지막 관측 |

XID episode 생성 규칙:

- All-XID를 사용한다.
- GPU별 XID가 clean 상태에서 non-zero로 전환되는 시점을 onset으로 본다.
- 5분 bucket 기준 10분 이하 gap은 같은 episode로 merge한다.
- 같은 episode 안의 지속 관측은 여러 failure로 세지 않는다.
- simulator failure event의 timestamp는 `episode_start`다.
- XID가 다시 clean 상태가 된 뒤 새로운 onset이 생기면 새 episode다.

### 5.4 Job trace

파일: `job_trace.parquet`

필수 컬럼:

| 컬럼 | 의미 |
|---|---|
| `job_id` | trace Job ID |
| `arrival_time` | Job arrival time |
| `duration_seconds` | 정상 실행시간 |
| `gpu_demand` | Gang GPU demand |
| `gang_id` | Gang-Job 식별자, 단일 Job이면 job_id와 동일 |
| `checkpoint_interval_seconds` | 있으면 실제값, 없으면 null |

실제 trace에 checkpoint interval이 없으면 recovery config의 고정값을 사용한다. 이 경우 manifest에 `checkpoint_source: fixed_default`를 기록한다.

### 5.5 GPU topology manifest

파일: `gpu_manifest.parquet` 또는 `gpu_manifest.json`

```text
gpu_uid
server_id
local_gpu_id
node_index
rack_id (optional)
cohort_index
```

선택 규칙:

- 실제 manifest를 고정된 정렬 기준으로 정렬한다.
- `cohort_index`가 0부터 99인 연속 100 GPU를 primary cohort로 사용한다.
- 모든 policy/run은 동일한 cohort를 사용한다.
- server packing은 `server_id` 기준으로 계산한다.
- server mapping이 없는 GPU는 server placement 결과에 사용하지 않는다.

### 5.6 Run manifest

파일: `runs/<run_id>/manifest.json`

필수 기록:

```json
{
  "run_id": "...",
  "source_commit": "...",
  "model_manifest_sha256": "...",
  "probability_tape_sha256": "...",
  "failure_tape_sha256": "...",
  "job_trace_sha256": "...",
  "gpu_manifest_sha256": "...",
  "cohort_size": 100,
  "decision_interval_seconds": 300,
  "threshold": 0.72,
  "target_recall": 0.99,
  "scheduler": "Las",
  "placement_policy": "packed_eligible",
  "active_migration": false,
  "server_correlated_failure": false,
  "random_seed": 17
}
```

## 6. Risk와 eligibility 명세

### 6.1 Core decision

Core placement은 continuous probability ranking이 아니라 binary mask를 사용한다.

```python
excluded = probability >= threshold
eligible = not excluded
```

display용 `low/caution/high/critical` class는 dashboard와 분석을 위해 기록할 수 있지만, core allocation은 `eligible/excluded` 값만 사용한다.

### 6.2 Probability-based failure sensitivity

primary run에서는 실제 `failure_tape`를 사용한다. probability 기반 failure 생성은 별도 모드로만 지원한다.

24시간 probability를 5분 event probability로 직접 사용하는 것은 금지한다. 필요하면 별도 hazard 변환 가정과 변환식을 manifest에 기록해야 한다. 변환 근거가 없는 상태에서는 probability-based failure simulation을 실행 결과로 보고하지 않는다.

## 7. Gang-Job 실행 명세

### 7.1 Fixed-size synchronous semantics

실제 framework 정보가 없는 trace를 위해 fixed-size synchronous Gang-Job을 기본 모델로 둔다.

```text
Gang-Job demand = g
실행 조건 = g개 GPU가 동시에 allocation되어야 함
GPU 하나의 collective failure
  → 전체 Gang-Job interruption
  → 전체 assignment release
  → checkpoint loss + recovery delay
  → queue 복귀
  → 새 g개 GPU로 relaunch
```

NCCL 문서는 rank/communicator 오류를 communicator 전체에서 abort하고 재생성하는 fault-tolerance 흐름을 설명한다. TensorFlow의 synchronous multi-worker 문서도 한 worker가 unavailable해지면 다른 worker가 실패할 수 있고, 실패한 worker와 영향을 받은 worker를 restart해야 한다고 설명한다. PyTorch의 fault-tolerance 예시는 snapshot에서 모든 process를 자동 restart하는 방식을 사용한다.

출처:

- [NCCL Fault Tolerance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
- [PyTorch fault-tolerant distributed training](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
- [TensorFlow multi-worker fault tolerance](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)

### 7.2 Job state

```text
FUTURE
  └─ arrival → QUEUED
QUEUED
  └─ allocation → RUNNING
RUNNING
  ├─ normal completion → COMPLETED
  ├─ XID/failure → RECOVERING → QUEUED
  └─ preventive group migration → MIGRATING → RECOVERING → QUEUED
```

`MIGRATING`은 failure recovery와 preventive migration을 event-level에서 구분하기 위한 상태다. 둘 다 전체 Gang-Job을 release하고 다시 배치하지만, loss ledger의 원인이 다르다.

### 7.3 GPU state

```text
AVAILABLE
  └─ launch → RUNNING
RUNNING
  ├─ normal completion → AVAILABLE
  ├─ XID → FAILED
  └─ preventive exclusion → DRAINED
DRAINED
  └─ maintenance slot → MAINTENANCE
FAILED
  └─ maintenance slot → MAINTENANCE
MAINTENANCE
  └─ maintenance complete → COOLDOWN
COOLDOWN
  └─ cooldown complete → AVAILABLE
```

위험도가 다시 낮아지는 것만으로 GPU를 available로 되돌리지 않는다. maintenance와 cooldown을 거쳐야 다시 eligible candidate가 된다.

## 8. PlacementPolicy 명세

### 8.1 Core packed eligible policy

입력:

```text
ordered queued jobs
GPU state snapshot
GPU → server topology
binary eligibility mask
```

처리:

```text
for job in scheduler_order:
    candidates = free GPUs with eligible == true
    if total candidates < job.gpu_demand:
        leave job queued
        continue

    server_candidates = group candidates by server_id
    choose server subset that satisfies demand and minimizes:
        (number_of_servers,
         unused_selected_capacity,
         fragmentation,
         deterministic_server_id_order)

    choose deterministic GPU IDs inside selected servers
    emit atomic Gang placement
```

100 GPU cohort가 일반적으로 8 GPU/server라면 약 13개 server subset이므로 exact subset search 또는 capacity dynamic programming이 가능하다. 전체 cluster로 확장할 때는 서버별 eligible capacity를 이용한 dynamic programming으로 전환한다.

### 8.2 Supported comparison policies

| Policy | 설명 |
|---|---|
| `risk_blind_packed` | risk mask를 무시하고 최소 서버 수로 배치 |
| `risk_mask_packed` | high/excluded GPU를 제거하고 최소 서버 수로 배치, core |
| `risk_mask_random` | 동일 eligible 집합에서 seeded random 배치 |
| `risk_mask_packed_nonsticky` | active Gang-Job 전체를 checkpoint/relaunch해 재배치 |
| `risk_mask_probability_tiebreak` | eligible 내부 probability를 tie-break로 사용, sensitivity |

핵심 비교에서는 scheduler를 하나 고정하고 위 PlacementPolicy를 비교한다. FIFO/LAS/Tiresias2Q 변경은 별도의 scheduler sensitivity로 둔다.

### 8.3 No-fit 정책

eligible GPU로 Gang-Job을 만들 수 없으면 fallback하지 않는다.

```text
no feasible eligible Gang placement
→ Job stays QUEUED
→ no excluded GPU allocation
→ queue delay/starvation recorded separately
```

threshold relaxation, least-risk excluded GPU fallback은 향후 정책 비교로만 둔다.

## 9. Failure, migration, maintenance 명세

### 9.1 Reactive failure

```text
XID episode arrives at t
  ├─ GPU idle: GPU → FAILED, no running Job loss
  └─ GPU running:
       owning Gang-Job → RECOVERING
       all assigned GPUs release
       failure_checkpoint_loss added
       failure_recovery_delay applied
       GPU → FAILED
       Job returns to QUEUED
```

### 9.2 Preventive group migration

active migration policy가 켜져 있고 running Gang-Job의 GPU 하나가 excluded로 바뀌면:

```text
running Gang-Job
  → checkpoint all workers
  → release all GPUs
  → record preventive checkpoint/migration overhead
  → GPU(s) DRAINED
  → Job RECOVERING
  → Job QUEUED
  → new full Gang placement
```

GPU 한 장만 교체한 뒤 나머지 worker를 계속 실행하는 모델은 core에 사용하지 않는다. 이는 fixed-size synchronous semantics와 맞지 않으며, elastic communicator/rendezvous가 필요한 별도 연구다.

### 9.3 Maintenance capacity

현재 RiskLab의 maintenance capacity와 duration 개념은 재사용한다.

- `DRAINED` 또는 `FAILED` GPU만 maintenance queue에 들어간다.
- 동시 maintenance 수는 config로 고정한다.
- maintenance 완료 후 cooldown을 적용한다.
- cooldown 중 GPU는 physically available로 표시할 수 있지만 Placement candidate에서는 제외한다.

## 10. LossLedger 명세

### 10.1 분리 지표

```text
failure_lost_time_seconds
  = failure_checkpoint_loss_seconds
  + failure_recovery_delay_seconds
  + failure_gpu_unavailable_seconds

preventive_overhead_seconds
  = preventive_checkpoint_seconds
  + preventive_relaunch_seconds
  + maintenance_unavailable_seconds

scheduling_metrics
  = queue_wait_seconds
  + jct_seconds
  + makespan_seconds
```

### 10.2 기록 원칙

- reactive failure와 preventive migration은 서로 다른 event type으로 기록한다.
- 같은 checkpoint loss를 두 번 더하지 않는다.
- queue wait는 failure loss에 포함하지 않는다.
- composite metric은 보조 출력으로만 남긴다.
- Job-level, Gang-level, GPU-level 집계를 모두 보존한다.

### 10.3 출력 예시

```json
{
  "failure_lost_time_seconds": 1840,
  "failure_checkpoint_loss_seconds": 900,
  "failure_recovery_delay_seconds": 600,
  "failure_gpu_unavailable_seconds": 340,
  "preventive_overhead_seconds": 720,
  "maintenance_gpu_hours": 2.4,
  "avg_queue_wait_seconds": 810,
  "p95_queue_wait_seconds": 1900,
  "avg_jct_seconds": 2400,
  "completed_jobs": 87,
  "starved_jobs": 3,
  "migrations": 5
}
```

## 11. 파일별 구현계획

### 11.1 `risklab/domain.py`

수정 내용:

- GPU에 `server_id`, `risk_probability`, `eligible`, `risk_class`, `cooldown_until` 추가
- GPU state에 `cooldown` 추가 또는 `available + cooldown_until` invariant 명시
- Job에 `gang_id`, `failure_checkpoint_loss`, `preventive_checkpoint_loss`, `recovery_seconds`, `migration_count` 추가
- `MIGRATING` Job state 추가
- Event에 `source`, `job_id`, `gpu_ids`, `episode_id` optional field 추가

### 11.2 `risklab/feeds.py` 신규

클래스:

```python
class RiskTapeProvider:
    def snapshot(self, at_time: int) -> dict[str, float]: ...

class FailureTapeProvider:
    def events_at(self, at_time: int) -> list[FailureEvent]: ...

class JobTraceProvider:
    def jobs(self) -> list[JobSpec]: ...

class TopologyProvider:
    def cohort(self, size: int) -> list[GPUSpec]: ...
```

공통 기능:

- input schema 검증
- timestamp 정렬 검증
- missing/duplicate row 검증
- checksum 계산
- cohort 고정
- manifest 정보 제공

### 11.3 `risklab/placement_policy.py` 신규

클래스:

```python
class PlacementPolicy(Protocol):
    def allocate(self, ordered_jobs, jobs, gpus, topology, *, seed): ...

class RiskMaskPackedPlacement:
    ...

class RiskMaskRandomPlacement:
    ...

class RiskBlindPackedPlacement:
    ...

class RiskMaskPackedNonStickyPlacement:
    ...
```

반드시 반환해야 하는 정보:

```json
{
  "job_id": 17,
  "gpu_ids": [3, 4, 11, 12],
  "server_ids": ["server-01", "server-02"],
  "server_count": 2,
  "eligibility_rule": "risk_mask",
  "placement_policy": "risk_mask_packed",
  "risk_values_used_for_selection": false
}
```

### 11.4 `risklab/loss_ledger.py` 신규

책임:

- event별 loss component 기록
- Job/Gang/GPU level aggregate
- failure/preventive/queue 분리
- summary JSON 생성
- duplicate event 방지

### 11.5 `risklab/blox_adapter.py`

기존 `decide()`를 내부적으로 다음 두 단계로 분리한다.

```python
order = adapter.schedule_order(job_state, cluster_state, scheduler_name)
decision = placement_policy.allocate(order, jobs, gpus, topology, seed)
```

Blox scheduler가 반환한 Job order는 보존한다. 기본 Blox placement는 reference path로 남기되, actual-data run에서는 custom PlacementPolicy가 최종 allocation을 생성한다.

### 11.6 `risklab/engine.py`

변경 내용:

- synthetic `_risk_for()`를 actual-data path에서 호출하지 않음
- `RiskTapeProvider`, `FailureTapeProvider`, `JobTraceProvider`, `PlacementPolicy`, `LossLedger`를 constructor dependency로 주입
- event loop 순서를 명시적으로 고정
- reactive failure와 preventive migration 분기
- Gang-Job 전체 recovery 구현
- no-fit queue 정책 구현
- state transition invariant 검사

### 11.7 `risklab/replay_runner.py` 신규

CLI 책임:

```text
manifest 읽기
→ input checksum 확인
→ provider 생성
→ scheduler/placement 조합 확인
→ RiskLabSimulation 실행
→ event/decision/summary artifact 저장
→ HTML report 생성
```

예정 명령 형태:

```powershell
python -m risklab.replay_runner `
  --manifest inputs/runs/example_manifest.json `
  --output runs/<run_id>
```

### 11.8 `risklab/controller.py`와 dashboard

dashboard는 HTTP 요청에서 simulator를 직접 실행하지 않는다.

```text
POST /runs
  → run_id 생성
  → replay_worker subprocess 시작
  → 202 Accepted 반환

GET /runs/{run_id}
  → status.json + latest snapshot

GET /runs/{run_id}/events?cursor=...
  → events.jsonl tail

GET /runs/{run_id}/summary
  → summary.json

POST /runs/{run_id}/stop
  → worker 종료 요청
```

Windows 환경의 process spawn을 고려해 worker entrypoint는 import side effect가 없고, `if __name__ == "__main__"` guard를 사용한다. UI는 전체 event log를 메모리에 올리지 않고 cursor/tail 방식으로 읽는다.

## 12. Synthetic 경로 처리

사용자 실행 경로에서는 synthetic preset을 제거한다.

### 제거 대상

- dashboard preset catalog에서 synthetic risk scenario 노출 제거
- `build_workload()`의 synthetic workload를 production/replay entry point에서 호출하지 않음
- scheduled/stochastic failure preset을 actual-data runner의 기본값으로 사용하지 않음

### 보존 대상

- `tests/fixtures/`의 4·6·16 GPU synthetic fixture
- Blox adapter unit test input
- state transition regression test input
- missing tape/duplicate timestamp/no-fit placement test data

Synthetic fixture는 실제 결과를 만드는 실행 경로가 아니라, 코드 동작을 검증하는 테스트 전용이다.

## 13. Build plan

### Phase 0 — baseline lock

목표: 현재 PoC를 변경하기 전에 동작 기준을 보존한다.

- 현재 Blox import와 RiskLab dashboard smoke 확인
- 기존 run result schema 백업
- current Blox source commit 기록
- legacy synthetic fixture를 테스트 fixture 위치로 복사
- baseline test command 정의

산출물:

- baseline manifest
- 기존 결과 schema snapshot
- regression fixture

### Phase 1 — ML output contract

목표: ML과 simulator의 경계를 먼저 고정한다.

- probability tape schema 생성
- 99% validation recall threshold 선택 함수 추가
- threshold/model manifest 생성
- GPU ID mapping version 기록
- test label과 simulator 결과가 threshold 선택에 들어가지 않는지 검증

통과 조건:

- 100 GPU cohort의 모든 timestamp가 존재
- duplicate/missing/NaN 검출
- threshold manifest만으로 RiskLab이 mask를 재생성

### Phase 2 — actual input providers

목표: 실제 tape를 읽되, engine 로직과 분리한다.

- `RiskTapeProvider`
- `FailureTapeProvider`
- `JobTraceProvider`
- `TopologyProvider`
- checksum과 manifest validation
- All-XID episode 생성 및 10분 gap merge

통과 조건:

- 같은 input checksum에서 같은 snapshot 생성
- 미래 timestamp를 provider가 반환하지 않음
- 100 GPU mapping이 실제 node/server와 일치

### Phase 3 — explicit Blox/RiskLab boundary

목표: monkey-patching 의존을 제거하고 scheduler/placement를 분리한다.

- Blox scheduler order API
- custom PlacementPolicy API
- actual-data engine dependency injection
- legacy external risk path 분리
- Blox risk-blind reference path 검증

통과 조건:

- risk-blind policy에서 기존 scheduler order가 보존
- Blox가 반환한 Job order와 custom placement가 정상 결합
- invalid allocation이 launch되지 않음

### Phase 4 — core Placement

목표: binary eligibility와 minimum-server Gang placement를 구현한다.

- `risk_blind_packed`
- `risk_mask_packed`
- `risk_mask_random`
- no-fit queue
- exact/dynamic-programming server subset 선택
- server count/fragmentation decision log

통과 조건:

- excluded GPU 신규 allocation 0건
- Gang demand와 allocated GPU count 일치
- 가능한 경우 minimum server count 검증
- 불가능한 경우 queue 유지

### Phase 5 — failure/recovery/loss

목표: 실제 XID replay와 group-level recovery를 구현한다.

- reactive group failure
- preventive group migration
- maintenance/cooldown
- checkpoint source: actual 또는 fixed default
- LossLedger
- failure/preventive/queue 분리 summary

통과 조건:

- GPU 하나의 XID가 전체 Gang-Job recovery를 유발
- 같은 checkpoint loss가 중복 계산되지 않음
- preventive migration이 failure loss로 잘못 집계되지 않음

### Phase 6 — dashboard full replay

목표: dashboard에서 전체 실제 tape를 background process로 실행한다.

- run creation endpoint
- worker process
- status file
- event tail API
- stop API
- progress/result UI
- large event log streaming

통과 조건:

- HTTP 요청이 simulator completion까지 block하지 않음
- worker failure가 run status에 기록됨
- 재실행 시 새로운 run_id와 manifest 생성

### Phase 7 — 100 GPU scale validation

목표: 실제 연속 100 GPU cohort와 full tape를 처리한다.

- memory profile
- provider read throughput
- placement decision latency
- event ledger size
- dashboard polling overhead
- deterministic rerun

이 단계 전에는 full-scale experiment result를 논문 수치로 사용하지 않는다.

## 14. Build와 실행 환경

실행 환경은 현재 RiskLab의 Python environment와 Blox checkout을 기준으로 한다.

예정 검증 명령:

```powershell
python -m compileall blox_repo_actual\blox-risklab\risklab
pytest -q blox_repo_actual\blox-risklab\tests
python -m risklab.replay_runner --help
python -m risklab.replay_runner --manifest <manifest> --output <run_dir>
```

실제 구현 시 Python executable, Blox source root, dependency lock, CUDA/NCCL는 run manifest에 기록한다. simulator 자체는 GPU를 사용하지 않는 offline replay를 primary로 한다.

## 15. 테스트 명세

### 15.1 Feed validation

- probability가 `[0,1]` 밖이면 실패
- NaN/inf probability 거부
- duplicate `(timestamp,gpu_uid)` 거부
- decision time 누락 거부
- GPU count 불일치 거부
- future failure event를 현재 provider가 반환하지 않음
- 실제 manifest에 없는 GPU tape 거부

### 15.2 Threshold

- validation recall 0.99 이상 threshold 선택
- 가장 높은 threshold 선택 규칙 검증
- test label 미사용 검증
- threshold manifest 재생성 결과 일치
- 99% recall이 달성되지 않는 경우 명시적 `infeasible` 반환

### 15.3 Placement

- 1 server에서 가능한 Gang은 1 server에 배치
- 여러 server가 필요한 Gang은 최소 server subset 선택
- excluded GPU는 신규 allocation에 포함되지 않음
- no-fit Job은 queue 유지
- random policy는 고정 seed로 재현
- 동일 GPU가 두 Job에 할당되지 않음
- Job demand와 allocation count가 일치
- 모든 Gang GPU가 동일 decision round에 할당됨

### 15.4 State transition

- `AVAILABLE → RUNNING → AVAILABLE`
- `RUNNING → FAILED → MAINTENANCE → COOLDOWN → AVAILABLE`
- `RUNNING → DRAINED → MAINTENANCE → COOLDOWN → AVAILABLE`
- cooldown 중 신규 allocation 금지
- risk가 낮아졌다는 이유만으로 immediate available 복귀 금지

### 15.5 Failure/recovery

- idle GPU XID는 GPU failure로 기록하되 Job loss는 0
- running GPU XID는 전체 Gang-Job recovery
- 한 episode가 여러 failure로 중복 집계되지 않음
- group recovery에서 모든 assignment release
- preventive migration과 reactive failure의 ledger 분리
- recovery 후 Job은 queue로 복귀

### 15.6 End-to-end

- 4 GPU actual-format fixture
- 16 GPU actual-format fixture
- 100 GPU topology fixture
- dashboard background worker run
- batch runner와 dashboard 결과 동일
- 동일 manifest/seed 결과 byte-level 또는 metric-level 재현
- worker crash 시 run status `failed`와 원인 기록

## 16. 결과 artifact 구조

```text
runs/<run_id>/
├─ manifest.json
├─ status.json
├─ decisions.jsonl
├─ events.jsonl
├─ snapshots.jsonl
├─ metrics.parquet
├─ summary.json
├─ job_timeline.parquet
├─ gpu_timeline.parquet
└─ report.html
```

`decisions.jsonl`에는 매 decision round의 다음 내용을 기록한다.

```json
{
  "time": 3600,
  "scheduler": "Las",
  "placement_policy": "risk_mask_packed",
  "eligible_gpu_count": 82,
  "excluded_gpu_count": 18,
  "queued_jobs": 21,
  "launches": [
    {
      "job_id": 12,
      "gpu_ids": [3, 4, 11, 12],
      "server_ids": ["server-01", "server-02"],
      "server_count": 2
    }
  ],
  "no_fit_jobs": [18, 20]
}
```

`events.jsonl`에는 다음 event type을 사용한다.

```text
risk_snapshot
risk_mask_change
job_admitted
job_launch
job_complete
xid_episode
gang_failure
gang_checkpoint
preventive_migration
gpu_drained
maintenance_start
maintenance_complete
cooldown_start
cooldown_complete
job_recovered
job_relaunch
job_no_fit
```

## 17. 성능·확장성 요구사항

100 GPU 전체 replay에서 모든 input tape를 config memory에 한 번에 적재하지 않는다.

- Parquet row group 또는 time partition 단위로 읽는다.
- provider는 현재 round와 필요한 lookahead만 보유한다.
- event log는 JSONL append 방식으로 기록한다.
- dashboard는 event 전체가 아니라 cursor/tail만 읽는다.
- metrics는 round별 append 후 Parquet로 저장한다.
- placement는 server capacity만으로 후보를 줄인다.
- full cluster 확장에서는 server subset DP를 사용한다.

성능 목표는 구현 후 실제 측정한다. PAL 논문의 placement overhead 수치를 우리 목표값으로 복사하지 않는다. 다만 5분 decision interval 안에서 placement 계산이 안정적으로 완료되는지와 input 읽기·serialization·UI polling이 simulator 전체 시간의 병목이 아닌지를 측정한다.

## 18. 주요 failure mode와 대응

| failure mode | 영향 | 방어 |
|---|---|---|
| GPU ID mapping 불일치 | 잘못된 GPU risk/Failure 연결 | manifest checksum와 교집합 검사 |
| probability horizon 오해 | failure rate 과대계상 | primary는 XID replay, horizon manifest 강제 |
| threshold를 simulator로 재조정 | evaluation leakage | validation-only manifest, run-time read-only |
| XID persistence 중복 집계 | failure/loss 과대계상 | GPU별 onset episode + 10분 gap merge |
| eligible GPU 부족 | queue starvation | no-fit event와 p95/max wait 보고 |
| active GPU 한 장만 교체 | synchronous Gang semantics 위반 | 전체 group checkpoint/relaunch |
| risk 하락 즉시 GPU 복귀 | 정비 전 재노출 | maintenance/cooldown state |
| server correlation 미모델링 | packing 효과 과대해석 가능 | core 제외를 명시하고 sensitivity 수행 |
| dashboard request block | timeout/서비스 중단 | background worker subprocess |
| run 중 worker crash | 결과 손실 | status/error artifact와 partial event 보존 |
| queue wait를 failure loss에 포함 | 목적함수 해석 혼합 | LossLedger component 분리 |
| Blox placement와 custom placement 충돌 | 잘못된 launch | scheduler order와 placement 분리 |

## 19. Not in scope

- 실제 GPU에 drain/maintenance 명령을 보내는 운영 controller
- PyTorch/TensorFlow framework 내부 checkpoint 구현
- NCCL communicator shrink/grow를 실제 실행하는 elastic training
- server correlated failure의 ML 모델링
- Themis auction/Gurobi allocator
- Gandiva time-slicing/grow-shrink
- ML model architecture 교체 또는 ensemble confidence 구현
- simulator 결과를 이용한 threshold 자동 최적화
- 기존 모든 synthetic 소스 코드의 삭제
- 전체 1,992 GPU 또는 전체 249-node cluster를 첫 구현에서 primary로 다루는 것

## 20. 완료 기준

구현 완료는 다음 조건을 모두 만족해야 한다.

### 입력과 재현성

- 실제 100 GPU cohort가 고정되어 있다.
- 실제 server mapping이 manifest에 저장된다.
- probability/failure/job/topology tape에 checksum이 있다.
- threshold는 validation-only manifest로 고정된다.
- 같은 manifest와 seed에서 같은 decision/event/summary가 재생된다.

### 정책

- high/excluded GPU 신규 allocation이 0건이다.
- eligible 부족 시 fallback 없이 queue에 남는다.
- Gang-Job allocation이 atomic하다.
- 최소 server count가 검증된다.
- scheduler와 placement 정책이 독립적으로 교체된다.

### Failure와 loss

- All-XID episode가 10분 gap 규칙으로 merge된다.
- synchronous Gang failure가 전체 Gang-Job recovery를 유발한다.
- active migration도 전체 Gang-Job 단위로 처리된다.
- failure loss, preventive overhead, queue wait가 분리된다.
- maintenance/cooldown 없이는 GPU가 복귀하지 않는다.

### 운영 경로

- dashboard에서 전체 replay가 background worker로 실행된다.
- batch runner와 dashboard가 같은 engine/provider를 사용한다.
- worker failure와 partial result가 기록된다.
- 실행 경로에서 synthetic preset이 사용자에게 노출되지 않는다.
- synthetic fixture는 회귀 테스트에 남아 있다.

## 21. Traceability matrix

| 결정 | 구현 위치 | 검증 artifact |
|---|---|---|
| 99% validation recall | ML threshold selector, manifest | `threshold_manifest.json`, threshold test |
| actual XID primary | `FailureTapeProvider`, `engine.py` | `failure_tape.parquet`, `events.jsonl` |
| binary eligibility | `RiskTapeProvider`, `PlacementPolicy` | `decisions.jsonl` |
| 100 GPU actual cohort | `TopologyProvider` | `gpu_manifest`, run manifest |
| min server placement | `placement_policy.py` | placement unit/property tests |
| no-fit queue | `engine.py`, `decisions.jsonl` | no-fit integration test |
| group recovery | `engine.py`, `loss_ledger.py` | gang failure test |
| preventive overhead separation | `loss_ledger.py` | loss component test |
| XID episode merge | feed preprocessing/provider | episode fixture and event log |
| dashboard background run | `controller.py`, `replay_worker.py` | worker integration test |
| synthetic execution removal | `config.py`, dashboard catalog | UI/API regression test |

## 22. 구현 전 최종 확인사항

계획은 확정되었지만 실제 구현을 시작하기 전에 다음 파일·데이터만 read-only로 확인한다.

1. Branch 1·2 최종 probability artifact의 실제 경로와 column name
2. actual GPU manifest의 경로와 server mapping completeness
3. `trace_seren.csv`에서 `arrival`, `duration`, `gpu_num`의 실제 의미
4. checkpoint interval 정보의 존재 여부
5. Blox source commit과 현재 RiskLab import path
6. dashboard가 사용하는 Python runtime과 worker process 시작 방식

이 확인은 설계를 변경하는 것이 아니라, 계획서의 경로·컬럼명·실행 명령을 실제 저장소에 맞게 고정하기 위한 사전 검증이다.

## 23. 참고자료

- [Blox paper: A Modular Toolkit for Deep Learning Schedulers](https://arxiv.org/abs/2312.12621)
- [Blox source repository](https://github.com/msr-fiddle/blox)
- [PAL paper](https://arxiv.org/abs/2408.11919)
- [Tiresias paper](https://www.usenix.org/system/files/nsdi19-gu.pdf)
- [Themis paper](https://www.usenix.org/system/files/nsdi20-paper-mahajan.pdf)
- [Gandiva paper](https://www.usenix.org/system/files/osdi18-xiao.pdf)
- [NCCL communicator fault tolerance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
- [PyTorch fault-tolerant distributed training](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
- [TensorFlow multi-worker fault tolerance](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)
- [Project-local Blox RiskLab README](<blox_repo_actual/blox-risklab/README.md>)

## 24. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-09-22 | 질문 기반 설계 확정본 작성 |

# GPU Telemetry 고장 예측 연구 미팅 브리프

**미팅:** 2026-09-02  
**목적:** 검증 완료 사항, 낮은 성능의 원인, 다음 실험 방향을 결정한다.

## 1. 결론

AcmeTrace XID 43의 24시간 **개별 GPU 텔레메트리 예측에는 약한 순위화 신호가 있지만 높은 정확도를 기대하기 어렵다.** XID 43은 물리 GPU 고장보다 애플리케이션/워크로드 오류인 경우가 많고, 동일 노드의 여러 GPU에 거의 동시에 기록되기 때문이다.

다음 실험은 XID 43을 **노드 단위 운영 이상 사건**으로 재정의하고, 실제 하드웨어 장애 라벨이 있는 Summit DBE에서 같은 파이프라인을 검증하는 것이 가장 설득력 있다.

## 2. 완료한 작업

- 반복 XID를 onset episode로 정리하고 AcmeTrace 텔레메트리와 결합
- 10분 leakage buffer, 시간순 60/20/20 분할, 36시간 purge 적용
- `향후 24시간 내 동일 GPU의 새 XID` 이진 분류로 정의
- GBDT, MLP, 1D CNN, TCN 네 모델 비교
- XID 43/Any-XID, fixed/sliding, parallel ensemble, 긴 입력 구간 비교
- Nie et al.의 다중 시간축 특징 재현

## 3. 핵심 결과

| 실험 | 최고 모델 | PR-AUC | AP/양성률 | Recall@Top 2% |
|---|---:|---:|---:|---:|
| XID 43, fixed | GBDT | 0.00752 | 1.55배 | 5.70% |
| Any-XID, fixed | CNN | 0.01153 | 1.56배 | - |
| Any-XID, sliding | TCN | 0.01020 | 1.38배 | - |
| XID 43, Nie-style | MLP | 0.00757 | - | - |
| XID 43, 12시간 입력 | Logistic | 0.01140 | - | - |

XID 43 평가 양성률은 약 0.486%다. PR-AUC는 양성률의 영향을 받으므로 AP/양성률도 함께 봐야 한다.

기존 탐색적 OOF의 AP 0.0127~0.0190, 최고 Lift@Top 100 3.33배는 XID 구성·샘플링·fold·평가 단위가 다르다. 최종 주장은 **시간 분할, purge, 자연 양성률을 고정한 위 확인 실험**을 기준으로 한다.

- Parallel hard intersection은 표본과 recall이 붕괴했다. rank averaging 또는 validation stacking이 적합하다.
- Sliding은 GBDT에는 도움, MLP/CNN에는 악화되어 보편적 개선책이 아니었다.

## 4. 낮은 성능의 핵심 원인

NVIDIA는 XID 43을 일반적으로 **사용자 애플리케이션이 GPU에서 fault를 일으킨 사건**으로 설명한다. AcmeTrace에서도 다음을 확인했다.

- 확정 XID 43: 1,538 episode, 910 GPU
- 1분 이내 동일 노드의 다른 GPU에도 발생: 1,259건(81.9%)
- 동일 노드·1분으로 묶으면 1,538 GPU episode가 524 node-time incident로 축소
- 테스트 GPU episode의 90.6%가 multi-GPU incident

하나의 작업/노드 사건이 여러 GPU 양성으로 복제된 구조다. 입력은 GPU별 utilization, temperature, power, framebuffer이고 job/application context는 없다. **노드·워크로드 수준 라벨과 개별 GPU 수준 특징의 단위 불일치**가 핵심 병목이다.

Nie et al.은 application·topology를 포함한 job-level SBE 문제다. 유사 연구의 `Precision@Top 2%`도 PR-AUC와 다른 지표이며 양성률도 다르므로 직접 비교할 수 없다.

## 5. 논문 주장과 한계

> 같은 telemetry 파이프라인도 fault semantics와 관측 가능한 context에 따라 predictability가 크게 달라진다.

AcmeTrace의 결과는 application-induced fault 대조군으로 의미가 있다. 다만 현재 결과만으로 예측 기반 스케줄링이 Lost GPU-Hours/JCT를 개선했다고 주장할 수는 없다. Blox 반사실적 시뮬레이션 후 검증해야 한다.

## 6. 다음 실험과 결정사항

1. AcmeTrace XID 43을 동일 노드·1분 incident로 통합하고 node aggregate telemetry로 예측
2. Summit DBE/ClusterWise의 하드웨어 DBE를 같은 24시간 horizon·네 모델·지표로 비교
3. 공통 telemetry와 node/job context 추가 실험을 분리해 context 기여도 측정
4. PR-AUC, AP/양성률, Recall@Top K%, Precision@Top K%를 함께 보고
5. risk score가 유의미할 때 Blox에서 Lost GPU-Hours와 JCT 평가

- **A안(권장):** AcmeTrace를 application/node incident 대조군으로 유지하고 Summit DBE를 주 실험으로 추가
- **B안:** AcmeTrace node-level XID 43 prioritization으로 범위 축소
- **C안:** per-GPU XID 43을 하드웨어 고장 예측으로 유지 — 라벨 의미상 비권장

## 7. 60초 설명

“XID 43의 24시간 예측에서 최고 PR-AUC는 0.0075, 양성률 대비 약 1.55배였습니다. Sliding, 라벨 통합, Nie 방식 특징도 결정적 개선을 만들지 못했습니다. XID 43의 81.9%가 1분 안에 같은 노드의 다른 GPU와 함께 발생했고 NVIDIA도 이를 주로 애플리케이션 fault로 봅니다. 노드·작업 사건을 개별 GPU 물리 텔레메트리로 맞히려 한 단위 불일치가 핵심 병목입니다. 다음에는 AcmeTrace를 node incident로 재정의하고 실제 하드웨어 DBE가 있는 Summit에서 같은 파이프라인을 검증하겠습니다.”

## 참고

- [NVIDIA XID Errors](https://docs.nvidia.com/deploy/xid-errors/)
- [AcmeTrace 원 논문](https://www.usenix.org/conference/nsdi24/presentation/hu)
- [ClusterWise 공개 데이터](https://huggingface.co/datasets/AI4System/ClusterWise)
- 초기 모델 비교: [`ML/reports/all_models_comparison_report.md`](ML/reports/all_models_comparison_report.md)
- 초기 OOF 결과: [`ML/reports/branch1_report.md`](ML/reports/branch1_report.md)

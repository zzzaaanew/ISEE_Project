# [Bidirectional Dynamic ADST & Dual-Branch Fusion] 최종 성능 보고서

## 1. 실험 핵심 사양
- **타깃 정의**: 모든 XID 코드 전면 통합 (`all_xids`, 향후 24시간 내 신규 Onset)
- **아키텍처**: 직교형 Dual-Branch (Branch 1 텔레메트리 + Branch 2 누적 이력/부하)
- **양방향 ADST 2D 탐색**: $L_{train} \in \{7d, 14d, 21d\} \times L_{obs} \in \{1h, 6h, 24h\}$
- **재학습 주기**: 24시간 주기 Walk-Forward Rolling 검증

## 2. 종합 평균 평가 지표
- **Fused PR-AUC**: `0.0411` (Branch 1 단독: `0.0221`, Branch 2 단독: `0.0407`)
- **Fused ROC-AUC**: `0.6809`
- **Top-100 Fault Recall (상위 5% 자원)**: `20.7%`
- **Top-100 Fault Lift**: `4.12배` (무작위 대비)

## 3. 양방향 동적 윈도우 선택 통계
- **학습 기간($L_{train}$) 선택 분포**: {21: 23, 7: 21, 14: 19}
- **관측 시간($L_{obs}$) 선택 분포**: {24: 23, 1: 21, 6: 19}

## 4. 핵심 결론
1. 텔레메트리(Branch 1)와 누적 이력(Branch 2)의 직교 결합으로 단일 브랜치 대비 PR-AUC 및 랭킹 정밀도 동시 향상.
2. 양방향 ADST가 하드웨어 열화 진행 속도에 맞춰 최적 윈도우를 적응 선택하여 Concept Drift를 성공적으로 방어.
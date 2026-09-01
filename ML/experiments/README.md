# Confirmatory ML experiments

이 폴더는 2026-09-02 미팅 브리프의 핵심 수치를 만든 leakage-aware 실험 코드만 모은다. 저장소 루트의 기존 `ML/run_*.py`는 초기 탐색적 OOF 실험이며 평가 프로토콜이 다르다.

## 파일

- `run_telemetry_24h_experiment.py`: 24시간 onset 라벨, 10분 buffer, 시간 분할·purge를 생성하는 공통 파이프라인
- `run_xid_incident_fixed.py`: XID 43 및 Any-XID의 GBDT/MLP/CNN/TCN fixed 비교와 parallel 평가
- `run_xid_incident_sliding.py`: 동일 네 모델의 sliding training 비교
- `run_nie_xid43_reproduction.py`: Nie-style 다중 시간축 특징 재현
- `run_nie_xid43_reproduction_linear_svm.py`: 대규모 평가가 가능한 LinearSVC 수정판
- `run_telemetry_window_comparison.py`: 1/6/12시간 입력 구간 비교

## 데이터와 의존성

`DATA/`와 `outputs/`는 크기 때문에 커밋하지 않는다. AcmeTrace 원본을 저장소 루트의 `DATA/`에 배치하고 루트 `requirements.txt`를 설치한다. 모든 기본 경로는 저장소 루트에서 실행할 때를 기준으로 한다.

## 실행 순서

```bash
python ML/experiments/run_telemetry_24h_experiment.py --rebuild-labels
python ML/experiments/run_xid_incident_fixed.py
python ML/experiments/run_xid_incident_sliding.py
python ML/experiments/run_nie_xid43_reproduction_linear_svm.py --output-dir outputs/nie_xid43_reproduction_linear_svm
python ML/experiments/run_telemetry_window_comparison.py
```

각 스크립트의 기본 seed, sampling 수, epoch와 출력 경로는 당시 확인 실험 설정이다. 다른 데이터나 설정과 비교할 때는 PR-AUC뿐 아니라 양성률, AP/양성률, Recall/Precision@Top K를 함께 기록한다.

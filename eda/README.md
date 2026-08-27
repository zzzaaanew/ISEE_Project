# AcmeTrace XID EDA

AcmeTrace에서 XID episode 전후의 GPU 텔레메트리 변화를 분석하고, 동일 GPU의 요일·시간대·사전 활동량을 맞춘 비사건 control과 비교한 분석 파이프라인입니다. 이후 Blox counterfactual replay에 사용할 Job Tape, Fault Tape, Risk Tape의 입력 적합성을 점검하고 5분 단위 OOF Risk Tape를 생성합니다.

## 데이터

원본 데이터는 저장소에 포함하지 않습니다. [AcmeTrace Hugging Face 데이터셋](https://huggingface.co/datasets/Qinghao/AcmeTrace)에서 아래 파일을 내려받아 저장소 루트에 둡니다.

- `DRAM_ACTIVE.csv`
- `FB_USED.csv`
- `GPU_TEMP.csv`
- `GPU_UTIL.csv`
- `POWER_USAGE.csv`
- `XID_ERRORS.csv`
- `trace_seren.csv`
- `MEMORY_TEMP.csv`
- `MEM_CLOCK.csv`
- `GPU_AB_Power.csv`
- `GPU_C_Power.csv`

## 설치

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## 실행 순서

각 단계의 결과는 저장소 루트의 `outputs/` 디렉토리 아래에 생성됩니다.

```bash
# 1. 기본 데이터 품질 및 XID episode EDA
python eda/run_eda.py
python eda/finalize_eda.py

# 2. XID 전후 event window
python eda/run_event_eda_v2.py
python eda/finalize_event_analysis.py

# 3. 동일 GPU matched non-event control 비교
python eda/eda_matched_controls.py

# 4. XID 43 기록 시각 지연 분석
python eda/eda_xid43_timestamp_lag.py

# 5. 메모리 및 노드 전력 보조 분석
python eda/run_extended_metrics.py
python eda/postprocess_extended_metrics.py

# 6. Blox replay 입력 적합성 EDA
python eda/eda_blox_inputs.py

# 7. XID 결측·관측성 및 Fault Tape 생성
python eda/run_xid_observability_compact.py --rescan

# 8. 시간순 OOF Risk Tape 생성
python eda/build_risk_tape_5m.py
python eda/finalize_risk_tape_5m.py
```

## 핵심 분석 주의사항

- **30초 Episode 병합**: 30초 이내 반복 XID는 하나의 episode로 묶고 최초 Onset 시점만 사용합니다.
- **XID 43 Logging Delay 방지**: XID 43 기록 직전 10분의 텔레메트리 하락에는 이미 GPU가 멈춘 뒤의 정보가 포함될 수 있으므로 예측 입력 구간에서 배제합니다.
- **Data Leakage 원천 차단**: 조기예측과 발생 후 incident 탐지를 엄격히 구분하고, 시간순 OOF(Out-of-Fold) 분할로 미래 정보 유입을 막습니다.
- **외생적 3-Tape Replay**: Blox replay에서는 과거 XID를 외생적 Fault Tape로 사용하고, Risk Tape는 GPU의 상대 위험 순위로 해석합니다.
- **대용량 파일 제외**: 원본 CSV와 대용량 중간 산출물은 `.gitignore`로 관리됩니다.

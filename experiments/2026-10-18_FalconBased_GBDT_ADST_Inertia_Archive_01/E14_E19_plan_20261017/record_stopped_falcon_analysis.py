"""Analyze preserved traces after the user's stop; never imports or starts ML models."""
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean
from collections import defaultdict
import csv
import hashlib
import html
import json
import re

PLAN = Path(__file__).resolve().parent
ROOT = PLAN / 'runtime/migration/e19-reproduction-gpu-all'
RUN = ROOT / 'experiments/2026-10-17_E14ToE19_GPU_All_Rerun_01'
E14 = RUN / 'e14'
status = json.loads((PLAN / 'gpu_all_status.json').read_text(encoding='utf-8'))
assert status['status'] == 'STOPPED_BY_USER', 'This analysis must not interrupt or restart a running job.'
parts = sorted((E14 / 'metrics_parts').glob('metrics_part_*.json'))
metrics = [json.loads(path.read_text(encoding='utf-8')) for path in parts]
indices = {row['origin_idx'] for row in metrics}
assert indices == set(range(13))

def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

fit = [v for v in jsonl(E14 / 'training_fit_trace.jsonl') if v.get('state_idx') in indices]
backend = [v for v in jsonl(E14 / 'gpu_backend_trace.jsonl') if v.get('state_idx') in indices]
features = jsonl(E14 / 'performance_events.jsonl')
training_summary = {}
for event in sorted({v['event'] for v in fit}):
    vals = [v['elapsed_seconds'] for v in fit if v['event'] == event and 'elapsed_seconds' in v]
    training_summary[event] = {'count': sum(v['event'] == event for v in fit),
                               'sum_seconds': sum(vals) if vals else None}
groups = defaultdict(list)
for row in backend:
    if row['event'].endswith('_fit'):
        key = f"{row['model']}/rounds={row.get('rounds')}/features={row.get('features')}"
        groups[key].append(row['elapsed_seconds'])
model_times = {key: {'count': len(values), 'seconds': sum(values)} for key, values in groups.items()}
forest_seconds = model_times['GPUForest/rounds=1/features=22']['seconds']
b1_boost_seconds = model_times['GPUBoost/rounds=120/features=22']['seconds']
cascade_boost_seconds = model_times['GPUBoost/rounds=160/features=26']['seconds']
cnn_seconds = training_summary['_fit_cnn']['sum_seconds']
linear_seconds = sum(v['elapsed_seconds'] for v in backend if v['event'] == 'linear_fit')
feature_seconds = sum(v['elapsed_seconds'] for v in features)
journal = json.loads((RUN / 'pipeline.json').read_text(encoding='utf-8'))
start = datetime.fromisoformat(journal['started_utc']).timestamp()
stop = datetime.fromisoformat(status['stopped_utc']).timestamp()
last_checkpoint = max(path.stat().st_mtime for path in parts)
wall_completed = last_checkpoint - start
wall_to_stop = stop - start
with (PLAN / 'runtime/migration/e19-reproduction-gpu-opt/experiments/2026-10-17_E14ToE19_GPU_Optimized_Rerun_01/e19/origin_metrics.csv').open(encoding='utf-8', newline='') as handle:
    baseline = [v for v in csv.DictReader(handle) if int(v['origin_idx']) in indices]
current_ap = {model: mean(v[key] for v in metrics) for model, key in
              [('B1', 'pr_auc_b1'), ('B2', 'pr_auc_b2'), ('E14_Fusion', 'pr_auc_fused')]}
baseline_ap = {model: mean(float(v['pr_auc']) for v in baseline if v['model'] == key)
               for model, key in [('B1', 'B1'), ('B2', 'B2'), ('E14_Fusion', 'Experiment14_Fusion')]}
reference = json.loads((PLAN / 'optimization_gpu_acceptance.json').read_text(encoding='utf-8'))
result = {
    'status': 'PARTIAL_PIPELINE_USER_STOPPED', 'stopped_utc': status['stopped_utc'],
    'remaining_experiment_processes_at_stop_verification': 0,
    'completed_origins': len(metrics), 'interrupted_origin_index': 13,
    'score_parts': len(list((E14 / 'full_population').glob('*.parquet'))),
    'decision_times': sum(v['decision_time_count'] for v in metrics),
    'score_rows': sum(v['decision_time_count'] * v['gpu_count_evaluated'] for v in metrics),
    'positive_rows': sum(v['positives'] for v in metrics), 'e19_started': False,
    'wall_completed_first_13_seconds': wall_completed,
    'wall_started_until_user_stop_seconds': wall_to_stop,
    'wall_incomplete_14th_until_stop_seconds': stop - last_checkpoint,
    'training_complete_origins': training_summary, 'backend_complete_origins_by_model': model_times,
    'feature_events_through_stop_including_partial_14th_seconds': feature_seconds,
    'tree_fit_seconds_complete_13': forest_seconds + b1_boost_seconds + cascade_boost_seconds,
    'tree_fit_share_of_wall_completed_13': (forest_seconds + b1_boost_seconds + cascade_boost_seconds) / wall_completed,
    'current_variant_origin_mean_AP_first_13': current_ap,
    'completed_baseline_origin_mean_AP_same_13': baseline_ap,
    'scope_cautions': ['partial metrics do not establish full-period performance',
                       'models and CUDA precision/RNG changed; not a pure device ablation',
                       'GPU contention during initial baseline overlap; no controlled CPU-vs-GPU wall comparison',
                       'nested CNN/family/Cascade/backend times cannot be added across levels',
                       'state_fitted and score_cycle events record completion counts but no duration',
                       'feature traces contain incomplete origin14 and have no origin IDs'],
    'verified_previous_optimization': {
        'scope': 'same first origin, sequential runs, same existing hybrid GPU implementation',
        'reference_seconds': reference['reference_seconds'], 'optimized_seconds': reference['optimized_seconds'],
        'time_saved_percent': reference['time_saved_percent'], 'speedup': reference['end_to_end_speedup'],
        'all_scores_numerically_equivalent': reference['all_scores_numerically_equivalent'],
        'metrics_exactly_equal': reference['metrics_exactly_equal']},
    'falcon_audit': {'module': 'ML/branch1_enhanced_features.py', 'feature_count': 45,
                     'feature_groups': {'base': 22, 'cross_metric': 7, 'topology': 8, 'peer_relative': 5, 'temporal': 3},
                     'self_check_only': {'sample_bins': [2000, 2001, 2002], 'sample_gpus': [0, 1, 7],
                                         'observation_hours': 6, 'model_training': False,
                                         'checks': ['8 new feature columns exist', 'no NaN/Inf', 'peer rank in 1..8'],
                                         'execution_log_available': False},
                     'benchmark_feature_path': 'UnifiedADSTPipeline.extract_enhanced_branch1_features: old30+cross7+node8',
                     'new_falcon_engine_imported_by_benchmark': False,
                     'new_module_used_by_stopped_gpu_run': False,
                     'module_end_to_end_performance_status': 'NOT_VERIFIED'},
    'trace_sha256': {str(path.relative_to(PLAN)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in [E14 / 'training_fit_trace.jsonl', E14 / 'gpu_backend_trace.jsonl', E14 / 'performance_events.jsonl']},
}
(PLAN / 'gpu_all_stopped_runtime_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

md = PLAN.parent / '실험 기록/실험_연대기.md'
content = md.read_text(encoding='utf-8')
assert '## Experiment 6 ' not in content
entry = f'''

## Experiment 6 = GPU 모델 변형 중단·Falcon 실험 경로와 시간 병목 검토 = 분석

[요약]

- 사용자 지시에 따라 Experiment2의 현재 E14 GPU 모델 변형을 2026-10-17 18:23:24 KST에 중단했다. 이후 작업은 Falcon 모듈 실험 분석과 저장 trace의 ML 시간 단축 분석이다. 새 학습이나 특징 검사는 실행하지 않았다.
- Experiment3·4의 전체 완료 기준 결과는 보존한다. 이번 중단 결과는 PARTIAL_PIPELINE_USER_STOPPED이며 원본 수치 재현·전체 성능 개선·E19 완료를 주장하지 않는다.

[실행]

- 감독·pipeline·학습 Python의 실제 명령줄과 부모/자식 PID를 대조하여 이 실행의6개 프로세스만 종료했다. 종료 후 해당 학습/감독 명령줄의 남은 프로세스0개를 확인했다. 원래 status/pipeline JSON은 before_user_stop 사본으로 보존하고 STOPPED_BY_USER를 기록했다. 원본 Parquet·모델 코드·완료 점수 파일은 변경하지 않았다.
- 승인 범위 G-004/G-007/G-008. 동일 label/36h purge/64 origins/B1 ADST/B2 21일 ST/E14 MoE 유지, CUDA model 변형의 frozen launch identity 기준이다. Falcon은 이 실행에 적용하지 않았으며 B1 tabular22개·CNN7채널을 사용했다.
- 진행 상태 파일이 RUNNING에 머물러 있던 점 때문에 프로세스 존재·실제 CUDA backend trace·완료 metrics/state/score part를 함께 확인했다. Falcon 신규 모듈은 코드·자체 검사·benchmark 호출·결과 CSV를 읽어 분석했다.
- 시간 분석 진입점 record_stopped_falcon_analysis.py, 증거 gpu_all_stopped_runtime_analysis.json. 완료13개 origin만의 모델 fit 시간을 분리했다. feature trace는 origin ID가 없어 미완료14번째까지 포함한 별도 범위다.

[결과]

- 완료13/64개 origins, 624판단 시점, 1,243,008 GPU-time 점수 행, 양성13,196행. 14번째(origin_idx13) 학습 중 중단했으며 E19 단계는 시작하지 않았다. 전체 GPU pipeline 감사·Top100 전수 감사·최종 결과는 미완료다.
- 완료13개 CNN CUDA39회, B1 family39회, Cascade13회, B2 final13회. 부분14번째까지 CNN41회가 기록됐으나 완료13개 지표에 넣지 않았다. 실제 CUDA tree fit104회(39forest+39B1boost+26Cascadeboost), linear fit62회(B1 39+B2 13+E14 gate10)를 완료13개 trace에서 확인했다.
- 시작16:02:06→완료13번째 체크포인트17:50:23은 {wall_completed/60:.2f}분이다. 중단까지 {wall_to_stop/60:.2f}분, 이 중 미완료14번째에 {((stop-last_checkpoint)/60):.2f}분이 추가로 경과했다. 이 미완료 구간을 완료 origin의 평균 속도에 포함하지 않는다.

### 완료13개 origin의 모델 학습 시간

| 실제 backend / 역할 | fit 수 | 학습 합계 | 설명 |
| --- | ---: | ---: | --- |
| GPUForest / B1 | 39 | {forest_seconds/60:.2f}분 | CPU ExtraTrees를 CUDA XGBoost RF100개·depth12로 교체 |
| GPUBoost120 / B1 | 39 | {b1_boost_seconds/60:.2f}분 | CUDA histogram boosting |
| GPUBoost160 / Cascade | 26 | {cascade_boost_seconds/60:.2f}분 | 두 모델씩13개 Cascade |
| CUDA CNN | 39 | {cnn_seconds/60:.2f}분 | 날짜마다 OOF2회+최종1회 |
| CUDA logistic 전체 | 62 | {linear_seconds/60:.2f}분 | B1·B2·E14 gate 포함 |

- 위 tree fit 합계 {(forest_seconds+b1_boost_seconds+cascade_boost_seconds)/60:.2f}분은 완료13개 경과시간의 {(forest_seconds+b1_boost_seconds+cascade_boost_seconds)/wall_completed*100:.1f}%다. model fit 행들은 역할을 분리했다. 상위 _fit_families {training_summary['_fit_families']['sum_seconds']/60:.2f}분은 CNN/tree/linear를 포함하므로 표에 다시 더하지 않는다. fit_cascade {training_summary['fit_cascade']['sum_seconds']/60:.2f}분도 GPUBoost160과 중첩된다.
- 전체 feature 계측 합계 {feature_seconds:.1f}초({feature_seconds/60:.2f}분)는 미완료14번째까지 포함한다. 이를 완료13개 wall의 정확한 CPU 시간 비율로 해석하지 않는다. state_fitted/score_cycle는 소요시간 필드가 없어0초가 아니라 미기록이다.

### 부분 성능 참고값

| 모델 | 완료 GPU 기준의 같은13일 평균 AP | 중단 CUDA 모델 변형의13일 평균 AP |
| --- | ---: | ---: |
| B1 | {baseline_ap['B1']:.6f} | {current_ap['B1']:.6f} |
| B2 | {baseline_ap['B2']:.6f} | {current_ap['B2']:.6f} |
| E14 MoE Fusion | {baseline_ap['E14_Fusion']:.6f} | {current_ap['E14_Fusion']:.6f} |

- 같은13개 날짜 기준 참고 비교다. 기존 전체64개 평균과 직접 비교하지 않으며 새 변형의 전체 E19 성능·시간은 미측정이다.

### Falcon 모듈에서 실제 확인되는 실행

1. 모듈 자체 main은 test_falcon_enhanced_features를 호출한다. bins2000/2001/2002, GPU0/1/7의3개 표본에서6시간 특징을 생성한다. peer5+temporal3 열 존재, NaN/Inf 없음, rank1~8 범위를 검사한다. 학습·ADST·Cascade·B2 ST·Fusion·성능 평가를 호출하지 않는다. 이 검사의 실행/통과 로그는 저장소에서 확인되지 않았다.
2. 신규 engine은 base22+cross7+topology8+peer5+temporal3=45개 표 특징을 만든다. peer zscore는 같은 노드의 나머지7개 GPU 대비 온도/전력/사용률 편차, rank는8개 내 상대 순위다. slope는 delta/관측시간이고 persistence는 (last−mean)/std다. 임계 초과 지속시간 계산은 아니다. CNN tensor는7채널 그대로다.
3. 보고 수치의 제공된 실행기는 scratch_compute_all_origins_top100.py다. frozen5분 Parquet/cache/onset ledger를 읽고 미제공 unified_adst_metrics.csv의 선택 train/obs 창을 받아63개 날짜를 평가한다. upstream unified 선택 후보는 train7/14/21일×obs1/6/24시간, 직전3일 validation, logistic B1/B2의50:50 validation AP와 momentum skip-hold 규칙이다.
4. exact scratch는 날짜마다 B1 HGB150회, B2 balanced logistic+HGB120회를 새 학습한다. 두 B2 확률을50:50으로 섞고 표본/모집단 prior 보정 후 lambda=min(0.60,1/(k+1)), k=origin−10min 이전30일 XID 횟수로 Pareto 결합한다. 30분마다 전체1,992GPU의48개 판단 시점을 평가하고 전체 점수 기반 AP·Top100 hits를 CSV에 남긴다. B2 고정21일 ST·원본 네 family/Cascade·E19 residual Fusion은 이 경로에 없다.
5. exact scratch는 신규 EnhancedTelemetryEngine을 쓰지 않고 기존 UnifiedADSTPipeline의30+7+8 특징을 호출한다. 보고된 B1 AP0.051201/hits11572, Fusion AP0.060910/hits12287는 최신 CSV 요약이며 신규 Falcon 모듈의 end-to-end 효과는 NOT_VERIFIED다. 별도 scratch_check_stability는 저장100GPU cohort만 읽어 대안 가중치를 비교하므로 전수 AP와 혼용하지 않는다.

[분석]

- 현재 GPU 변형은 모든 모델 CUDA 사용이 확인됐지만 시간 단축을 입증하지 못했다. 실제 병목은 GPU tree histogram/forest 학습이다. GTX1060 3GB, CPU2threads, 초기 기준 실행과의 GPU 경쟁, ExtraTrees/HGB→XGBoost 모델 교체, optimizer/RNG/precision 변경이 포함돼 있다. 부분 평균을 순수 CPU→GPU 가속비로 해석하지 않는다.
- B2 final 상위 학습은13개 합계 {training_summary['_fit_final_b2']['sum_seconds']:.1f}초라 전체 시간을 줄일 여지가 작다. E19 CUDA component의4보정+4결합 학습·추론4.767초는 연결 pilot이며 전체 E19 학습시간이나 속도 개선비가 아니다.
- 검증된 시간 단축은 Experiment1의 같은 첫 origin 순차 비교다. 기존 GPU 기준202.787초→최적화155.320초,23.4% 감소/1.306배. CNN100.609→57.734초(42.6% 감소), DataLoader 준비3.095→0.071초. B1 최대 점수차1.8e-16·모든48개Top100 순위·지표 일치다. CPU tree를 유지한 상태에서 CPU 데이터 준비와 복사를 줄여 얻은 결과다.43배는 loader 준비만의 수치다.
- Falcon을 현재 구현 그대로 추가하면 sample×1,992GPU 전체 row gather 때문에 CPU/RAM 준비 비용이 커질 수 있다. 같은 node8개만 직접 gather하면 해당 float32 배열은249분의1,300k sample 예시2.3904GB→9.6MB다. 메모리 계산이며 실측 속도 향상은 아니다. chunking·결측 규칙·실제 물리 node mapping 확인이 선행돼야 한다.
- 같은1h에서 slope와 delta가 중복되고 headroom도 base delta와 같다. 유용한 새 특징 효과와 중복을 ablation으로 구분해야 한다. report의 feature 설명과 구현을 맞추고 36h purge/성숙 validation을 먼저 적용해야 개선 주장을 검증할 수 있다.

[실행할 것]

후속 제안이며 이번 턴에서 실행하지 않는다. 사용자의 중단 지시를 유지한다.

1. Falcon을 별도 ablation에 명시 연결한다. 기존 E14→E19 계약의 A0 base22 / A1 cross29 / A2 topology37 / A3 peer·temporal45를 비교하고 actual feature manifest·전수 score tape를 남긴다. 첫 단계에서 B2 ST·Cascade·Fusion을 바꾸지 않는다. 자체3표본 검사를 전체 모델 결과로 취급하지 않는다.
2. 먼저 같은-node8개 직접 gather와 chunk window reduction을 구현 대상으로 삼는다. 일부 원점의 CPU·GPU 특징 값/전송량/peak memory/시간을 비교하며3GB 안에 맞는 배치를 선정한다. 전체 GPU 캐시 상주를 무조건 적용하지 않는다.
3. 48개 시점의 B1/B2 특징·점수 계산을 묶어 CUDA 전송과 반복 predict 호출을 줄이는 방안을 검토한다. 예측 점수는 batch로 만들고 maturity 기반 ADST/MoE·E19 state 업데이트는 시간 순서로 유지한다. GPU tensor→CPU array→GPU의 모델 사이 왕복을 계측해 제거한다.
4. 가장 큰 GPUForest/Boost/Cascade fit 병목을 별도 비교한다. 입력 matrix/OOF 재사용·중복 전처리 제거는 같은 모델의 최적화 후보다. RF tree수/depth/rounds 감소, 단일GBDT, Cascade 제거, 재학습 빈도 감소는 학습 예산·구조 변경이므로 별도 방법 arm으로 기록하고 성능 비열화 조건을 사전 고정한다.
5. 동일 첫 origin/같은 모델·예산/병렬 학습 없는 순차 측정에서 component와전체 wall time을 확인한다. feature·transfer·CNN·tree·Cascade·B2·Fusion·serialization 시간과 score/rank 차이를 함께 기록한다. 현재 중단13개를64개에 외삽해 GPU 가속 완료를 주장하지 않는다.

근거: gpu_all_stopped_runtime_analysis.json, gpu_all_stopped_processes.json, preserved pipeline/status snapshots, training_fit_trace.jsonl, gpu_backend_trace.jsonl, performance_events.jsonl, optimization_gpu_acceptance.json, ISEE main1eac1f0의branch1_enhanced_features.py/unified/scratch 실행기. Experiment5의 공개 코드 링크와 평가 문제 분석을 이어받는다.
'''
content += entry

def inline(text):
    text = html.escape(text)
    text = re.sub(r'\[([^\]]+)\]\((?:&lt;)?(.*?)(?:&gt;)?\)',
                  lambda m: '<a href="' + m[2] + '">' + m[1] + '</a>', text)
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

rows = content.splitlines()
output = []
i = 0
while i < len(rows):
    line = rows[i]
    i += 1
    if not line.strip():
        continue
    if line.startswith('#'):
        level = len(line) - len(line.lstrip('#'))
        output.append(f'<h{level}>' + inline(line[level:].strip()) + f'</h{level}>')
    elif line.startswith('|'):
        table = [line]
        while i < len(rows) and rows[i].startswith('|'):
            table.append(rows[i]); i += 1
        output.append('<div class="table"><table>')
        for j, row in enumerate(table):
            if j == 1:
                continue
            tag = 'th' if j == 0 else 'td'
            output.append('<tr>' + ''.join(f'<{tag}>' + inline(c.strip()) + f'</{tag}>' for c in row.strip('|').split('|')) + '</tr>')
        output.append('</table></div>')
    elif line.startswith('- ') or re.match(r'\d+\. ', line):
        ordered = not line.startswith('- ')
        tag = 'ol' if ordered else 'ul'
        items = [line]
        while i < len(rows) and (bool(re.match(r'\d+\. ', rows[i])) if ordered else rows[i].startswith('- ')):
            items.append(rows[i]); i += 1
        output.append(f'<{tag}>' + ''.join('<li>' + inline(re.sub(r'^(?:- |\d+\. )', '', value)) + '</li>' for value in items) + f'</{tag}>')
    else:
        output.append('<p>' + inline(line) + '</p>')
page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실험 연대기</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#20252b}h2{margin-top:48px;border-top:1px solid #bbb;padding-top:20px}h3{margin-top:32px}.table{overflow:auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:8px;vertical-align:top}th{background:#f3f5f7}li,p,td{overflow-wrap:anywhere}code{background:#f2f2f2}a{color:#165ab1}</style><main>' + ''.join(output) + '</main></html>'
assert page.count('<h2>Experiment ') == len(re.findall(r'^## Experiment ', content, re.M)) == 6
for value in ['13/64', '18:23:24', f'{forest_seconds/60:.2f}분', 'NOT_VERIFIED', '23.4%', '249분의1']:
    assert value in content and value in page
md.write_text(content, encoding='utf-8')
(md.parent / '실험_연대기.html').write_text(page, encoding='utf-8')
print(json.dumps({'status': result['status'], 'completed_origins': len(metrics),
                  'wall_completed_minutes': wall_completed / 60, 'wall_to_stop_minutes': wall_to_stop / 60,
                  'tree_fit_minutes': result['tree_fit_seconds_complete_13'] / 60,
                  'tree_fit_wall_share': result['tree_fit_share_of_wall_completed_13'],
                  'cnn_fit_minutes': cnn_seconds / 60, 'feature_through_stop_minutes': feature_seconds / 60,
                  'logbook': 'Experiment 6'}, indent=2))

"""Independently audit persisted scores and append the shared research log."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
BASES = ['S7_0', 'S7_I', 'S14_0', 'S14_I', 'S21_0', 'S21_I', 'A_0', 'A_I']
ARMS = [f'{base}_{fusion}' for base in BASES for fusion in ['P', 'E']]
EPS = 1e-6


def clipped_logit(value):
    return logit(np.clip(value, EPS, 1 - EPS))


def verify_predictions(frame, models, config, previous_memory, previous_time):
    """Reconstruct causal inertia and both fusion formulas independently."""
    n = 1992
    nt = len(frame) // n
    times = frame.timestamp.astype('int64').to_numpy().reshape(nt, n)[:, 0]
    memory_times = frame.memory_previous_timestamp.astype('int64').to_numpy().reshape(nt, n)
    nat_b2 = frame.b2.to_numpy().reshape(nt, n)
    lam = frame.pareto_lambda.to_numpy().reshape(nt, n)
    if not np.isfinite(lam).all() or (lam <= 0).any() or (lam > 1).any() or not np.array_equal(lam, np.broadcast_to(lam[0], lam.shape)):
        raise ValueError('Pareto weights invalid or changed within origin')
    if not np.allclose(1 / lam, np.rint(1 / lam), atol=1e-8, rtol=1e-8):
        raise ValueError('Pareto lambda does not equal 1/(integer count+1)')
    for policy in ['S7', 'S14', 'S21', 'A']:
        prior = frame[f'b1_prior_{policy}'].to_numpy()
        if ((prior > 0) & (prior < 1)).all():
            p = np.clip(expit(clipped_logit(frame[f'b1_model_raw_{policy}'].to_numpy()) + clipped_logit(prior)), EPS, 1-EPS)
            if not np.allclose(p, frame[f'b1_{policy}'].to_numpy(), atol=1e-12, rtol=1e-10):
                raise ValueError('B1 natural prior restoration differs')
    prior = frame.b2_prior.to_numpy()
    if ((prior > 0) & (prior < 1)).all():
        p = np.clip(expit(clipped_logit(frame.b2_model_raw.to_numpy()) + clipped_logit(prior)), EPS, 1-EPS)
        if not np.allclose(p, frame.b2.to_numpy(), atol=1e-12, rtol=1e-10):
            raise ValueError('B2 natural prior restoration differs')
    raw_z = np.stack([clipped_logit(frame[f'b1_{p}'].to_numpy().reshape(nt, n)) for p in ['S7', 'S14', 'S21', 'A']])
    stored_memory = np.stack([frame[f'memory_logit_{p}'].to_numpy().reshape(nt, n) for p in ['S7', 'S14', 'S21', 'A']])
    for j, timestamp in enumerate(times):
        reset = previous_time is None or timestamp - previous_time >= 24 * 3600 * 10 ** 9
        expected = raw_z[:, j] if reset else previous_memory
        expected_time = np.iinfo(np.int64).min if reset else previous_time
        if not np.all(memory_times[j] == expected_time) or not np.allclose(stored_memory[:, j], expected, atol=1e-12, rtol=1e-10):
            raise ValueError('Inertia memory is not the strictly past causal state')
        delta = .5 if previous_time is None else (timestamp - previous_time) / (3600 * 10 ** 9)
        rho = 2 ** (-delta / config['inertia_half_life_hours'])
        previous_memory = raw_z[:, j].copy() if reset else rho * previous_memory + (1-rho) * raw_z[:, j]
        previous_time = timestamp
    for base in BASES:
        policy, flag = base.rsplit('_', 1)
        z = clipped_logit(frame[f'b1_{policy}'].to_numpy())
        m = frame[f'memory_logit_{policy}'].to_numpy()
        eta = z if flag == '0' else (z + config['inertia_gamma'] * m) / (1 + config['inertia_gamma'])
        q = np.clip(expit(eta), EPS, 1-EPS)
        if not np.allclose(q, frame[f'q_{base}'].to_numpy(), atol=1e-12, rtol=1e-10):
            raise ValueError('KL inertia inference differs from approved closed form')
        pareto = lam.ravel() * q + (1-lam.ravel()) * frame.b2.to_numpy()
        if not np.allclose(pareto, frame[f'risk_{base}_P'].to_numpy(), atol=1e-12, rtol=1e-10):
            raise ValueError('Pareto output does not consume actual inertia B1')
        model = models[base]
        expected = nat_b2.ravel()
        if model['status'] == 'trained':
            c1, c2 = model['calibrators']
            p1 = np.clip(expit(c1['intercept'] + c1['slope'] * eta / 6), EPS, 1-EPS)
            p2 = np.clip(expit(c2['intercept'] + c2['slope'] * clipped_logit(nat_b2.ravel()) / 6), EPS, 1-EPS)
            a, b = clipped_logit(p1) / 6, clipped_logit(p2) / 6
            basis = np.column_stack([np.ones(len(a)), a, b, a*a, b*b, a*b])
            expected = np.clip(expit(clipped_logit(p2) + basis @ np.asarray(model['beta'])), EPS, 1-EPS)
        if not np.allclose(expected, frame[f'risk_{base}_E'].to_numpy(), atol=1e-12, rtol=1e-10):
            raise ValueError('E19 residual output differs from saved fitted model')
    t = frame.timestamp.astype('int64').to_numpy()
    if not np.all(t-frame.feature_cutoff_time.astype('int64').to_numpy() == 10*60*10**9):
        raise ValueError('Nominal feature cutoff changed')
    if not np.all(t-frame.latest_telemetry_time.astype('int64').to_numpy() == 15*60*10**9):
        raise ValueError('Actual latest telemetry lag changed')
    return previous_memory, previous_time


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    payload=json.dumps(value, ensure_ascii=False, indent=2)
    deadline=time.monotonic()+30
    while True:
        try:
            temp.write_text(payload, encoding='utf-8')
            temp.replace(path)
            return
        except PermissionError:
            # Windows readers/sync clients can briefly deny atomic replacement.
            # Retry only the metadata write, never an ML training operation.
            if time.monotonic()>=deadline:
                raise
            time.sleep(.2)


def score_metrics(score, labels) -> dict:
    # The frame has already been sorted by timestamp and the stable GPU-ID order.
    probabilities = np.asarray(score, dtype=np.float64).reshape(-1, 1992)
    y = np.asarray(labels, dtype=np.uint8).reshape(-1, 1992)
    order = np.argsort(-probabilities, axis=1, kind='stable')[:, :100]
    relevance = np.take_along_axis(y, order, axis=1)
    hits = int(relevance.sum())
    positives = int(y.sum())
    discounts = 1 / np.log2(np.arange(2, 102))
    ideal = np.r_[0., np.cumsum(discounts)][np.minimum(y.sum(axis=1), 100).astype(int)]
    dcg = relevance @ discounts
    ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
    return {'pr_auc': float(average_precision_score(y.ravel(), probabilities.ravel())) if positives else 0.,
            'hits_at_100': hits, 'positives': positives,
            'recall_at_100': hits / positives if positives else 0.,
            'precision_at_100': hits / (len(y) * 100), 'decision_times': len(y),
            'gpu_count': 1992, 'ndcg_at_100': float(ndcg[ideal > 0].mean()) if (ideal > 0).any() else 0.}


def inspect(output: Path) -> dict:
    contract = json.loads((output / 'execution_contract.json').read_text(encoding='utf-8'))
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    fidelity = json.loads((output / 'fidelity_audit.json').read_text(encoding='utf-8'))
    feature = json.loads((output / 'feature_manifest.json').read_text(encoding='utf-8'))
    with (output / 'metrics_by_origin.csv').open(encoding='utf-8-sig', newline='') as stream:
        metrics = list(csv.DictReader(stream))
    lookup = {}
    for row in metrics:
        key = (int(row['origin']), row['arm'].removeprefix('risk_'))
        if key in lookup:
            raise ValueError(f'Duplicate stored metric key: {key}')
        lookup[key] = row
    parts = sorted((output / 'scores').glob('origin_*.parquet'))
    if not parts:
        raise ValueError('No full-population score parts')
    expected_fidelity = 'FULL_PIPELINE_INTERNAL_AUDIT' if len(parts) == 64 else 'PARTIAL_PIPELINE_INTERNAL_AUDIT'
    if fidelity.get('status') != expected_fidelity:
        raise ValueError('Internal causal fidelity audit missing or incomplete')
    for field in ['purge_violations', 'label_maturity_violations']:
        if fidelity.get(field) != 0:
            raise ValueError(f'Internal causal audit rejected: {field}')
    if fidelity.get('feature_count') != 45 or fidelity.get('arms') != 16 or fidelity.get('b2_input_count') != 2:
        raise ValueError('Internal feature/model arm contract differs')
    if fidelity.get('source_feature_extractor') != 'frozen_falcon_features.EnhancedTelemetryEngine.extract_branch1_features':
        raise ValueError('Different feature extractor was used')
    if not fidelity.get('inertia_input_is_strictly_lagged') or fidelity.get('pareto_cap', 1) is not None or fidelity.get('pareto_floor', 0) is not None:
        raise ValueError('Inertia/Pareto internal contract differs')
    if feature.get('feature_count') != 45 or len(feature.get('features', [])) != 45:
        raise ValueError('Actual feature manifest does not contain 45 columns')
    if contract['source_sha256'] != digest(ROOT / 'ML' / 'run_experiment.py'):
        raise ValueError('Current main source differs from run contract')
    for filename, expected_hash in contract.get('vendor_sha256', {}).items():
        if digest(ROOT / 'vendor' / filename) != expected_hash or feature['vendor_sha256'].get(filename) != expected_hash:
            raise ValueError('Vendored feature/math source differs from execution contract')
    totals = {arm: [] for arm in ARMS}
    rows = decisions = positive_rows = 0
    hashes = {}
    last_time = None
    gpu_order = None
    memory, memory_time = None, None
    for part in parts:
        index = int(part.stem.split('_')[-1])
        original_digest = digest(part)
        frame = pq.read_table(part).to_pandas()
        required = ['timestamp', 'gpu_uid', 'label'] + [f'risk_{arm}' for arm in ARMS]
        if not set(required).issubset(frame.columns):
            raise ValueError(f'{part.name}: missing required score columns')
        if frame.duplicated(['timestamp', 'gpu_uid']).any():
            raise ValueError(f'{part.name}: duplicated score keys')
        groups = frame.groupby('timestamp', sort=False)
        times = np.array(list(groups.groups))
        if len(times) == 0 or not times[1:].__ge__(times[:-1]).all():
            raise ValueError(f'{part.name}: unsorted timestamps')
        if last_time is not None and times[0] <= last_time:
            raise ValueError(f'{part.name}: overlapping origin decision times')
        last_time = times[-1]
        if not groups.size().eq(1992).all():
            raise ValueError(f'{part.name}: incomplete GPU population')
        for _, query in groups:
            actual_order = query.gpu_uid.astype(str).to_numpy()
            if gpu_order is None:
                gpu_order = actual_order.copy()
            if not np.array_equal(actual_order, gpu_order):
                raise ValueError(f'{part.name}: GPU order changed across queries')
        if len(np.unique(gpu_order)) != 1992:
            raise ValueError('Nonunique GPU population')
        labels = frame.label.to_numpy(dtype=np.uint8)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('Nonbinary label')
        branch = output / 'branches' / part.name
        branch_frame = pq.read_table(branch, columns=['timestamp', 'gpu_uid', 'label']).to_pandas()
        if not branch_frame.equals(frame[['timestamp', 'gpu_uid', 'label']]):
            raise ValueError(f'{part.name}: branch and fusion population/labels differ')
        models = json.loads((output / 'fusion_models' / f'origin_{index:04d}.json').read_text(encoding='utf-8'))
        memory, memory_time = verify_predictions(frame, models, contract['config'], memory, memory_time)
        for arm in ARMS:
            probability = frame[f'risk_{arm}'].to_numpy()
            if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
                raise ValueError(f'{part.name}/{arm}: invalid probability')
            fresh = score_metrics(probability, labels)
            stored = lookup[(index, arm)]
            for name, value in fresh.items():
                if name in stored and stored[name] != '' and not np.isclose(float(stored[name]), value, atol=1e-9, rtol=1e-7):
                    raise ValueError(f'{part.name}/{arm}/{name}: stored={stored[name]}, fresh={value}')
            totals[arm].append(fresh)
        if digest(part) != original_digest:
            raise ValueError('Scores modified during independent audit')
        hashes[part.name] = original_digest
        rows += len(frame)
        decisions += len(times)
        positive_rows += int(labels.sum())
    if len(lookup) != len(parts) * 16:
        raise ValueError('Metric count does not match 16 arms per persisted origin')
    if len(parts) == 64 and (rows, decisions, positive_rows) != (6081576, 3053, 52945):
        raise ValueError('Full-period evaluation coverage differs from approved E19 ground truth')
    trace = [json.loads(line) for line in (output / 'fit_trace.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not trace:
        raise ValueError('No fit trace: cannot verify fresh ML')
    if 'cuda' not in json.dumps(trace).lower():
        raise ValueError('CUDA fit evidence missing')
    completed_b1 = [row for row in trace if row.get('stage') == 'fit_done' and row.get('component') == 'B1']
    completed_b2 = [row for row in trace if row.get('stage') == 'fit_done' and row.get('component') == 'B2']
    expected_b1 = {(origin, days) for origin in range(len(parts)) for days in [7, 14, 21]}
    if {(int(row['origin']), int(row['days'])) for row in completed_b1} != expected_b1:
        raise ValueError('Missing fresh B1 fits for ST7/14/21')
    if {int(row['origin']) for row in completed_b2} != set(range(len(parts))):
        raise ValueError('Missing fresh B2 Logistic fits')
    for row in completed_b1:
        if not str(row.get('actual_backend', '')).startswith('cuda') and row.get('actual_backend') != 'constant_fallback':
            raise ValueError('B1 backend silently changed')
        if str(row.get('actual_backend', '')).startswith('cuda') and row.get('tree_count') != 150:
            raise ValueError('GBDT tree count changed')
    actual_cuda = sum(str(row.get('actual_backend', '')).startswith('cuda') for row in completed_b1)
    if actual_cuda != fidelity.get('actual_cuda_b1_fits'):
        raise ValueError('Actual CUDA fit count differs from main audit')
    aggregates = {}
    for arm, values in totals.items():
        hits = sum(v['hits_at_100'] for v in values)
        aggregates[arm] = {'origin_mean_pr_auc': float(np.mean([v['pr_auc'] for v in values])),
                           'origin_mean_ndcg_at_100': float(np.mean([v['ndcg_at_100'] for v in values])),
                           'hits_at_100': hits, 'pooled_recall_at_100': hits / positive_rows if positive_rows else 0.,
                           'pooled_precision_at_100': hits / (decisions * 100)}
    result = {'status': 'SCORE_ARTIFACTS_VERIFIED', 'audited_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'origins': len(parts), 'score_rows': rows, 'decision_times': decisions,
              'positive_rows': positive_rows, 'gpu_count': 1992, 'arm_count': 16,
              'metric_rows_recomputed': len(lookup), 'fit_trace_records': len(trace),
              'cuda_b1_fit_records': actual_cuda, 'b1_unique_fits': len(expected_b1),
              'b2_unique_fits': len(parts),
              'inertia_and_fusion_equations_reconstructed': True,
              'source_fidelity_audit': fidelity, 'feature_manifest': feature,
              'arms': aggregates, 'score_sha256': hashes,
              'execution_contract_sha256': digest(output / 'execution_contract.json'),
              'main_summary_status': summary.get('status', '미기록'),
              'scope': 'Independent persisted score/coverage/metric audit; training-time causal audit remains main fidelity_audit.json'}
    write_json(output / 'independent_audit.json', result)
    return result


@contextmanager
def log_lock(directory: Path):
    path = directory / '.experiment_logbook.lock'
    deadline = time.monotonic() + 180
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, json.dumps({'pid': os.getpid(), 'created': time.time()}).encode())
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError('Research log writer lock remained active')
            time.sleep(2)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def render(content: str) -> str:
    # Reuse the established standalone renderer, without executing its result-writing code.
    source = (WORKSPACE / 'E14_E19_plan_20261017/record_stopped_falcon_analysis.py').read_text(encoding='utf-8')
    start = source.index('def inline(text):')
    end = source.index("assert page.count('<h2>Experiment ')", start)
    namespace = {'html': html, 're': re, 'content': content}
    exec(source[start:end], namespace)
    page = namespace['page']
    if page.count('<h2>Experiment ') != len(re.findall(r'^## Experiment ', content, re.M)):
        raise ValueError('Markdown and HTML experiment sections differ')
    return page


def record(output: Path, result: dict) -> None:
    directory = WORKSPACE / '실험 기록'
    md = directory / '실험_연대기.md'
    marker = f'실행 증거: {output.relative_to(WORKSPACE).as_posix()}/independent_audit.json (origins={result["origins"]}; status={result["status"]})'
    with log_lock(directory):
        # Always reread after acquiring the lock; do not append to a cached log snapshot.
        original = md.read_text(encoding='utf-8')
        if marker in original:
            return
        numbers = list(map(int, re.findall(r'^## Experiment (\d+)', original, re.M)))
        number = max(numbers, default=0) + 1
        complete = result['origins'] == 64
        entry = f'\n\n## Experiment {number} = FalconStyle GBDT·ADST·예측 Inertia·Fusion 비교 = {"전체 점수 검증" if complete else "부분 실행"}\n\n[요약]\n\n'
        entry += f'- 승인된 명세 v1.1 구현 실행의 저장 결과 {result["origins"]}/64 origins를 독립 재계산했다. 16개 조건·{result["score_rows"]:,}행·{result["decision_times"]:,}시점, 양성 GPU-time {result["positive_rows"]:,}행이다.\n'
        entry += '- 독립 판정은 SCORE_ARTIFACTS_VERIFIED이며 학습·피드백 시간 인과성 판정은 같은 폴더 fidelity_audit.json의 실제 상태를 함께 확인한다.\n\n[실행]\n\n'
        entry += '- G-004·G-007·G-008. 신규 GitHub EnhancedTelemetryEngine 45열, XGBoost CUDA 단일 GBDT, B2 이력2열 Logistic/ST21. ST7·14·21 및 성숙 피드백 ADST를 동일 원시 점수에서 비교한다.\n'
        entry += '- All-XID24h inclusive-bin 라벨, 36h purge, 24h 라벨 성숙, 직전10분 제외/실제 마지막 telemetry15분 lag, 30분 전수1,992GPU scoring. KL logit 결합 Inertia는 예측에 반영하며 raw score로 상태를 예측 후 갱신한다.\n'
        entry += '- Pareto lambda=1/(XID30d count+1), cap/floor 없음. E19 기반 Fusion은 과거7일 calibration+7일 B2-offset residual이며 Inertia별 입력을 따로 학습한다. 원본 E19 dual-inertia 입력을 중복 추가하지 않는다.\n'
        entry += '- 진입점 FalconStyle_GBDT_ADST_Inertia/ML/run_experiment.py. execution_contract.json·manifest·fit_trace·feature_manifest와 branch/fusion 전수 점수를 보존한다. 새 독립 held-out 기간이 아닌 기존 개발64-origin 비교다.\n\n[결과]\n\n'
        entry += '| 조건 | origin 평균 AP | pooled Recall@100 | Top100 GPU-time 적중 |\n| --- | ---: | ---: | ---: |\n'
        for arm, value in result['arms'].items():
            entry += f'| {arm} | {value["origin_mean_pr_auc"]:.6f} | {value["pooled_recall_at_100"]:.6f} | {value["hits_at_100"]:,} |\n'
        if (output/'comparison.json').exists() and (output/'operational_metrics.json').exists():
            comparisons=json.loads((output/'comparison.json').read_text(encoding='utf-8'))
            operations=json.loads((output/'operational_metrics.json').read_text(encoding='utf-8'))
            entry += f'\n- 사전 지정 paired 비교 {len(comparisons["comparisons"])}개(full/trained subset 포함), 3일·7일 moving-block bootstrap 각5000회/seed20260905 결과는 comparison.json에 기록했다. 개발기간 기술적 신뢰구간이며 다중비교 보정 가설검정은 아니다.\n'
            entry += f'- 새로운 운영 정의의 완전24h 경고창을 가진 고유 GPU-onset {operations["eligible_unique_onsets"]}개를 평가했다. 각조건 onset포착·선행시간·Top100 turnover·고유추천GPU수는 operational_metrics.json에 보존했다. 기존E19 onset지표와 동일하다고 주장하지 않는다.\n'
        entry += '\n[분석]\n\n- 16개 조건은 같은 모집단·라벨·시점에서 비교한다. A는 ADST, I는 B1 Inertia, P는 상한 없는 Pareto, E는 E19 기반 residual이다. GPU-time 적중 수는 고유 고장 사건 수가 아니다.\n'
        entry += '- 모델·특징·Inertia 적용 위치가 원본 E19와 달라 원본 E19 전체 대비 차이를 Fusion만의 효과로 해석하지 않는다. peer는 gpu_idx//8 그룹 가정이며 실제 host/task 검증은 미완료다.\n'
        entry += '- 수학적 결합식의 해가 유일하다는 사실은 위험 순위 향상을 보장하지 않는다. 효과·채택은 날짜별 paired 차이, 불확실성, 진입 지연·이벤트 지표를 함께 검토해 판단한다.\n\n[실행할 것]\n\n1. 날짜별 paired 차이·3/7일 block bootstrap과 fallback 포함/학습가능 공통 origin 결과를 검토한다.\n2. ADST·Inertia·Fusion의 개별 효과와 상호작용을 확인한 뒤 신규 기간 검증 범위를 결정한다.\n\n' + marker + '\n'
        content = original + entry
        page = render(content)
        md_temp = md.with_name(md.name + f'.{os.getpid()}.tmp')
        html_path = md.with_suffix('.html')
        html_temp = html_path.with_name(html_path.name + f'.{os.getpid()}.tmp')
        md_temp.write_text(content, encoding='utf-8')
        html_temp.write_text(page, encoding='utf-8')
        if md.read_text(encoding='utf-8') != original:
            md_temp.unlink(); html_temp.unlink()
            raise RuntimeError('Concurrent legacy writer changed log; retry recording from latest source')
        md_temp.replace(md)
        html_temp.replace(html_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--no-record', action='store_true')
    parser.add_argument('--record-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    try:
        result = json.loads((output/'independent_audit.json').read_text(encoding='utf-8')) if args.record_only else inspect(output)
        if args.record_only and result['origins'] != len(list((output/'scores').glob('origin_*.parquet'))):
            raise ValueError('Stored audit is stale; rerun independent audit first')
        if not args.no_record:
            record(output, result)
    except Exception as error:
        write_json(output / 'independent_audit_failure.json', {'status': 'AUDIT_FAILED', 'error': f'{type(error).__name__}: {error}'})
        raise
    print(json.dumps({key: value for key, value in result.items() if key not in ['feature_manifest', 'score_sha256', 'source_fidelity_audit']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

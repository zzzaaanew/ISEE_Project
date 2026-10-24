"""Descriptive paired evaluation; writes only reports/comparison.{json,csv}."""
from __future__ import annotations

import csv
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
ARMS = ('fifo', 'tiresias', 'joint')
N_JOBS, N_GPUS, N_DECISIONS = 140786, 100, 18317
START = datetime.fromisoformat('2023-06-15T09:40:00+00:00').timestamp()
END = datetime.fromisoformat('2023-08-18T00:05:00+00:00').timestamp()
NOTE = ('Descriptive single frozen-period comparison; joint is an approximate '
        'surrogate, not globally optimal. No statistical significance or '
        'independent generalization is established.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def field(summary, *aliases):
    for key in aliases:
        if key in summary:
            return summary[key]
    flat = flatten(summary)
    found = [(key, value) for key, value in flat.items() if key.rsplit('.', 1)[-1] in aliases]
    require(len(found) == 1, f'Missing or ambiguous summary field {aliases}: {found}')
    return found[0][1]


def overhead(summary):
    """Preserve explicit units; never add job-seconds to GPU-seconds."""
    groups = {'scheduling': {}, 'failure': {}, 'periodic_checkpoint': {}}
    unclassified = {}
    for key, value in flatten(summary).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        low = key.lower()
        unit = ('GPU-seconds' if 'gpu_seconds' in low else
                'job-seconds' if 'job_seconds' in low else
                'seconds' if low.endswith('_seconds') else None)
        if unit is None:
            continue
        if 'periodic' in low or 'periodic_cp' in low:
            group = 'periodic_checkpoint'
        elif any(token in low for token in ('failure', 'fault', 'xid', 'retry', 'rollback')):
            group = 'failure'
        elif any(token in low for token in ('schedul', 'preempt', 'migration')):
            group = 'scheduling'
        else:
            if any(token in low for token in ('checkpoint', 'restart', 'overhead', 'blocked')):
                unclassified[key] = {'value': value, 'unit': unit}
            continue
        groups[group][key] = {'value': value, 'unit': unit}
    return {'by_cause': groups, 'unclassified_explicit_unit_fields': unclassified,
            'unavailable_causes': [key for key, value in groups.items() if not value],
            'note': 'Original counters preserved separately; fields may overlap and are not summed. Missing is not zero.'}


def stats(series):
    return {'mean_seconds': float(series.mean()) if len(series) else None,
            'p95_seconds': float(series.quantile(.95)) if len(series) else None}


def comparison():
    import numpy as np
    import pandas as pd

    frames, summaries, rows, complete_ids = {}, {}, {}, {}
    reference = None
    for arm in ARMS:
        directory = ROOT / 'runs' / arm
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8-sig'))
        summaries[arm] = summary
        require(int(field(summary, 'decision_count', 'decisions', 'num_decisions', 'n_decisions', 'regular_decisions', 'decision_steps')) == N_DECISIONS,
                f'{arm}: expected {N_DECISIONS} decisions')
        require(int(field(summary, 'gpu_count', 'gpus', 'num_gpus', 'n_gpus')) == N_GPUS,
                f'{arm}: expected {N_GPUS} GPUs')
        require(int(field(summary, 'total_jobs', 'job_count', 'jobs_total', 'num_jobs', 'jobs')) == N_JOBS,
                f'{arm}: expected {N_JOBS} jobs')
        start = float(field(summary, 'start_time', 'start_epoch', 'start_ts'))
        end = float(field(summary, 'end_time', 'end_epoch', 'end_ts'))
        require(start == START and end == END and end - start == N_DECISIONS * 300,
                f'{arm}: full UTC horizon mismatch: {start}, {end}')
        if 'risk_on' in summary:
            require(summary['risk_on'] in (True, 1), f'{arm}: risk must be ON')
        frame = pd.read_parquet(directory / 'jobs.parquet')
        required = {'job_id', 'arrival', 'completed', 'duration', 'progress', 'demand',
                    'state', 'queue_wait', 'interruptions', 'launches', 'started'}
        require(required <= set(frame.columns), f'{arm}: missing columns {required - set(frame.columns)}')
        require(len(frame) == N_JOBS and frame.job_id.notna().all() and frame.job_id.is_unique,
                f'{arm}: invalid job population')
        frame = frame.set_index('job_id').sort_index()
        for column in ('arrival', 'duration', 'progress', 'demand', 'queue_wait', 'interruptions', 'launches'):
            frame[column] = pd.to_numeric(frame[column], errors='raise')
            require(np.isfinite(frame[column].to_numpy(dtype=float)).all(), f'{arm}: nonfinite {column}')
        for column in ('completed', 'started'):
            frame[column] = pd.to_numeric(frame[column], errors='raise')
            present = frame[column].dropna()
            require(np.isfinite(present.to_numpy(dtype=float)).all(), f'{arm}: invalid {column}')
            require(((present >= frame.loc[present.index, 'arrival']) & (present <= end)).all(),
                    f'{arm}: out-of-horizon {column}')
        require(((frame.arrival >= start) & (frame.arrival < end)).all(), f'{arm}: arrival outside horizon')
        require(((frame.duration > 0) & (frame.demand >= 1) & (frame.demand <= N_GPUS) &
                 (frame.demand == np.floor(frame.demand))).all(), f'{arm}: invalid duration/demand')
        require(((frame.progress >= -1e-7) & (frame.progress <= frame.duration + 1e-6) &
                 (frame.queue_wait >= -1e-7)).all(), f'{arm}: invalid progress/wait')
        inputs = frame[['arrival', 'duration', 'demand']]
        if reference is None:
            reference = inputs
        else:
            pd.testing.assert_frame_equal(inputs, reference, check_dtype=False, check_exact=True)
        completed = frame.completed.notna()
        require(np.allclose(frame.loc[completed, 'progress'], frame.loc[completed, 'duration'], rtol=0, atol=1e-6),
                f'{arm}: completed work is not fully progressed')
        failed = frame.state.astype(str).str.lower().isin(('failed', 'failure', 'permanent_failure'))
        require(not (completed & failed).any(), f'{arm}: failed/completed state conflict')
        complete_ids[arm] = set(frame.index[completed])
        jct = frame.loc[completed, 'completed'] - frame.loc[completed, 'arrival']
        restricted = np.minimum(frame.completed.fillna(end), end) - frame.arrival
        retained = float((frame.progress * frame.demand).sum())
        useful_completed = float((frame.loc[completed, 'duration'] * frame.loc[completed, 'demand']).sum())
        rows[arm] = {'arm': arm, 'total_jobs': len(frame), 'completed': int(completed.sum()),
                     'unfinished': int((~completed).sum()), 'failed': int(failed.sum()),
                     'unfinished_nonfailed': int((~completed & ~failed).sum()),
                     'never_started': int(frame.started.isna().sum()),
                     'restricted_total_flow_job_seconds': float(restricted.sum()),
                     'restricted_mean_flow_seconds': float(restricted.mean()),
                     'completed_jct_mean_seconds': stats(jct)['mean_seconds'],
                     'completed_jct_p95_seconds': stats(jct)['p95_seconds'],
                     'queue_wait_p95_seconds_all_jobs': float(frame.queue_wait.quantile(.95)),
                     'useful_completed_gpu_seconds': useful_completed,
                     'retained_gpu_seconds': retained,
                     'retained_unfinished_gpu_seconds': retained - useful_completed,
                     'remaining_gpu_seconds': float(((frame.duration - frame.progress) * frame.demand).sum()),
                     'interruptions_count': int(frame.interruptions.sum()),
                     'launches_count': int(frame.launches.sum())}
        frames[arm] = frame
    intersection = set.intersection(*complete_ids.values())
    union = set.union(*complete_ids.values())
    common = {}
    for arm in ARMS:
        # Common job IDs ensure that mean differences describe the same jobs.
        subset = frames[arm].loc[sorted(intersection)]
        common[arm] = stats(subset.completed - subset.arrival)
    pairs = {}
    for left, right in itertools.combinations(ARMS, 2):
        ids = sorted(complete_ids[left] & complete_ids[right])
        a, b = frames[left].loc[ids], frames[right].loc[ids]
        delta = (b.completed - b.arrival) - (a.completed - a.arrival)
        pairs[f'{right}_minus_{left}'] = {
            'common_completed': len(ids), 'completion_union': len(complete_ids[left] | complete_ids[right]),
            'left_only_completed': len(complete_ids[left] - complete_ids[right]),
            'right_only_completed': len(complete_ids[right] - complete_ids[left]),
            'paired_jct_delta_mean_seconds': float(delta.mean()) if len(delta) else None,
            'paired_jct_delta_p95_seconds': float(delta.quantile(.95)) if len(delta) else None,
            'paired_jct_faster_count': int((delta < -1e-7).sum()),
            'paired_jct_equal_count': int((abs(delta) <= 1e-7).sum()),
            'paired_jct_slower_count': int((delta > 1e-7).sum())}
    return {'status': 'PASS_DESCRIPTIVE_APPROXIMATE_SURROGATE', 'note': NOTE,
            'horizon': {'start_epoch_utc': START, 'end_epoch_utc_exclusive': END,
                        'elapsed_seconds': END - START, 'decisions': N_DECISIONS, 'gpus': N_GPUS},
            'definitions': {'restricted_total_flow': 'sum(min(completed or T_end,T_end)-arrival) over ALL jobs',
                            'unfinished': 'all jobs lacking completion, including terminal failures',
                            'retained_gpu_seconds': 'sum(progress*demand), including completed jobs',
                            'remaining_gpu_seconds': 'sum((duration-progress)*demand), including unfinished failed jobs',
                            'queue_wait': 'engine cumulative queue wait through the common horizon; all jobs',
                            'quantiles': 'linear interpolation, pandas quantile(0.95)'},
            'arms': rows, 'completion_intersection': len(intersection), 'completion_union': len(union),
            'all_three_common_completed_jct': common, 'paired_comparisons': pairs,
            'overhead': {arm: overhead(summary) for arm, summary in summaries.items()},
            'source_summaries': summaries}, list(rows.values())


def main():
    reports = ROOT / 'reports'
    json_path, csv_path = reports / 'comparison.json', reports / 'comparison.csv'
    if json_path.exists() or csv_path.exists():
        raise FileExistsError('Refusing to overwrite existing comparison report')
    reports.mkdir(exist_ok=True)
    try:
        result, rows = comparison()
        exit_code = 0
    except Exception as exc:
        result = {'status': 'ERROR', 'note': NOTE, 'error_type': type(exc).__name__,
                  'error': str(exc), 'traceback': traceback.format_exc(),
                  'input_paths': {arm: str(ROOT / 'runs' / arm) for arm in ARMS}}
        rows = [{'status': 'ERROR', 'error_type': type(exc).__name__, 'error': str(exc)}]
        exit_code = 1
    with json_path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with csv_path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'EVALUATION status={result["status"]} report={json_path}', flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

"""Post-run read-only cross checks; only writes its own exclusive evidence JSON.

This analysis helper is not imported by the running simulation or any policy.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def close(a, b, label):
    if abs(a - b) > max(.01, abs(b) * 1e-10):
        raise AssertionError(f'{label}: {a} != {b}')


def main():
    target = ROOT / 'reports/final_verification.json'
    if target.exists():
        raise FileExistsError('No implicit rerun/overwrite')
    monitor = read(ROOT / 'reports/monitor.json')
    if monitor['status'] != 'COMPLETED':
        raise RuntimeError('All runs and descriptive evaluator must finish first')
    import pandas as pd
    began = time.monotonic()
    report = dict(status='FAIL', full_replay=True, arms={})
    try:
        old = read(ROOT.parent / '2026-10-23_JointQueuePlacement_Preparation_01/reports/input_audit.json')
        expected = old['preservation']['before']
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = dict(zip(expected, pool.map(sha, expected)))
        if actual != expected:
            raise AssertionError('Frozen source/input/result files changed')
        report['preserved_files'] = len(expected)
        report['source_preservation_exact'] = True
        population = None
        common_code = None
        for arm in ('fifo', 'tiresias', 'joint'):
            folder = ROOT / 'runs' / arm
            m, summary = read(folder / 'manifest.json'), read(folder / 'summary.json')
            for name, value in m['code_sha256'].items():
                if sha(ROOT / name) != value:
                    raise AssertionError(f'Executed code changed: {name}')
            if common_code is None:
                common_code = m['code_sha256']
            elif common_code != m['code_sha256']:
                raise AssertionError('Code differs between arms')
            for item in m['config']['inputs'].values():
                if sha(item['path']) != item['sha256']:
                    raise AssertionError('Runtime input hash mismatch')
            provenance = m['config']['score_provenance']
            if sha(provenance['path']) != provenance['sha256']:
                raise AssertionError('Score provenance changed')
            assert summary['status'] == 'completed' and summary['decision_count'] == 18317
            assert summary['total_jobs'] == 140786 and summary['gpu_count'] == 100 and summary['risk_on']
            jobs = pd.read_parquet(folder / 'jobs.parquet').set_index('job_id').sort_index()
            current = jobs[['arrival', 'duration', 'demand']]
            if population is None:
                population = current
            else:
                pd.testing.assert_frame_equal(current, population, check_exact=True)
            completion = jobs.completed.notna()
            close(float(jobs.queue_wait.sum()), summary['queue_job_seconds'], f'{arm} queue integral')
            close(float(jobs.periodic_checkpoint_seconds.sum()), summary['periodic_checkpoint_job_seconds'], f'{arm} periodic cost')
            retained = float((jobs.progress * jobs.demand).sum())
            close(retained, summary['retained_useful_gpu_seconds'], f'{arm} retained work')
            close(float((jobs.loc[completion, 'duration'] * jobs.loc[completion, 'demand']).sum()),
                  summary['completed_useful_gpu_seconds'], f'{arm} completed work')
            close(summary['executed_gpu_seconds'], retained + summary['observed_rollback_gpu_seconds'], f'{arm} execution/rollback')
            close(float((jobs.duration * jobs.demand).sum()), retained + summary['unfinished_remaining_gpu_seconds'], f'{arm} input work')
            flow = float((jobs.completed.fillna(summary['end_time']) - jobs.arrival).sum())
            close(flow, summary['restricted_total_flow_seconds'], f'{arm} restricted flow')
            gpu = pd.read_parquet(folder / 'gpu_state_time.parquet')
            elapsed = summary['end_time'] - summary['start_time']
            assert gpu.gpu_uid.nunique() == 100
            for uid, value in gpu.groupby('gpu_uid').seconds.sum().items():
                close(float(value), elapsed, f'{arm}/{uid} full horizon')
            busy = float(gpu.loc[gpu.state.isin(['running', 'loading', 'checkpointing']), 'seconds'].sum())
            close(busy, summary['busy_gpu_seconds'], f'{arm} allocation time')
            report['arms'][arm] = dict(full_population_and_period=True, code_inputs_exact=True,
                queue_integral=True, periodic_cost=True, work_and_rollback=True,
                restricted_flow=True, gpu_capacity_time=True,
                simulation_stage_peak_memory_bytes=summary['peak_memory_bytes'],
                final_progress_peak_memory_bytes=read(folder / 'progress.json').get('peak_memory_bytes'),
                wall_seconds=summary['elapsed_seconds'], decision_count=18317)
        comparison = read(ROOT / 'reports/comparison.json')
        assert comparison['status'] == 'PASS_DESCRIPTIVE_APPROXIMATE_SURROGATE'
        report.update(status='PASS', same_population=True, same_code=True,
                      statistical_or_generalization_claim=False,
                      independent_reproduction_performed=False,
                      reproducibility_note='seed17 recorded; PYTHONHASHSEED separately not fixed/recorded; joint tied proposal order can depend on set iteration')
    except Exception as exc:
        import traceback
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    report['elapsed_seconds'] = time.monotonic() - began
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=True), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

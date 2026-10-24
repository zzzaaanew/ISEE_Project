"""Describe existing results only; no policy fitting, rerun, or retuning."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'reports/limitations_evidence.json'
    if target.exists():
        raise FileExistsError('Refusing implicit repeat/overwrite')
    comparison = json.loads((ROOT / 'reports/comparison.json').read_text(encoding='utf-8'))
    rows = {}
    for arm in ('fifo', 'tiresias', 'joint'):
        jobs = pd.read_parquet(ROOT / 'runs' / arm / 'jobs.parquet')
        groups = {}
        for demand, subset in jobs.groupby('demand'):
            groups[str(int(demand))] = dict(jobs=len(subset),
                completed=int(subset.completed.notna().sum()),
                never_started=int(subset.started.isna().sum()))
        large = jobs.loc[jobs.demand >= 32]
        summary = comparison['source_summaries'][arm]
        rows[arm] = dict(by_gpu_demand=groups,
            gpu_demand_ge32=dict(jobs=len(large), completed=int(large.completed.notna().sum()),
                                 never_started=int(large.started.isna().sum())),
            unfinished=int(jobs.completed.isna().sum()),
            never_started=int(jobs.started.isna().sum()),
            max_accumulated_queue_wait_seconds=float(jobs.queue_wait.max()),
            scheduler_preemptions=summary['event_counts'].get('scheduler_preemption', 0),
            active_xid_interruptions=summary['active_xid_job_interruptions'],
            failure_rollback_gpu_seconds=summary['observed_rollback_gpu_seconds'],
            hardware_recoveries=summary['confirmed_hardware_recoveries'],
            preventive_checkpoint_seconds=summary['preventive_checkpoint_seconds'],
            migrations=summary['migrations'])
    offered = float((jobs.duration * jobs.demand).sum())
    capacity = comparison['horizon']['elapsed_seconds'] * comparison['horizon']['gpus']
    result = dict(status='DESCRIPTIVE_EXISTING_RESULTS_ONLY', goal='G-009', experiment=30,
        arms=rows, offered_gpu_seconds=offered, physical_capacity_gpu_seconds=capacity,
        offered_load_ratio=offered/capacity,
        new_simulation_or_tuning=False, independent_reproduction=False,
        interpretation='Demand-stratified descriptive comparison; no causal isolation of ML contribution or statistical/generalization claim.')
    with target.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=True, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()

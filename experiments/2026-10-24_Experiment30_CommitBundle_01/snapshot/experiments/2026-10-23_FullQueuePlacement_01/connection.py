"""One-shot unit + actual six-hour connection gate, followed by approved full runs.

Does not select parameters from observed performance. Failures are preserved.
"""
from dataclasses import fields
import itertools
import random
import time
from types import SimpleNamespace
import pandas as pd
from common import *
from engine import FullSimulation
from policies import JobView, choose, capacity_dp
from run import build


class ConstantRisk:
    def __init__(self, ids, start):
        self.ids, self.at = ids, start

    def snapshot(self, at):
        self.at = at
        return {g: (i + 1) / 100 for i, g in enumerate(self.ids)}


class Failures:
    def __init__(self, events=()):
        self.events = list(events)
        self.index = 0

    def next_time(self):
        return self.events[self.index]['time'] if self.index < len(self.events) else float('inf')

    def events_at(self, at):
        result = []
        while self.index < len(self.events) and self.events[self.index]['time'] <= at:
            result.append(self.events[self.index])
            self.index += 1
        return result


def fixture(arm, records, *, duration=1800, cp=300, n=3, fail_at=None):
    m = manifest(arm)
    start = pd.Timestamp('2023-01-01T00:00:00Z').timestamp()
    m.update(start_time=pd.Timestamp(start, unit='s', tz='UTC').isoformat(),
             end_time=pd.Timestamp(start + duration, unit='s', tz='UTC').isoformat(),
             cohort_size=n, fixture_only=True)
    ids = [f'g{k}' for k in range(n)]
    top = SimpleNamespace(frame=pd.DataFrame([dict(gpu_uid=g, server_id='s0', local_gpu_id=i)
                                             for i, g in enumerate(ids)]))
    jobs = SimpleNamespace(records=[dict(job_id=jid, gang_id=jid, arrival=start + at,
        duration_seconds=work, gpu_demand=demand, checkpoint_interval_seconds=cp)
        for jid, at, work, demand in records])
    events = [] if fail_at is None else [dict(time=start + fail_at, episode_id='fixture-xid43',
                                            gpu_uid='g0', parsed_xid_codes=[43])]
    trace = []
    sim = FullSimulation(m, top, ConstantRisk(ids, start), Failures(events), jobs,
                         dict(threshold=.5), emit=lambda kind, row: trace.append((kind, row)))
    return sim, trace


def main():
    output = ROOT / 'reports/connection.json'
    if output.exists():
        raise FileExistsError('No implicit connection rerun/overwrite')
    began = time.monotonic()
    report = dict(status='FAIL', experiment=30, goal='G-009', tests=[], actual_connection=[],
                  full_run_required_after_pass=True, seed=17, command=[sys.executable, '-B', __file__])
    try:
        for arm in ('fifo', 'tiresias', 'joint'):
            sim, _ = fixture(arm, [('a', 0, 400, 2)], n=2, duration=600)
            summary = sim.run()
            assert sim.jobs['a'].completed - sim.start == 460
            assert summary['periodic_checkpoint_job_seconds'] == 60
            assert summary['busy_gpu_seconds'] == 920
            report['tests'].append(f'{arm}: periodic_pause60_and_capacity')
        sim, _ = fixture('joint', [('a', 0, 400, 2)], n=2, fail_at=350)
        summary = sim.run()
        assert sim.jobs['a'].completed - sim.start == 1660
        assert summary['observed_rollback_gpu_seconds'] == 600
        assert summary['active_xid_job_interruptions'] == 1
        report['tests'].append('failure_during_checkpoint_rollback_and_retry660')
        sim, trace = fixture('joint', [('large', 0, 10000, 3), ('a', 100, 100, 1),
                                     ('b', 100, 100, 1), ('c', 100, 100, 1)], duration=600, cp=10000)
        summary = sim.run()
        launches = [row for kind, row in trace if kind == 'events' and row['type'] == 'job_launch' and row['job_id'] != 'large']
        assert len(launches) == 3 and all(row['time'] == 360 for row in launches)
        assert summary['scheduling_preemption_overhead_seconds'] == 300  # 60 + censored restart240
        report['tests'].append('joint_selection_checkpoint_GPU_reserved60_then_commit')
        assert not {'duration', 'completed', 'future_failure', 'label'} & {f.name for f in fields(JobView)}
        report['tests'].append('policy_view_no_future_runtime_fields')
        rng = random.Random(17)
        for _ in range(40):
            js = [JobView(str(k), rng.randint(1, 4), k, None, 0, 0, (), 0) for k in range(5)]
            dp = capacity_dp(js, 6, {str(k): .1 for k in range(6)})
            oracle = max(sum(300 for j in subset) for r in range(6) for subset in itertools.combinations(js, r)
                         if sum(j.demand for j in subset) <= 6)
            assert len(dp) * 300 == oracle
        report['tests'].append('40_seed17_DP_capacity_oracle_cases')
        js = [JobView('large', 5, 0, None, 0, 0, (), 0), JobView('small', 1, 1, None, 0, 0, (), 0)]
        p, _ = choose('fifo', js, {str(k): .1 for k in range(4)})
        assert p == {}
        js = [JobView(str(k), (1, 2, 3, 4, 8, 32, 64)[k % 7], k, None, 0, 0, (), 0) for k in range(300)]
        p, evidence = choose('joint', js, {str(k): .01 * (k % 10) for k in range(100)})
        gs = [g for v in p.values() for g in v]
        assert len(gs) == len(set(gs)) <= 100 and evidence['solver_status'] == 'FEASIBLE_APPROXIMATE'
        report['tests'].append('100GPU_300readyjobs_and_strictFIFO_HOL')
        for arm in ('fifo', 'tiresias', 'joint'):
            print('ACTUAL_CONNECTION ' + arm, flush=True)
            sim = build(arm, hours=6)
            try:
                result = sim.run()
                assert result['decision_count'] == 72 and result['gpu_count'] == 100
                assert result['progress_conservation'] and result['flow_conservation'] and result['gpu_time_conservation']
                report['actual_connection'].append(dict(arm=arm, decisions=result['decision_count'],
                    jobs=result['total_jobs'], completed=result['completed_jobs'],
                    mean_planning_seconds=result['mean_planning_seconds'], max_planning_seconds=result['max_planning_seconds']))
            finally:
                sim.risk.close()
        report['status'] = 'PASS'
    except BaseException as exc:
        import traceback
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    report['elapsed_seconds'] = time.monotonic() - began
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=True), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

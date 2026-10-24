"""Full approved actual-data run; no output overwrite or automatic retry."""
import argparse
from collections import Counter
import gzip
import os
import shutil
import time
from datetime import datetime, timezone
import pandas as pd
from common import *
from engine import FullSimulation
from risklab.feeds import load_inputs
from risklab.runtime_metrics import peak_memory_bytes


def build(arm, *, hours=None, emit=None, progress=None):
    m = manifest(arm)
    _, topology, risk, failures, trace, threshold = load_inputs(E27 / 'configs/risk1.json')
    if hours is not None:
        end = pd.Timestamp(m['start_time']) + pd.Timedelta(hours=hours)
        m['end_time'] = end.isoformat()
        m['fixture_only'] = True
        trace.records = [r for r in trace.records if r['arrival'] < end.timestamp()]
        failures.events = [r for r in failures.events if r['time'] < end.timestamp()]
    return FullSimulation(m, topology, risk, failures, trace, threshold, emit=emit, progress=progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=['fifo', 'tiresias', 'joint'])
    args = parser.parse_args()
    gate = read_json(ROOT / 'reports/connection.json')
    if gate['status'] != 'PASS':
        raise AssertionError('Connection gate not passed')
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise OSError('Need at least 10 GiB free for full traces')
    out = ROOT / 'runs' / args.arm
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    streams = {name: gzip.open(out / f'{name}.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)
               for name in ('events', 'decisions')}
    counters = Counter()
    last_progress = [-float('inf')]
    sim = None

    def emit(stream, row):
        if stream in streams:
            streams[stream].write(json.dumps(row, ensure_ascii=True, separators=(',', ':')) + '\n')
        if stream == 'events':
            counters[row['type']] += 1

    def progress(row):
        wall = time.monotonic()
        if wall - last_progress[0] >= 10 or row['progress'] >= 1:
            row = {**row, 'elapsed_seconds': wall - began, 'pid': os.getpid(),
                   'peak_memory_bytes': peak_memory_bytes()}
            write_json(out / 'progress.json', row, exclusive=False)
            print(json.dumps(row, ensure_ascii=True), flush=True)
            last_progress[0] = wall

    contract = manifest(args.arm)
    write_json(out / 'manifest.json', dict(
        config=contract, command=[sys.executable, '-B', *sys.argv],
        started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        code_sha256={p.name: sha(p) for p in ROOT.glob('*.py')},
        inputs_sha256={k: v['sha256'] for k, v in contract['inputs'].items()},
        score_provenance=contract['score_provenance'], seed=17, full_replay_run=True))
    try:
        sim = build(args.arm, emit=emit, progress=progress)
        summary = sim.run()
        if summary['decision_count'] != 18317 or summary['total_jobs'] != 140786 or len(sim.gpus) != 100:
            raise AssertionError('Not the full approved population/period')
        summary.update(elapsed_seconds=time.monotonic() - began,
                       peak_memory_bytes=peak_memory_bytes(), event_counts=dict(counters))
        pd.DataFrame([dict(job_id=j.id, arrival=j.arrival, completed=j.completed,
            duration=j.duration, progress=j.progress, demand=j.demand, state=j.state,
            queue_wait=j.queue_wait, interruptions=j.interruptions, launches=j.launches,
            started=j.started, periodic_checkpoint_seconds=sim.periodic_costs[j.id])
            for j in sim.jobs.values()]).to_parquet(out / 'jobs.parquet', index=False)
        pd.DataFrame([dict(gpu_uid=uid, state=state, seconds=seconds)
                      for (uid, state), seconds in sim.gpu_time.items()]).to_parquet(out / 'gpu_state_time.parquet', index=False)
        write_json(out / 'summary.json', summary)
        progress({**sim.snapshot(), 'progress': 1.0})
        print(json.dumps({'event': 'COMPLETED', 'arm': args.arm, 'summary': summary}), flush=True)
    except BaseException as exc:
        write_json(out / 'failure.json', dict(type=type(exc).__name__, error=str(exc),
            elapsed_seconds=time.monotonic() - began, snapshot=sim.snapshot() if sim else None))
        raise
    finally:
        for stream in streams.values():
            stream.close()
        if sim is not None:
            sim.risk.close()


if __name__ == '__main__':
    main()

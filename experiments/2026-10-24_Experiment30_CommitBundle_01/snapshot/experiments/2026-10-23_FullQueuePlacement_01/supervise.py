"""One-shot, gated supervisor for the three approved full-period ON arms.

Invoke only after the main runner announces FROZEN/GATEPASS:
    REPO/.venv/Scripts/python.exe -B supervise.py --gate FROZEN/GATEPASS
No retry, no overwrite, no intervention except the 7200-second hard timeout.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PYTHON = REPO / '.venv' / 'Scripts' / 'python.exe'
ARMS = ('fifo', 'tiresias', 'joint')
TIMEOUT = 7200
INTERVAL = 30
EXPECTED = ('summary.json', 'jobs.parquet', 'progress.json', 'manifest.json',
            'events.jsonl.gz', 'decisions.jsonl.gz')


def utc():
    return datetime.now(timezone.utc).isoformat()


def free_ram():
    if os.name != 'nt':
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in
                      ('total_phys', 'avail_phys', 'total_page', 'avail_page',
                       'total_virtual', 'avail_virtual', 'avail_extended')]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return status.avail_phys
    return None


def read_progress(arm):
    path = ROOT / 'runs' / arm / 'progress.json'
    try:
        return {'data': json.loads(path.read_text(encoding='utf-8-sig')),
                'mtime_ns': path.stat().st_mtime_ns}
    except (OSError, ValueError) as exc:
        return {'read_error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', required=True, choices=['FROZEN/GATEPASS'])
    parser.parse_args()
    reports = ROOT / 'reports'
    connection = json.loads((reports / 'connection.json').read_text(encoding='utf-8-sig'))
    if connection.get('status') != 'PASS':
        raise RuntimeError('Connection test must report status PASS before launch')
    for path in (PYTHON, ROOT / 'run.py', ROOT / 'evaluate.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    protected = [ROOT / 'runs' / arm for arm in ARMS]
    protected += [reports / name for name in ('monitor.json', 'comparison.json', 'comparison.csv')]
    protected += [reports / f'{arm}.{stream}.log'
                  for arm in ARMS for stream in ('stdout', 'stderr')]
    conflicts = [str(path) for path in protected if path.exists()]
    if conflicts:
        raise FileExistsError('Refusing a repeat/overwrite: ' + ', '.join(conflicts))
    reports.mkdir(exist_ok=True)
    state = {'status': 'STARTING', 'started_at': utc(), 'connection': connection,
             'free_ram_bytes_at_start': free_ram(), 'hard_timeout_seconds': TIMEOUT,
             'poll_seconds': INTERVAL, 'arms': {}, 'errors': []}
    monitor = reports / 'monitor.json'
    # Exclusive claim prevents two supervisors from starting the same experiment.
    with monitor.open('x', encoding='utf-8') as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)

    def publish():
        state['updated_at'] = utc()
        try:
            with monitor.open('w', encoding='utf-8') as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
        except OSError as exc:
            print(f'MONITOR_WRITE_ERROR {exc}', flush=True)

    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    children, streams, clocks, previous, stalled = {}, {}, {}, {}, {}
    for arm in ARMS:
        command = [str(PYTHON), '-B', str(ROOT / 'run.py'), '--arm', arm]
        record = {'command': command, 'cwd': str(ROOT), 'started_at': utc(),
                  'stdout_path': str(reports / f'{arm}.stdout.log'),
                  'stderr_path': str(reports / f'{arm}.stderr.log'),
                  'pid': None, 'exit_code': None, 'elapsed_seconds': 0,
                  'status': 'STARTING'}
        state['arms'][arm] = record
        try:
            out = open(record['stdout_path'], 'x', encoding='utf-8')
            streams[arm] = [out]
            err = open(record['stderr_path'], 'x', encoding='utf-8')
            streams[arm].append(err)
            clocks[arm] = time.monotonic()
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=out, stderr=err, shell=False)
            children[arm] = process
            record.update(pid=process.pid, status='RUNNING')
            print(f'START arm={arm} pid={process.pid} command={subprocess.list2cmdline(command)}', flush=True)
        except Exception:
            record.update(status='LAUNCH_FAILED', ended_at=utc(), error=traceback.format_exc())
            for stream in streams.get(arm, []):
                stream.close()
            print(f'LAUNCH_FAILED arm={arm} {record["error"]}', flush=True)
        publish()

    state['status'] = 'RUNNING'
    active = set(children)
    while active:
        for arm in tuple(sorted(active)):
            process = children[arm]
            record = state['arms'][arm]
            record['elapsed_seconds'] = round(time.monotonic() - clocks[arm], 3)
            code = process.poll()
            if code is None and record['elapsed_seconds'] >= TIMEOUT:
                # Flush notification BEFORE the only permitted automatic kill.
                print(f'HARD_TIMEOUT arm={arm} pid={process.pid} elapsed={record["elapsed_seconds"]}s; killing now', flush=True)
                record['status'] = 'HARD_TIMEOUT'
                record['timeout_notified_at'] = utc()
                publish()
                try:
                    process.kill()
                except OSError as exc:
                    record['kill_error'] = str(exc)
                code = process.poll()
            record['alive'] = code is None
            record['progress'] = read_progress(arm)
            signature = record['progress'].get('mtime_ns')
            stalled[arm] = stalled.get(arm, 0) + 1 if signature == previous.get(arm) else 0
            previous[arm] = signature
            if code is None and stalled[arm] in (3, 5):
                print(f'STALL_ADVISORY arm={arm} unchanged_checks={stalled[arm]} (continuing)', flush=True)
            if code is not None:
                active.remove(arm)
                record.update(exit_code=code, ended_at=utc())
                if record['status'] != 'HARD_TIMEOUT':
                    record['status'] = 'COMPLETED' if code == 0 else 'CRASHED'
                record['outputs'] = {}
                for name in EXPECTED:
                    path = ROOT / 'runs' / arm / name
                    record['outputs'][name] = {'exists': path.is_file(),
                                                'bytes': path.stat().st_size if path.is_file() else None}
                if code == 0 and not all(item['exists'] and item['bytes'] > 0
                                          for item in record['outputs'].values()):
                    record['status'] = 'OUTPUT_MISSING'
                for stream in streams.get(arm, []):
                    stream.close()
                try:
                    record['stderr_tail'] = Path(record['stderr_path']).read_text(
                        encoding='utf-8', errors='replace').splitlines()[-50:]
                except OSError as exc:
                    record['stderr_tail_error'] = str(exc)
            progress = record['progress'].get('data', {})
            compact = {key: value for key, value in progress.items()
                       if isinstance(value, (int, float, str, bool, type(None)))} if isinstance(progress, dict) else progress
            print(f'CHECK arm={arm} pid={record["pid"]} status={record["status"]} '
                  f'elapsed={record["elapsed_seconds"]} progress={json.dumps(compact, ensure_ascii=False)}', flush=True)
        publish()
        if active:
            try:
                time.sleep(INTERVAL)
            except KeyboardInterrupt:
                print('INTERRUPT_ADVISORY: approved runs continue; only hard timeout auto-kills', flush=True)

    if all(record['status'] == 'COMPLETED' for record in state['arms'].values()):
        state['status'] = 'EVALUATING'
        publish()
        try:
            import evaluate
            evaluation_code = evaluate.main()
            state['evaluation_exit_code'] = evaluation_code
            state['status'] = 'COMPLETED' if evaluation_code == 0 else 'EVALUATION_FAILED'
        except Exception:
            state['status'] = 'EVALUATION_FAILED'
            state['errors'].append(traceback.format_exc())
    else:
        state['status'] = 'RUN_FAILED'
    state['ended_at'] = utc()
    publish()
    print(f'FINAL status={state["status"]} monitor={monitor}', flush=True)
    return 0 if state['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    sys.exit(main())

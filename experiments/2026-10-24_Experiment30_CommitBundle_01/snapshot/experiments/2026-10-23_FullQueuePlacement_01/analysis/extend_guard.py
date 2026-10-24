"""User-approved external six-hour guard; never changes/restarts the workers.

The old supervisor has a loaded 7200-second constant. Suspend only that
supervisor, retain process handles, then resume it after ALL children exit so
its existing collector/evaluator can finish. Workers remain untouched until
the explicitly approved six-hour deadline measured from the original start.
"""
import ctypes
from ctypes import wintypes as W
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
SUPERVISOR = 33160
EXPECTED = {'fifo': (57920, 12536), 'tiresias': (38940, 44028),
            'joint': (42004, 9656)}
LIMIT = 21600
K = ctypes.WinDLL('kernel32', use_last_error=True)
N = ctypes.WinDLL('ntdll')
K.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
K.OpenProcess.restype = W.HANDLE
K.CloseHandle.argtypes = [W.HANDLE]
K.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
K.WaitForSingleObject.restype = W.DWORD
K.GetProcessTimes.argtypes = [W.HANDLE] + [ctypes.POINTER(W.FILETIME)] * 4
K.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
N.NtSuspendProcess.argtypes = [W.HANDLE]
N.NtSuspendProcess.restype = W.LONG
N.NtResumeProcess.argtypes = [W.HANDLE]
N.NtResumeProcess.restype = W.LONG


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def opened(pid, access=0x100000 | 0x1000):
    handle = K.OpenProcess(access, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def creation(handle):
    values = [W.FILETIME() for _ in range(4)]
    if not K.GetProcessTimes(handle, *[ctypes.byref(v) for v in values]):
        raise ctypes.WinError(ctypes.get_last_error())
    return values[0].dwLowDateTime | (values[0].dwHighDateTime << 32)


def done(handle):
    result = K.WaitForSingleObject(handle, 0)
    if result not in (0, 258):
        raise ctypes.WinError(ctypes.get_last_error())
    return result == 0


def parents():
    class Entry(ctypes.Structure):
        _fields_ = [('dwSize', W.DWORD), ('cntUsage', W.DWORD),
                    ('pid', W.DWORD), ('heap', ctypes.c_size_t),
                    ('module', W.DWORD), ('threads', W.DWORD), ('ppid', W.DWORD),
                    ('priority', W.LONG), ('flags', W.DWORD), ('name', W.WCHAR * 260)]
    K.CreateToolhelp32Snapshot.restype = W.HANDLE
    K.Process32FirstW.argtypes = [W.HANDLE, ctypes.POINTER(Entry)]
    K.Process32NextW.argtypes = [W.HANDLE, ctypes.POINTER(Entry)]
    snap = K.CreateToolhelp32Snapshot(2, 0)
    if snap == W.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    rows = {}
    try:
        entry = Entry(); entry.dwSize = ctypes.sizeof(entry)
        ok = K.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            rows[entry.pid] = entry.ppid
            ok = K.Process32NextW(snap, ctypes.byref(entry))
    finally:
        K.CloseHandle(snap)
    return rows


def main():
    claim = REPORTS / 'time_budget_extension.json'
    if claim.exists():
        raise RuntimeError('Extension already claimed; refusing duplicate guard')
    monitor = read(REPORTS / 'monitor.json')
    assert monitor['status'] == 'RUNNING'
    started = datetime.fromisoformat(monitor['started_at']).timestamp()
    graph = parents()
    supervisor = opened(SUPERVISOR, 0x100000 | 0x1000 | 0x0800)
    handles = {}
    suspended = False
    state = {'status': 'STARTING', 'approved_change': 'Keep current runs; maximum six hours',
             'original_started_at': monitor['started_at'], 'hard_timeout_seconds': LIMIT,
             'deadline_utc': datetime.fromtimestamp(started + LIMIT, timezone.utc).isoformat(),
             'supervisor_pid': SUPERVISOR, 'supervisor_creation': creation(supervisor),
             'original_timeout_seconds': 7200, 'workers_restarted': False,
             'worker_or_policy_changes': False, 'children': {}, 'created_at': utc()}
    try:
        assert not done(supervisor)
        for arm, (launcher, worker) in EXPECTED.items():
            assert monitor['arms'][arm]['pid'] == launcher
            assert read(ROOT / 'runs' / arm / 'progress.json')['pid'] == worker
            assert graph[launcher] == SUPERVISOR and graph[worker] == launcher
            for pid in (launcher, worker):
                handles[pid] = opened(pid)
                assert not done(handles[pid])
            state['children'][arm] = {str(pid): {'creation': creation(handles[pid])}
                                      for pid in (launcher, worker)}
        with claim.open('x', encoding='utf-8') as out:
            json.dump(state, out, ensure_ascii=False, indent=2)
        result = N.NtSuspendProcess(supervisor)
        if result != 0:
            raise RuntimeError(f'NtSuspendProcess failed: {result}')
        suspended = True
        state.update(status='GUARD_RUNNING', supervisor_suspended_at=utc())
        save(claim, state)
        print('GUARD_ACTIVE: old monitor suspended; all six child processes untouched', flush=True)
        while True:
            state['elapsed_seconds'] = round(time.time() - started, 3)
            state['updated_at'] = utc()
            all_done = True
            for arm, pids in EXPECTED.items():
                for pid in pids:
                    ended = done(handles[pid])
                    state['children'][arm][str(pid)]['exited'] = ended
                    all_done &= ended
                try:
                    state['children'][arm]['progress'] = read(ROOT / 'runs' / arm / 'progress.json')
                except (OSError, ValueError) as exc:
                    state['children'][arm]['progress_read_error'] = str(exc)
            save(REPORTS / 'monitor_extension.json', state)
            if all_done:
                state['status'] = 'CHILDREN_EXITED'
                break
            if time.time() >= started + LIMIT:
                print('SIX_HOUR_DEADLINE: terminating only verified owned live children', flush=True)
                state.update(status='SIX_HOUR_TIMEOUT', timeout_notified_at=utc())
                save(REPORTS / 'monitor_extension.json', state)
                for arm, (launcher, worker) in EXPECTED.items():
                    for pid in (worker, launcher):
                        if not done(handles[pid]):
                            kill_handle = opened(pid, 0x1000 | 0x0001)
                            try:
                                assert creation(kill_handle) == state['children'][arm][str(pid)]['creation']
                                if not K.TerminateProcess(kill_handle, 124):
                                    raise ctypes.WinError(ctypes.get_last_error())
                            finally:
                                K.CloseHandle(kill_handle)
                for handle in handles.values():
                    if K.WaitForSingleObject(handle, 10000) != 0:
                        raise RuntimeError('Owned child did not exit after deadline termination')
                break
            print('GUARD_CHECK elapsed=' + str(state['elapsed_seconds']) + ' ' +
                  ' '.join(f"{arm}={state['children'][arm].get('progress', {}).get('progress')}"
                           for arm in EXPECTED), flush=True)
            time.sleep(30)
    except BaseException as exc:
        state.update(status='GUARD_ERROR', error=repr(exc), updated_at=utc())
        if claim.exists():
            save(REPORTS / 'monitor_extension.json', state)
        raise
    finally:
        if suspended:
            result = N.NtResumeProcess(supervisor)
            state.update(supervisor_resumed_at=utc(), supervisor_resume_status=result)
            save(REPORTS / 'monitor_extension.json', state)
            save(claim, state)
            print(f'OLD_SUPERVISOR_RESUMED status={result}; original collector/evaluator retained', flush=True)
        for handle in handles.values():
            K.CloseHandle(handle)
        K.CloseHandle(supervisor)


if __name__ == '__main__':
    main()

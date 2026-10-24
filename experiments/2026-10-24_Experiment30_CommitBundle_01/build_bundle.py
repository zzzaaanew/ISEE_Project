"""One-shot archival copy, input inventory, integrity and syntax checks.

Never imports or executes experiment code. Never copies raw input Parquet,
event/decision archives, environments, caches, or unrelated dirty changes.
"""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROJECT = REPO.parent
E30 = REPO / 'experiments/2026-10-23_FullQueuePlacement_01'
E25 = REPO / 'experiments/2026-10-18_Experiment25_FullPeriod_StressPrep_01'
E27 = REPO / 'experiments/2026-10-22_Causal5m_BLOX_01'
E29 = REPO / 'experiments/2026-10-23_JointQueuePlacement_Preparation_01'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(name, value):
    with (ROOT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)


def git(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={REPO.as_posix()}',
                                   '-C', str(REPO), *args])


def main():
    if {p.name for p in ROOT.iterdir()} != {'README.md', 'build_bundle.py'}:
        raise FileExistsError('Fresh archive boundary required; no implicit overwrite/retry')
    assert read(E30 / 'reports/final_verification.json')['status'] == 'PASS'
    assert read(E30 / 'reports/monitor.json')['status'] == 'COMPLETED'
    assert git('branch', '--show-current').decode().strip() == 'research/xid-recovery-30m-validation'
    assert not git('diff', '--cached', '--name-only').strip(), 'Index must initially be empty'
    baseline = []
    for name in git('diff', '--name-only', '-z', 'HEAD').decode('utf-8').split('\0'):
        if name:
            path = REPO / name
            baseline.append(dict(path=name, sha256=sha(path) if path.is_file() else None))
    write('preexisting_tracked_changes.json', baseline)
    copied, excluded = [], []

    def copy(source, relative):
        destination = ROOT / relative
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha(source)
        assert digest == sha(destination)
        copied.append(dict(source=str(source), archived_path=relative.as_posix(), sha256=digest,
                           bytes=source.stat().st_size))

    allowed = {'.py', '.md', '.json', '.csv', '.log', '.parquet'}
    for source in sorted(E30.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts:
            continue
        relative = source.relative_to(REPO)
        if source.suffix == '.gz':
            excluded.append(dict(source=str(source), sha256=sha(source), bytes=source.stat().st_size,
                                 reason='large event/decision trace retained locally'))
        elif source.suffix in allowed:
            copy(source, Path('snapshot') / relative)
        else:
            raise ValueError(f'Unexpected E30 file: {source}')
    for source in sorted((E25 / 'src').rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts:
            assert source.suffix in {'.py', '.html', '.js', '.json'}
            copy(source, Path('snapshot') / source.relative_to(REPO))
    for source in (E27 / 'configs/risk1.json', E27 / 'inputs/threshold_manifest.json',
                   E29 / 'reports/input_audit.json'):
        copy(source, Path('snapshot') / source.relative_to(REPO))
    for name in ('프로젝트 목표 및 결정사항.md', '프로젝트 진행 상황.md',
                 '실험 계획/2026-10-23_BLOX_위험기반_공동최적화_장기계획.md',
                 '실험 기록/실험_연대기.md', '실험 기록/실험_연대기.html',
                 '실험 기록/render_experiment_log.cjs'):
        copy(PROJECT / name, Path('docs') / name)
    manifest = read(E30 / 'runs/joint/manifest.json')['config']
    inventory = {**manifest['inputs'], 'score_provenance': manifest['score_provenance']}
    for key, entry in inventory.items():
        source = Path(entry['path'])
        assert sha(source) == entry['sha256'], f'Input changed: {key}'
        entry['bytes'] = source.stat().st_size
        entry['included'] = key == 'threshold_manifest'
    write('input_inventory.json', inventory)
    versions = {name: importlib.metadata.version(name) for name in ('pandas', 'numpy', 'pyarrow')}
    write('archive_environment.json', dict(python=sys.version, executable=sys.executable,
                                          platform=platform.platform(), packages=versions,
                                          historical_hash_seed='not separately fixed/recorded'))
    python_count = 0
    credential_patterns = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                           r'ghp_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}',
                           r'AKIA[0-9A-Z]{16}', r'sk-proj-[A-Za-z0-9_-]{25,}',
                           r'https?://[^\s/@]+:[^\s/@]+@github\.com']
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            python_count += 1
        if path.suffix in {'.py', '.md', '.json', '.csv', '.log', '.html', '.js', '.cjs'}:
            content = path.read_text(encoding='utf-8-sig')
            for pattern in credential_patterns:
                if re.search(pattern, content):
                    raise RuntimeError(f'Credential-like content: {path.relative_to(ROOT)}')
        assert path.stat().st_size < 50 * 1024**2, 'Unexpected large upload'
        files.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                          sha256=sha(path)))
    for entry in baseline:
        path = REPO / entry['path']
        assert (sha(path) if path.is_file() else None) == entry['sha256'], 'Unrelated dirty file changed'
    write('ARCHIVE_MANIFEST.json', dict(status='PASS_ARCHIVE_INTEGRITY_SYNTAX_AND_RESULT_AUDIT',
        goal='G-009', experiment=30, archived_at=datetime.now(timezone.utc).isoformat(),
        source_head=git('rev-parse', 'HEAD').decode().strip(),
        target='origin/research/xid-recovery-30m-validation',
        file_count_excluding_this_manifest=len(files), total_bytes_excluding_this_manifest=sum(f['bytes'] for f in files),
        files=files, copies=copied, excluded_traces=excluded,
        python_AST_checks=python_count, credential_pattern_scan='PASS',
        preexisting_tracked_changes_preserved=True, original_runs_untouched=True,
        raw_input_parquet_included=False, independent_reproduction=False,
        review_note='ship checklist absent; workflow stopped and research archive/manual source review used',
        manifest_hash_scope='all files except this manifest; new generation only, no experiment run'))
    print(json.dumps(dict(status='PASS', files=len(files)+1,
                          bytes=sum(f['bytes'] for f in files), python_syntax_checks=python_count,
                          excluded_traces=len(excluded)), ensure_ascii=True))


if __name__ == '__main__':
    main()

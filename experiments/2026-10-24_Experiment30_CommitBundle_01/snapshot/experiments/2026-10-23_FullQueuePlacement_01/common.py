"""Frozen inputs, explicit output boundaries for the approved E30 full runs."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parent / '2026-10-18_Experiment25_FullPeriod_StressPrep_01'
E27 = ROOT.parent / '2026-10-22_Causal5m_BLOX_01'
E29 = ROOT.parent / '2026-10-23_JointQueuePlacement_Preparation_01'
sys.path.insert(0, str(PREP / 'src'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)


def manifest(arm):
    if arm not in ('fifo', 'tiresias', 'joint'):
        raise ValueError('Unknown arm')
    m = read_json(E27 / 'configs/risk1.json')
    m.update(research_experiment_id='Experiment30', experiment_label=f'Full Queue Placement {arm}',
             event_fill=False, full_arm=arm)
    m['optimization'].update(risk_enabled=True, top_k_quarantine=0)
    m['full_contract'] = dict(
        tiresias_service_gpu_seconds=3600, promote_ratio=2,
        periodic_checkpoint_pause_seconds=60, scheduling_checkpoint_seconds=60,
        scheduling_restart_seconds=300, queued_candidate_limit=256,
        risk_semantics='uncalibrated_score_surrogate_not_failure_probability',
        algorithm='feasible_joint_candidate_search_with_capacity_DP',
        global_optimality_claim=False, duration_oracle_in_planner=False,
        notes='scenario parameters fixed before runs, not tuned to results')
    return m

"""Scalable ON-only schedule search. All scores are uncalibrated surrogates.

Job records have no duration/label/future events. Capacity DP produces candidate
job sets; each complete GPU mapping is re-evaluated for actual lease changes and
checkpoint-release delay before choosing a feasible schedule. No global gap claim.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class JobView:
    id: str
    demand: int
    arrival: float
    first_start: float | None
    service: float
    queue: int
    previous: tuple
    checkpoint_age: float


def fifo_key(j):
    return j.arrival, j.id


def tiresias_key(j):
    return j.queue, j.first_start is None, j.first_start if j.first_start is not None else j.arrival, j.arrival, j.id


def pack(selected, scores):
    """Retain selected incumbents, then give high observable exposure lower risk."""
    free = set(scores)
    plan = {}
    for j in selected:
        if j.previous:
            if len(j.previous) != j.demand or not set(j.previous) <= free:
                return None
            plan[j.id] = j.previous
            free.difference_update(j.previous)
    queued = sorted((j for j in selected if not j.previous),
                    key=lambda j: (-(j.checkpoint_age + 660) * j.demand, j.arrival, j.id))
    ranked = sorted(free, key=lambda g: (scores[g], g))
    for j in queued:
        if len(ranked) < j.demand:
            return None
        gs, ranked = ranked[:j.demand], ranked[j.demand:]
        plan[j.id] = tuple(gs)
    return plan


def plan_value(jobs, plan, scores):
    retained = {g for j in jobs if j.id in plan and j.previous for g in j.previous}
    occupied_by_victims = {g for j in jobs if j.previous and j.id not in plan for g in j.previous}
    flow = risk = 0.0
    changes = 0
    for j in jobs:
        if j.id not in plan:
            flow += 300
            if j.previous:
                flow += 360
                changes += 1
        else:
            gs = plan[j.id]
            if not j.previous and occupied_by_victims.intersection(gs):
                flow += 60  # Commit waits for old GPU checkpoint to finish.
            risk += (j.checkpoint_age + 660) * sum(scores[g] for g in gs)
    return flow, risk, changes


def capacity_dp(jobs, capacity, scores):
    """0/1 capacity proposal: retained job benefit660, newly selected job300.

    Secondary proxy only generates a proposal; actual mapped objective rechecked.
    Exact for this proposal subproblem only, not the entire online problem.
    """
    values = np.zeros(capacity + 1, dtype=np.int64)
    risks = np.zeros(capacity + 1)
    take = np.zeros((len(jobs), capacity + 1), dtype=bool)
    mean = float(np.mean(list(scores.values()))) if scores else 0.0
    for i, j in enumerate(jobs):
        d = j.demand
        if d > capacity:
            continue
        value = 660 if j.previous else 300
        risk = (j.checkpoint_age + 660) * (sum(scores[g] for g in j.previous) if j.previous else d * mean)
        candidate = values[:-d] + value
        candidate_risk = risks[:-d] + risk
        choose = (candidate > values[d:]) | ((candidate == values[d:]) & (candidate_risk < risks[d:] - 1e-12))
        values[d:] = np.where(choose, candidate, values[d:])
        risks[d:] = np.where(choose, candidate_risk, risks[d:])
        take[i, d:] = choose
    selected = []
    remaining = capacity
    for i in range(len(jobs) - 1, -1, -1):
        if take[i, remaining]:
            selected.append(jobs[i])
            remaining -= jobs[i].demand
    return selected


def choose(arm, jobs, scores, *, candidate_limit=256):
    if len({j.id for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
        raise ValueError('Invalid raw risk')
    capacity = len(scores)
    running = [j for j in jobs if j.previous]
    queued = sorted((j for j in jobs if not j.previous), key=fifo_key)
    if arm == 'fifo':
        selected = list(running)
        room = capacity - sum(j.demand for j in running)
        for j in queued:
            if j.demand > room:
                break
            selected.append(j)
            room -= j.demand
        plan = pack(selected, scores)
        return plan, {'solver_status': 'BASELINE', 'candidate_jobs': len(jobs)}
    if arm == 'tiresias':
        selected = []
        room = capacity
        for j in sorted(jobs, key=tiresias_key):
            if j.demand > room:
                break
            selected.append(j)
            room -= j.demand
        if not selected and running:
            selected = running  # A blocked large head is not a reason to idle leases.
        return pack(selected, scores), {'solver_status': 'BASELINE_STYLE', 'candidate_jobs': len(jobs)}
    if arm != 'joint':
        raise ValueError('Unknown policy')
    # Keep oldest queue prefix, plus the oldest representative of each Gang demand.
    reps = {}
    for j in queued:
        reps.setdefault(j.demand, j)
    pool = {j.id: j for j in queued[:candidate_limit]}
    pool.update({j.id: j for j in reps.values()})
    candidates = running + list(pool.values())
    proposals = []
    # A lease-preserving feasible incumbent is always included as fallback.
    base, _ = choose('fifo', candidates, scores)
    proposals.append(base)
    for order in (candidates, list(reversed(candidates))):
        proposals.append(pack(capacity_dp(order, capacity, scores), scores))
    valid = [p for p in proposals if p is not None]
    if not valid:
        raise AssertionError('No feasible fallback')
    best = min(valid, key=lambda p: (*plan_value(candidates, p, scores), tuple(sorted(p))))
    value = plan_value(candidates, best, scores)
    return best, dict(solver_status='FEASIBLE_APPROXIMATE', gap=None, bound=None,
                      candidate_jobs=len(candidates), full_ready_jobs=len(jobs),
                      candidate_queue_truncated=max(0, len(queued) - len(pool)),
                      candidate_schedules=len(valid), surrogate_flow_seconds=value[0],
                      raw_score_exposure=value[1], planned_preemptions=value[2],
                      selected_differs_from_fifo=set(best) != set(base),
                      global_optimality_claim=False)

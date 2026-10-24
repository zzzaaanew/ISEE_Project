"""Actual full trace adapter over the frozen E25 incident/event environment.

All three arms share periodic checkpoint pauses, physical checkpoint reservations,
and recovery. Policies receive JobView rather than environment durations.
"""
from collections import defaultdict
import math
import time
from types import SimpleNamespace
from common import *
from risklab.replay_engine import ActualRiskLabSimulation
from policies import JobView, choose


class FullSimulation(ActualRiskLabSimulation):
    def __init__(self, m, topology, risk, failures, trace, threshold, *, emit=None, progress=None):
        arm = m['full_arm']
        super().__init__(m, topology, risk, failures, trace, threshold,
                         SimpleNamespace(name=arm, label=f'E30 {arm} explicit ON controller'),
                         emit=emit, progress=progress)
        self.arm = arm
        self.checkpointing = set()
        self.pending = {}
        self.phase_start = {}
        self.policy_resets = defaultdict(lambda: (0.0, 0.0))
        self.promotions = 0
        self.completed_count = self.failed_count = 0
        self.failure_times = {}
        self.periodic_costs = defaultdict(float)
        self.periodic_job_seconds = 0.0
        self.executed_gpu_seconds = self.lost_gpu_seconds = 0.0
        self.active_job_seconds = 0.0
        self.gpu_time = defaultdict(float)
        self.planning_seconds = []
        self.joint_selection_changes = 0
        self.candidate_truncated_decisions = 0
        self.pending_starts = 0

    def event(self, kind, **details):
        if kind == 'job_complete':
            self.completed_count += 1
        elif kind == 'job_retry_exhausted':
            self.failed_count += 1
            self.failure_times[details['job_id']] = self.time
        return super().event(kind, **details)

    def release(self, j):
        for uid in j.assigned:
            g = self.gpus[uid]
            g.job_id = None
            if g.state in ('running', 'loading', 'checkpointing'):
                g.state = 'available'
                g.until = math.inf
        j.assigned = []

    def _normalize_failure_state(self, j):
        if j is not None and (j.id in self.checkpointing or
                              (j.state == 'recovering' and j.assigned)):
            self.checkpointing.discard(j.id)
            self.recovering.discard(j.id)
            j.state = 'running'
            self.running.add(j.id)
            for uid in j.assigned:
                self.gpus[uid].state = 'running'
                self.gpus[uid].until = math.inf

    def _evidence_restart(self, j, failure, *, app_failure=True):
        self._normalize_failure_state(j)
        return super()._evidence_restart(j, failure, app_failure=app_failure)

    def interrupt(self, j, cause):
        if cause == 'failure':
            self._normalize_failure_state(j)
        if j.state not in ('running', 'loading'):
            return
        assigned, saved = list(j.assigned), j.saved_progress
        if cause == 'failure':
            self.lost_gpu_seconds += max(0.0, j.progress - j.saved_progress) * j.demand
        super().interrupt(j, cause)
        if cause != 'failure' and j.recovery_checkpoint_remaining > 0:
            # Checkpoint is not saved until its physical pause has completed.
            j.saved_progress = saved
            j.assigned = assigned
            for uid in assigned:
                g = self.gpus[uid]
                if g.job_id is not None:
                    raise AssertionError('Checkpoint reservation collision')
                g.state, g.job_id = 'checkpointing', j.id
                g.until = self.time + j.recovery_checkpoint_remaining

    def _advance(self, target):
        # Split at the first periodic checkpoint boundary instead of saving for free.
        target = min([target] + [self.time + max(1e-7, self.jobs[k].checkpoint_interval - self.jobs[k].since_checkpoint)
                                  for k in self.running])
        dt = target - self.time
        if dt < 0:
            raise AssertionError('Time reversed')
        ncp = len(self.checkpointing)
        reserved = sum(self.jobs[k].demand for k in self.recovering if self.jobs[k].assigned)
        self.busy_gpu_seconds += dt * (sum(self.jobs[k].demand for k in self.running | self.loading | self.checkpointing) + reserved)
        self.queue_job_seconds += dt * (len(self.active) - len(self.running) - len(self.recovering) - len(self.loading) - ncp)
        self.active_job_seconds += dt * len(self.active)
        self.periodic_job_seconds += dt * ncp
        for uid, g in self.gpus.items():
            self.gpu_time[(uid, g.state)] += dt
        for ident in self.checkpointing:
            self.periodic_costs[ident] += dt
        for ident in tuple(self.running):
            j = self.jobs[ident]
            risk_sum = sum(self.gpus[u].probability for u in j.assigned)
            self.ledger.add_exposure(dt * risk_sum, dt * (j.since_checkpoint + 660) * risk_sum)
            amount = min(dt, j.duration - j.progress)
            j.progress += amount
            j.since_checkpoint += amount
            j.attained_service += amount * j.demand
            self.executed_gpu_seconds += amount * j.demand
        for ident in tuple(self.loading):
            j = self.jobs[ident]
            amount = min(dt, j.active_load_remaining)
            j.active_load_remaining -= amount
            if amount:
                self.ledger.add(f'load:{self.time}:{target}:{ident}', 'failure_checkpoint_load_seconds', amount, job_id=ident)
        for ident in tuple(self.recovering):
            j = self.jobs[ident]
            left = dt
            for attribute, component in (
                ('recovery_checkpoint_remaining', f'{j.recovery_cause}_checkpoint_seconds'),
                ('recovery_retry_remaining', 'failure_retry_wait_seconds'),
                ('recovery_load_remaining', 'failure_checkpoint_load_seconds'),
                ('recovery_relaunch_remaining', 'failure_relaunch_seconds' if j.recovery_cause == 'failure' else f'{j.recovery_cause}_relaunch_seconds')):
                amount = min(left, getattr(j, attribute))
                setattr(j, attribute, getattr(j, attribute) - amount)
                left -= amount
                if amount:
                    self.ledger.add(f'{self.time}:{target}:{ident}', component, amount, job_id=ident)
        self.time = target

    def process_events(self):
        # Finish checkpoints BEFORE same-time failures/dispatch, with half-open phases.
        for ident in tuple(self.checkpointing):
            j = self.jobs[ident]
            if self.gpus[j.assigned[0]].until <= self.time + 1e-8:
                j.saved_progress, j.since_checkpoint = j.progress, 0.0
                j.state = 'running'
                self.checkpointing.remove(ident)
                self.running.add(ident)
                for uid in j.assigned:
                    self.gpus[uid].state, self.gpus[uid].until = 'running', math.inf
                self.event('periodic_checkpoint_complete', job_id=ident, gpu_ids=list(j.assigned))
        for ident in tuple(self.recovering):
            j = self.jobs[ident]
            if j.assigned and j.recovery_checkpoint_remaining <= 1e-8:
                j.saved_progress, j.since_checkpoint = j.progress, 0.0
                released = list(j.assigned)
                self.release(j)
                self.event('scheduler_checkpoint_complete', job_id=ident, gpu_ids=released)
        super().process_events()
        # A committed start is an execution of the previous plan, not a fresh solve.
        launches = []
        for ident, gs in list(self.pending.items()):
            j = self.jobs[ident]
            if j.state != 'queued':
                self.pending.pop(ident)
                continue
            if all(self.gpus[u].state == 'available' and self.gpus[u].job_id is None for u in gs):
                self._launch_job(j, list(gs), launches)
                self.pending.pop(ident)
                self.pending_starts += 1
            elif any(self.gpus[u].state not in ('available', 'checkpointing') for u in gs):
                self.pending.pop(ident)  # Environment changed; wait for next 5m solve.
        # Start new periodic pauses only after completions/failures at this instant.
        pause = self.manifest['full_contract']['periodic_checkpoint_pause_seconds']
        for ident in tuple(self.running):
            j = self.jobs[ident]
            if j.since_checkpoint >= j.checkpoint_interval - 1e-7 and j.progress < j.duration - 1e-8:
                j.state = 'checkpointing'
                self.running.remove(ident)
                self.checkpointing.add(ident)
                for uid in j.assigned:
                    self.gpus[uid].state, self.gpus[uid].until = 'checkpointing', self.time + pause
                self.event('periodic_checkpoint_start', job_id=ident, gpu_ids=list(j.assigned), seconds=pause)

    def _views(self):
        views = []
        threshold = self.manifest['full_contract']['tiresias_service_gpu_seconds']
        ratio = self.manifest['full_contract']['promote_ratio']
        for ident in self.active:
            j = self.jobs[ident]
            if j.state not in ('queued', 'running'):
                continue
            reset_service, reset_wait = self.policy_resets[ident]
            service = j.attained_service - reset_service
            wait = j.queue_wait + (self.time - j.queued_since if j.queued_since is not None else 0)
            q = int(service >= threshold)
            if q and service > 0 and wait - reset_wait >= ratio * service / j.demand:
                self.policy_resets[ident] = (j.attained_service, wait)
                service, q = 0.0, 0
                self.promotions += 1
            views.append(JobView(j.id, j.demand, j.arrival, j.started, service, q,
                                 tuple(j.assigned) if j.state == 'running' else (), j.since_checkpoint))
        return views

    def decision(self, refresh=True, fill_only=False):
        began = time.perf_counter()
        if refresh:
            self.refresh_risk()
        self.maintenance_step()
        views = self._views()
        scores = {u: g.probability for u, g in self.gpus.items() if g.state in ('available', 'running')}
        plan, evidence = choose(self.arm, views, scores,
                                candidate_limit=self.manifest['full_contract']['queued_candidate_limit'])
        if plan is None:
            raise AssertionError('Missing feasible plan')
        chosen = [u for gs in plan.values() for u in gs]
        if len(chosen) != len(set(chosen)) or not set(chosen) <= set(scores):
            raise AssertionError('Nonfeasible target mapping')
        # Do not change an incumbent's GPU set unless it leaves the running set.
        self.pending.clear()
        for ident in tuple(self.running):
            if ident not in plan:
                self.interrupt(self.jobs[ident], 'scheduling')
            elif set(plan[ident]) != set(self.jobs[ident].assigned):
                raise AssertionError('Uncharged migration')
        launches = []
        for ident, gs in plan.items():
            j = self.jobs[ident]
            if len(gs) != j.demand:
                raise AssertionError('Partial Gang')
            if j.state == 'queued':
                if all(self.gpus[u].state == 'available' and self.gpus[u].job_id is None for u in gs):
                    self._launch_job(j, list(gs), launches)
                else:
                    self.pending[ident] = gs
        self.decision_count += 1
        elapsed = time.perf_counter() - began
        self.planning_seconds.append(elapsed)
        self.max_placement_seconds = max(self.max_placement_seconds, elapsed)
        self.joint_selection_changes += int(evidence.get('selected_differs_from_fifo', False))
        self.candidate_truncated_decisions += int(evidence.get('candidate_queue_truncated', 0) > 0)
        self.emit_callback('decisions', dict(time=self.time, index=self.decision_count, arm=self.arm,
            risk_on=True, risk_gpus=len(self.gpus), ready_jobs=len(views),
            target_jobs=list(plan), target_gpu_count=len(chosen), pending_jobs=len(self.pending),
            planning_seconds=elapsed, **evidence))
        self.progress_callback(self.snapshot())

    def snapshot(self):
        counts = dict(future=len(self.jobs) - self._arrival_index,
                      queued=len(self.active) - len(self.running) - len(self.recovering) - len(self.loading) - len(self.checkpointing),
                      running=len(self.running), recovering=len(self.recovering), loading=len(self.loading),
                      checkpointing=len(self.checkpointing), completed=self.completed_count, failed=self.failed_count)
        return dict(status=self.status, progress=(self.time - self.start) / (self.end - self.start),
                    timestamp=self.time, start_time=self.start, end_time=self.end, arm=self.arm,
                    decisions=self.decision_count, jobs=counts)

    def run(self):
        summary = super().run()
        retained = sum(j.progress * j.demand for j in self.jobs.values())
        restricted = sum((j.completed if j.completed is not None else self.end) - j.arrival for j in self.jobs.values())
        failed_tail = sum(self.end - at for at in self.failure_times.values())
        tol = max(1e-4, self.executed_gpu_seconds * 1e-10)
        if abs(self.executed_gpu_seconds - retained - self.lost_gpu_seconds) > tol:
            raise AssertionError('Progress/rollback conservation failed')
        if abs(self.active_job_seconds + failed_tail - restricted) > max(1e-3, restricted * 1e-10):
            raise AssertionError('Restricted flow integral failed')
        elapsed = self.end - self.start
        if abs(sum(self.gpu_time.values()) - elapsed * len(self.gpus)) > 1e-3:
            raise AssertionError('GPU state time capacity failed')
        summary.update(arm=self.arm, start_time=self.start, end_time=self.end, gpu_count=len(self.gpus), risk_on=True,
            periodic_checkpoint_job_seconds=self.periodic_job_seconds,
            total_job_interruption_waste_including_periodic_seconds=summary['total_job_interruption_waste_seconds'] + self.periodic_job_seconds,
            restricted_total_flow_seconds=restricted, restricted_mean_flow_seconds=restricted / len(self.jobs),
            progress_conservation=True, flow_conservation=True, gpu_time_conservation=True,
            executed_gpu_seconds=self.executed_gpu_seconds, observed_rollback_gpu_seconds=self.lost_gpu_seconds,
            periodic_checkpoint_jobs=len(self.periodic_costs), checkpoint_lease_contract='physical_cp60_then_restart300; periodic_pause60',
            mean_planning_seconds=sum(self.planning_seconds) / len(self.planning_seconds),
            max_planning_seconds=max(self.planning_seconds), joint_selection_changes=self.joint_selection_changes,
            candidate_truncated_decisions=self.candidate_truncated_decisions, pending_committed_starts=self.pending_starts,
            policy_promotions=self.promotions, full_replay_run=True, global_optimality_claim=False,
            risk_semantics='raw_24h_surrogate_not_5m_probability', duration_oracle_in_planner=False)
        return summary

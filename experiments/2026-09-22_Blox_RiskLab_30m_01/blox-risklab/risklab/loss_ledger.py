"""Auditable component ledger: wall-clock job seconds and GPU seconds stay distinct."""
from __future__ import annotations
from collections import defaultdict


class LossLedger:
    def __init__(self):
        self.total=defaultdict(float); self.jobs=defaultdict(lambda:defaultdict(float)); self.gpus=defaultdict(lambda:defaultdict(float)); self.seen=set()
        self.counters=defaultdict(int)

    def add(self,event_id,component,seconds,*,job_id=None,gpu_id=None):
        if seconds<0: raise ValueError('Negative loss')
        key=(event_id,component,job_id,gpu_id)
        if key in self.seen: raise ValueError('Duplicate loss event')
        self.seen.add(key); self.total[component]+=float(seconds)
        if job_id is not None: self.jobs[job_id][component]+=float(seconds)
        if gpu_id is not None: self.gpus[gpu_id][component]+=float(seconds)

    def add_exposure(self,gpu_risk_seconds,vuln_risk_seconds=0.):
        self.total['risk_exposure_gpu_seconds']+=float(gpu_risk_seconds)
        self.total['vuln_weighted_risk_exposure']+=float(vuln_risk_seconds)

    def increment(self,counter_name,delta=1):
        self.counters[counter_name]+=int(delta)

    def summary(self):
        v=dict(self.total)
        for key in ['failure_checkpoint_loss_seconds','failure_recovery_delay_seconds','failure_gpu_unavailable_seconds','preventive_checkpoint_seconds','preventive_relaunch_seconds','maintenance_unavailable_seconds','scheduling_checkpoint_seconds','scheduling_relaunch_seconds','soft_xid_reset_seconds','risk_exposure_gpu_seconds','vuln_weighted_risk_exposure']:
            v.setdefault(key,0.)
        v['failure_lost_time_seconds']=sum(v[k] for k in ['failure_checkpoint_loss_seconds','failure_recovery_delay_seconds','failure_gpu_unavailable_seconds'])
        v['job_failure_lost_seconds']=sum(v[k] for k in ['failure_checkpoint_loss_seconds','failure_recovery_delay_seconds'])
        v['preventive_overhead_seconds']=sum(v[k] for k in ['preventive_checkpoint_seconds','preventive_relaunch_seconds','maintenance_unavailable_seconds'])
        v['scheduling_preemption_overhead_seconds']=sum(v[k] for k in ['scheduling_checkpoint_seconds','scheduling_relaunch_seconds'])
        v['total_job_interruption_waste_seconds']=v['job_failure_lost_seconds']+v['scheduling_preemption_overhead_seconds']+v['preventive_checkpoint_seconds']+v['preventive_relaunch_seconds']
        v['maintenance_gpu_hours']=v['maintenance_unavailable_seconds']/3600
        for cname in ['avoided_xid_episodes_on_idle','active_xid_job_interruptions','soft_xid_resets','hard_xid_maintenances','soft_xid_ignored_events']:
            v[cname]=int(self.counters.get(cname,0))
        v['units_note']='Checkpoint/relaunch: job wall seconds; GPU unavailability: GPU seconds. Component sums are accounting totals, not elapsed makespan.'
        return v

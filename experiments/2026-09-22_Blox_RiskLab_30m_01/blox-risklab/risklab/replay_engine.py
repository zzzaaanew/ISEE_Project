"""Continuous event replay with discrete 30-minute risk/scheduler decisions.

This explicit path never imports legacy monkey patches or synthetic presets.
"""
from __future__ import annotations
import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass,field,asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from .feeds import seconds
from .placement_policy import PlacementPolicy,compute_job_vulnerability,effective_gpu_risk
from .loss_ledger import LossLedger


@dataclass
class ReplayGPU:
    uid:str
    server_id:str
    local_id:int
    state:str='available'
    job_id:str|None=None
    probability:float=0.
    b1_probability:float=0.
    b2_probability:float=0.
    eligible:bool=True
    risk_alert:bool=False
    until:float=math.inf
    unavailable_since:float|None=None
    cause:str|None=None


@dataclass
class ReplayJob:
    id:str
    gang_id:str
    arrival:float
    duration:float
    demand:int
    checkpoint_interval:float
    state:str='future'
    progress:float=0.
    saved_progress:float=0.
    since_checkpoint:float=0.
    assigned:list=field(default_factory=list)
    started:float|None=None
    completed:float|None=None
    ready:float=math.inf
    queue_wait:float=0.
    attained_service:float=0.
    interruptions:int=0
    migrations:int=0
    launches:int=0
    recovery_cause:str|None=None
    recovery_checkpoint_remaining:float=0.
    recovery_relaunch_remaining:float=0.
    queued_since:float|None=None


class NativeScheduler:
    """Execute upstream scheduling policies without importing deployment services."""
    def __init__(self,source,name='Las',quantum=3600):
        if name not in ('Fifo','Las','Tiresias2Q'): raise ValueError('Unsupported scheduler')
        self.name=name; self.quantum=quantum
        root=Path(source).resolve(); init=root/'schedulers/__init__.py'
        if not init.exists(): raise FileNotFoundError(f'Blox schedulers missing: {root}')
        package='_risklab_native_schedulers_'+__import__('hashlib').sha256(str(root).encode()).hexdigest()[:12]
        if package not in sys.modules:
            spec=importlib.util.spec_from_file_location(package,init,submodule_search_locations=[str(init.parent)])
            module=importlib.util.module_from_spec(spec);sys.modules[package]=module;spec.loader.exec_module(module)
        module=sys.modules[package]
        self.policy=getattr(module,name)(argparse.Namespace()) if name!='Tiresias2Q' else None
        self.source=str(root)
        self.label=f'Blox schedulers.{name}' if self.policy else 'Blox-compatible Tiresias2Q adapter (existing RiskLab semantics)'

    def order(self,jobs,gpus):
        active={j.id:{'job_priority':999,'submit_time':j.arrival,'num_GPUs':j.demand,'is_running':j.state=='running','tracked_metrics':{'attained_service':j.attained_service,'per_iter_time':1.}} for j in jobs.values() if j.state in ('queued','running')}
        if self.policy:
            state=SimpleNamespace(active_jobs=active)
            cluster=SimpleNamespace(server_map={},gpu_df=pd.DataFrame())
            return [k for k,_ in self.policy.schedule(state,cluster)['job_order']]
        return [k for k,v in sorted(active.items(),key=lambda kv:(0 if kv[1]['tracked_metrics']['attained_service']<self.quantum else 1,kv[1]['submit_time'] if kv[1]['tracked_metrics']['attained_service']<self.quantum else kv[1]['tracked_metrics']['attained_service'],kv[0]))]


class ActualRiskLabSimulation:
    def __init__(self,manifest,topology,risk,failures,trace,threshold,scheduler,*,emit=None,stop=None,progress=None):
        self.manifest=manifest; self.risk=risk; self.failures=failures; self.scheduler=scheduler
        self.start=seconds(manifest['start_time']);self.end=seconds(manifest['end_time']);self.time=self.start
        self.interval=int(manifest['decision_interval_seconds']); self.threshold=float(threshold['threshold'])
        if self.end<=self.start or self.interval<=0: raise ValueError('Invalid replay interval')
        self.opt_config=dict(manifest.get('optimization',{}))
        self.policy=PlacementPolicy(manifest['placement_policy'],manifest['random_seed'],opt_config=self.opt_config)
        self.config=manifest['recovery']; self.maintenance=manifest['maintenance']
        if any(float(self.config[k])<0 for k in ('checkpoint_interval_seconds','checkpoint_cost_seconds','relaunch_seconds')) or self.config['checkpoint_interval_seconds']<=0: raise ValueError('Invalid recovery config')
        if self.maintenance['capacity']<1 or self.maintenance['duration_seconds']<=0 or self.maintenance['cooldown_seconds']<=0: raise ValueError('Invalid maintenance config')
        self.xid_severity_mode=str(manifest.get('xid_severity_mode','uniform_hardware'))
        self.xid43_interrupt_prob=float(manifest.get('xid43_interrupt_prob',1.0))
        self.soft_reset_seconds=float(manifest.get('soft_reset_seconds',60.0))
        self._xid_rng=np.random.default_rng(int(manifest.get('random_seed',17))+99991)
        self.gpus={r['gpu_uid']:ReplayGPU(r['gpu_uid'],str(r['server_id']),int(r['local_gpu_id'])) for r in topology.frame.to_dict('records')}
        self.jobs={r['job_id']:ReplayJob(r['job_id'],r['gang_id'],r['arrival'],r['duration_seconds'],r['gpu_demand'],r['checkpoint_interval_seconds'] or self.config['checkpoint_interval_seconds']) for r in trace.records}
        if any(j.arrival<self.start or j.arrival>=self.end for j in self.jobs.values()): raise ValueError('Jobs outside replay window')
        if failures.next_time()<self.start: raise ValueError('Failure tape must be clipped to replay interval')
        self.ledger=LossLedger();self.emit_callback=emit or (lambda stream,row:None);self.stop=stop or (lambda:False);self.progress_callback=progress or (lambda state:None)
        self.event_seq=0;self.decision_count=0;self.no_fit_count=0;self.excluded_allocations=0;self.max_placement_seconds=0.
        self._arrival_queue=sorted(self.jobs.values(),key=lambda j:(j.arrival,j.id));self._arrival_index=0
        self.active=set();self.running=set();self.recovering=set();self.status='running'

    def event(self,kind,**details):
        self.event_seq+=1
        row={'event_id':self.event_seq,'time':self.time-self.start,'timestamp':pd.Timestamp(self.time,unit='s',tz='UTC').isoformat(),'type':kind,**details}
        self.emit_callback('events',row);return str(self.event_seq)

    def _advance(self,target):
        dt=target-self.time
        if dt<0:raise AssertionError('Time reversed')
        for ident in sorted(self.running | self.recovering):
            j=self.jobs[ident]
            if j.state=='running':
                if dt>0 and j.assigned:
                    risk_sum=sum(self.gpus[u].probability for u in j.assigned)
                    vuln=compute_job_vulnerability(j,self.config)
                    self.ledger.add_exposure(dt*risk_sum,dt*vuln*risk_sum)
                j.progress=min(j.duration,j.progress+dt);j.since_checkpoint+=dt;j.attained_service+=dt*j.demand
                if j.since_checkpoint>=j.checkpoint_interval:
                    residual=j.since_checkpoint%j.checkpoint_interval
                    j.saved_progress=j.progress-residual;j.since_checkpoint=residual
            elif j.state=='recovering':
                left=dt
                cp=min(left,j.recovery_checkpoint_remaining);j.recovery_checkpoint_remaining-=cp;left-=cp
                rel=min(left,j.recovery_relaunch_remaining);j.recovery_relaunch_remaining-=rel
                if cp:self.ledger.add(f'{self.event_seq}:{self.time}:{target}:{ident}',f'{j.recovery_cause}_checkpoint_seconds',cp,job_id=ident)
                if rel:
                    component='failure_recovery_delay_seconds' if j.recovery_cause=='failure' else f'{j.recovery_cause}_relaunch_seconds'
                    self.ledger.add(f'{self.event_seq}:{self.time}:{target}:{ident}',component,rel,job_id=ident)
        self.time=target

    def release(self,j):
        for uid in j.assigned:
            g=self.gpus[uid];g.job_id=None
            if g.state=='running':g.state='available'
        j.assigned=[]

    def interrupt(self,j,cause):
        if j.state!='running':return
        assigned=list(j.assigned)
        eid=self.event('gang_failure' if cause=='failure' else ('preventive_migration' if cause=='preventive' else 'scheduler_preemption'),job_id=j.id,gang_id=j.gang_id,gpu_ids=assigned)
        if cause=='failure':
            loss=max(0.,j.progress-j.saved_progress);j.progress=j.saved_progress;j.since_checkpoint=0
            self.ledger.add(eid,'failure_checkpoint_loss_seconds',loss,job_id=j.id);cp=0.
        else:
            j.saved_progress=j.progress;j.since_checkpoint=0.;cp=float(self.config['checkpoint_cost_seconds'])
            self.event('gang_checkpoint',job_id=j.id,gpu_ids=assigned,cause=cause)
        self.release(j);self.running.discard(j.id);self.recovering.add(j.id);j.state='recovering';j.interruptions+=1;j.migrations+=int(cause=='preventive')
        j.recovery_cause=cause;j.recovery_checkpoint_remaining=cp;j.recovery_relaunch_remaining=float(self.config['relaunch_seconds']);j.ready=self.time+cp+j.recovery_relaunch_remaining

    def unavailable(self,g,state,cause):
        if g.unavailable_since is None:g.unavailable_since=self.time;g.cause=cause
        elif cause=='failure' and g.cause!='failure':
            comp='soft_xid_reset_seconds' if g.cause=='soft_xid' else 'maintenance_unavailable_seconds'
            self.ledger.add(f'switch:{g.uid}:{self.time}',comp,self.time-g.unavailable_since,gpu_id=g.uid)
            g.unavailable_since=self.time;g.cause='failure'
        g.state=state;g.until=math.inf

    def maintenance_step(self):
        for g in self.gpus.values():
            if g.state=='maintenance' and g.until<=self.time:
                g.state='cooldown';g.until=self.time+self.maintenance['cooldown_seconds'];self.event('maintenance_complete',gpu_id=g.uid);self.event('cooldown_start',gpu_id=g.uid)
            elif g.state=='cooldown' and g.until<=self.time:
                if g.cause=='soft_xid':
                    component='soft_xid_reset_seconds'
                elif g.cause=='failure':
                    component='failure_gpu_unavailable_seconds'
                else:
                    component='maintenance_unavailable_seconds'
                self.ledger.add(f'gpu:{g.uid}:{g.unavailable_since}:{self.time}',component,self.time-g.unavailable_since,gpu_id=g.uid)
                g.unavailable_since=None;g.cause=None;g.state='available';g.until=math.inf;self.event('cooldown_complete',gpu_id=g.uid)
        slots=self.maintenance['capacity']-sum(g.state=='maintenance' for g in self.gpus.values())
        waiting=sorted((g for g in self.gpus.values() if g.state in ('failed','drained')),key=lambda g:(0 if g.state=='failed' else 1,g.unavailable_since,g.uid))
        for g in waiting[:max(0,slots)]:
            g.state='maintenance';g.until=self.time+self.maintenance['duration_seconds'];self.event('maintenance_start',gpu_id=g.uid,cause=g.cause)

    def process_events(self):
        for ident in sorted(self.running | self.recovering):
            j=self.jobs[ident]
            if j.state=='running' and j.progress>=j.duration-1e-8:
                completed_gpus=list(j.assigned);self.release(j);j.state='completed';j.completed=self.time;self.active.remove(ident);self.running.discard(ident);self.event('job_complete',job_id=ident,gpu_ids=completed_gpus,server_ids=sorted({self.gpus[u].server_id for u in completed_gpus}))
            elif j.state=='recovering' and j.ready<=self.time:
                j.state='queued';j.queued_since=self.time;j.ready=math.inf;self.recovering.discard(ident);self.event('job_recovered',job_id=ident)
        self.maintenance_step()
        while self._arrival_index<len(self._arrival_queue) and self._arrival_queue[self._arrival_index].arrival<=self.time:
            j=self._arrival_queue[self._arrival_index];self._arrival_index+=1;j.state='queued';j.queued_since=self.time;self.active.add(j.id);self.event('job_admitted',job_id=j.id,demand=j.demand)
        for failure in self.failures.events_at(self.time):
            g=self.gpus[failure['gpu_uid']]
            codes=failure.get('parsed_xid_codes') or [43]
            self.event('xid_episode',gpu_id=g.uid,episode_id=str(failure['episode_id']))
            if g.job_id is None:
                self.ledger.increment('avoided_xid_episodes_on_idle')
            is_soft_xid43=(self.xid_severity_mode=='differentiated_43_31' and set(codes)=={43})
            if is_soft_xid43:
                # Solution B: XID 43 is an application/driver-level error.
                # GPU hardware remains intact (brief soft reset, no 900s hardware replacement).
                if g.job_id is not None:
                    should_interrupt=(self.xid43_interrupt_prob>=1.0) or (float(self._xid_rng.random())<self.xid43_interrupt_prob)
                    if should_interrupt:
                        self.ledger.increment('active_xid_job_interruptions')
                        self.interrupt(self.jobs[g.job_id],'failure')
                        if self.soft_reset_seconds>0 and g.state=='available':
                            self.ledger.increment('soft_xid_resets')
                            g.unavailable_since=self.time;g.cause='soft_xid';g.state='cooldown';g.until=self.time+self.soft_reset_seconds
                            self.event('cooldown_start',gpu_id=g.uid)
                    else:
                        self.ledger.increment('soft_xid_ignored_events')
                else:
                    if self.soft_reset_seconds>0 and g.state=='available':
                        self.ledger.increment('soft_xid_resets')
                        g.unavailable_since=self.time;g.cause='soft_xid';g.state='cooldown';g.until=self.time+self.soft_reset_seconds
                        self.event('cooldown_start',gpu_id=g.uid)
            else:
                self.ledger.increment('hard_xid_maintenances')
                if g.job_id is not None:
                    self.ledger.increment('active_xid_job_interruptions')
                    self.interrupt(self.jobs[g.job_id],'failure')
                self.unavailable(g,'failed','failure');self.event('gpu_failed',gpu_id=g.uid)
        self.maintenance_step()

    def _select_preemption_victims(self,job,lower,deficit):
        """Cost-Benefit Preemption Guard & Best-Fit Victim Selector."""
        # 1. Tail Completion Protection: do not preempt a running job finishing within 180s
        min_rem=float(self.opt_config.get('preemption_min_remaining_seconds',180.))
        candidates=[
            v for v in lower
            if (v.duration-v.progress)>min_rem
            and (v.attained_service/max(1,v.demand)-job.attained_service/max(1,job.demand))>=float(self.opt_config.get('preemption_service_margin',300.))
        ]
        avail_dets=[sum(self.gpus[u].eligible for u in v.assigned) for v in candidates]
        if sum(avail_dets)<deficit:
            return []
        # 2. Avoid killing a massive Gang job (e.g. 32 or 64 GPUs) for a tiny job if smaller lower-priority victims suffice
        small_first=[v for v in candidates if v.demand<=max(job.demand*2,job.demand+4)]
        small_pot=sum(sum(self.gpus[u].eligible for u in v.assigned) for v in small_first)
        pool=small_first if small_pot>=deficit else candidates
        victims=[]; gained=0
        for v in pool:
            gained+=sum(self.gpus[u].eligible for u in v.assigned)
            victims.append(v)
            if gained>=deficit: break
        return victims if gained>=deficit else []

    def _launch_job(self,j,chosen,launches):
        if len(chosen)!=j.demand or len(set(chosen))!=len(chosen):raise AssertionError('Non-atomic Gang allocation')
        j.assigned=chosen;j.state='running';j.launches+=1;self.running.add(j.id)
        j.queue_wait+=self.time-j.queued_since;j.queued_since=None
        if j.started is None:j.started=self.time
        for uid in chosen:
            g=self.gpus[uid]
            if g.state!='available' or g.job_id is not None:raise AssertionError('Double allocation')
            if self.policy.hard_mask and not g.eligible:self.excluded_allocations+=1;raise AssertionError('Excluded allocation')
            g.state='running';g.job_id=j.id
        servers=sorted({self.gpus[u].server_id for u in chosen})
        launches.append({'job_id':j.id,'gpu_ids':chosen,'server_ids':servers,'server_count':len(servers)})
        self.event('job_launch' if j.launches==1 else 'job_relaunch',job_id=j.id,gpu_ids=chosen,server_ids=servers)

    def decision(self):
        import time
        begin=time.perf_counter()
        if hasattr(self.risk,'branch_snapshot'):
            values,b1_vals,b2_vals=self.risk.branch_snapshot(self.time)
        else:
            values=self.risk.snapshot(self.time); b1_vals=values; b2_vals=values
        for uid,p in values.items():
            g=self.gpus[uid];previous=g.eligible
            g.probability=p;g.b1_probability=b1_vals.get(uid,p);g.b2_probability=b2_vals.get(uid,p)
            g.risk_alert=p>=self.threshold
            g.eligible=not self.policy.hard_mask or not g.risk_alert
            if previous!=g.eligible:self.event('risk_mask_change',gpu_id=uid,eligible=g.eligible,probability=p)

        if self.policy.hard_mask:
            if self.policy.name=='risk_mask_packed_nonsticky':
                for ident in sorted(self.active):
                    j=self.jobs[ident]
                    if j.state=='running' and any(not self.gpus[u].eligible for u in j.assigned):self.interrupt(j,'preventive')
            for g in self.gpus.values():
                if g.state=='available' and not g.eligible:
                    self.unavailable(g,'drained','preventive');self.event('gpu_drained',gpu_id=g.uid)
        self.maintenance_step()
        order=self.scheduler.order({k:self.jobs[k] for k in sorted(self.active)},self.gpus)
        launches=[];no_fit=[];allocation_cache={};preemption_blocked=set()
        ranks={ident:i for i,ident in enumerate(order)}

        # Solution A: Slack-aware Budgeted Top-K Quarantine
        top_k_q=self.policy.top_k_quarantine
        if top_k_q>0 and not self.policy.hard_mask:
            total_avail=sum(g.state=='available' and g.job_id is None for g in self.gpus.values())
            total_queued_demand=sum(self.jobs[ident].demand for ident in order if self.jobs[ident].state=='queued')
            slack=max(0,total_avail-total_queued_demand)
            actual_q=min(top_k_q,slack)
            if actual_q>0:
                avail_candidates=[g for g in self.gpus.values() if g.state=='available' and g.job_id is None]
                avail_candidates.sort(key=lambda g:(-max(g.probability,g.b1_probability),g.uid))
                quarantined_uids={g.uid for g in avail_candidates[:actual_q]}
                for g in self.gpus.values():
                    g.eligible=(g.uid not in quarantined_uids)
            else:
                for g in self.gpus.values():
                    g.eligible=True

        if self.policy.is_batch_policy:
            # Step 1: Priority-Respecting Admission & Preemption Resolution
            admitted_jobs=[]
            reserved_gpus=0
            avail_eligible=sum(g.state=='available' and g.job_id is None and g.eligible for g in self.gpus.values())
            can_preempt=(self.scheduler.name!='Fifo' and self.manifest.get('preemption',True))

            for ident in order:
                j=self.jobs[ident]
                if j.state!='queued': continue
                free_now=avail_eligible-reserved_gpus
                if free_now>=j.demand:
                    admitted_jobs.append(j)
                    reserved_gpus+=j.demand
                    continue
                if can_preempt and j.demand not in preemption_blocked:
                    deficit=j.demand-free_now
                    lower=sorted((self.jobs[k] for k in self.running if ranks.get(k,-1)>ranks[ident]),key=lambda v:ranks[v.id],reverse=True)
                    if self.policy.use_preemption_guard:
                        victims=self._select_preemption_victims(j,lower,deficit)
                    else:
                        victims=[]; gained=0
                        for v in lower:
                            gained+=sum(self.gpus[u].eligible for u in v.assigned)
                            victims.append(v)
                            if gained>=deficit: break
                        if gained<deficit: victims=[]
                    if victims:
                        for victim in victims:
                            self.interrupt(victim,'scheduling')
                        preemption_blocked.clear()
                        avail_eligible=sum(g.state=='available' and g.job_id is None and g.eligible for g in self.gpus.values())
                        if (avail_eligible-reserved_gpus)>=j.demand:
                            admitted_jobs.append(j)
                            reserved_gpus+=j.demand
                            continue
                    else:
                        preemption_blocked.add(j.demand)
                no_fit.append(ident);self.no_fit_count+=1

            # Step 2: Joint Batch Co-Optimization (Vulnerability-Weighted Exact DP + Metaheuristic 2-Opt/SA)
            if admitted_jobs:
                batch_plan=self.policy.choose_batch(admitted_jobs,self.gpus,self.config)
                for j in admitted_jobs:
                    chosen=batch_plan.get(j.id) or self.policy.choose(j.demand,self.gpus,job=j)
                    if chosen:
                        self._launch_job(j,chosen,launches)
                    else:
                        no_fit.append(j.id);self.no_fit_count+=1
        else:
            # Original single-job loop (100% bit-exact parity for legacy policies)
            for ident in order:
                j=self.jobs[ident]
                if j.state!='queued':continue
                if j.demand not in allocation_cache:allocation_cache[j.demand]=self.policy.choose(j.demand,self.gpus)
                chosen=allocation_cache[j.demand]
                if not chosen and j.demand not in preemption_blocked and self.scheduler.name!='Fifo' and self.manifest.get('preemption',True):
                    lower=sorted((self.jobs[k] for k in self.running if ranks.get(k,-1)>ranks[ident]),key=lambda j:ranks[j.id],reverse=True)
                    potential=sum(g.state=='available' and (g.eligible or not self.policy.hard_mask) for g in self.gpus.values())
                    victims=[]
                    for victim in lower:
                        potential+=sum(self.gpus[u].eligible or not self.policy.hard_mask for u in victim.assigned);victims.append(victim)
                        if potential>=j.demand:break
                    if potential>=j.demand:
                        for victim in victims:self.interrupt(victim,'scheduling')
                        allocation_cache.clear();preemption_blocked.clear()
                        chosen=self.policy.choose(j.demand,self.gpus)
                    else:preemption_blocked.add(j.demand)
                if not chosen:
                    no_fit.append(ident);self.no_fit_count+=1;continue
                allocation_cache.clear();preemption_blocked.clear()
                self._launch_job(j,chosen,launches)

        self.decision_count+=1;elapsed=time.perf_counter()-begin;self.max_placement_seconds=max(self.max_placement_seconds,elapsed)
        row={'time':self.time-self.start,'scheduler':self.scheduler.name,'placement_policy':self.policy.name,'eligible_gpu_count':sum(g.eligible for g in self.gpus.values()),'excluded_gpu_count':sum(not g.eligible for g in self.gpus.values()),'queued_jobs':sum(self.jobs[k].state=='queued' for k in self.active),'launches':launches,'no_fit_jobs':no_fit,'scheduler_order':order}
        self.emit_callback('decisions',row)
        self.event('risk_snapshot',eligible_gpu_count=row['eligible_gpu_count'],excluded_gpu_count=row['excluded_gpu_count'])
        if no_fit:self.event('job_no_fit',count=len(no_fit),job_ids=no_fit[:100],full_list='decisions.jsonl')
        snapshot=self.snapshot();self.emit_callback('snapshots',snapshot);self.progress_callback(snapshot)

    def snapshot(self):
        states={name:sum(j.state==name for j in self.jobs.values()) for name in ('future','queued','running','recovering','completed')}
        return {'status':self.status,'timestamp':pd.Timestamp(self.time,unit='s',tz='UTC').isoformat(),'start_timestamp':pd.Timestamp(self.start,unit='s',tz='UTC').isoformat(),'placement_policy':self.policy.name,'time':self.time-self.start,'progress':(self.time-self.start)/(self.end-self.start),'jobs':states,'gpus':[asdict(g) | {'until':None if not math.isfinite(g.until) else g.until} for g in self.gpus.values()],'decisions':self.decision_count}

    def run(self):
        next_decision=self.start
        while self.time<self.end:
            if self.stop():self.status='stopped';break
            self.process_events()
            if self.time>=next_decision-1e-8:
                self.decision();next_decision+=self.interval
            candidates=[next_decision,self.end,self.failures.next_time()]
            if self._arrival_index<len(self._arrival_queue):candidates.append(self._arrival_queue[self._arrival_index].arrival)
            for ident in sorted(self.running | self.recovering):
                j=self.jobs[ident]
                if j.state=='running':candidates.append(self.time+j.duration-j.progress)
                elif j.state=='recovering':candidates.append(j.ready)
            candidates.extend(g.until for g in self.gpus.values())
            future=[t for t in candidates if t>self.time+1e-8]
            if not future:raise AssertionError('Event loop made no progress')
            self._advance(min(future))
        if self.status!='stopped':
            # Completions exactly at the horizon count; arrivals/failures at the
            # exclusive bound do not create new work.
            for ident in sorted(self.running | self.recovering):
                j=self.jobs[ident]
                if j.state=='running' and j.progress>=j.duration-1e-8:
                    completed_gpus=list(j.assigned);self.release(j);j.state='completed';j.completed=self.time;self.active.remove(ident);self.running.discard(ident);self.event('job_complete',job_id=ident,gpu_ids=completed_gpus,server_ids=sorted({self.gpus[u].server_id for u in completed_gpus}))
            self.status='completed'
        for g in self.gpus.values():
            if g.unavailable_since is not None:
                if g.cause=='soft_xid':
                    component='soft_xid_reset_seconds'
                elif g.cause=='failure':
                    component='failure_gpu_unavailable_seconds'
                else:
                    component='maintenance_unavailable_seconds'
                self.ledger.add(f'end:{g.uid}',component,self.time-g.unavailable_since,gpu_id=g.uid)
        for j in self.jobs.values():
            if j.queued_since is not None:j.queue_wait+=self.time-j.queued_since;j.queued_since=None
        completed=[j for j in self.jobs.values() if j.completed is not None]
        waits=[j.queue_wait for j in self.jobs.values() if j.arrival<=self.time]
        summary={**self.ledger.summary(),'status':self.status,'scheduler':self.scheduler.name,'scheduler_implementation':self.scheduler.label,'placement_policy':self.policy.name,'xid_severity_mode':self.xid_severity_mode,'xid43_interrupt_prob':self.xid43_interrupt_prob,'decision_interval_seconds':self.interval,'cohort_size':len(self.gpus),'decision_count':self.decision_count,'completed_jobs':len(completed),'total_jobs':len(self.jobs),'unfinished_jobs':len(self.jobs)-len(completed),'starved_jobs':sum(j.started is None and j.arrival<self.time for j in self.jobs.values()),'avg_queue_wait_seconds':float(np.mean(waits)) if waits else 0.,'p95_queue_wait_seconds':float(np.quantile(waits,.95)) if waits else 0.,'max_queue_wait_seconds':max(waits,default=0.),'avg_jct_seconds':float(np.mean([j.completed-j.arrival for j in completed])) if completed else None,'makespan_seconds':max((j.completed-self.start for j in completed),default=0.),'makespan_censored':len(completed)<len(self.jobs),'migrations':sum(j.migrations for j in self.jobs.values()),'no_fit_decisions':self.no_fit_count,'excluded_allocations':self.excluded_allocations,'event_count':self.event_seq,'replay_seconds':self.time-self.start}
        return summary

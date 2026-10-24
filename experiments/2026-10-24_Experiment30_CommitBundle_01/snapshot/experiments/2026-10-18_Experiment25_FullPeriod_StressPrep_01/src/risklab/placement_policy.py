"""Atomic fixed-size Gang placement with exact minimum-server packing and vulnerability-weighted metaheuristic batch optimization."""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import math
import random

POLICIES=(
    'risk_prefer_packed',
    'risk_blind_packed',
    'risk_mask_packed',
    'risk_mask_random',
    'risk_mask_packed_nonsticky',
    'risk_mask_probability_tiebreak',
    'vuln_weighted_packed',
    'metaheuristic_packed',
    'topk_quarantine_packed',
    'full_co_optimized',
)


def compute_job_vulnerability(job,config=None):
    """Duration-aware heuristic weight. Not a calibrated expected-loss estimate.

    Uses observed remaining trace duration (oracle runtime); retain this policy
    as a controlled comparator until a past-only duration estimate is supplied.
    """
    if job is None: return 1.0
    cfg=config or {}
    ckpt_interval=float(getattr(job,'checkpoint_interval',300.) or 300.)
    relaunch=float(cfg.get('relaunch_seconds',300.))
    eta=float(cfg.get('duration_scale_eta',1.0))
    rem=max(1.0,float(job.duration)-float(job.progress))
    demand=int(job.demand)
    expected_ckpt_loss=0.5*min(rem,ckpt_interval)
    # Gang failure amplifies interruption penalty across all demand GPUs
    return float(demand)*(expected_ckpt_loss+relaunch)*(1.0+eta*(rem/3600.0))


def effective_gpu_risk(g,branch_mode='fused_plus_b1',b1_weight=0.35):
    """Allocation score; fused_plus_b1 may exceed one and is not a probability."""
    p=float(getattr(g,'probability',0.))
    if branch_mode=='b1_only':
        return float(getattr(g,'b1_probability',p))
    if branch_mode=='b2_only':
        return float(getattr(g,'b2_probability',p))
    if branch_mode=='fused_plus_b1':
        p1=float(getattr(g,'b1_probability',p))
        return p+b1_weight*p1
    return p


class PlacementPolicy:
    def __init__(self,name='risk_mask_packed',seed=17,opt_config=None):
        if name not in POLICIES: raise ValueError(f'Unknown placement policy: {name}')
        self.name=name; self.rng=random.Random(seed)
        self.opt_config=dict(opt_config or {})

    def allocation_risk(self,g):
        """Only policy-visible risk channel; reporting retains observed scores."""
        if not self.opt_config.get('risk_enabled',True): return 0.0
        return effective_gpu_risk(g,self.opt_config.get('branch_mode','fused_plus_b1'),
                                  self.opt_config.get('b1_acute_weight',0.35))

    @property
    def hard_mask(self):
        return self.name in ('risk_mask_packed','risk_mask_random','risk_mask_packed_nonsticky','risk_mask_probability_tiebreak')

    @property
    def is_batch_policy(self):
        return self.name in ('vuln_weighted_packed','metaheuristic_packed','topk_quarantine_packed','full_co_optimized')

    @property
    def use_preemption_guard(self):
        return self.name=='full_co_optimized' or bool(self.opt_config.get('preemption_guard',False))

    @property
    def top_k_quarantine(self):
        if self.name in ('topk_quarantine_packed','full_co_optimized'):
            return int(self.opt_config.get('top_k_quarantine',2))
        return int(self.opt_config.get('top_k_quarantine',0))

    def _exact_server_dp(self,demand,candidates,score_fn):
        """Exact DP: fewest servers, then sum of score_fn(g), then lexical GPU IDs."""
        if len(candidates)<demand: return []
        groups=defaultdict(list)
        for g in candidates: groups[g.server_id].append(g)
        states={(0,0):(0.,())}
        for server in sorted(groups):
            items=sorted(groups[server],key=lambda g:(score_fn(g),g.uid))
            prefix=[(0.,())]
            for g in items: prefix.append((prefix[-1][0]+score_fn(g),prefix[-1][1]+(g.uid,)))
            updated=dict(states)
            for (used,count),(score,ids) in states.items():
                for k in range(1,min(len(items),demand-used)+1):
                    key=(used+k,count+1); candidate=(score+prefix[k][0],ids+prefix[k][1])
                    if key not in updated or candidate<updated[key]: updated[key]=candidate
            states=updated
        count,score,ids=min((count,score,ids) for (used,count),(score,ids) in states.items() if used==demand)
        return list(ids)

    def choose(self,demand,gpus,job=None):
        candidates=[g for g in gpus.values() if g.state=='available' and g.job_id is None and (not self.hard_mask or g.eligible)]
        if len(candidates)<demand: return []
        if self.name=='risk_mask_random': return [g.uid for g in self.rng.sample(sorted(candidates,key=lambda g:g.uid),demand)]
        groups=defaultdict(list)
        for g in candidates: groups[g.server_id].append(g)
        for items in groups.values(): items.sort(key=lambda g:(g.probability if self.name=='risk_mask_probability_tiebreak' else 0,g.uid))
        if self.name=='risk_prefer_packed':
            return self._exact_server_dp(demand,candidates,lambda g:self.allocation_risk(g))
        if self.is_batch_policy:
            branch_mode=self.opt_config.get('branch_mode','fused_plus_b1')
            b1_w=float(self.opt_config.get('b1_acute_weight',0.35))
            # Short 1-GPU jobs preserve full 8-GPU servers when partial servers exist
            rem=float(job.duration-job.progress) if job is not None else 3600.
            is_short_small=(demand<=2 and rem<1200.)
            server_free={s:len(items) for s,items in groups.items()}
            def score_fn(g):
                r=self.allocation_risk(g)
                if is_short_small:
                    # Prefer partially occupied servers (free < 8) to avoid breaking 8-GPU blocks for 8-GPU Gang jobs
                    full_server_penalty=0.05 if server_free.get(g.server_id,0)>=8 else 0.0
                    return full_server_penalty+r
                return r
            return self._exact_server_dp(demand,candidates,score_fn)
        # DP by total capacity. For a capacity, retain minimum number of servers
        # then lexical tuple. This is exact for primary objective and deterministic.
        dp={0:()}
        for server in sorted(groups):
            for capacity,subset in list(dp.items()):
                total=capacity+len(groups[server]); candidate=subset+(server,)
                if total not in dp or (len(candidate),candidate)<(len(dp[total]),dp[total]): dp[total]=candidate
        feasible=[(len(subset),capacity-demand,subset) for capacity,subset in dp.items() if capacity>=demand]
        _,_,servers=min(feasible)
        if self.name=='risk_mask_probability_tiebreak':
            k=len(servers);options=[]
            for subset in combinations(sorted(groups),k):
                pool=[g for server in subset for g in groups[server]]
                if len(pool)<demand:continue
                chosen=sorted(pool,key=lambda g:(g.probability,g.uid))[:demand]
                options.append((len(pool)-demand,sum(g.probability for g in chosen),subset,[g.uid for g in chosen]))
            return min(options)[3]
        selected=[]
        for server in servers:
            selected.extend(g.uid for g in groups[server][:max(0,demand-len(selected))])
        return selected

    def choose_batch(self,queued_jobs,gpus,recovery_config=None):
        """
        Batch Co-Optimization over candidate queued jobs in priority order.
        Step 1: Select maximal admissible set of jobs in scheduler order.
        Step 2: Greedy vulnerability order with exact DP for each single job.
        This does not solve the multi-job assignment to global optimality.
        Step 3: If policy uses metaheuristics ('metaheuristic_packed' or 'full_co_optimized'),
                refine the batch assignment via Warm-Start 2-Opt & Simulated Annealing.
        Returns dict {job_id: [gpu_uid, ...]}.
        """
        avail_map={g.uid:g for g in gpus.values() if g.state=='available' and g.job_id is None and g.eligible}
        if not avail_map or not queued_jobs: return {}

        # Step 1: Identify which jobs can be admitted in priority order
        admitted=[]
        remaining_Pool=dict(avail_map)
        for j in queued_jobs:
            if len(remaining_Pool)<j.demand: continue
            # Tentative check
            chosen=self._exact_server_dp(j.demand,list(remaining_Pool.values()),lambda g:self.allocation_risk(g))
            if not chosen: continue
            admitted.append(j)
            for u in chosen: remaining_Pool.pop(u,None)
            if not remaining_Pool: break

        if not admitted: return {}
        if len(admitted)==1 and self.name not in ('metaheuristic_packed','full_co_optimized'):
            j=admitted[0]
            return {j.id:self.choose(j.demand,gpus,job=j)}

        # Step 2: Vulnerability-prioritized Exact DP warm-start assignment
        cfg={**(recovery_config or {}),**self.opt_config}
        branch_mode=cfg.get('branch_mode','fused_plus_b1')
        b1_w=float(cfg.get('b1_acute_weight',0.35))
        vulns={j.id:compute_job_vulnerability(j,cfg) for j in admitted}

        # Sort admitted jobs by descending vulnerability V_j(t), tie-breaking by original scheduler rank
        rank_map={j.id:idx for idx,j in enumerate(admitted)}
        vuln_order=sorted(admitted,key=lambda j:(-vulns[j.id],rank_map[j.id]))

        pool=dict(avail_map)
        assignment={}
        for j in vuln_order:
            groups=defaultdict(int)
            for g in pool.values(): groups[g.server_id]+=1
            rem=max(1.0,float(j.duration-j.progress))
            is_short_small=(j.demand<=2 and rem<1200.)
            def s_fn(g):
                r=self.allocation_risk(g)
                if is_short_small and groups.get(g.server_id,0)>=8:
                    return r+0.05
                return r
            chosen=self._exact_server_dp(j.demand,list(pool.values()),s_fn)
            assignment[j.id]=chosen
            for u in chosen: pool.pop(u,None)

        if self.name not in ('metaheuristic_packed','full_co_optimized'):
            return assignment

        # Step 3: Warm-Start Metaheuristic Optimization (2-Opt Swap + Simulated Annealing)
        return self._metaheuristic_refine(admitted,assignment,avail_map,vulns,branch_mode,b1_w)

    def _metaheuristic_refine(self,admitted,initial_assignment,avail_map,vulns,branch_mode,b1_w):
        pack_w=float(self.opt_config.get('pack_weight',1000.0))
        risk_w=float(self.opt_config.get('risk_weight',1.0))
        risk_lookup={uid:self.allocation_risk(g) for uid,g in avail_map.items()}
        server_lookup={uid:g.server_id for uid,g in avail_map.items()}
        demand_map={j.id:j.demand for j in admitted}
        min_servers_map={j.id:max(1,math.ceil(j.demand/8.0)) for j in admitted}

        def job_cost(jid,uids):
            scount=len({server_lookup[u] for u in uids})
            frag=pack_w*max(0,scount-min_servers_map[jid])
            rcost=risk_w*vulns[jid]*sum(risk_lookup[u] for u in uids)
            return frag+rcost

        current={jid:list(uids) for jid,uids in initial_assignment.items()}
        costs={jid:job_cost(jid,uids) for jid,uids in current.items()}
        total_cost=sum(costs.values())

        best={jid:list(uids) for jid,uids in current.items()}
        best_cost=total_cost

        used_uids={u for uids in current.values() for u in uids}
        free_uids=sorted(set(avail_map.keys())-used_uids)
        job_ids=[j.id for j in admitted]

        if len(job_ids)<2 and not free_uids:
            return best

        # Fast deterministic-seeded Simulated Annealing + 2-Opt local search
        max_iters=int(self.opt_config.get('metaheuristic_iters',120))
        temp=float(self.opt_config.get('sa_initial_temp',0.5))
        cooling=float(self.opt_config.get('sa_cooling',0.95))

        for step in range(max_iters):
            if free_uids and (len(job_ids)<2 or self.rng.random()<0.4):
                # Move Type A: Swap an assigned GPU of a job with a free GPU
                jid=self.rng.choice(job_ids)
                idx=self.rng.randrange(len(current[jid]))
                f_idx=self.rng.randrange(len(free_uids))
                old_u=current[jid][idx]; new_u=free_uids[f_idx]
                candidate_uids=list(current[jid]); candidate_uids[idx]=new_u
                new_c=job_cost(jid,candidate_uids)
                delta=new_c-costs[jid]
                if delta<=0 or (temp>1e-6 and self.rng.random()<math.exp(-delta/max(temp,1e-6))):
                    current[jid]=candidate_uids; free_uids[f_idx]=old_u
                    costs[jid]=new_c; total_cost+=delta
                    if total_cost<best_cost-1e-12:
                        best_cost=total_cost
                        best={k:list(v) for k,v in current.items()}
            elif len(job_ids)>=2:
                # Move Type B: 2-Opt GPU Swap between two jobs in the batch
                j1,j2=self.rng.sample(job_ids,2)
                i1=self.rng.randrange(len(current[j1]))
                i2=self.rng.randrange(len(current[j2]))
                u1=current[j1][i1]; u2=current[j2][i2]
                cand1=list(current[j1]); cand1[i1]=u2
                cand2=list(current[j2]); cand2[i2]=u1
                c1=job_cost(j1,cand1); c2=job_cost(j2,cand2)
                delta=(c1+c2)-(costs[j1]+costs[j2])
                if delta<=0 or (temp>1e-6 and self.rng.random()<math.exp(-delta/max(temp,1e-6))):
                    current[j1]=cand1; current[j2]=cand2
                    costs[j1]=c1; costs[j2]=c2; total_cost+=delta
                    if total_cost<best_cost-1e-12:
                        best_cost=total_cost
                        best={k:list(v) for k,v in current.items()}
            temp*=cooling

        # Canonicalize GPU ID ordering per job
        return {jid:sorted(uids) for jid,uids in best.items()}

"""Atomic fixed-size Gang placement with exact minimum-server packing."""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import random

POLICIES=('risk_prefer_packed','risk_blind_packed','risk_mask_packed','risk_mask_random','risk_mask_packed_nonsticky','risk_mask_probability_tiebreak')


class PlacementPolicy:
    def __init__(self,name='risk_mask_packed',seed=17):
        if name not in POLICIES: raise ValueError(f'Unknown placement policy: {name}')
        self.name=name; self.rng=random.Random(seed)

    @property
    def hard_mask(self):
        return self.name not in ('risk_blind_packed','risk_prefer_packed')

    def choose(self,demand,gpus):
        candidates=[g for g in gpus.values() if g.state=='available' and g.job_id is None and (not self.hard_mask or g.eligible)]
        if len(candidates)<demand: return []
        if self.name=='risk_mask_random': return [g.uid for g in self.rng.sample(sorted(candidates,key=lambda g:g.uid),demand)]
        groups=defaultdict(list)
        for g in candidates: groups[g.server_id].append(g)
        for items in groups.values(): items.sort(key=lambda g:(g.probability if self.name=='risk_mask_probability_tiebreak' else 0,g.uid))
        if self.name=='risk_prefer_packed':
            # Exact DP: fewest servers, then sum of scores, then GPU IDs.
            # For k GPUs from one server, its k lowest scores are optimal.
            states={(0,0):(0.,())}
            for server in sorted(groups):
                items=sorted(groups[server],key=lambda g:(g.probability,g.uid))
                prefix=[(0.,())]
                for g in items:prefix.append((prefix[-1][0]+g.probability,prefix[-1][1]+(g.uid,)))
                updated=dict(states)
                for (used,count),(score,ids) in states.items():
                    for k in range(1,min(len(items),demand-used)+1):
                        key=(used+k,count+1);candidate=(score+prefix[k][0],ids+prefix[k][1])
                        if key not in updated or candidate<updated[key]:updated[key]=candidate
                states=updated
            count,score,ids=min((count,score,ids) for (used,count),(score,ids) in states.items() if used==demand)
            return list(ids)
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

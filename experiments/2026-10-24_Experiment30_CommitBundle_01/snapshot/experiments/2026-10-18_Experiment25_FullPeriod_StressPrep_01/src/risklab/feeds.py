"""Validated, causal actual-data feeds. No synthetic fallbacks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def seconds(value):
    stamp=pd.Timestamp(value)
    if stamp.tzinfo is None: raise ValueError('Timestamps must have an explicit timezone')
    return stamp.timestamp()


class TopologyProvider:
    def __init__(self,path,size=100):
        self.path=Path(path); self.frame=pd.read_parquet(path)
        required={'gpu_uid','server_id','local_gpu_id','cohort_index'}
        if not required.issubset(self.frame): raise ValueError('Incomplete GPU topology')
        f=self.frame.sort_values('cohort_index').reset_index(drop=True)
        if len(f)!=size or f.gpu_uid.duplicated().any() or f[list(required)].isna().any().any(): raise ValueError('Invalid cohort count, duplicate GPU or missing mapping')
        if list(f.cohort_index)!=list(range(size)): raise ValueError('Non-contiguous cohort indices')
        if f[['server_id','local_gpu_id']].duplicated().any(): raise ValueError('Duplicate server/local GPU mapping')
        self.frame=f; self.ids=list(f.gpu_uid); self.by_uid=f.set_index('gpu_uid').to_dict('index')


class RiskTapeProvider:
    def __init__(self,path,topology,interval=1800):
        self.path=Path(path); self.ids=set(topology.ids); self.interval=interval
        with pq.ParquetFile(self.path) as metadata:
            names=set(metadata.schema_arrow.names)
            present=names & {'b1_probability','b2_probability'}
            if present and len(present)!=2: raise ValueError('Supply both branch scores or neither')
            self.has_branch_scores=len(present)==2
        self.times=[]
        for t, values, _, _ in self._groups():
            if set(values)!=self.ids: raise ValueError(f'Missing/unknown GPU at {t}')
            if self.times and abs(t-self.times[-1]-interval)>1e-6: raise ValueError('Missing decision time or incorrect cadence')
            if t%300: raise ValueError('Timestamp is not on a 5-minute source grid')
            self.times.append(t)
        if not self.times: raise ValueError('Empty probability tape')
        self._iterator=iter(self._groups()); self._next=next(self._iterator,None)
        self.current=None; self.current_b1=None; self.current_b2=None; self.at=None

    def _groups(self):
        with pq.ParquetFile(self.path) as f:
            names=set(f.schema_arrow.names)
            if not {'timestamp','gpu_uid','probability'}.issubset(names): raise ValueError('Probability tape schema mismatch')
            has_branches={'b1_probability','b2_probability'}.issubset(names)
            cols=['timestamp','gpu_uid','probability']+(['b1_probability','b2_probability'] if has_branches else [])
            current=None; values={}; b1_vals={}; b2_vals={}
            for batch in f.iter_batches(batch_size=8192,columns=cols):
                for row in batch.to_pandas().itertuples(index=False,name=None):
                    timestamp,uid,p=row[0],row[1],row[2]
                    t=seconds(timestamp); p=float(p)
                    if not np.isfinite(p) or not 0<=p<=1: raise ValueError('Invalid probability')
                    p1=float(row[3]) if has_branches else p
                    p2=float(row[4]) if has_branches else p
                    if any(not np.isfinite(v) or not 0<=v<=1 for v in (p1,p2)):
                        raise ValueError('Invalid branch score')
                    if current is not None and t!=current:
                        if t<current: raise ValueError('Probability tape is not chronological')
                        yield current,values,b1_vals,b2_vals
                        values={}; b1_vals={}; b2_vals={}
                    current=t
                    suid=str(uid)
                    if suid in values: raise ValueError('Duplicate GPU/time probability')
                    values[suid]=p; b1_vals[suid]=p1; b2_vals[suid]=p2
            if current is not None: yield current,values,b1_vals,b2_vals

    def close(self):
        self._iterator.close()

    def snapshot(self,at_time):
        if self.at is not None and at_time<self.at: raise ValueError('Risk provider cannot travel backwards')
        while self._next is not None and self._next[0]<=at_time:
            self.at,self.current,self.current_b1,self.current_b2=self._next
            self._next=next(self._iterator,None)
        if self.current is None: raise ValueError('No risk observation available yet')
        if at_time-self.at>=self.interval: raise ValueError('Stale risk snapshot')
        return dict(self.current)

    def branch_snapshot(self,at_time):
        self.snapshot(at_time)
        return dict(self.current),dict(self.current_b1),dict(self.current_b2)


def _parse_xid_codes(raw_val):
    values=raw_val
    if isinstance(raw_val,str):
        s=raw_val.strip()
        try: values=json.loads(s) if s.startswith('[') else s.split(',')
        except (ValueError,TypeError) as exc: raise ValueError('Malformed XID codes') from exc
    if not isinstance(values,(list,tuple,set,np.ndarray)): values=[values]
    parsed=[]
    for value in values:
        try: number=float(value)
        except (TypeError,ValueError) as exc: raise ValueError('Malformed or missing XID code') from exc
        if isinstance(value,bool) or not np.isfinite(number) or number<=0 or int(number)!=number:
            raise ValueError('Malformed or missing XID code')
        parsed.append(int(number))
    if not parsed: raise ValueError('Missing XID code; no synthetic XID43 fallback')
    return sorted(set(parsed))


class FailureTapeProvider:
    def __init__(self,path,topology):
        df=pd.read_parquet(path)
        required={'timestamp','gpu_uid','episode_id','episode_start','episode_end','xid_codes'}
        if not required.issubset(df): raise ValueError('Failure tape schema mismatch')
        if df.episode_id.duplicated().any(): raise ValueError('Duplicate XID episode')
        if not set(df.gpu_uid).issubset(topology.ids): raise ValueError('Unknown failure GPU')
        self.events=[]
        for r in df.sort_values(['timestamp','gpu_uid','episode_id']).to_dict('records'):
            r['time']=seconds(r['timestamp'])
            if r['time']!=seconds(r['episode_start']) or seconds(r['episode_end'])<r['time']: raise ValueError('Invalid XID episode interval')
            r['parsed_xid_codes']=_parse_xid_codes(r['xid_codes'])
            self.events.append(r)
        self.index=0

    def next_time(self):
        return self.events[self.index]['time'] if self.index<len(self.events) else float('inf')

    def events_at(self,at_time):
        result=[]
        while self.index<len(self.events) and self.events[self.index]['time']<=at_time:
            result.append(self.events[self.index]); self.index+=1
        return result


class JobTraceProvider:
    def __init__(self,path):
        df=pd.read_parquet(path)
        required={'job_id','arrival_time','duration_seconds','gpu_demand','gang_id','checkpoint_interval_seconds'}
        if not required.issubset(df) or df.job_id.duplicated().any(): raise ValueError('Invalid job trace schema or duplicate job')
        self.records=[]
        for r in df.to_dict('records'):
            d=float(r['duration_seconds']); g=float(r['gpu_demand'])
            if not np.isfinite(d) or d<=0 or not np.isfinite(g) or g<1 or int(g)!=g: raise ValueError('Invalid duration/Gang demand')
            r['job_id']=str(r['job_id']); r['gang_id']=str(r['gang_id']); r['gpu_demand']=int(g); r['arrival']=seconds(r['arrival_time'])
            cp=r['checkpoint_interval_seconds']
            r['checkpoint_interval_seconds']=None if pd.isna(cp) else float(cp)
            if r['checkpoint_interval_seconds'] is not None and r['checkpoint_interval_seconds']<=0: raise ValueError('Invalid checkpoint interval')
            self.records.append(r)
        self.records.sort(key=lambda r:(r['arrival'],r['job_id']))


def load_inputs(manifest_path):
    path=Path(manifest_path).resolve(); m=json.loads(path.read_text(encoding='utf-8'))
    files={}
    for name in ['probability_tape','failure_tape','gpu_manifest','job_trace','threshold_manifest']:
        ref=m['inputs'][name]; file=(path.parent/ref['path']).resolve()
        if sha256(file)!=ref['sha256']: raise ValueError(f'Checksum mismatch: {name}')
        files[name]=file
    threshold=json.loads(files['threshold_manifest'].read_text(encoding='utf-8'))
    if threshold.get('selection_split')!='validation_only' or threshold.get('heldout_used_for_selection') is not False: raise ValueError('Threshold provenance must be validation-only')
    risk_interval=int(m.get('risk_interval_seconds',m['decision_interval_seconds']))
    if risk_interval<=0 or int(m['decision_interval_seconds'])<=0: raise ValueError('Invalid cadence')
    if threshold.get('decision_interval_seconds')!=risk_interval: raise ValueError('Threshold/risk cadence mismatch')
    value=float(threshold['threshold'])
    if not np.isfinite(value) or not 0<=value<=1: raise ValueError('Invalid threshold')
    top=TopologyProvider(files['gpu_manifest'],m['cohort_size'])
    risk=RiskTapeProvider(files['probability_tape'],top,risk_interval)
    try:
        if m.get('require_branch_scores') and not risk.has_branch_scores:
            raise ValueError('This policy requires explicit B1 and B2 scores')
        failures=FailureTapeProvider(files['failure_tape'],top); jobs=JobTraceProvider(files['job_trace'])
        if seconds(m['start_time'])!=risk.times[0] or not risk.times[-1]<seconds(m['end_time'])<=risk.times[-1]+risk_interval: raise ValueError('Replay bounds do not match tape')
        if threshold.get('validation_end') and seconds(threshold['validation_end'])>=risk.times[0]: raise ValueError('Threshold validation overlaps replay')
    except BaseException:
        risk.close()
        raise
    return m,top,risk,failures,jobs,threshold

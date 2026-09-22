"""Derive exact state intervals from the append-only event log."""
import json
from pathlib import Path
import pandas as pd


def write_timelines(output,sim):
    output=Path(output)
    gpu={uid:(0.,'available',None) for uid in sim.gpus}
    job={};assignments={};gpu_rows=[];job_rows=[]
    def change_gpu(uid,t,state,job_id=None):
        before,old,owner=gpu[uid]
        if t>before:gpu_rows.append({'gpu_uid':uid,'server_id':sim.gpus[uid].server_id,'start_seconds':before,'end_seconds':t,'state':old,'job_id':owner})
        gpu[uid]=(t,state,job_id)
    def change_job(ident,t,state):
        if ident in job:
            before,old=job[ident]
            if t>before:job_rows.append({'job_id':ident,'gang_id':sim.jobs[ident].gang_id,'start_seconds':before,'end_seconds':t,'state':old})
        job[ident]=(t,state)
    with (output/'events.jsonl').open(encoding='utf-8') as f:
        for line in f:
            e=json.loads(line);t=e['time'];kind=e['type'];ident=e.get('job_id')
            if kind=='job_admitted':change_job(ident,t,'queued')
            elif kind in ('job_launch','job_relaunch'):
                change_job(ident,t,'running');assignments[ident]=e['gpu_ids']
                for uid in e['gpu_ids']:change_gpu(uid,t,'running',ident)
            elif kind in ('gang_failure','preventive_migration','scheduler_preemption','job_complete'):
                change_job(ident,t,'completed' if kind=='job_complete' else 'recovering')
                for uid in assignments.pop(ident,[]):change_gpu(uid,t,'available')
            elif kind=='job_recovered':change_job(ident,t,'queued')
            elif kind in ('gpu_failed','gpu_drained','maintenance_start','cooldown_start','cooldown_complete'):
                state={'gpu_failed':'failed','gpu_drained':'drained','maintenance_start':'maintenance','cooldown_start':'cooldown','cooldown_complete':'available'}[kind]
                change_gpu(e['gpu_id'],t,state)
    end=sim.time-sim.start
    for uid,(start,state,owner) in list(gpu.items()):change_gpu(uid,end,state,owner)
    for ident,(start,state) in list(job.items()):change_job(ident,end,state)
    pd.DataFrame(gpu_rows,columns=['gpu_uid','server_id','start_seconds','end_seconds','state','job_id']).to_parquet(output/'gpu_timeline.parquet',index=False)
    pd.DataFrame(job_rows,columns=['job_id','gang_id','start_seconds','end_seconds','state']).to_parquet(output/'job_timeline.parquet',index=False)

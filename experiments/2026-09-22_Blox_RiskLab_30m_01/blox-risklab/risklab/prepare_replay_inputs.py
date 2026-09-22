"""Build actual XID episodes and observed job trace for the frozen ML cohort."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .feeds import sha256
from .replay_runner import atomic_json


def build_failure_tape(raw_path,ids,start,end):
    # 15-second raw observations are grouped into five-minute buckets. Only
    # <=10-minute gaps between nonzero buckets are merged; the timestamp is the
    # first actual observation (never rounded earlier than the observed onset).
    active={};episodes=[]
    def finish(uid):
        item=active.pop(uid);episodes.append({'timestamp':item['start'],'gpu_uid':uid,'episode_id':f'{uid}:{item["start"].isoformat()}','episode_start':item['start'],'episode_end':item['end'],'xid_codes':json.dumps(sorted(item['codes']))})
    previous_time=None
    for chunk in pd.read_csv(raw_path,usecols=['Time']+ids,chunksize=10000):
        times=pd.to_datetime(chunk['Time'],utc=True)
        if not times.is_monotonic_increasing or (previous_time is not None and times.iloc[0]<previous_time):raise ValueError('Raw XID observations are not chronological')
        previous_time=times.iloc[-1]
        for uid in ids:
            values=pd.to_numeric(chunk[uid],errors='coerce')
            mask=values.notna() & values.ne(0)
            for stamp,code in zip(times[mask],values[mask]):
                if stamp>=end:continue
                bucket=stamp.floor('5min')
                old=active.get(uid)
                if old is not None and (bucket-old['bucket']).total_seconds()>600:finish(uid);old=None
                if old is None:active[uid]={'start':stamp,'end':stamp,'bucket':bucket,'codes':{int(code)}}
                else:old['end']=stamp;old['bucket']=bucket;old['codes'].add(int(code))
        if times.iloc[0]>=end:break
    for uid in list(active):finish(uid)
    columns=['timestamp','gpu_uid','episode_id','episode_start','episode_end','xid_codes']
    frame=pd.DataFrame(episodes,columns=columns)
    if len(frame):frame=frame[(frame.timestamp>=start)&(frame.timestamp<end)].sort_values(['timestamp','gpu_uid'])
    return frame


def prepare(project,inputs):
    project=Path(project).resolve();inputs=Path(inputs).resolve()
    complete_path=inputs/'export_complete.json'
    if complete_path.exists():
        complete=json.loads(complete_path.read_text(encoding='utf-8'))
    else:
        source=project/'ISEE_Project_GitHubBase/experiments/2026-09-21_All-XID_ReportFaithful_ParetoMomentumCascade_01/experiment_manifest.json'
        metadata=json.loads(source.read_text(encoding='utf-8'))
        complete={'start':metadata['test_start_time'],'end_exclusive':metadata['test_end_time']}
        grid=np.load(project/'ISEE_Project_GitHubBase/outputs/branch1/cache/grid_meta.npz',allow_pickle=True)
        uids=list(map(str,grid['gpu_ids'][:100]));grid.close()
        topology=pd.DataFrame([{'gpu_uid':uid,'server_id':uid.rsplit('-',1)[0],'local_gpu_id':int(uid.rsplit('-',1)[1]),'cohort_index':i} for i,uid in enumerate(uids)])
        topology['node_index']=pd.factorize(topology.server_id,sort=True)[0]
        topology.to_parquet(inputs/'gpu_manifest.parquet',index=False)
    top=pd.read_parquet(inputs/'gpu_manifest.parquet');ids=list(top.gpu_uid)
    start=pd.Timestamp(complete['start']);end=pd.Timestamp(complete['end_exclusive'])
    raw=project/'DATA/AcmeTrace/data/utilization/seren/XID_ERRORS.csv'
    node=project/'DATA/AcmeTrace/data/utilization/seren/NODE_CPU_UTILIZATION.csv'
    with raw.open(encoding='utf-8-sig') as f:columns=next(csv.reader(f))
    with node.open(encoding='utf-8-sig') as f:servers=next(csv.reader(f))
    if not set(ids).issubset(columns) or not set(top.server_id).issubset(servers):raise ValueError('GPU/server mapping not supported by raw telemetry headers')
    if (inputs/'raw_inputs_ready.json').exists():
        prior=json.loads((inputs/'raw_inputs_ready.json').read_text(encoding='utf-8'))
        if prior['cohort']!=ids or prior['start']!=str(start) or prior['end']!=str(end):raise ValueError('Raw input cache contract changed')
        failures=pd.read_parquet(inputs/'failure_tape.parquet')
    else:
        failures=build_failure_tape(raw,ids,start,end);failures.to_parquet(inputs/'failure_tape.parquet',index=False)
    job_source=project/'DATA/AcmeTrace/data/job_trace/trace_seren.csv'
    raw_jobs=pd.read_csv(job_source);arrivals=pd.to_datetime(raw_jobs.submit_time,utc=True)
    in_window=(arrivals>=start)&(arrivals<end)
    # Only completed jobs provide uncensored execution durations. CPU-only and
    # jobs exceeding the 100-GPU cluster are not silently shortened/resized.
    mask=in_window & raw_jobs.state.eq('COMPLETED') & raw_jobs.gpu_num.between(1,len(ids)) & raw_jobs.duration.gt(0)
    selected=raw_jobs[mask].copy()
    jobs=pd.DataFrame({'job_id':selected.job_id.astype(str),'gang_id':selected.job_id.astype(str),'arrival_time':arrivals[mask],'duration_seconds':selected.duration.astype(float),'gpu_demand':selected.gpu_num.astype(int),'checkpoint_interval_seconds':np.nan})
    jobs.sort_values(['arrival_time','job_id']).to_parquet(inputs/'job_trace.parquet',index=False)
    assumptions={
        'workload':'All completed GPU jobs arriving in the replay interval with 1..100 GPU demand; no resampling or time compression. This replays cluster-wide arrivals on a smaller cohort, not a measured production load for these 100 GPUs.',
        'duration':'Observed completed-job execution seconds treated as useful work; original unsuccessful/cancelled durations are censored and excluded.',
        'initial_state':'Empty job queue and initially healthy GPUs at replay start; no carry-in reconstruction.',
        'job_arrivals_in_interval':int(in_window.sum()),'included_jobs':len(jobs),'excluded_jobs':int(in_window.sum()-mask.sum()),
        'trace_last_arrival':str(arrivals.max()),'failure_episodes':len(failures),
        'checkpoint_source':'fixed_default; job trace has no checkpoint field',
        'recovery_parameters':'Checkpoint interval 300s, preventive/scheduler checkpoint cost 60s, relaunch 300s; assumptions, not observed durations.',
        'maintenance_parameters':'One concurrent GPU, 900s maintenance, 300s cooldown; assumptions.',
        'episode_rule':'Merge nonzero five-minute buckets with gap <=600s. Gap rule takes precedence over transient clean observations; timestamp preserves first raw observation.',
        'unavailability':'Reactive downtime and preventive maintenance are disjoint GPU-second components; queue time is separate.',
        'cohort':'First 100 lexical GPU IDs in original telemetry grid; server prefix verified against node telemetry header.',
        'risk_probability':'Raw source model score; calibration quality is not asserted. Failure generation uses observed XID only.'}
    atomic_json(inputs/'raw_inputs_ready.json',{'cohort':ids,'start':str(start),'end':str(end),'assumptions':assumptions})
    if not complete_path.exists():
        return {'status':'waiting_for_predictions','assumptions':assumptions}
    inputs_dict={name:{'path':filename,'sha256':sha256(inputs/filename)} for name,filename in [('probability_tape','probability_tape.parquet'),('failure_tape','failure_tape.parquet'),('gpu_manifest','gpu_manifest.parquet'),('job_trace','job_trace.parquet'),('threshold_manifest','threshold_manifest.json')]}
    manifest={'version':1,'start_time':start.isoformat(),'end_time':end.isoformat(),'cohort_size':100,'decision_interval_seconds':1800,'random_seed':17,'scheduler':'Las','placement_policy':'risk_mask_packed','preemption':True,'blox_source_root':str(project/'blox_repo_actual/blox-main'),'server_correlated_failure':False,'inputs':inputs_dict,'recovery':{'checkpoint_interval_seconds':300,'checkpoint_cost_seconds':60,'relaunch_seconds':300},'maintenance':{'capacity':1,'duration_seconds':900,'cooldown_seconds':300},'assumptions':assumptions,'source_data':{'job_trace_sha256':sha256(job_source),'raw_xid_sha256':sha256(raw),'node_telemetry_header_sha256':__import__('hashlib').sha256(','.join(servers).encode()).hexdigest()}}
    atomic_json(inputs/'run_manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True,type=Path);p.add_argument('--inputs',required=True,type=Path);a=p.parse_args();print(json.dumps(prepare(a.project,a.inputs)['assumptions'],ensure_ascii=False,indent=2))

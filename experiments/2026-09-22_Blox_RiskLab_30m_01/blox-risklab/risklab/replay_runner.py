"""CLI and worker entry point sharing the actual-data replay engine."""
from __future__ import annotations
import argparse
import dataclasses
import html
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
import pandas as pd
from .feeds import load_inputs,sha256
from .replay_engine import ActualRiskLabSimulation,NativeScheduler


def atomic_json(path,value):
    # Windows readers do not share delete access: a short status read can
    # temporarily block os.replace. Preserve atomic reads, retry only the
    # transient PermissionError, and keep concurrent writers' temp files apart.
    import uuid
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        for attempt in range(40):
            try:
                tmp.replace(path);return
            except PermissionError:
                if attempt==39:raise
                time.sleep(.025)
    finally:
        tmp.unlink(missing_ok=True)


class ArtifactWriter:
    def __init__(self,out):
        self.out=out;self.files={name:(out/f'{name}.jsonl').open('w',encoding='utf-8') for name in ('events','decisions','snapshots','round_metrics')}
    def emit(self,stream,row):
        if stream=='snapshots':
            metric={'time_seconds':row['time'],'progress':row['progress'],'decision':row['decisions']}
            metric.update({f'jobs_{k}':v for k,v in row['jobs'].items()})
            metric['eligible_gpus']=sum(g['eligible'] for g in row['gpus'])
            for state in ('available','running','drained','failed','maintenance','cooldown'):
                metric[f'gpus_{state}']=sum(g['state']==state for g in row['gpus'])
            self.emit('round_metrics',metric)
        self.files[stream].write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    def flush(self):
        for f in self.files.values():f.flush()
    def close(self):
        for f in self.files.values():f.close()


def execute(manifest_path,out,policy=None,scheduler_name=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Output already has a run; choose a new directory')
    start=time.perf_counter();writer=None;risk=None
    atomic_json(out/'status.json',{'status':'validating','progress':0,'pid':os.getpid()})
    try:
        m,top,risk,failures,jobs,threshold=load_inputs(manifest_path)
        if policy:m['placement_policy']=policy
        if scheduler_name:m['scheduler']=scheduler_name
        source=Path(m['blox_source_root'])
        if not source.is_absolute():source=(Path(manifest_path).resolve().parent/source).resolve()
        native=NativeScheduler(source,m['scheduler'])
        source_hashes={str(p.relative_to(source)):sha256(p) for p in sorted((source/'schedulers').glob('*.py'))}
        m.update({'run_id':out.name,'python':sys.executable,'python_version':platform.python_version(),'source_manifest_sha256':sha256(manifest_path),'blox_scheduler_sha256':source_hashes,'source_commit':m.get('source_commit','unavailable: local Blox checkout has no .git metadata'),'scheduler_implementation':native.label})
        m['implementation_sha256']={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['replay_engine.py','replay_runner.py','replay_web.py','feeds.py','placement_policy.py','loss_ledger.py','timelines.py','thresholds.py','runtime_metrics.py']}
        from importlib.metadata import version
        m['dependencies']={name:version(name) for name in ['pandas','numpy','pyarrow']}
        atomic_json(out/'manifest.json',m);writer=ArtifactWriter(out)
        last_update=[0.]
        def update(snapshot):
            writer.flush()
            atomic_json(out/'status.json',{'timestamp':snapshot['timestamp'],'start_timestamp':snapshot['start_timestamp'],'placement_policy':snapshot['placement_policy'],'status':'running','progress':snapshot['progress'],'time':snapshot['time'],'jobs':snapshot['jobs'],'gpus':snapshot['gpus'],'decisions':snapshot['decisions'],'pid':os.getpid()})
        sim=ActualRiskLabSimulation(m,top,risk,failures,jobs,threshold,native,emit=writer.emit,stop=lambda:(out/'stop.request').exists(),progress=update)
        simulation_start=time.perf_counter()
        summary=sim.run();writer.flush()
        simulation_seconds=time.perf_counter()-simulation_start
        atomic_json(out/'summary.json',summary)
        # Performance is kept out of deterministic summary/event artifacts.
        from .runtime_metrics import peak_memory_bytes
        rows=[]
        for j in sim.jobs.values():
            row=dataclasses.asdict(j);row['ready']=None if row['ready']==float('inf') else row['ready'];row['loss_components']=json.dumps(dict(sim.ledger.jobs.get(j.id,{})))
            rows.append(row)
        pd.DataFrame(rows).to_parquet(out/'job_summary.parquet',index=False)
        pd.DataFrame([{'gpu_uid':uid,**dict(sim.ledger.gpus.get(uid,{}))} for uid in top.ids]).to_parquet(out/'gpu_loss.parquet',index=False)
        metrics_path=out/'round_metrics.jsonl'
        round_metrics=pd.read_json(metrics_path,lines=True) if metrics_path.stat().st_size else pd.DataFrame(columns=['time_seconds','progress','decision'])
        round_metrics.to_parquet(out/'metrics.parquet',index=False)
        from .timelines import write_timelines
        write_timelines(out,sim)
        report='<!doctype html><meta charset="utf-8"><title>RiskLab replay result</title><style>body{font:16px system-ui;max-width:1000px;margin:48px auto;background:#f4f6fa;color:#132435}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}code{white-space:pre-wrap}</style><h1>RiskLab · 30-minute replay</h1><p>Observed XID replay · fixed cohort · validation-only threshold</p><table>'+''.join('<tr><th>'+html.escape(k)+'</th><td>'+html.escape(str(v))+'</td></tr>' for k,v in summary.items())+'</table><h2>Input and workload assumptions</h2><code>'+html.escape(json.dumps(m.get('assumptions',{}),ensure_ascii=False,indent=2))+'</code>'
        (out/'report.html').write_text(report,encoding='utf-8')
        atomic_json(out/'performance.json',{'elapsed_seconds':time.perf_counter()-start,'simulation_seconds':simulation_seconds,'input_setup_seconds':simulation_start-start,'artifact_seconds':time.perf_counter()-simulation_start-simulation_seconds,'max_decision_seconds':sim.max_placement_seconds,'process_peak_working_set_bytes':peak_memory_bytes()})
        atomic_json(out/'status.json',{**sim.snapshot(),'status':summary['status'],'summary':summary,'pid':os.getpid()})
        return summary
    except BaseException as exc:
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        try:atomic_json(out/'status.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}','pid':os.getpid()})
        except OSError:pass  # Original traceback and worker stderr remain available.
        raise
    finally:
        if writer:writer.close()
        if risk:risk.close()


def main():
    p=argparse.ArgumentParser(description='Actual-data Blox/RiskLab replay')
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--policy',choices=['risk_prefer_packed','risk_blind_packed','risk_mask_packed','risk_mask_random','risk_mask_packed_nonsticky','risk_mask_probability_tiebreak'])
    p.add_argument('--scheduler',choices=['Fifo','Las','Tiresias2Q'])
    a=p.parse_args();print(json.dumps(execute(a.manifest,a.output,a.policy,a.scheduler),ensure_ascii=False))


if __name__=='__main__':main()

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
from .placement_policy import POLICIES
from .replay_engine import ActualRiskLabSimulation,NativeScheduler
from .incident_policy import incident_defaults


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
            for state in ('available','running','drained','failed','maintenance','cooldown','diagnosing','quarantined','loading'):
                metric[f'gpus_{state}']=sum(g['state']==state for g in row['gpus'])
            self.emit('round_metrics',metric)
        self.files[stream].write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    def flush(self):
        for f in self.files.values():f.flush()
    def close(self):
        for f in self.files.values():f.close()


def execute(manifest_path,out,policy=None,scheduler_name=None,xid_severity_mode=None,xid43_interrupt_prob=None,opt_overrides=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Output already has a run; choose a new directory')
    start=time.perf_counter();writer=None;risk=None
    atomic_json(out/'status.json',{'status':'validating','progress':0,'pid':os.getpid()})
    try:
        m,top,risk,failures,jobs,threshold=load_inputs(manifest_path)
        if policy:m['placement_policy']=policy
        if scheduler_name:m['scheduler']=scheduler_name
        if xid_severity_mode is not None:m['xid_severity_mode']=xid_severity_mode
        if xid43_interrupt_prob is not None:m['xid43_interrupt_prob']=float(xid43_interrupt_prob)
        if opt_overrides:
            m['optimization']={**m.get('optimization',{}),**opt_overrides}
        m['risk_interval_seconds']=int(m.get('risk_interval_seconds',m['decision_interval_seconds']))
        m['xid_severity_mode']=m.get('xid_severity_mode','uniform_hardware')
        m['xid43_interrupt_prob']=float(m.get('xid43_interrupt_prob',1.0))
        m['soft_reset_seconds']=float(m.get('soft_reset_seconds',60.0))
        m['incident_policy']=incident_defaults(m.get('incident_policy'))
        m['branch_scores_available']=bool(risk.has_branch_scores)
        m['risk_score_semantics']='Model score; calibration not asserted; score-time exposure is a surrogate'
        m['checkpoint_execution_model']=m.get('checkpoint_execution_model','Periodic saves have no simulated pause; checkpoint_cost applies to preventive/scheduler interruptions only')
        m['runtime_information']=m.get('runtime_information','Observed trace duration is known to simulation; duration-aware allocation is an oracle-runtime scenario')
        m['response_scope_note']='Unmapped application events require explicit gpu_local_sensitivity to affect a simulated job'
        source=Path(m['blox_source_root'])
        if not source.is_absolute():source=(Path(manifest_path).resolve().parent/source).resolve()
        native=NativeScheduler(source,m['scheduler'])
        source_hashes={str(p.relative_to(source)):sha256(p) for p in sorted((source/'schedulers').glob('*.py'))}
        m.update({'run_id':out.name,'python':sys.executable,'python_version':platform.python_version(),'source_manifest_sha256':sha256(manifest_path),'blox_scheduler_sha256':source_hashes,'source_commit':m.get('source_commit','unavailable: local Blox checkout has no .git metadata'),'scheduler_implementation':native.label})
        m['implementation_sha256']={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['replay_engine.py','replay_runner.py','replay_web.py','feeds.py','placement_policy.py','loss_ledger.py','timelines.py','thresholds.py','runtime_metrics.py']}
        for name in ('incident_policy.py','recovery_models.py'):
            m['implementation_sha256'][name]=sha256(Path(__file__).parent/name)
        import subprocess
        checkout=next((p for p in Path(__file__).resolve().parents if (p/'.git').exists()),None)
        if checkout:
            revision=subprocess.run(['git','-c',f'safe.directory={checkout.as_posix()}','-C',str(checkout),'rev-parse','HEAD'],capture_output=True,text=True)
            m['upstream_commit']=revision.stdout.strip() if revision.returncode==0 else 'unavailable'
        from importlib.metadata import version
        m['dependencies']={name:version(name) for name in ['pandas','numpy','pyarrow']}
        atomic_json(out/'manifest.json',m);writer=ArtifactWriter(out)
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
        report='<!doctype html><meta charset="utf-8"><title>RiskLab replay result</title><style>body{font:16px system-ui;max-width:1000px;margin:48px auto;background:#f4f6fa;color:#132435}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}code{white-space:pre-wrap}</style><h1>RiskLab · '+str(m['decision_interval_seconds']/60)+'-minute replay</h1><p>Risk refresh '+str(m['risk_interval_seconds'])+'s · XID mode '+html.escape(m['xid_severity_mode'])+' · job seconds and GPU seconds reported separately</p><table>'+''.join('<tr><th>'+html.escape(k)+'</th><td>'+html.escape(str(v))+'</td></tr>' for k,v in summary.items())+'</table><h2>Input and workload assumptions</h2><code>'+html.escape(json.dumps(m.get('assumptions',{}),ensure_ascii=False,indent=2))+'</code>'
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
    p.add_argument('--policy',choices=list(POLICIES))
    p.add_argument('--scheduler',choices=['Fifo','Las','Tiresias2Q'])
    p.add_argument('--xid-severity-mode',choices=['uniform_hardware','differentiated_43_31','evidence_based'],default=None)
    p.add_argument('--xid43-interrupt-prob',type=float,default=None)
    a=p.parse_args();print(json.dumps(execute(a.manifest,a.output,a.policy,a.scheduler,a.xid_severity_mode,a.xid43_interrupt_prob),ensure_ascii=False))


if __name__=='__main__':main()

from __future__ import annotations
import itertools
import json
import random
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
import pandas as pd
from risklab.feeds import TopologyProvider,RiskTapeProvider,FailureTapeProvider,JobTraceProvider,load_inputs,sha256
from risklab.loss_ledger import LossLedger
from risklab.placement_policy import PlacementPolicy
from risklab.replay_engine import ActualRiskLabSimulation,NativeScheduler,ReplayGPU
from risklab.replay_runner import execute,atomic_json
from risklab.replay_web import make_server

ROOT=Path(__file__).resolve().parents[1]
START=pd.Timestamp('2023-08-05T06:45:00Z')


def fixture(root,n=4,policy='risk_blind_packed',failure_times=(),high_from=None,duration=5000,end=7200):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    topology=pd.DataFrame([{'gpu_uid':f's{i//2}-{i%2}','server_id':f's{i//2}','local_gpu_id':i%2,'cohort_index':i} for i in range(n)])
    topology.to_parquet(root/'gpu_manifest.parquet',index=False)
    rows=[]
    for t in range(0,end,1800):
        for uid in topology.gpu_uid:rows.append({'timestamp':START+pd.Timedelta(seconds=t),'gpu_uid':uid,'probability':.9 if high_from is not None and t>=high_from else .1})
    pd.DataFrame(rows).to_parquet(root/'probability_tape.parquet',index=False)
    fs=[{'timestamp':START+pd.Timedelta(seconds=t),'gpu_uid':uid,'episode_id':str(i),'episode_start':START+pd.Timedelta(seconds=t),'episode_end':START+pd.Timedelta(seconds=t),'xid_codes':'[31]'} for i,(t,uid) in enumerate(failure_times)]
    pd.DataFrame(fs,columns=['timestamp','gpu_uid','episode_id','episode_start','episode_end','xid_codes']).to_parquet(root/'failure_tape.parquet',index=False)
    pd.DataFrame([{'job_id':'j1','gang_id':'g1','arrival_time':START,'duration_seconds':duration,'gpu_demand':2,'checkpoint_interval_seconds':300}]).to_parquet(root/'job_trace.parquet',index=False)
    atomic_json(root/'threshold_manifest.json',{'threshold':.5,'selection_split':'validation_only','heldout_used_for_selection':False,'decision_interval_seconds':1800,'validation_end':'2023-08-01T00:00:00Z'})
    inputs={name:{'path':name+('.json' if name=='threshold_manifest' else '.parquet'),'sha256':sha256(root/(name+('.json' if name=='threshold_manifest' else '.parquet')))} for name in ['gpu_manifest','probability_tape','failure_tape','job_trace','threshold_manifest']}
    manifest={'start_time':START.isoformat(),'end_time':(START+pd.Timedelta(seconds=end)).isoformat(),'cohort_size':n,'decision_interval_seconds':1800,'random_seed':17,'scheduler':'Fifo','placement_policy':policy,'blox_source_root':str(ROOT.parent/'blox-main'),'inputs':inputs,'recovery':{'checkpoint_interval_seconds':300,'checkpoint_cost_seconds':60,'relaunch_seconds':300},'maintenance':{'capacity':4,'duration_seconds':900,'cooldown_seconds':300}}
    atomic_json(root/'manifest.json',manifest);return root/'manifest.json'


class ReplayTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def run_sim(self,**kwargs):
        path=fixture(self.root/'input',**kwargs);m,t,r,f,j,h=load_inputs(path);events=[]
        sim=ActualRiskLabSimulation(m,t,r,f,j,h,NativeScheduler(m['blox_source_root'],m['scheduler']),emit=lambda s,row:events.append((s,row)))
        return sim,sim.run(),events
    def test_true_completion_time(self):
        sim,s,e=self.run_sim(duration=117)
        self.assertEqual(s['avg_jct_seconds'],117);self.assertEqual(s['failure_lost_time_seconds'],0)
    def test_atomic_failure_and_queue_separation(self):
        sim,s,e=self.run_sim(failure_times=[(650,'s0-0'),(650,'s0-1')])
        self.assertEqual(sum(r['type']=='gang_failure' for st,r in e if st=='events'),1)
        self.assertEqual(s['failure_checkpoint_loss_seconds'],50)
        self.assertEqual(s['failure_recovery_delay_seconds'],300)
        self.assertEqual(s['failure_gpu_unavailable_seconds'],2400)
        self.assertGreater(s['avg_queue_wait_seconds'],0)
        self.assertEqual(s['failure_lost_time_seconds'],2750)
    def test_idle_failure_no_job_loss(self):
        sim,s,e=self.run_sim(duration=117,failure_times=[(650,'s1-1')])
        self.assertEqual(s['failure_checkpoint_loss_seconds'],0);self.assertEqual(s['failure_recovery_delay_seconds'],0)
    def test_no_fit_no_fallback(self):
        sim,s,e=self.run_sim(policy='risk_mask_packed',high_from=0)
        self.assertEqual(s['completed_jobs'],0);self.assertEqual(s['excluded_allocations'],0);self.assertEqual(s['starved_jobs'],1)
        self.assertEqual(s['failure_lost_time_seconds'],0)
    def test_group_migration_separate(self):
        sim,s,e=self.run_sim(policy='risk_mask_packed_nonsticky',high_from=1800)
        self.assertEqual(s['migrations'],1);self.assertEqual(s['failure_lost_time_seconds'],0)
        self.assertEqual(s['preventive_checkpoint_seconds'],60);self.assertEqual(s['preventive_relaunch_seconds'],300)
        self.assertTrue(any(r['type']=='cooldown_start' for st,r in e if st=='events'))
    def test_recovery_censored_at_end(self):
        sim,s,e=self.run_sim(failure_times=[(7100,'s0-0')],duration=20000)
        self.assertEqual(s['failure_recovery_delay_seconds'],100)
        self.assertEqual(s['failure_gpu_unavailable_seconds'],100)
    def test_exact_packing_and_seed(self):
        rng=random.Random(7)
        for _ in range(25):
            counts=[rng.randint(1,8) for k in range(5)]
            gs={f'{i}-{k}':ReplayGPU(f'{i}-{k}',str(i),k) for i,c in enumerate(counts) for k in range(c)}
            for demand in range(1,sum(counts)+1):
                picked=PlacementPolicy().choose(demand,gs)
                actual=len({gs[u].server_id for u in picked})
                oracle=min(len(combo) for length in range(1,6) for combo in itertools.combinations(range(5),length) if sum(counts[i] for i in combo)>=demand)
                self.assertEqual(len(picked),demand);self.assertEqual(actual,oracle)
        self.assertEqual(PlacementPolicy('risk_mask_random',17).choose(3,gs),PlacementPolicy('risk_mask_random',17).choose(3,gs))
    def test_invalid_feeds(self):
        path=fixture(self.root/'input');top=TopologyProvider(path.parent/'gpu_manifest.parquet',4)
        file=path.parent/'probability_tape.parquet';original=pd.read_parquet(file)
        for broken in [pd.concat([original.iloc[:1],original]),original.iloc[1:],original.assign(probability=float('nan')),original.assign(probability=1.1),original.iloc[::-1]]:
            broken.to_parquet(file,index=False)
            with self.assertRaises(ValueError):RiskTapeProvider(file,top)
    def test_causal_provider(self):
        path=fixture(self.root/'input',failure_times=[(650,'s0-0')]);m,t,r,f,j,h=load_inputs(path)
        with self.assertRaises(ValueError):r.snapshot(START.timestamp()-1)
        self.assertEqual(f.events_at(START.timestamp()+649),[])
        self.assertEqual(len(f.events_at(START.timestamp()+650)),1)
        self.assertEqual(f.events_at(START.timestamp()+651),[]);r.close()
    def test_duplicate_loss_rejected(self):
        ledger=LossLedger();ledger.add('a','loss',1)
        with self.assertRaises(ValueError):ledger.add('a','loss',1)
    def test_deterministic_cli_engine_100_gpu(self):
        path=fixture(self.root/'input',n=100,failure_times=[(650,'s0-0')])
        a=execute(path,self.root/'a');b=execute(path,self.root/'b');self.assertEqual(a,b)
        for name in ['events.jsonl','decisions.jsonl','summary.json']:
            self.assertEqual((self.root/'a'/name).read_bytes(),(self.root/'b'/name).read_bytes())
    def test_background_worker_and_cursor(self):
        path=fixture(self.root/'input',n=16,duration=117)
        jobs=pd.read_parquet(path.parent/'job_trace.parquet')
        clones=[jobs.assign(job_id=f'j{i}',gang_id=f'g{i}') for i in range(8)]
        pd.concat(clones).to_parquet(path.parent/'job_trace.parquet',index=False)
        manifest=json.loads(path.read_text(encoding='utf-8'));manifest['inputs']['job_trace']['sha256']=sha256(path.parent/'job_trace.parquet');atomic_json(path,manifest)
        server=make_server(path,self.root/'runs',0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            req=urllib.request.Request(base+'/runs',data=json.dumps({'policy':'risk_blind_packed','scheduler':'Fifo'}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req) as response:self.assertEqual(response.status,202);ident=json.load(response)['run_id']
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                with urllib.request.urlopen(base+'/runs/'+ident) as response:state=json.load(response)
                if state['status'] in ('completed','failed'):break
                time.sleep(.2)
            self.assertEqual(state['status'],'completed',state)
            with urllib.request.urlopen(base+'/runs/'+ident+'/events?cursor=0') as response:events=json.load(response)
            self.assertTrue(events['events']);self.assertGreater(events['cursor'],0)
            direct=execute(path,self.root/'direct');self.assertEqual(state['summary'],direct)
            for name in ['events.jsonl','decisions.jsonl']:
                self.assertEqual((self.root/'direct'/name).read_bytes(),(self.root/'runs'/ident/name).read_bytes())
        finally:
            server.shutdown();server.server_close();thread.join()
            for proc in server.controller.processes.values():proc.wait(timeout=10)


if __name__=='__main__':unittest.main()

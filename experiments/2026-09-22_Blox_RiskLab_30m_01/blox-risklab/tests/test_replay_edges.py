import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from test_actual_replay import fixture,START,ROOT
from risklab.feeds import load_inputs,sha256
from risklab.replay_runner import execute,atomic_json
from risklab.replay_engine import ActualRiskLabSimulation,NativeScheduler,ReplayJob,ReplayGPU
from risklab.replay_web import ReplayController
from risklab.thresholds import select_threshold
from risklab.placement_policy import PlacementPolicy
from risklab.prepare_replay_inputs import build_failure_tape


class EdgeTests(unittest.TestCase):
    def test_threshold_highest_and_no_test_access(self):
        scores=[.2,.3,.3,.8,.9];labels=[1,1,1,1,0]
        r=select_threshold(scores,labels,.75)
        self.assertEqual(r['threshold'],.3);self.assertEqual(r['recall'],.75)
        self.assertEqual(select_threshold([.1],[0])['status'],'infeasible')
        with self.assertRaises(ValueError):select_threshold(scores,labels,split='test')
    def test_probability_tiebreak_is_sensitivity_only(self):
        g={'a':ReplayGPU('a','a',0,probability=.8),'b':ReplayGPU('b','b',0,probability=.1)}
        self.assertEqual(PlacementPolicy('risk_mask_packed').choose(1,g),['a'])
        self.assertEqual(PlacementPolicy('risk_mask_probability_tiebreak').choose(1,g),['b'])
    def test_real_scheduler_order(self):
        jobs={'old':ReplayJob('old','old',0,100,1,300,state='running',attained_service=5000),'new':ReplayJob('new','new',100,100,1,300,state='queued',attained_service=0)}
        self.assertEqual(NativeScheduler(ROOT.parent/'blox-main','Fifo').order(jobs,{}),['old','new'])
        self.assertEqual(NativeScheduler(ROOT.parent/'blox-main','Las').order(jobs,{}),['new','old'])
        self.assertEqual(NativeScheduler(ROOT.parent/'blox-main','Tiresias2Q').order(jobs,{}),['new','old'])
    def test_episode_gap_and_persistent_observations(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'xid.csv'
            pd.DataFrame({'Time':[START+pd.Timedelta(seconds=s) for s in [15,30,300,615,1815]],'s0-0':[31,31,0,43,31]}).to_csv(raw,index=False)
            f=build_failure_tape(raw,['s0-0'],START,START+pd.Timedelta(hours=1))
            self.assertEqual(len(f),2);self.assertEqual(f.iloc[0].timestamp,START+pd.Timedelta(seconds=15));self.assertEqual(json.loads(f.iloc[0].xid_codes),[31,43])
    def test_stop_and_failed_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=fixture(root/'inputs');out=root/'stopped';out.mkdir();(out/'stop.request').touch()
            result=execute(manifest,out);self.assertEqual(result['status'],'stopped');self.assertEqual(result['decision_count'],0)
            bad=json.loads(manifest.read_text(encoding='utf-8'));bad['inputs']['probability_tape']['sha256']='bad';atomic_json(manifest,bad)
            with self.assertRaises(ValueError):execute(manifest,root/'failed')
            self.assertEqual(json.loads((root/'failed/status.json').read_text(encoding='utf-8'))['status'],'failed')
            self.assertTrue((root/'failed/error.txt').exists())
    def test_worker_crash_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=fixture(root/'inputs');bad=json.loads(manifest.read_text(encoding='utf-8'));bad['inputs']['failure_tape']['sha256']='bad';atomic_json(manifest,bad)
            c=ReplayController(manifest,root/'runs');r=c.start({});c.processes[r['run_id']].wait(timeout=20)
            self.assertEqual(c.status(r['run_id'])['status'],'failed')
    def test_scheduler_preemption_is_not_failure_loss(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=fixture(root/'inputs',n=2,duration=20000)
            jobs=pd.read_parquet(path.parent/'job_trace.parquet');new=jobs.copy();new['job_id']='new';new['gang_id']='new';new['arrival_time']=START+pd.Timedelta(seconds=100);new['duration_seconds']=50
            pd.concat([jobs,new]).to_parquet(path.parent/'job_trace.parquet',index=False)
            m=json.loads(path.read_text(encoding='utf-8'));m['scheduler']='Las';m['inputs']['job_trace']['sha256']=sha256(path.parent/'job_trace.parquet');atomic_json(path,m)
            result=execute(path,root/'out')
            self.assertEqual(result['failure_lost_time_seconds'],0);self.assertGreater(result['scheduling_checkpoint_seconds'],0)
            events=(root/'out/events.jsonl').read_text(encoding='utf-8');self.assertIn('scheduler_preemption',events)


if __name__=='__main__':unittest.main()

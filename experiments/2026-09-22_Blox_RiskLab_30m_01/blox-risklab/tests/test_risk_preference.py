import itertools,json,random,tempfile,unittest
from pathlib import Path
from test_actual_replay import fixture
from risklab.placement_policy import PlacementPolicy
from risklab.replay_engine import ReplayGPU
from risklab.replay_runner import execute
from risklab.replay_web import ReplayController

class RiskPreferenceTests(unittest.TestCase):
 def test_exact_server_and_score_oracle(self):
  rng=random.Random(19)
  for trial in range(15):
   gs={f's{i//2}-{i%2}':ReplayGPU(f's{i//2}-{i%2}',f's{i//2}',i%2,probability=rng.random(),eligible=False) for i in range(8)}
   gs['s3-1'].state='maintenance'
   for demand in range(1,8):
    ids=PlacementPolicy('risk_prefer_packed').choose(demand,gs)
    oracle=min((len({gs[u].server_id for u in combo}),sum(gs[u].probability for u in combo)) for combo in itertools.combinations([u for u,g in gs.items() if g.state=='available'],demand))
    self.assertEqual(len(ids),demand);self.assertEqual(len({gs[u].server_id for u in ids}),oracle[0]);self.assertAlmostEqual(sum(gs[u].probability for u in ids),oracle[1]);self.assertNotIn('s3-1',ids)
 def test_high_scores_allow_work_without_preventive_drain(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);m=fixture(root/'in',policy='risk_prefer_packed',high_from=0)
   result=execute(m,root/'out');self.assertEqual(result['completed_jobs'],1);self.assertEqual(result['preventive_overhead_seconds'],0);self.assertEqual(result['migrations'],0)
   events=[json.loads(x) for x in (root/'out/events.jsonl').read_text(encoding='utf-8').splitlines()]
   self.assertFalse(any(e['type']=='gpu_drained' for e in events))
   complete=next(e for e in events if e['type']=='job_complete');self.assertEqual(complete['server_ids'],['s0']);self.assertEqual(len(complete['gpu_ids']),2)
   status=json.loads((root/'out/status.json').read_text(encoding='utf-8'));self.assertEqual(status['timestamp'],'2023-08-05T08:45:00+00:00');self.assertTrue(all(g['risk_alert'] and g['eligible'] for g in status['gpus']))
 def test_real_failure_still_interrupts_and_recovers(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);m=fixture(root/'in',policy='risk_prefer_packed',high_from=0,failure_times=[(650,'s0-0')]);s=execute(m,root/'out')
   self.assertGreater(s['failure_lost_time_seconds'],0);self.assertEqual(s['preventive_overhead_seconds'],0)
 def test_bounded_latest_event_tail(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);out=root/'run_test';out.mkdir();p=out/'events.jsonl'
   p.write_text(''.join(json.dumps({'event_id':i,'timestamp':'2023-08-05T06:45:00Z'})+'\n' for i in range(1000)),encoding='utf-8')
   r=ReplayController(root/'missing',root).events('run_test',0,tail=True)
   self.assertEqual(len(r['events']),100);self.assertEqual(r['events'][0]['event_id'],900);self.assertEqual(r['events'][-1]['event_id'],999)

if __name__=='__main__':unittest.main()

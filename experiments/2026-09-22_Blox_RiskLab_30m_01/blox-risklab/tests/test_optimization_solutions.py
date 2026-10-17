import json
import tempfile
import unittest
import pandas as pd
from pathlib import Path
from test_actual_replay import fixture
from risklab.feeds import sha256
from risklab.placement_policy import PlacementPolicy, compute_job_vulnerability
from risklab.replay_engine import ReplayGPU, ReplayJob
from risklab.replay_runner import execute, atomic_json


class OptimizationSolutionTests(unittest.TestCase):
    def test_vulnerability_weight_prioritizes_large_long_gang_jobs(self):
        short_1gpu = ReplayJob('j_short', 'j_short', 0., 300., 1, 300.)
        long_8gpu = ReplayJob('j_long', 'j_long', 0., 14400., 8, 300.)
        v_short = compute_job_vulnerability(short_1gpu)
        v_long = compute_job_vulnerability(long_8gpu)
        self.assertGreater(v_long, v_short * 20.0)

    def test_batch_policies_place_vulnerable_gang_on_safer_server(self):
        # Server s0 has low risk (0.01), Server s1 has high risk (0.80)
        gpus = {
            's0-0': ReplayGPU('s0-0', 's0', 0, probability=0.01, b1_probability=0.01, b2_probability=0.01),
            's0-1': ReplayGPU('s0-1', 's0', 1, probability=0.01, b1_probability=0.01, b2_probability=0.01),
            's1-0': ReplayGPU('s1-0', 's1', 0, probability=0.80, b1_probability=0.85, b2_probability=0.75),
            's1-1': ReplayGPU('s1-1', 's1', 1, probability=0.80, b1_probability=0.85, b2_probability=0.75),
        }
        # Job 1 arrives first in queue (1 GPU, short 100s), Job 2 is second in queue (2 GPUs, long 7200s)
        j_tiny = ReplayJob('j1_tiny', 'j1_tiny', 0., 100., 1, 300., state='queued', queued_since=0.)
        j_gang = ReplayJob('j2_gang', 'j2_gang', 1., 7200., 2, 300., state='queued', queued_since=1.)

        for pol_name in ('vuln_weighted_packed', 'metaheuristic_packed', 'full_co_optimized'):
            pol = PlacementPolicy(pol_name, seed=17)
            batch = pol.choose_batch([j_tiny, j_gang], gpus)
            self.assertIn('j1_tiny', batch)
            self.assertIn('j2_gang', batch)
            # The 2-GPU long Gang job MUST get the safe server s0 (['s0-0', 's0-1']),
            # while the 1-GPU tiny job acts as shock absorber on s1!
            self.assertEqual(sorted(batch['j2_gang']), ['s0-0', 's0-1'])
            self.assertIn(batch['j1_tiny'][0], ['s1-0', 's1-1'])

    def test_solution_b_differentiated_xid43_vs_xid31(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # 1) XID 31 in fixture -> triggers full 1200s hardware maintenance even in differentiated mode
            m31 = fixture(root / 'in31', policy='full_co_optimized', high_from=0, failure_times=[(650, 's0-0')])
            res_xid31 = execute(m31, root / 'out31', policy='full_co_optimized', xid_severity_mode='differentiated_43_31')
            self.assertEqual(res_xid31['failure_gpu_unavailable_seconds'], 1200.0)
            self.assertEqual(res_xid31['hard_xid_maintenances'], 1)

            # 2) XID 43 -> skips 1200s hardware maintenance and applies 60s soft reset in differentiated mode
            m43 = fixture(root / 'in43', policy='full_co_optimized', high_from=0, failure_times=[(650, 's0-0')])
            ft_path = root / 'in43' / 'failure_tape.parquet'
            ft = pd.read_parquet(ft_path)
            ft['xid_codes'] = '[43]'
            ft.to_parquet(ft_path, index=False)
            manifest = json.loads(m43.read_text(encoding='utf-8'))
            manifest['inputs']['failure_tape']['sha256'] = sha256(ft_path)
            atomic_json(m43, manifest)

            res_uniform = execute(m43, root / 'out43_uniform', policy='full_co_optimized', xid_severity_mode='uniform_hardware')
            res_diff = execute(m43, root / 'out43_diff', policy='full_co_optimized', xid_severity_mode='differentiated_43_31', xid43_interrupt_prob=1.0)
            self.assertEqual(res_uniform['failure_gpu_unavailable_seconds'], 1200.0)
            self.assertEqual(res_diff['failure_gpu_unavailable_seconds'], 0.0)
            self.assertEqual(res_diff['soft_xid_reset_seconds'], 60.0)
            self.assertEqual(res_diff['soft_xid_resets'], 1)


if __name__ == '__main__':
    unittest.main()

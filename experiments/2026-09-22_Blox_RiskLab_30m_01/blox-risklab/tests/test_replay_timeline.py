import tempfile
import unittest
from pathlib import Path
import pandas as pd
from test_actual_replay import fixture
from risklab.replay_runner import execute


class TimelineTests(unittest.TestCase):
    def test_gpu_time_conservation_and_queue_integral(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=fixture(root/'input',n=100,failure_times=[(650,'s0-0')])
            summary=execute(path,root/'out')
            gpu=pd.read_parquet(root/'out/gpu_timeline.parquet');jobs=pd.read_parquet(root/'out/job_timeline.parquet')
            metrics=pd.read_parquet(root/'out/metrics.parquet')
            self.assertEqual(len(metrics),summary['decision_count'])
            self.assertEqual(list(metrics.time_seconds),[0,1800,3600,5400])
            self.assertTrue((metrics.filter(regex='^gpus_').sum(axis=1)==100).all())
            durations=gpu.end_seconds-gpu.start_seconds
            self.assertEqual(durations.sum(),7200*100)
            for uid,rows in gpu.groupby('gpu_uid'):
                self.assertEqual(rows.iloc[0].start_seconds,0);self.assertEqual(rows.iloc[-1].end_seconds,7200)
                self.assertEqual(list(rows.end_seconds.iloc[:-1]),list(rows.start_seconds.iloc[1:]))
            waits=jobs[jobs.state=='queued'];self.assertEqual((waits.end_seconds-waits.start_seconds).sum(),summary['avg_queue_wait_seconds'])


if __name__=='__main__':unittest.main()

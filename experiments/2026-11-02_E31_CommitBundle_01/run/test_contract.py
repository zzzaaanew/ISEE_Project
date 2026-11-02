from __future__ import annotations

import contextlib
import ast
import ctypes
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from source_bridge import FastGridFeatureStore,basic30,apply_kl_inertia_chain,new_controller,update_adst,feature_names
from run_experiment import train_interval,flush_feedback,model_b1,models_b2,close_feature_maps
from evaluate_scores import inertia_stepwise
from supervise import persist
import finish_handoff

ROOT=Path(__file__).resolve().parent
FIXTURES=ROOT/"attempts/resume_01/test_fixtures"


def engine(matrices,times=None):
    B,G=matrices["util"].shape
    nodes={}
    for m in ["util","temp","power"]:
        v=matrices[m].reshape(B,G//8,8)
        cnt=np.isfinite(v).sum(axis=2)
        nodes[m]=np.divide(np.nansum(v,axis=2),cnt,out=np.full(cnt.shape,np.nan,dtype=np.float32),where=cnt>0)
    return SimpleNamespace(num_bins=B,num_gpus=G,num_nodes=G//8,matrices=matrices,node_matrices=nodes,
                           bin_start_ns=np.arange(B,dtype=np.int64)*300_000_000_000 if times is None else times)


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FIXTURES.mkdir(exist_ok=True)

    @unittest.skipUnless(os.name=="nt","Windows file sharing regression")
    def test_windows_monitor_lock_old_fails_new_retries(self):
        from ctypes import wintypes
        kernel=ctypes.windll.kernel32
        kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
                                    wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        original=ast.parse((ROOT/"attempts/resume_01/before/supervise.py").read_text(encoding="utf-8"))
        node=next(x for x in original.body if isinstance(x,ast.FunctionDef) and x.name=="persist")
        env={"json":json}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"old_supervise.persist","exec"),env)
        with tempfile.TemporaryDirectory(dir=FIXTURES) as directory:
            target=Path(directory)/"monitor.json"
            target.write_text('{"version":0}',encoding="utf-8")
            def lock():
                handle=kernel.CreateFileW(str(target),0x80000000,3,None,3,0,None)
                self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
                return handle
            handle=lock()
            try:
                with self.assertRaises(PermissionError):
                    env["persist"](target,{"version":1})
            finally:
                kernel.CloseHandle(handle)
            handle=lock()
            thread=threading.Thread(target=lambda:(time.sleep(.15),kernel.CloseHandle(handle)))
            thread.start()
            try:
                self.assertTrue(persist(target,{"version":2}))
            finally:
                thread.join()
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["version"],2)

    def test_noncritical_monitor_failure_does_not_abort(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as directory:
            target=Path(directory)/"monitor.json"
            with patch.object(Path,"replace",side_effect=PermissionError("simulated share lock")),patch("supervise.time.sleep"):
                self.assertFalse(persist(target,{"status":"RUNNING"},critical=False))

    def test_mmap_release_preserves_values(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as directory:
            target=Path(directory)/"features.npy"
            expected=np.random.default_rng(42).normal(size=(72,16,53)).astype(np.float32)
            np.save(target,expected)
            mapped=np.load(target,mmap_mode="r")
            copied=mapped[20:30].copy()
            close_feature_maps(mapped)
            self.assertTrue(mapped._mmap.closed)
            reopened=np.load(target,mmap_mode="r")
            np.testing.assert_array_equal(reopened[20:30],copied)
            np.testing.assert_array_equal(reopened,expected)
            close_feature_maps(reopened)

    def test_research_methods_and_config_unchanged(self):
        def nodes(path):
            return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(x,ast.FunctionDef)}
        old=nodes(ROOT/"attempts/resume_01/before/run_experiment.py")
        new=nodes(ROOT/"run_experiment.py")
        for name in ["model_b1","models_b2","sampled","train_interval","flush_feedback","adst_feedback",
                     "schedule","score_models","resource_check"]:
            self.assertEqual(old[name],new[name],name)
        self.assertEqual((ROOT/"config.json").read_bytes(),(ROOT/"attempts/resume_01/before/config.json").read_bytes())

    def test_resume_completion_preserves_old_failure_record(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as directory:
            target=Path(directory)/"record.md"
            target.write_text("## recent\n<!-- E31_COMPLETION_HANDOFF -->\nold failure\n",encoding="utf-8")
            with patch.object(finish_handoff,"MARKER","<!-- E31_COMPLETION_HANDOFF_resume_01 -->"):
                finish_handoff.insert_text(target,"## recent","<!-- E31_COMPLETION_HANDOFF_resume_01 -->\nresume")
            text=target.read_text(encoding="utf-8")
            self.assertIn("old failure",text)
            self.assertIn("resume",text)

    def test_chunked_features_equal_frozen_full_grid(self):
        rng=np.random.default_rng(7)
        mats={m:rng.uniform(1,80,(400,16)).astype(np.float32) for m in ["util","temp","power","fb"]}
        mats["temp"][55:59,2]=np.nan
        full=engine(mats)
        with contextlib.redirect_stdout(io.StringIO()):
            original=FastGridFeatureStore(full,{})
            for start,end in [(75,144),(144,288),(288,400)]:
                lo=start-75
                chunk=engine({m:v[lo:end] for m,v in mats.items()},full.bin_start_ns[lo:end])
                bounded=FastGridFeatureStore(chunk,{})
                bins=np.repeat(np.arange(start,end),16);gpus=np.tile(np.arange(16),end-start)
                np.testing.assert_allclose(original.extract_b1(bins,gpus,6),bounded.extract_b1(bins-lo,gpus,6),rtol=3e-5,atol=3e-5)
                np.testing.assert_allclose(basic30(original,bins,gpus,6),basic30(bounded,bins-lo,gpus,6),rtol=3e-5,atol=3e-5)
                np.testing.assert_array_equal(original.extract_b2(bins,gpus),bounded.extract_b2(bins-lo,gpus))

    def test_basic30_matches_direct_windows(self):
        rng=np.random.default_rng(8)
        mats={m:rng.uniform(1,80,(180,16)).astype(np.float32) for m in ["util","temp","power","fb"]}
        e=engine(mats)
        bins=np.array([80,100,170]);gpus=np.array([0,7,15])
        with contextlib.redirect_stdout(io.StringIO()):
            actual=basic30(FastGridFeatureStore(e,{}),bins,gpus,6)
        for i,(b,g) in enumerate(zip(bins,gpus)):
            for mi,m in enumerate(["util","temp","power","fb"]):
                w=mats[m][b-74:b-2,g]
                expected=[w[-1],w.mean(),w.max(),w.min(),w.std(),w[-1]-w.mean()]
                np.testing.assert_allclose(actual[i,mi*6:mi*6+6],expected,rtol=3e-5,atol=3e-5)
        self.assertEqual(len(feature_names()["falcon45"]),45)
        self.assertEqual(len(feature_names()["basic30"]),30)

    def test_inertia_stream_causality_and_legacy_parity(self):
        p=np.random.default_rng(1).uniform(.001,.2,(100,16)).astype(np.float32)
        for minutes in [5,30]:
            expected=apply_kl_inertia_chain(p,minutes,2.,1.)
            a,state=inertia_stepwise(p[:44],minutes,1.,2.)
            b,state=inertia_stepwise(p[44:],minutes,1.,2.,state)
            np.testing.assert_array_equal(np.vstack([a,b]),expected)
            changed=p.copy();changed[50:]=.99
            q,_=inertia_stepwise(changed,minutes,1.,2.)
            np.testing.assert_array_equal(q[:50],expected[:50])

    def test_purge_and_feedback_boundaries(self):
        config=dict(purge_hours=36,horizon_hours=24,observation_hours=6)
        start,end=train_interval(9000,21,config)
        self.assertEqual(end,9000-432)
        self.assertEqual(end-start,21*288)
        c=new_controller()
        pending=[dict(origin=8000,latest_score_bin=8500,train_days=14,ap=.04,prevalence=.01)]
        pending=flush_feedback(c,pending,8931,config)
        self.assertEqual(len(c.adst_history),0)
        pending=flush_feedback(c,pending,8932,config)
        self.assertEqual(len(c.adst_history),1)
        self.assertEqual(len(pending),0)
        self.assertLessEqual(c.adst_history[0]["feedback_label_maturity_bin"],8932)

    def test_controller_actually_changes_window(self):
        c=new_controller()
        for i in range(3):
            update_adst(c,.08,.01,dict(origin=i,selected=dict(selected_train_days=c.current_train_days)))
        self.assertEqual(c.current_train_days,21)
        for i in range(3,12):
            update_adst(c,.001,.01,dict(origin=i,selected=dict(selected_train_days=c.current_train_days)))
        self.assertLess(c.current_train_days,21)

    def test_classifiers_fit_and_predict(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(400,5)); y=(x[:,0]+x[:,1]>1).astype(np.uint8)
        with threadpool_limits(limits=2):
            for m in [model_b1(),*models_b2()]:
                m.fit(x,y)
                p=m.predict_proba(x[:8])[:,1]
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p>=0)&(p<=1)).all())


if __name__=="__main__":
    unittest.main(verbosity=2)

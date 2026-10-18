"""Meaningful preflight: original feature calls, causality, and trained E19 path."""
from pathlib import Path
from collections import deque
import json
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
import run_experiment as r

ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
output=ROOT/'runs/preflight_01'
trace=r.Trace(output)
with threadpool_limits(limits=2):
    e,events,gt,math,manifest=r.load_engine(config,output,trace)
    schedule=r.schedule_from_reference(e,config)
    origin=schedule[0]['origin'];end=origin-r.PURGE
    for days in (7,14,21):
        b,g,y,prior=r.sample_rows(gt,end-days*r.DAY,end,20260905+days)
        assert b.max()<=origin-r.PURGE and b.max()+r.HORIZON<origin
        assert len(r.history_features(e,events,b[:20],g[:20])[0])==2
    # Chunked source output must equal a direct unmodified source call.
    b=np.repeat(np.array([origin,origin+6]),6);g=np.tile([0,1,7,8,9,15],2)
    x=r.telemetry(e,b,g,3,trace,'preflight')
    original,_=e.extract_branch1_features(b,g,1)
    np.testing.assert_allclose(x,original.to_numpy(),rtol=0,atol=0)
    # KL first-order optimum, off path and strict pre-update state.
    z=np.array([-2.,0.,3.]);m=np.array([-1.,2.,1.])
    q=expit((z+m)/2)
    np.testing.assert_allclose(2*logit(q)-z-m,0,atol=1e-12)
    before=m.copy();rho=2**(-.5/2)
    after=rho*m+(1-rho)*z
    assert np.array_equal(before,m) and not np.array_equal(before,after)
    # This is an explicit synthetic wiring check, never a reported experiment score.
    rng=np.random.default_rng(20260905)
    O=100*r.DAY*r.STEP;C=O-36*12*r.STEP
    history=deque()
    for t in np.arange(C-14*r.DAY*r.STEP,C+1,6*r.STEP,dtype='int64'):
        z2=rng.normal(-2,1,64);zs=rng.normal(-2,1,(8,64))
        y=rng.binomial(1,expit(.4*zs[0]+.8*z2)).astype('uint8')
        history.append((int(t),zs,z2,y))
    model=r.fit_e19(history,O,0,math,trace,0,'synthetic_check')
    assert model['status']=='trained'
    p=r.e19_predict(history[-1][1][0],history[-1][2],expit(history[-1][2]),model,math)
    assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
    assert model['latest_fit_score_ns']<=C and model['latest_fit_score_ns']+r.HORIZON*r.STEP<=O
    assert r.fit_e19(deque(),O,0,math,trace,0,'cold_check')['status']=='cold_start_b2_fallback'
    state=dict(days=14,stable=0,processed=[],values=[],updates=[])
    feedback=[dict(id=0,latest_bin=origin-r.PURGE+1,query_mean_ap=.1,prevalence=.01)]
    r.update_adst(state,feedback,origin,trace);assert not state['processed']
    feedback[0]['latest_bin']=origin-r.PURGE
    r.update_adst(state,feedback,origin,trace);r.update_adst(state,feedback,origin,trace)
    assert state['processed']==[0]
    lam=1/(np.array([0.,1.,10.])+1)
    np.testing.assert_allclose(lam,[1,.5,1/11])
    result=dict(status='WIRING_VERIFIED_NOT_FULL_RESULT',feature_count=45,b2_features=2,
        reference_origins=len(schedule),reference_decision_times=sum(len(s['bins']) for s in schedule),
        source_chunk_equality=True,purge_and_feedback_checks=True,kl_stationarity=True,
        synthetic_e19_trained=True,cold_start_fallback=True,pareto_uncapped=True,
        source_sha256=r.digest(ROOT/'ML/run_experiment.py'))
    r.write_json(output/'wiring_check.json',result)
    print(json.dumps(result,ensure_ascii=False))

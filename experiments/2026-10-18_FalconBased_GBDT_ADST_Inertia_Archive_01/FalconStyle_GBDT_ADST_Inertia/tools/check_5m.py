"""Small preflight checks; never runs the conditional experiment before its gate."""
import ast
import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ML'))
import run_experiment as r
import run_5m as five
import conditional_5m as watcher

def main():
    for file in ['ML/run_5m.py','tools/audit_5m.py','tools/conditional_5m.py']:
        ast.parse((ROOT/file).read_text(encoding='utf-8'))
    source=ROOT/'runs/full_01'
    contract=json.loads((source/'execution_contract.json').read_text(encoding='utf-8'))
    assert r.digest(ROOT/'ML/run_experiment.py')==contract['source_sha256']
    config=contract['config']
    with tempfile.TemporaryDirectory(prefix='five_minute_preflight_') as temp:
        engine,events,gt,math,manifest=r.load_engine(config,Path(temp),r.Trace(Path(temp)))
        old=r.schedule_from_reference(engine,config);new=five.expand_schedule(old)
        assert sum(len(s['bins']) for s in new)==18313
        assert sum(len(s['bins']) for s in new)*engine.num_gpus==36479496
        assert np.all(np.diff(np.concatenate([s['bins'] for s in new]))==1)
        assert old[0]['bins'][0]==new[0]['bins'][0] and old[-1]['bins'][-1]==new[-1]['bins'][-1]
        # Tiny CPU diagnostic avoids contention with the running CUDA learner.
        bins=np.repeat([old[0]['origin'],old[0]['origin']+1],8).astype('int32')
        g=np.tile(np.arange(8,dtype='int32'),2)
        x=r.telemetry(engine,bins,g,config['feature_chunk'],r.Trace(Path(temp)),'preflight16')
        model=xgb.XGBClassifier();model.load_model(source/'models/b1_origin_0000_days_14.json');model.set_params(device='cpu',n_jobs=2)
        raw=model.predict_proba(x)[:,1]
        info=json.loads((source/'branches/origin_0000.json').read_text(encoding='utf-8'))['fit_info'][1]
        p=r.natural_probability(raw,info['natural_prior'])
        ref=pq.read_table(source/'branches/origin_0000.parquet',columns=['b1_S14']).column(0).to_numpy()[:8]
        assert np.allclose(p[:8],ref,atol=1e-6,rtol=0)
        rho=2**(-(1/12)/config['inertia_half_life_hours']);assert abs(rho-.9715319411536059)<1e-14
        # Verify frozen-model E19 prediction against an independent formula.
        from scipy.special import expit,logit
        z=math.clipped_logit(np.array([.01,.02,.05]));z2=math.clipped_logit(np.array([.03,.04,.07]))
        fused=dict(status='trained',calibrators=[dict(intercept=.1,slope=.8),dict(intercept=-.1,slope=1.2)],beta=[.1,.2,.3,.4,.5,.6])
        actual=r.e19_predict(z,z2,expit(z2),fused,math)
        a=logit(np.clip(expit(.1+.8*z/6),r.EPS,1-r.EPS))/6
        p2=np.clip(expit(-.1+1.2*z2/6),r.EPS,1-r.EPS);b=logit(p2)/6
        expected=np.clip(expit(logit(p2)+np.column_stack([np.ones(3),a,b,a*a,b*b,a*b])@fused['beta']),r.EPS,1-r.EPS)
        assert np.allclose(actual,expected,atol=1e-12,rtol=0)
    result=dict(status='PREFLIGHT_PASSED',queries_5m=18313,rows_5m=36479496,
        common_time_raw_score_max_difference=float(np.max(abs(p[:8]-ref))),
        regular_5m_rho=rho,diagnostic_rows=16,diagnostic_device='cpu',
        production_device='cuda',conditional_gate_not_bypassed=True,
        verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256={name:r.digest(ROOT/name) for name in ['ML/run_5m.py','tools/audit_5m.py','tools/conditional_5m.py']})
    r.write_json(ROOT/'policies/preflight_5m.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    main()

"""Freeze approved input paths and source snapshots without modifying their sources."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
W=ROOT.parent
v=ROOT/'vendor';v.mkdir(exist_ok=True)
sources=[W/'ISEE_Project_latest_20261017/ML/run_bidirectional_adst_fusion.py',
         W/'ISEE_Project_latest_20261017/ML/branch1_enhanced_features.py',
         W/'E14_E19_plan_20261017/runtime/migration/e19-reproduction-gpu-opt/ML/run_experiment19_30m_dual_inertia_fusion.py']
for source in sources:
    target=v/source.name
    if target.exists() and target.read_bytes()!=source.read_bytes():
        raise ValueError('Frozen source already exists with different content')
    if not target.exists():
        shutil.copy2(source,target)
previous=json.loads((W/'Falcon_AcmeTrace_ST_Top100/config.json').read_text(encoding='utf-8'))
config=dict(data_directory=previous['data_directory'],data_inventory=previous['data_inventory'],
    reference_directory=previous['reference_directory'],
    cache_directory=str(W/'E14_E19_plan_20261017/runtime/migration/e19-reproduction-gpu-opt/outputs/branch1/cache'),
    verified_parquet_cache=str(W/'Falcon_AcmeTrace_ST_Top100/cache'),
    specification=str(W/'실험 명세/FalconStyle_GBDT_ADST_Inertia_명세_v1.md'),
    seed=20260905,cpu_threads=2,feature_chunk=2048,purge_hours=36,label_maturity_hours=24,
    inertia_half_life_hours=2.,inertia_gamma=1.,fusion_cap=None,fusion_floor=None,fusion_alpha=1.,
    b2_feature_set='history2',b2_train_days=21,gbdt_backend='xgboost_cuda')
path=ROOT/'config.json'
if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=config and (ROOT/'runs/full_01/execution_contract.json').exists():
    raise ValueError('Existing experiment config differs')
path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PREPARED',config=str(path),vendor_files=len(sources)),ensure_ascii=False))

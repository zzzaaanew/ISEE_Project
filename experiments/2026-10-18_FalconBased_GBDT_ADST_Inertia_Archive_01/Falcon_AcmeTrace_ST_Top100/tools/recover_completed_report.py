"""Recover reporting after all 64 fits; never fit or alter frozen scores/models."""
from pathlib import Path
import hashlib
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ML'))
from run_falcon_acmetrace_st_top100 import digest, write_json
from record_run import inspect, mean_zero_filled


def independent_metrics(scores, y):
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores')
    labels = y.reshape(-1, 1992)
    ranks = np.argsort(-scores.reshape(-1, 1992), kind='stable', axis=1)[:, :100]
    hits = int(np.take_along_axis(labels, ranks, axis=1).sum())
    positives = int(y.sum())
    return {'pr_auc': float(average_precision_score(y, scores)) if positives else None,
            'hits_at_100': hits, 'positives': positives,
            'recall_at_100': hits/positives if positives else None}


def main():
    started = time.perf_counter()
    output = ROOT/'runs/full_01'
    contract = json.loads((output/'execution_contract.json').read_text(encoding='utf-8'))
    config = contract['config']
    rows = json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    trace = [json.loads(s) for s in (output/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    if [r['origin_index'] for r in rows] != list(range(64)):
        raise ValueError('Not all origins complete')
    expected_source = contract['source_sha256']
    for source in [ROOT/'ML/run_falcon_acmetrace_st_top100.py', output/'provenance/runner_at_launch.py']:
        if digest(source) != expected_source:
            raise ValueError('Executed training source identity changed')
    for name in ['status.json', 'supervisor_status.json']:
        backup = output/(name.removesuffix('.json')+'.before_report_repair.json')
        if not backup.exists():
            backup.write_bytes((output/name).read_bytes())
    # Rebuild labels independently from raw onset ledger and the frozen grid.
    with np.load(config['grid_meta'], allow_pickle=True) as grid:
        gpus = grid['gpu_ids'].astype(str)
        times = grid['bin_start_ns'].astype('int64')
    step = 300_000_000_000
    gpu_index = {gpu: i for i, gpu in enumerate(gpus)}
    gt = np.zeros((len(times), len(gpus)), dtype='uint8')
    ledger = pq.read_table(Path(config['data_directory'])/'xid_onsets_metadata.parquet').to_pandas()
    for event in ledger.itertuples():
        onset_bin = (pd.Timestamp(event.onset_time).value-int(times[0]))//step
        gt[max(0,onset_bin-288):min(len(times),onset_bin+1),gpu_index[event.gpu_id]] = 1
    data_events = [r for r in trace if r['stage']=='data_ready']
    if not data_events or any(d['input_sha256'] != data_events[0]['input_sha256'] for d in data_events):
        raise ValueError('Input identities differ between pilot and continuation')
    if digest(Path(config['data_directory'])/'xid_onsets_metadata.parquet') != data_events[0]['input_sha256']['xid_onsets_metadata.parquet']:
        raise ValueError('Raw onset ledger changed')
    part_evidence = []
    for row in rows:
        name = f'score_part_{row["origin_index"]:04d}'
        part = output/'full_population'/f'{name}.parquet'
        frame = pq.read_table(part).to_pandas()
        if digest(part) != row['score_sha256']:
            raise ValueError('Frozen score changed')
        ns = frame.timestamp.astype('int64').to_numpy()
        bins = (ns-times[0])//step
        if not np.array_equal(ns, np.repeat(np.unique(ns),1992)):
            raise ValueError('Invalid decision ordering')
        if not np.array_equal(frame.gpu_uid.to_numpy(),np.tile(gpus,len(frame)//1992)):
            raise ValueError('Invalid GPU ordering')
        if len(np.unique(ns)) > 1 and not np.all(np.diff(np.unique(ns)) == 6*step):
            raise ValueError('Invalid scoring interval')
        if not np.all(frame.feature_cutoff_time.astype('int64').to_numpy() == ns-2*step):
            raise ValueError('Invalid feature cutoff')
        labels = gt[bins, np.tile(np.arange(1992),len(frame)//1992)]
        if not np.array_equal(labels,frame.label.to_numpy()):
            raise ValueError('Labels differ from independently rebuilt raw-ledger GT')
        fresh = independent_metrics(frame.falcon_risk_score.to_numpy(),labels)
        if any(fresh[k] != row[k] for k in fresh):
            raise ValueError('Fresh Falcon metrics mismatch')
        refpart = Path(config['reference_directory'])/part.name
        ref = pq.read_table(refpart).to_pandas()
        if not np.array_equal(ref.timestamp.astype('int64').to_numpy(),ns) or not np.array_equal(ref.gpu_uid.to_numpy(),frame.gpu_uid.to_numpy()):
            raise ValueError('Reference keys differ')
        for column, saved in row['reference'].items():
            recomputed = independent_metrics(ref[column].to_numpy(),labels)
            if any(recomputed[k] != saved[k] for k in recomputed):
                raise ValueError(f'Reference metrics mismatch {column}')
        modelpath = output/'checkpoints'/f'{name}.model.joblib'
        bundle = joblib.load(modelpath)
        if bundle['backend'] != 'cuda:0' or bundle['model'].get_booster().num_boosted_rounds() != 300 or bundle['imputer'].n_features_in_ != 143:
            raise ValueError('Model checkpoint differs from executed architecture')
        part_evidence.append({'origin_index':row['origin_index'],'score_sha256':row['score_sha256'],
                              'model_sha256':digest(modelpath),'reference_sha256':digest(refpart)})
        del frame,ref,bundle
        if row['origin_index'] % 16 == 15:
            print(f'Independent raw-ledger / score / reference / checkpoint verification {row["origin_index"]+1}/64', flush=True)
    fits = [r for r in trace if r['stage']=='fit_done' and r.get('tag','').startswith('origin_')]
    if {r['tag'] for r in fits} != {f'origin_{i}' for i in range(64)} or len(fits)!=64 or any(r['actual_backend']!='cuda:0' for r in fits):
        raise ValueError('64 fresh CUDA fits not proven')
    origin_done = [r for r in trace if r['stage']=='origin_done']
    pilot = json.loads((ROOT/'runs/pilot_01/audit.json').read_text(encoding='utf-8'))
    continued_seconds = origin_done[-1]['elapsed_seconds']
    total_seconds = pilot['elapsed_seconds']+continued_seconds
    selection = json.loads((output/'selected_learner.json').read_text(encoding='utf-8'))
    summary = {'status':'REPORT_RECOVERED_PENDING_AUDIT','origins_completed':64,'expected_full_origins':64,
               'score_rows':sum(r['rows'] for r in rows),'decision_times':sum(r['decision_times'] for r in rows),
               'selected_learner':selection['selected']['family'],'backend':selection['selected']['backend'],
               'origin_mean_pr_auc':mean_zero_filled([r['pr_auc'] for r in rows]),
               'hits_at_100':sum(r['hits_at_100'] for r in rows),
               'pooled_recall_at_100':sum(r['hits_at_100'] for r in rows)/sum(r['positives'] for r in rows),
               'elapsed_seconds':total_seconds,'pilot_seconds':pilot['elapsed_seconds'],
               'continuation_seconds':continued_seconds,'timing_scope':'pilot plus resumed process; excludes idle handoff, environment setup and report repair',
               'report_repaired':True,'training_rerun':False,
               'aggregation_policy':'64-origin macro; undefined zero-positive metrics zero-filled, matching original E19',
               'reference_origin_mean_pr_auc':{k:mean_zero_filled([r['reference'][k]['pr_auc'] for r in rows]) for k in rows[0]['reference']},
               'fidelity':'All-XID/24h/ST/Top100 adaptation, not full original Falcon'}
    write_json(output/'summary.json',summary)
    evidence = {'status':'INDEPENDENT_INPUT_GT_KEYS_METRICS_CHECKPOINTS_VERIFIED','origins':64,
                'training_source_sha256':expected_source,'parts':part_evidence,
                'gt_policy':'onset bin minus 288 through onset bin inclusive, raw ledger independently rebuilt',
                'input_sha256_at_training':data_events[0]['input_sha256'],
                'cuda_fit_evidence':'All 64 final fit_done records actual_backend cuda:0; saved bundle backend corroborates trace. Loading XGB checkpoint may reset current device, which is not training evidence.',
                'raw_metrics_preserved':True,'training_rerun':False,'utc':pd.Timestamp.now(tz='UTC').isoformat()}
    write_json(output/'independent_verification.json',evidence)
    audit,_ = inspect(output)
    summary['status'] = audit['status']
    write_json(output/'summary.json',summary)
    write_json(output/'status.json',summary)
    supervisor = json.loads((output/'supervisor_status.before_report_repair.json').read_text(encoding='utf-8'))
    supervisor.update(status='COMPLETED_AND_AUDITED_AFTER_REPORT_REPAIR',original_exit_code=supervisor.get('exit_code'),
                      report_repaired=True,training_rerun=False,audit=str(output/'audit.json'),
                      repair_utc=pd.Timestamp.now(tz='UTC').isoformat())
    write_json(output/'supervisor_status.json',supervisor)
    analysis = {'scope':'Same 64 development origins, same raw-ledger GT and keys; no independent heldout',
                'models':{},'timing':{},'paired_comparison':{},'subsets':{},'report_repair':evidence['status']}
    names = ['falcon']+list(rows[0]['reference'])
    positives = np.array([r['positives'] for r in rows])
    ap, hit = {},{}
    for name in names:
        records = [r if name=='falcon' else r['reference'][name] for r in rows]
        ap[name] = np.array([r['pr_auc'] or 0. for r in records])
        hit[name] = np.array([r['hits_at_100'] for r in records])
        analysis['models'][name] = {'origin_mean_ap':float(ap[name].mean()),
            'positive_origin_mean_ap':float(ap[name][positives>0].mean()),
            'hits':int(hit[name].sum()),'pooled_recall_at_100':float(hit[name].sum()/positives.sum()),
            'top100_precision':float(hit[name].sum()/(3053*100))}
    # Ensure the canonical original E19 summary is reproduced, including null convention.
    original = json.loads((Path(config['reference_directory']).parent/'summary.json').read_text(encoding='utf-8'))
    mapping = {'B1':'b1_probability','B2':'b2_probability','Experiment14_Fusion':'experiment14_fusion_probability',
               'no_inertia':'fusion_no_inertia','b1_inertia':'fusion_b1_inertia','b2_inertia':'fusion_b2_inertia','dual_inertia':'fusion_dual_inertia'}
    for old,new in mapping.items():
        current=analysis['models'][new]
        if abs(current['origin_mean_ap']-original[old]['origin_mean_pr_auc'])>1e-12 or current['hits']!=original[old]['hits_at_100']:
            raise ValueError('Original canonical E19 summary not reproduced')
    for name,indices in {'fallback_first16':np.arange(16),'trained_last48':np.arange(16,64)}.items():
        analysis['subsets'][name] = {'origins':len(indices),'positive_rows':int(positives[indices].sum()),
            'models':{n:{'origin_mean_ap':float(ap[n][indices].mean()),'hits':int(hit[n][indices].sum()),
                         'pooled_recall_at_100':float(hit[n][indices].sum()/positives[indices].sum())} for n in names}}
    comparison='fusion_dual_inertia'
    delta=ap['falcon']-ap[comparison]
    paired={'ap_delta_falcon_minus_e19':float(delta.mean()),'ap_relative_change':float(ap['falcon'].mean()/ap[comparison].mean()-1),
            'recall_delta':float((hit['falcon'].sum()-hit[comparison].sum())/positives.sum()),
            'ap_origin_wins':int((delta>0).sum()),'ap_origin_losses':int((delta<0).sum()),'ap_origin_ties':int((delta==0).sum()),
            'recall_origin_wins':int((hit['falcon']>hit[comparison]).sum()),'recall_origin_losses':int((hit['falcon']<hit[comparison]).sum()),
            'recall_origin_ties':int((hit['falcon']==hit[comparison]).sum()),'circular_block_bootstrap':{}}
    rng=np.random.default_rng(20260905)
    for block in [3,7]:
        starts=rng.integers(0,64,size=(10000,int(np.ceil(64/block))))
        indices=((starts[:,:,None]+np.arange(block))%64).reshape(10000,-1)[:,:64]
        ap_delta=delta[indices].mean(axis=1)
        recall_delta=(hit['falcon'][indices].sum(axis=1)-hit[comparison][indices].sum(axis=1))/positives[indices].sum(axis=1)
        paired['circular_block_bootstrap'][str(block)]={'replicates':10000,'ap_delta_percentile95':np.quantile(ap_delta,[.025,.975]).tolist(),
                    'pooled_recall_delta_percentile95':np.quantile(recall_delta,[.025,.975]).tolist(),
                    'scope':'Descriptive development-period uncertainty; overlapping labels and reused dataset preclude independent generalization claim'}
    analysis['paired_comparison']=paired
    selections=[r for r in trace if r['stage']=='fit_done' and r['tag']=='selection']
    features=[r for r in trace if r['stage']=='features' and r['tag'].startswith('origin_')]
    origin_seconds=np.array([r['seconds'] for r in rows])
    analysis['timing']={'end_to_end_compute_seconds':total_seconds,'pilot_seconds':pilot['elapsed_seconds'],
        'continuation_seconds':continued_seconds,'sum64_origin_seconds':float(origin_seconds.sum()),
        'final64_fit_including_imputation_seconds':sum(r['seconds'] for r in fits),
        'final64_feature_seconds':sum(r['seconds'] for r in features),
        'origin_mean_seconds':float(origin_seconds.mean()),'origin_median_seconds':float(np.median(origin_seconds)),
        'origin_min_seconds':float(origin_seconds.min()),'origin_max_seconds':float(origin_seconds.max()),
        'five_candidate_fit_seconds':sum(r['seconds'] for r in selections),
        'candidate_fit_by_family':{r['family']:r['seconds'] for r in selections},
        'note':'fit timer includes CPU median imputation and GPU training; CPU features/serialization/evaluation also present. No matched CPU rerun, so no speedup factor claimed.'}
    analysis['per_origin']=[{'origin_index':r['origin_index'],'positives':r['positives'],
        'falcon_ap':ap['falcon'][i],'e19_dual_ap':ap[comparison][i],
        'falcon_hits':hit['falcon'][i],'e19_hits':hit[comparison][i],'seconds':r['seconds']} for i,r in enumerate(rows)]
    analysis['postprocessing_seconds']=time.perf_counter()-started
    write_json(output/'result_analysis.json',analysis)
    print(json.dumps({k:v for k,v in analysis.items() if k!='per_origin'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()

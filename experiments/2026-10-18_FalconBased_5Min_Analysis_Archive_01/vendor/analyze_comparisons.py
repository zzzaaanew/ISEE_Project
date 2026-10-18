"""Paired descriptive uncertainty and operational metrics on the frozen period."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from record_run import ARMS, BASES, ROOT, write_json

SEED = 20260905


def comparison(frame, left, right, subset='all', replicates=5000):
    a = frame[frame.arm == left].set_index('origin').sort_index()
    b = frame[frame.arm == right].set_index('origin').sort_index()
    common = a.index.intersection(b.index)
    if subset == 'e19_trained':
        common = common[a.loc[common].fusion_status.eq('trained').to_numpy()]
    a, b = a.loc[common], b.loc[common]
    if not len(common):
        return dict(left=left, right=right, subset=subset, origins=[], status='NO_ELIGIBLE_ORIGINS')
    if not np.array_equal(a.positives.to_numpy(), b.positives.to_numpy()):
        raise ValueError('Paired comparisons have different labels/populations')
    delta = a.pr_auc.to_numpy() - b.pr_auc.to_numpy()
    hit_delta = a.hits_at_100.to_numpy() - b.hits_at_100.to_numpy()
    positives = a.positives.to_numpy()
    total = int(positives.sum())
    result = dict(left=left, right=right, subset=subset, origins=[int(v) for v in common],
        origin_count=len(common), mean_ap_delta=float(delta.mean()),
        pooled_recall_delta=float(hit_delta.sum()/total) if total else 0.,
        hits_delta=int(hit_delta.sum()), positive_rows=total,
        ap_wins=int((delta>1e-12).sum()), ap_losses=int((delta < -1e-12).sum()),
        ap_ties=int((np.abs(delta)<=1e-12).sum()), bootstrap={})
    for requested in [3, 7]:
        length = min(requested, len(common))
        rng = np.random.default_rng(SEED)
        starts = rng.integers(0, len(common)-length+1, size=(replicates, int(np.ceil(len(common)/length))))
        indices = (starts[:,:,None] + np.arange(length)).reshape(replicates,-1)[:,:len(common)]
        ap_samples = delta[indices].mean(axis=1)
        denominators = positives[indices].sum(axis=1)
        recall_samples = np.divide(hit_delta[indices].sum(axis=1), denominators,
                                   out=np.zeros(replicates), where=denominators>0)
        result['bootstrap'][str(requested)] = dict(block_length_used=length, replicates=replicates,
            seed=SEED, mean_ap_delta_ci95=np.quantile(ap_samples,[.025,.975]).tolist(),
            pooled_recall_delta_ci95=np.quantile(recall_samples,[.025,.975]).tolist())
    return result


def paired(output):
    frame = pd.read_csv(output/'metrics_by_origin.csv')
    comparisons=[]
    for base in BASES:
        for subset in ['all', 'e19_trained']:
            row = comparison(frame,f'{base}_E',f'{base}_P',subset)
            row['question']='E19_minus_Pareto'
            comparisons.append(row)
    for policy in ['S7','S14','S21','A']:
        for fusion in ['P','E']:
            row=comparison(frame,f'{policy}_I_{fusion}',f'{policy}_0_{fusion}')
            row['question']='Inertia_minus_no_Inertia'
            comparisons.append(row)
    for inertia in ['0','I']:
        for fusion in ['P','E']:
            row=comparison(frame,f'A_{inertia}_{fusion}',f'S21_{inertia}_{fusion}')
            row['question']='ADST_minus_ST21'
            comparisons.append(row)
    result=dict(status='DESCRIPTIVE_PAIRED_ANALYSIS',comparisons=comparisons,
        period='Existing 64-origin development period; not a new independent test set',
        uncertainty='Non-circular moving-block bootstrap on paired ordered origins; descriptive CI, no universal superiority or intervention claim',
        subset_note='E19-trained subset preserves retained origin order. If eligibility has gaps, adjacency is among retained origins.',
        multiple_comparisons='No multiplicity-adjusted hypothesis tests; report all prespecified contrasts')
    write_json(output/'comparison.json',result)
    return result


def operational(output, config):
    parts=sorted((output/'scores').glob('origin_*.parquet'))
    timestamp_sets=[]
    first_population=None
    for part in parts:
        table=pq.read_table(part,columns=['timestamp','gpu_uid'])
        f=table.to_pandas()
        timestamp_sets.extend(f.timestamp.astype('int64').unique().tolist())
        if first_population is None:
            first_population=f.gpu_uid.astype(str).to_numpy()[:1992]
    times=np.array(sorted(set(timestamp_sets)),dtype='int64')
    if len(times)<1:
        raise ValueError('No evaluated decision timestamps')
    step=30*60*10**9
    dense=bool(len(times)==1 or np.all(np.diff(times)==step))
    gpu_to_index={str(gpu):i for i,gpu in enumerate(first_population)}
    xid=pq.read_table(Path(config['data_directory'])/'xid_onsets_metadata.parquet').to_pandas()
    xid['onset_time']=pd.to_datetime(xid.onset_time,utc=True)
    xid['gpu_id']=xid.gpu_id.astype(str)
    original_events=len(xid)
    xid=xid.drop_duplicates(['gpu_id','onset_time']).copy()
    xid['onset_ns']=xid.onset_time.astype('int64')
    known=xid.gpu_id.isin(gpu_to_index)
    window_start=xid.onset_ns.to_numpy()-24*3600*10**9
    within=(window_start>=times[0]) & (xid.onset_ns.to_numpy()<=times[-1])
    eligible=known.to_numpy() & within
    # A discontinuous evaluation period requires complete query coverage in each window.
    if not dense:
        for i,row in enumerate(xid.itertuples()):
            if not eligible[i]:
                continue
            relevant=times[(times>=row.onset_ns-24*3600*10**9)&(times<row.onset_ns)]
            eligible[i]=len(relevant)==48 and np.all(np.diff(relevant)==step)
    events=xid.loc[eligible].copy().reset_index(drop=True)
    event_gpus=events.gpu_id.map(gpu_to_index).to_numpy(dtype='int32')
    event_ns=events.onset_ns.to_numpy(dtype='int64')
    starts=event_ns-24*3600*10**9
    first_hits={arm:np.full(len(events),np.iinfo(np.int64).min,dtype='int64') for arm in ARMS}
    unique={arm:np.zeros(1992,dtype=bool) for arm in ARMS}
    previous={arm:None for arm in ARMS}
    turnover_sums={arm:0. for arm in ARMS}
    turnover_counts={arm:0 for arm in ARMS}
    gap_transitions=0;previous_time=None;decisions=0
    for part in parts:
        columns=['timestamp','gpu_uid']+[f'risk_{arm}' for arm in ARMS]
        frame=pq.read_table(part,columns=columns).to_pandas()
        nt=len(frame)//1992
        current_times=frame.timestamp.astype('int64').to_numpy().reshape(nt,1992)[:,0]
        scores={arm:frame[f'risk_{arm}'].to_numpy().reshape(nt,1992) for arm in ARMS}
        for j,t in enumerate(current_times):
            contiguous=previous_time is not None and t-previous_time==step
            if previous_time is not None and not contiguous:
                gap_transitions+=1
            active=(starts<=t)&(t<event_ns)
            for arm in ARMS:
                selected=np.argsort(-scores[arm][j],kind='stable')[:100]
                unique[arm][selected]=True
                if contiguous:
                    turnover_sums[arm]+=(100-len(np.intersect1d(selected,previous[arm])))/100
                    turnover_counts[arm]+=1
                unseen=first_hits[arm]==np.iinfo(np.int64).min
                candidates=np.flatnonzero(active&unseen)
                if len(candidates):
                    detected=candidates[np.isin(event_gpus[candidates],selected)]
                    first_hits[arm][detected]=t
                previous[arm]=selected
            previous_time=t;decisions+=1
    outcomes={}
    for arm in ARMS:
        found=first_hits[arm]!=np.iinfo(np.int64).min
        lead=(event_ns[found]-first_hits[arm][found])/(3600*10**9)
        outcomes[arm]=dict(mean_top100_turnover=turnover_sums[arm]/turnover_counts[arm] if turnover_counts[arm] else None,
            turnover_transitions=turnover_counts[arm],unique_recommended_gpus=int(unique[arm].sum()),
            total_top100_recommendations=decisions*100,eligible_unique_onsets=len(events),
            captured_unique_onsets=int(found.sum()),onset_capture_rate=float(found.mean()) if len(events) else None,
            mean_earliest_selection_lead_hours=float(lead.mean()) if len(lead) else None,
            median_earliest_selection_lead_hours=float(np.median(lead)) if len(lead) else None)
    result=dict(status='OPERATIONAL_DESCRIPTIVE_METRICS',decision_times=decisions,
        evaluation_start_utc=pd.to_datetime(times[0],utc=True).isoformat(),
        evaluation_end_utc=pd.to_datetime(times[-1],utc=True).isoformat(),
        dense_30min_evaluation=dense,excluded_gap_transitions=gap_transitions,
        raw_xid_rows=original_events,deduplicated_gpu_onsets=len(xid),eligible_unique_onsets=len(events),
        event_definition='Unique (gpu_id,onset_time) across all XID codes; not confirmed hardware failure episodes',
        warning_window='[onset_time-24h,onset_time); full continuous 24h window must lie inside evaluated timestamp range; onset itself excluded',
        lead_definition='onset_time minus earliest Top100 selection inside the warning window, conditional on capture',
        turnover_definition='Fraction of 100 recommendations replaced since immediately preceding 30min query; gaps excluded',
        cautions=['More repeated prediction opportunities can increase any-event capture.',
                  'This onset metric is newly defined and is not claimed identical to earlier E19 operational metrics.',
                  'GPU-time hits and unique onset captures have different denominators.',
                  'No maintenance intervention/counterfactual failure reduction is measured.'],arms=outcomes)
    write_json(output/'operational_metrics.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=args.output.resolve(strict=True)
    audit=json.loads((output/'independent_audit.json').read_text(encoding='utf-8'))
    if audit.get('status')!='SCORE_ARTIFACTS_VERIFIED':
        raise ValueError('Independent score audit must finish before comparison analysis')
    config=json.loads((output/'execution_contract.json').read_text(encoding='utf-8'))['config']
    comparisons=paired(output);operations=operational(output,config)
    print(json.dumps({'comparisons':len(comparisons['comparisons']),
        'eligible_onsets':operations['eligible_unique_onsets'],
        'decision_times':operations['decision_times']},ensure_ascii=False))


if __name__=='__main__':
    main()

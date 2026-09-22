"""Regenerate a fixed-cohort raw Pareto tape without changing the source experiment.

Only local, generated checkpoints are deserialized. No model/grid selection is run.
Thresholds use a time-separated cascade model on development validation only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import run_bidirectional_adst_fusion as base
import run_fusion_diversified_b1_history_b2 as integrated
from branch1_enhanced_features import EnhancedTelemetryEngine
from run_report_faithful_pareto_momentum_fusion import ReportFaithfulParetoMomentumFusion, report_pareto_lambda
from run_pareto_momentum_enhanced_fusion import _pareto_counts


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        pickle.dump(obj, f, protocol=5)
    temp.replace(path)


def cached(path, build):
    if path.exists():
        with path.open('rb') as f:
            return pickle.load(f)
    result = build()
    dump(path, result)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class Exporter(ReportFaithfulParetoMomentumFusion):
    def score(self, b1, cascade, meta, b2, selected, bins, gpus):
        s = self.predict_parallel(b1, bins, gpus, selected['selected_family_weights'], selected['selected_input_weights'])
        tab, _ = self.engine.extract_branch1_features(bins, gpus, 24)
        x = self.stage2_features(tab, s['parallel'], s['disagreement'], s['uncertainty'], s['hardness'])
        p = base.adjusted_probability(cascade.predict_proba(x)[:, 1], meta['stage2_true_prior'], meta['stage2_sample_prior'])
        p1 = np.clip(s['parallel'] + selected['selected_cascade_alpha'] * s['hardness'] * (p-s['parallel']), 0, 1)
        x2 = self.engine.extract_branch2_features(bins, gpus, self.history_map)
        p2 = base.adjusted_probability(b2['model'].predict_proba(x2)[:, 1], b2['true_prior'], b2['sample_prior'])
        weights = np.zeros(len(bins))
        for b in np.unique(bins):
            mask = bins == b
            counts = _pareto_counts(self.history_map, int(self.engine.bin_start_ns[b]), self.engine.num_gpus)
            weights[mask] = report_pareto_lambda(counts[gpus[mask]], .5, .02, 1.)
        return weights*p1 + (1-weights)*p2

    def fixed_cascade(self, records, selected, seed):
        self.attach_parallel(records, selected['selected_family_weights'], selected['selected_input_weights'])
        x = pd.concat([r['stage2_x'] for r in records], ignore_index=True)
        y = np.concatenate([r['labels'] for r in records])
        h = np.concatenate([r['stage1']['hardness'] for r in records])
        model = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=31, l2_regularization=1., class_weight='balanced', random_state=seed)
        model.fit(x, y, sample_weight=.5+h)
        return model, {'stage2_true_prior': float(np.mean([r['stage1']['parallel'].mean() for r in records])), 'stage2_sample_prior': float(y.mean())}

    def export(self, source):
        selected = json.loads((source/'checkpoints/fusion_selection.json').read_text(encoding='utf-8'))
        original = json.loads((source/'experiment_manifest.json').read_text(encoding='utf-8'))
        if original['test_stride_bins'] != 6 or original['pareto']['primary_w_clean'] != .5:
            raise ValueError('Unexpected source experiment contract')
        source_digest = hashlib.sha256((source/'experiment_manifest.json').read_bytes()).hexdigest()
        contract = {'source_sha256': source_digest, 'seed': self.seed, 'decision_interval_seconds': 1800, 'cohort_size': 100, 'version': 1}
        contract_path = self.output_dir/'export_contract.json'
        if contract_path.exists() and json.loads(contract_path.read_text(encoding='utf-8')) != contract:
            raise ValueError('Existing export has a different contract')
        write_json(contract_path, contract)
        cache = self.output_dir/'model_cache'; cache.mkdir(exist_ok=True)
        warmup, test_start, test_end = self.split_bounds()
        cadence = int(self.retrain_cadence_hours*60/base.STEP_MINUTES)
        min_origin = warmup + int(21*integrated.DAY_NS//integrated.STEP_NS) + 2*integrated.PURGE_BINS + integrated.VALIDATION_BLOCKS*integrated.VALIDATION_BINS
        origins = np.arange(min_origin, test_start, cadence, dtype=np.int32)[-integrated.POOL_ORIGINS:]
        records = []
        for oi, origin in enumerate(origins):
            for block in self._selection_blocks(int(origin)):
                bi = int(block['block_idx'])
                path = cache/f'oof_{oi}_{bi}.pkl'
                def build():
                    bins, gpus, labels, start = self._sample_training(int(block['train_end']), int(selected['selected_train_days']), int(origin)+200003*(bi+1))
                    fitted = self._fit_families(bins,gpus,labels,int(block['train_end']),int(selected['selected_train_days']),int(selected['selected_half_life_days']),self.seed+300000+oi*10000+bi*100)
                    stage1 = self.predict_parallel(fitted,block['val_bins'],block['val_gpus'])
                    tab,_ = self.engine.extract_branch1_features(block['val_bins'],block['val_gpus'],24)
                    return {'origin_bin':int(origin),'block_idx':bi,'train_start':start,'train_end':int(block['train_end']),'val_start':int(block['val_start']),'val_end':int(block['val_end']),'labels':block['val_labels'],'bins':block['val_bins'],'gpus':block['val_gpus'],'stage1':stage1,'tabular_24h':tab}
                print(f'OOF {oi+1}/6 block {bi} ({"cached" if path.exists() else "fit"})',flush=True)
                records.append(cached(path,build))
        # Source weights, windows and alpha remain frozen. Threshold validation
        # uses an independently fitted, temporally purged cascade (no in-sample fit).
        val = max(records,key=lambda r:(r['val_end'],r['val_start']))
        old = [r for r in records if r['val_end'] <= val['val_start']-integrated.PURGE_BINS]
        if not old:
            raise ValueError('No time-separated cascade training records for threshold validation')
        def make_validation():
            c,m = self.fixed_cascade(old,selected,self.seed+710001)
            self.attach_parallel([val],selected['selected_family_weights'],selected['selected_input_weights'])
            s=val['stage1']; p=base.adjusted_probability(c.predict_proba(val['stage2_x'])[:,1],m['stage2_true_prior'],m['stage2_sample_prior'])
            p1=np.clip(s['parallel']+selected['selected_cascade_alpha']*s['hardness']*(p-s['parallel']),0,1)
            b2=self._fit_final_b2(int(selected['selected_b2_train_days']),val['train_end']+integrated.PURGE_BINS)
            x2=self.engine.extract_branch2_features(val['bins'],val['gpus'],self.history_map)
            p2=base.adjusted_probability(b2['model'].predict_proba(x2)[:,1],b2['true_prior'],b2['sample_prior'])
            w=np.zeros(len(p1))
            for b in np.unique(val['bins']):
                mask=val['bins']==b
                count=_pareto_counts(self.history_map,int(self.engine.bin_start_ns[b]),self.engine.num_gpus)
                w[mask]=report_pareto_lambda(count[val['gpus'][mask]],.5,.02,1.)
            return pd.DataFrame({'timestamp':pd.to_datetime(self.engine.bin_start_ns[val['bins']],utc=True),'gpu_uid':self.engine.gpu_ids[val['gpus']],'probability':w*p1+(1-w)*p2,'label':val['labels']})
        validation=cached(cache/'threshold_validation.pkl',make_validation)
        validation.to_parquet(self.output_dir/'validation_predictions.parquet',index=False)
        positives=np.sort(validation.loc[validation.label==1,'probability'].to_numpy())
        if not len(positives): raise ValueError('Threshold infeasible: no validation positives')
        sys.path.insert(0,str(base.PARENT_ROOT/'blox_repo_actual/blox-risklab'))
        from risklab.thresholds import select_threshold
        points={str(recall):select_threshold(validation.probability,validation.label,recall) for recall in [.9,.95,.99]}
        write_json(self.output_dir/'threshold_manifest.json',{
            'model_version':source.name,'prediction_horizon':'24h','decision_interval_seconds':1800,'selection_split':'validation_only','target_recall':.99,
            'threshold':points['0.99']['threshold'],'threshold_rule':'highest_threshold_with_validation_recall_at_least_target','sensitivity':points,
            'validation_start':str(validation.timestamp.min()),'validation_end':str(validation.timestamp.max()),'validation_samples':len(validation),'validation_positives':len(positives),
            'validation_population':'source negative-sampled development block; all positive examples retained','cascade_validation':'chronological earlier OOF records with 36h purge; separately fitted cascade',
            'cascade_training_max_time':str(pd.to_datetime(self.engine.bin_start_ns[max(r['val_end'] for r in old)-1],utc=True)),
            'heldout_used_for_selection':False,'source_manifest_sha256':source_digest})
        print('Fitting final frozen B1/B2/cascade',flush=True)
        cascade,meta=cached(cache/'final_cascade.pkl',lambda:self.fixed_cascade(records,selected,self.seed+700002))
        b1=cached(cache/'final_b1.pkl',lambda:self._fit_final_stage1(selected,test_start))
        b2=cached(cache/'final_b2.pkl',lambda:self._fit_final_b2(int(selected['selected_b2_train_days']),test_start))
        # Use the source's lexical GPU ordering, with the real server/local suffix.
        cohort=list(map(str,self.engine.gpu_ids[:100]))
        topology=pd.DataFrame([{'gpu_uid':uid,'server_id':uid.rsplit('-',1)[0],'local_gpu_id':int(uid.rsplit('-',1)[1]),'cohort_index':i} for i,uid in enumerate(cohort)])
        topology['node_index']=pd.factorize(topology.server_id,sort=True)[0]
        topology.to_parquet(self.output_dir/'gpu_manifest.parquet',index=False)
        pieces=[]
        for ci,start in enumerate(range(test_start,test_end,cadence)):
            path=self.output_dir/f'probability_part_{ci:04d}.parquet'
            if not path.exists():
                rows=[]
                for b in range(start,min(start+cadence,test_end),6):
                    # Hardness uses the full-GPU quantile in the source model.
                    # Predict full population before taking the fixed 100.
                    gs=np.arange(self.engine.num_gpus,dtype=np.int32); bs=np.full(len(gs),b,dtype=np.int32)
                    scores=self.score(b1,cascade,meta,b2,selected,bs,gs)
                    rows.append(pd.DataFrame({'timestamp':pd.to_datetime(np.repeat(self.engine.bin_start_ns[b],100),utc=True),'gpu_uid':cohort,'probability':scores[:100],'model_version':source.name}))
                pd.concat(rows,ignore_index=True).to_parquet(path,index=False)
            pieces.append(pd.read_parquet(path)); print(f'Prediction cycle {ci+1}/13',flush=True)
        tape=pd.concat(pieces,ignore_index=True)
        tape.to_parquet(self.output_dir/'probability_tape.parquet',index=False)
        write_json(self.output_dir/'export_complete.json',{'status':'completed',**contract,'rows':len(tape),'timestamps':tape.timestamp.nunique(),'gpu_count':tape.gpu_uid.nunique(),'start':str(tape.timestamp.min()),'end_exclusive':original['test_end_time'],'model_state':'regenerated using frozen source settings; original learned weights were not saved','python':sys.executable,'source_experiment':str(source),'selected':{k:v for k,v in selected.items() if k.startswith('selected_')}})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        base.seed_everything(20260905)
        cache=base.PROJECT_ROOT/'outputs/branch1/cache'
        engine=EnhancedTelemetryEngine(data_dir=base.find_data_dir(),cache_dir=cache,include_topology=False)
        _,history,gt=engine.load_all_xid_ledger()
        runner=Exporter(engine=engine,history_map=history,gt_matrix=gt,output_dir=a.output,retrain_cadence_hours=24,negative_ratio=10,test_stride_bins=6,resume=True,branch2_mode='history_0910',include_topology=False)
        runner.export(a.source.resolve())


if __name__=='__main__': main()

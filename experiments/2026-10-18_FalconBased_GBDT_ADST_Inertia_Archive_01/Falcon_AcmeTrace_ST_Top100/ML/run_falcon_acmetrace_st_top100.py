"""Falcon-inspired early feature fusion, frozen learner selection, 21-day ST.

This is an AcmeTrace adaptation, not a reproduction of Falcon's typed tickets.
No E14/E19 scores enter features, training, or learner selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

STEP = 300_000_000_000
DAY = 288
HORIZON = 288
PURGE = 432
BUFFER = 2
SEED = 20260905
METRICS = ('util', 'temp', 'power', 'fb')
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')
    tmp.replace(path)


class Trace:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()

    def emit(self, stage, **fields):
        row = {'stage': stage, 'utc': pd.Timestamp.now(tz='UTC').isoformat(),
               'elapsed_seconds': time.perf_counter() - self.start,
               'rss_mb': psutil.Process().memory_info().rss / 2**20, **fields}
        with (self.output / 'trace.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
        write_json(self.output / 'status.json', {'status': 'RUNNING', **row})


class DataEngine:
    def __init__(self, config, trace):
        self.config, self.trace = config, trace
        self.data = Path(config['data_directory'])
        self.cache = ROOT / 'cache'
        self.cache.mkdir(exist_ok=True)
        with np.load(config['grid_meta'], allow_pickle=True) as m:
            self.gpus = np.asarray(m['gpu_ids']).astype(str)
            self.times = np.asarray(m['bin_start_ns'], dtype=np.int64)
        if len(self.gpus) != 1992 or not np.all(np.diff(self.times) == STEP):
            raise ValueError('Reference population/grid mismatch')
        self.t0, self.n = int(self.times[0]), len(self.gpus)
        self.means, self.obs, self.input_identity = {}, {}, {}
        inventory = json.loads(Path(config['data_inventory']).read_text(encoding='utf-8'))
        expected = {Path(r['path']).name: r['sha256'] for r in inventory['lfs_objects']
                    if r['path'].endswith('.parquet') and '/data/' in r['path'].replace('\\', '/')}
        for metric in METRICS:
            source = self.data / f'telemetry_5m_{metric}.parquet'
            sha = digest(source)
            if sha != expected[source.name]:
                raise ValueError(f'Input hash mismatch: {source.name}')
            self.input_identity[source.name] = sha
            identity_path = self.cache / f'{metric}.identity.json'
            mean_path, obs_path = self.cache / f'{metric}.mean.npy', self.cache / f'{metric}.obs.npy'
            cached = (identity_path.exists() and mean_path.exists() and obs_path.exists()
                      and json.loads(identity_path.read_text())['source_sha256'] == sha)
            if not cached:
                self.trace.emit('cache_build_start', metric=metric)
                shape = (len(self.times), self.n)
                mean = np.lib.format.open_memmap(mean_path, mode='w+', dtype='float32', shape=shape)
                obs = np.lib.format.open_memmap(obs_path, mode='w+', dtype='float32', shape=shape)
                mean[:] = np.nan
                obs[:] = np.nan
                pf = pq.ParquetFile(source)
                rows = 0
                for batch in pf.iter_batches(batch_size=1_000_000,
                          columns=['Time_5m', 'gpu_id', f'{metric}_mean', f'{metric}_obs']):
                    ns = batch.column(0).cast(pa.int64()).to_numpy()
                    b = ((ns - self.t0) // STEP).astype(np.int32)
                    g = pc.index_in(batch.column(1), value_set=pa.array(self.gpus)).to_numpy()
                    if np.isnan(g).any() or (b < 0).any() or (b >= len(self.times)).any():
                        raise ValueError('Unmapped/out-of-range Parquet row')
                    g = g.astype(np.int32)
                    mean[b, g] = batch.column(2).to_numpy().astype('float32')
                    obs[b, g] = batch.column(3).to_numpy().astype('float32')
                    rows += len(batch)
                mean.flush(); obs.flush()
                del mean, obs
                write_json(identity_path, {'source_sha256': sha, 'rows': rows, 'shape': shape,
                    'scale': 'five-minute means; obs retained as supplied, not raw sample counts'})
                self.trace.emit('cache_build_done', metric=metric, rows=rows)
            self.means[metric] = np.load(mean_path, mmap_mode='r')
            self.obs[metric] = np.load(obs_path, mmap_mode='r')
        ledger_path = self.data / 'xid_onsets_metadata.parquet'
        sha = digest(ledger_path)
        if sha != expected[ledger_path.name]:
            raise ValueError('XID ledger hash mismatch')
        self.input_identity[ledger_path.name] = sha
        ledger = pq.read_table(ledger_path).to_pandas()
        ledger['idx'] = pd.Categorical(ledger.gpu_id, categories=self.gpus).codes
        if (ledger['idx'] < 0).any():
            raise ValueError('XID GPU absent from reference grid')
        ledger['ns'] = pd.to_datetime(ledger.onset_time, utc=True).astype('int64')
        self.events = {int(g): np.sort(v.ns.to_numpy()) for g, v in ledger.groupby('idx')}
        self.gt = np.zeros((len(self.times), self.n), dtype=bool)
        # Exactly preserve E19's inclusive onset-bin label convention.
        for r in ledger.itertuples():
            b = (int(r.ns) - self.t0) // STEP
            self.gt[max(0, b-HORIZON):min(len(self.times), b+1), int(r.idx)] = True
        self.trace.emit('data_ready', gpu_count=self.n, grid_bins=len(self.times),
                        input_sha256=self.input_identity, xid_rows=len(ledger))

    def sample(self, start, end, seed):
        if start < DAY or end <= start:
            raise ValueError('Training interval lacks 24-hour feature history')
        p_b, p_g = np.where(self.gt[start:end])
        p = (p_b.astype('int64')+start)*self.n+p_g
        if not len(p):
            raise ValueError('No positive training rows')
        total_neg = (end-start)*self.n-len(p)
        wanted = min(len(p)*10, 200_000, total_neg)
        rng = np.random.default_rng(seed)
        neg = np.empty(0, dtype='int64')
        while len(neg) < wanted:
            candidates = rng.integers(start*self.n, end*self.n,
                                      size=max(5000, 2*(wanted-len(neg))), dtype='int64')
            candidates = candidates[~self.gt[candidates//self.n, candidates%self.n]]
            neg = np.unique(np.r_[neg, candidates])
        neg = rng.choice(neg, size=wanted, replace=False)
        flat = np.sort(np.r_[p, neg])
        b, g = (flat//self.n).astype('int32'), (flat%self.n).astype('int32')
        y = self.gt[b, g].astype('uint8')
        # Restore the full-population loss after case/control sampling.
        w = np.where(y == 1, 1.0, total_neg/wanted).astype('float32')
        return b, g, y, w


class DenseTelemetryEncoder:
    suffixes = ('last', 'mean', 'std', 'min', 'max', 'delta', 'slope_per_hour',
                'valid_grid_bins', 'missing_grid_fraction', 'last_valid_lag_minutes', 'obs_mean_as_supplied')

    def __init__(self, engine):
        self.engine = engine
        self.names = [f'{m}_grid5m_{h}h_{s}' for m in METRICS for h in (1,6,24) for s in self.suffixes]

    def transform(self, b, g):
        output = []
        for m in METRICS:
            for hours in (1,6,24):
                width = hours*12
                index = b[:, None]-BUFFER-width+np.arange(width)[None, :]
                raw = np.asarray(self.engine.means[m][index, g[:, None]], dtype='float32')
                observed = np.asarray(self.engine.obs[m][index, g[:, None]], dtype='float32')
                valid = np.isfinite(raw)
                n = valid.sum(axis=1)
                positions = np.broadcast_to(np.arange(width, dtype='int32'), raw.shape)
                left = np.maximum.accumulate(np.where(valid, positions, -1), axis=1)
                right = np.minimum.accumulate(np.where(valid, positions, width)[:, ::-1], axis=1)[:, ::-1]
                lval = np.take_along_axis(raw, np.maximum(left, 0), axis=1)
                rval = np.take_along_axis(raw, np.minimum(right, width-1), axis=1)
                gap = np.maximum(right-left, 1)
                interp = lval + (rval-lval)*((positions-left)/gap)
                interp = np.where(left < 0, rval, np.where(right >= width, lval, interp))
                filled = np.where(valid, raw, interp).astype('float32')
                # Interpolation only sees bins in the allowed historical window.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    mean = np.nanmean(filled, axis=1)
                    std = np.nanstd(filled, axis=1)
                    minimum = np.nanmin(filled, axis=1)
                    maximum = np.nanmax(filled, axis=1)
                    obsmean = np.nanmean(observed, axis=1)
                x = np.arange(width, dtype='float32')/12
                valid_float = valid.astype('float32')
                safe = np.where(valid, raw, 0)
                sx = (valid_float*x).sum(axis=1)
                sy = safe.sum(axis=1)
                sxx = (valid_float*x*x).sum(axis=1)
                sxy = (safe*x).sum(axis=1)
                denom = n*sxx-sx*sx
                slope = np.divide(n*sxy-sx*sy, denom,
                    out=np.full(len(b), np.nan), where=(n >= 2)&(denom > 1e-8))
                last_index = np.max(np.where(valid, positions, -1), axis=1)
                lag = np.where(n > 0, (width-1-last_index)*5, width*5)
                output.extend([filled[:, -1], mean, std, minimum, maximum,
                    filled[:, -1]-filled[:, 0], slope, n, 1-n/width, lag, obsmean])
        return np.column_stack(output).astype('float32')


class XIDHazardEncoder:
    names = ['xid_count_1h', 'xid_count_6h', 'xid_count_24h', 'xid_count_7d', 'xid_count_30d',
             'days_since_xid_capped90', 'has_past_xid', 'last_interarrival_days_capped90',
             'xid_1h_share_of_24h', 'ledger_exposure_days_capped30', 'xid_rate_per_observed_day_30d']

    def __init__(self, engine):
        self.engine = engine

    def transform(self, b, g):
        c = self.engine.times[b]-BUFFER*STEP
        out = np.zeros((len(b), len(self.names)), dtype='float32')
        out[:, 5] = 90; out[:, 7] = 90
        exposure = np.clip((c-self.engine.t0)/(DAY*STEP), 0, 30)
        out[:, 9] = exposure
        for gpu in np.unique(g):
            rows = np.flatnonzero(g == gpu)
            events = self.engine.events.get(int(gpu))
            if events is None:
                continue
            cut = c[rows]
            right = np.searchsorted(events, cut, side='right')
            for col, bins in enumerate((12,72,288,7*DAY,30*DAY)):
                out[rows, col] = right-np.searchsorted(events, cut-bins*STEP, side='left')
            has = right > 0
            out[rows, 6] = has
            out[rows[has], 5] = np.minimum((cut[has]-events[right[has]-1])/(DAY*STEP), 90)
            has2 = right > 1
            out[rows[has2], 7] = np.minimum((events[right[has2]-1]-events[right[has2]-2])/(DAY*STEP), 90)
        out[:, 8] = out[:, 0]/np.maximum(out[:, 2], 1)
        out[:, 10] = out[:, 4]/np.maximum(exposure, 1)
        return out


class FeatureAssembler:
    def __init__(self, engine, trace):
        self.engine, self.trace = engine, trace
        self.dense, self.hazard = DenseTelemetryEncoder(engine), XIDHazardEncoder(engine)
        self.names = self.dense.names+self.hazard.names
        if len(set(self.names)) != len(self.names):
            raise ValueError('Duplicate feature names')

    def transform(self, b, g, tag):
        if len(b) == 0 or int(b.min()) < DAY+BUFFER:
            raise ValueError('Feature observation window out of range')
        start = time.perf_counter()
        x = np.empty((len(b), len(self.names)), dtype='float32')
        for left in range(0, len(b), 1024):
            right = min(left+1024, len(b))
            x[left:right] = np.column_stack([self.dense.transform(b[left:right], g[left:right]),
                                              self.hazard.transform(b[left:right], g[left:right])])
            if left and left % 65536 == 0:
                self.trace.emit('feature_progress', tag=tag, done=right, total=len(b))
        if np.isinf(x).any():
            raise ValueError('Infinite feature values')
        self.trace.emit('features', tag=tag, rows=len(b), columns=len(self.names),
            dense_columns=len(self.dense.names), hazard_columns=len(self.hazard.names),
            seconds=time.perf_counter()-start, latest_bin=int(b.max()-BUFFER-1),
            missing_values=int(np.isnan(x).sum()))
        return x


class EventPolicy:
    """Policy kernel only. It does not filter the Top100 score population."""
    def __init__(self, threshold, required, window, cooldown_steps):
        if not 1 <= required <= window:
            raise ValueError('Invalid persistence policy')
        self.threshold, self.required, self.window, self.cooldown = threshold, required, window, cooldown_steps

    def emit(self, scores):
        above = np.asarray(scores) >= self.threshold
        alerts = np.zeros_like(above, dtype=bool)
        history = np.zeros(above.shape[1], dtype='int32')
        last = np.full(above.shape[1], -self.cooldown, dtype='int32')
        for t in range(len(above)):
            history += above[t]
            if t >= self.window:
                history -= above[t-self.window]
            ready = (history >= self.required)&(t-last >= self.cooldown)
            alerts[t] = ready
            last[ready] = t
        return alerts


def model_factory(family, backend, seed):
    if family == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.05,
            tree_method='hist', device=backend, max_bin=64, n_jobs=2,
            subsample=.8, colsample_bytree=1., random_state=seed, eval_metric='logloss')
    if family == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=6, learning_rate=.05,
            loss_function='Logloss', task_type='GPU' if backend == 'cuda' else 'CPU',
            random_seed=seed, thread_count=2, verbose=False, allow_writing_files=False)
    if family == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=.05,
            max_bin=63, min_child_samples=50, device_type=backend,
            n_jobs=2, random_state=seed, verbosity=-1)
    if family == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=16,
            min_samples_leaf=5, n_jobs=2, random_state=seed)
    if family == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        return IsolationForest(n_estimators=200, max_samples=256, n_jobs=2, random_state=seed)
    raise ValueError(f'Unknown learner: {family}')


def fit_model(family, backend, x, y, weight, trace, tag):
    start = time.perf_counter()
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    xx = imputer.fit_transform(x)
    model = model_factory(family, backend, SEED)
    trace.emit('fit_start', tag=tag, family=family, backend=backend, rows=len(y), positives=int(y.sum()))
    if family == 'isolation_forest':
        model.fit(xx[y == 0])
    else:
        model.fit(xx, y, sample_weight=weight)
    actual = backend
    if family == 'xgboost':
        actual = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        if backend.startswith('cuda') and not actual.startswith('cuda'):
            raise ValueError('XGBoost silently fell back from CUDA')
    trace.emit('fit_done', tag=tag, family=family, requested_backend=backend,
               actual_backend=actual, seconds=time.perf_counter()-start,
               tree_count=int(model.tree_count_) if family == 'catboost' else 300 if family in ('xgboost','lightgbm') else 200)
    return {'model': model, 'imputer': imputer, 'family': family, 'backend': actual}


def predict(bundle, x):
    xx = bundle['imputer'].transform(x)
    if bundle['family'] == 'isolation_forest':
        return -bundle['model'].decision_function(xx)
    return bundle['model'].predict_proba(xx)[:, 1]


def score_metrics(score, labels):
    score, labels = np.asarray(score), np.asarray(labels)
    if score.shape != labels.shape or not np.isfinite(score).all():
        raise ValueError('Invalid full-population scores')
    order = np.argsort(-score, axis=1, kind='stable')[:, :100]
    chosen = np.take_along_axis(labels, order, axis=1)
    hits, positives = int(chosen.sum()), int(labels.sum())
    positives_t = labels.sum(axis=1)
    discount = 1/np.log2(np.arange(2,102))
    ideal = np.r_[0., np.cumsum(discount)]
    ok = positives_t > 0
    dcg = chosen@discount
    return {'pr_auc': float(average_precision_score(labels.ravel(), score.ravel())) if positives else None,
        'hits_at_100': hits, 'positives': positives, 'recall_at_100': hits/positives if positives else None,
        'precision_at_100': hits/(len(labels)*100), 'decision_times': len(labels),
        'gpu_count': labels.shape[1], 'positive_queries': int(ok.sum()),
        'ndcg_at_100': float(np.mean(dcg[ok]/ideal[np.minimum(positives_t[ok],100)])) if ok.any() else None}


def reference_schedule(engine, config):
    schedule = []
    for part in sorted(Path(config['reference_directory']).glob('score_part_*.parquet')):
        ns = pq.read_table(part, columns=['timestamp']).column(0).cast(pa.int64()).to_numpy()
        times = np.unique(ns)
        bins = ((times-engine.t0)//STEP).astype('int32')
        if len(ns) != len(bins)*engine.n or np.any(np.diff(bins) != 6):
            raise ValueError(f'Incomplete reference schedule {part.name}')
        schedule.append({'path': str(part), 'bins': bins, 'origin': int(bins[0]), 'id': part.stem})
    if len(schedule) != 64 or sum(len(s['bins']) for s in schedule) != 3053:
        raise ValueError('Reference is not the frozen 64-origin population')
    return schedule


def full_rows(bins, n):
    return np.repeat(bins, n), np.tile(np.arange(n, dtype='int32'), len(bins))


def reference_metrics(schedule, engine):
    names = ['b1_probability', 'b2_probability', 'experiment14_fusion_probability',
             'fusion_no_inertia', 'fusion_b1_inertia', 'fusion_b2_inertia', 'fusion_dual_inertia']
    df = pq.read_table(schedule['path'], columns=['timestamp', 'gpu_uid']+names).to_pandas()
    b = ((df.timestamp.astype('int64').to_numpy()-engine.t0)//STEP).astype('int32')
    g = pd.Categorical(df.gpu_uid, categories=engine.gpus).codes
    if (g < 0).any():
        raise ValueError('Unknown reference GPU')
    flat = (b-schedule['origin'])*engine.n+g
    order = np.argsort(flat, kind='stable')
    expected_b, expected_g = full_rows(schedule['bins'], engine.n)
    if not np.array_equal(b[order], expected_b) or not np.array_equal(g[order], expected_g):
        raise ValueError('Reference comparison keys differ')
    y = engine.gt[expected_b, expected_g].reshape(-1, engine.n)
    return {name: score_metrics(df[name].to_numpy()[order].reshape(-1, engine.n), y) for name in names}


def run(config, output, limit, resume):
    trace = Trace(output)
    output = Path(output)
    source_sha = digest(__file__)
    if (output/'execution_contract.json').exists():
        old = json.loads((output/'execution_contract.json').read_text(encoding='utf-8'))
        if old['source_sha256'] != source_sha or old['config'] != config:
            raise ValueError('Resume source/config identity changed')
        if not resume:
            raise ValueError('Output exists; explicit --resume required')
    contract = {'source_sha256': source_sha, 'config': config,
        'classification': 'Falcon-AcmeTrace-ST-Top100 adaptation; not original typed-ticket reproduction',
        'feature_fusion': 'concatenate dense and hazard features before one selected learner',
        'selection': 'one chronological 3-day validation block before first evaluation origin; frozen thereafter',
        'sampling': 'all positive 5m rows; negative min(10*positive,200000); inverse sampling weights',
        'imputation': 'within-lookback interpolation; training-only median for remaining missing values',
        'event_policy': 'kernel implemented, not tuned/executed; Top100 uses unfiltered risk',
        'deviations': ['All-XID instead of typed tickets', '24h instead of 6h target',
            '5m instead of 1m telemetry', '21d ST instead of original offline split',
            'AP-based learner selection instead of event-level F1', 'missing host/task peer and ECC/remap',
            'no multi-horizon fusion; exact paper hyperparameters unavailable'],
        'material_passport': {'type': 'code_experiment', 'goal_ids': ['G-004','G-007','G-008'],
            'approval': 'user approved design and requested execution', 'seed': SEED}}
    write_json(output/'execution_contract.json', contract)
    engine = DataEngine(config, trace)
    assembler = FeatureAssembler(engine, trace)
    write_json(output/'feature_manifest.json', {'features': assembler.names,
        'dense': assembler.dense.names, 'hazard': assembler.hazard.names,
        'peer_features': [], 'telemetry_latest_bin': 'decision_bin - 3',
        'hazard_cutoff': 'decision_time - 10 minutes; inclusive',
        'persistence_feature': 'not implemented; no undocumented proxy substituted'})
    schedule = reference_schedule(engine, config)
    selected_path = output/'selected_learner.json'
    if not selected_path.exists():
        first = schedule[0]['origin']
        validation_end = first-PURGE
        validation_start = validation_end-3*DAY
        train_end = validation_start-PURGE
        b, g, y, w = engine.sample(train_end-21*DAY, train_end, SEED)
        x = assembler.transform(b, g, 'selection_train')
        vb, vg = full_rows(np.arange(validation_start, validation_end, 6, dtype='int32'), engine.n)
        vx = assembler.transform(vb, vg, 'selection_validation')
        vy = engine.gt[vb, vg].reshape(-1, engine.n)
        if int(vb.max())+HORIZON > first or int(b.max())+HORIZON >= validation_start:
            raise ValueError('Selection label maturity violation')
        trace.emit('selection_contract', training_start=int(b.min()), training_end=int(b.max()),
            validation_start=validation_start, validation_end=validation_end,
            evaluation_start=first, train_rows=len(y), validation_rows=len(vb))
        records = []
        for family, backend in config['backends'].items():
            bundle = fit_model(family, backend, x, y, w, trace, 'selection')
            scores = predict(bundle, vx).reshape(-1, engine.n)
            metrics = score_metrics(scores, vy)
            records.append({'family': family, 'backend': backend, **metrics})
            trace.emit('candidate_validation', **records[-1])
            del bundle, scores
        best = max(records, key=lambda r: (r['pr_auc'], r['recall_at_100']))
        write_json(selected_path, {'selected': best, 'candidates': records,
            'validation_latest_score_bin': int(vb.max()), 'evaluation_first_bin': first})
        del x, vx, b, g, y, w, vb, vg, vy
    selected = json.loads(selected_path.read_text(encoding='utf-8'))['selected']
    (output/'full_population').mkdir(exist_ok=True)
    (output/'checkpoints').mkdir(exist_ok=True)
    rows = []
    for idx, state in enumerate(schedule[:limit]):
        metric_path = output/'checkpoints'/f'{state["id"]}.metrics.json'
        if resume and metric_path.exists():
            saved = json.loads(metric_path.read_text(encoding='utf-8'))
            if digest(output/'full_population'/f'{state["id"]}.parquet') != saved['score_sha256']:
                raise ValueError('Completed score checkpoint identity changed')
            rows.append(saved)
            continue
        start = time.perf_counter()
        origin, bins = state['origin'], state['bins']
        end = origin-PURGE
        b, g, y, w = engine.sample(end-21*DAY, end, SEED+idx)
        if int(b.max())+HORIZON >= origin:
            raise ValueError('Final training label maturity violation')
        trace.emit('origin_start', origin_index=idx, train_start_bin=end-21*DAY,
            train_end_exclusive=end, origin_bin=origin, train_rows=len(y), decisions=len(bins))
        x = assembler.transform(b, g, f'origin_{idx}_train')
        bundle = fit_model(selected['family'], selected['backend'], x, y, w, trace, f'origin_{idx}')
        joblib.dump(bundle, output/'checkpoints'/f'{state["id"]}.model.joblib')
        del x, b, g, y, w
        sb, sg = full_rows(bins, engine.n)
        sx = assembler.transform(sb, sg, f'origin_{idx}_score')
        scores = predict(bundle, sx)
        labels = engine.gt[sb, sg]
        metrics = score_metrics(scores.reshape(-1,engine.n), labels.reshape(-1,engine.n))
        # Comparison scores are read only AFTER fresh Falcon training and scoring.
        ref = reference_metrics(state, engine)
        frame = pd.DataFrame({'timestamp': pd.to_datetime(engine.times[sb], utc=True),
            'gpu_uid': engine.gpus[sg], 'model_state_id': state['id'],
            'feature_cutoff_time': pd.to_datetime(engine.times[sb]-BUFFER*STEP, utc=True),
            'falcon_risk_score': scores, 'label': labels.astype('uint8')})
        part = output/'full_population'/f'{state["id"]}.parquet'
        tmp = part.with_suffix('.parquet.tmp')
        frame.to_parquet(tmp, index=False, compression='zstd'); tmp.replace(part)
        row = {'origin_index': idx, 'origin_bin': origin,
               'family': selected['family'], 'backend': bundle['backend'],
               'seconds': time.perf_counter()-start, 'rows': len(frame), **metrics,
               'reference': ref, 'score_sha256': digest(part),
               'label_sha256': hashlib.sha256(labels.astype('uint8').tobytes()).hexdigest()}
        write_json(metric_path, row)
        rows.append(row)
        trace.emit('origin_done', **{k:v for k,v in row.items() if k != 'reference'})
        write_json(output/'metrics.json', rows)
        del bundle, sx, scores, frame
    summary = {'status': 'FULL_PIPELINE_VERIFIED' if len(rows) == 64 else 'STANDALONE_PILOT',
        'origins_completed': len(rows), 'expected_full_origins': 64,
        'score_rows': sum(r['rows'] for r in rows), 'decision_times': sum(r['decision_times'] for r in rows),
        'selected_learner': selected['family'], 'backend': selected['backend'],
        'origin_mean_pr_auc': float(np.mean([r['pr_auc'] for r in rows])),
        'hits_at_100': sum(r['hits_at_100'] for r in rows),
        'pooled_recall_at_100': sum(r['hits_at_100'] for r in rows)/sum(r['positives'] for r in rows),
        'elapsed_seconds': time.perf_counter()-trace.start,
        'reference_origin_mean_pr_auc': {k: float(np.mean([r['reference'][k]['pr_auc'] for r in rows]))
            for k in rows[0]['reference']},
        'fidelity': 'All-XID/24h/ST/Top100 adaptation; typed-fault, peer, multi-horizon and alert evaluation absent'}
    if len(rows) == 64 and (summary['score_rows'] != 6081576 or summary['decision_times'] != 3053):
        raise ValueError('Full-run coverage mismatch')
    write_json(output/'summary.json', summary)
    write_json(output/'status.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-origins', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if not 1 <= args.limit_origins <= 64:
        raise ValueError('Origin limit must be between 1 and 64')
    try:
        with threadpool_limits(limits=2):
            run(config, args.output, args.limit_origins, args.resume)
    except Exception as exc:
        write_json(args.output/'status.json', {'status': 'FAILED', 'error': repr(exc),
            'utc': pd.Timestamp.now(tz='UTC').isoformat()})
        raise


if __name__ == '__main__':
    main()

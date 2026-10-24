"""Source-labelled recovery scenarios with event-keyed common random numbers.

Published summaries are reference scenarios, not fits to the local workload.
Retry gaps and end-to-end downtime must never be added as independent stages.
"""
from __future__ import annotations
import hashlib
import math
from statistics import NormalDist


def event_uniform(seed, episode_id, component):
    key = f'{int(seed)}|{episode_id}|{component}'.encode('utf-8')
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')
    return (value + 0.5) / (2**64)


def sample_seconds(spec, seed, episode_id, component):
    if not isinstance(spec, dict) or not spec.get('source'):
        raise ValueError(f'{component}: recovery distribution needs a source')
    kind = spec.get('kind', 'fixed')
    u = event_uniform(seed, episode_id, component)
    if kind == 'fixed':
        value = float(spec['seconds'])
    elif kind == 'empirical':
        values = [float(x) for x in spec['samples_seconds']]
        if not values or any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError('Invalid empirical recovery observations')
        value = values[min(len(values)-1, int(u*len(values)))]
    elif kind == 'lognormal_summary_scenario':
        mean, median = float(spec['mean_seconds']), float(spec['median_seconds'])
        if not math.isfinite(mean) or not math.isfinite(median) or median <= 0 or mean < median:
            raise ValueError('Lognormal scenario requires finite mean >= median > 0')
        sigma = math.sqrt(2*math.log(mean/median))
        value = math.exp(math.log(median) + sigma*NormalDist().inv_cdf(min(1-1e-15, max(1e-15, u))))
    else:
        raise ValueError(f'Unsupported recovery distribution: {kind}')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid duration for {component}')
    return value


def fixed_scenario(seconds, source='Historical simulator assumption; not observed'):
    return {'kind': 'fixed', 'seconds': float(seconds), 'source': source}


def recovery_stages(config, seed, episode_id, attempt, legacy_relaunch, job_id=None):
    specs = config.get('durations', {})
    if config.get('random_key_version')=='job-attempt-v1':
        if job_id is None: raise ValueError('Job-keyed sampling requires job_id')
        episode_id=f'{episode_id}|job={job_id}|attempt={int(attempt)}'
    retry = sample_seconds(specs.get('retry_wait', fixed_scenario(legacy_relaunch)), seed, episode_id, 'retry_wait')
    load = sample_seconds(specs.get('checkpoint_load', fixed_scenario(0, 'Not separately observed; excluded from this scenario')), seed, episode_id, 'checkpoint_load')
    launch = sample_seconds(specs.get('relaunch', fixed_scenario(0, 'Included in retry-gap scenario')), seed, episode_id, 'relaunch')
    multiplier = float(config.get('backoff_multiplier', 1))
    cap = float(config.get('max_retry_wait_seconds', 86400))
    if not math.isfinite(multiplier) or multiplier < 1 or not math.isfinite(cap) or cap < 0:
        raise ValueError('Invalid retry/backoff policy')
    retry = min(cap, retry*(multiplier**max(0, attempt-1)))
    if config.get('duration_semantics', 'stages') == 'retry_gap_total' and (load or launch):
        raise ValueError('A total retry gap cannot be added to load/relaunch stages')
    semantics=config.get('duration_semantics','stages')
    if semantics not in ('stages','retry_gap_total','inter_session_gap_plus_loading'):
        raise ValueError('Unknown recovery phase boundary')
    if semantics=='inter_session_gap_plus_loading' and launch:
        raise ValueError('Inter-session gap already includes session relaunch overhead')
    return {'retry_wait': retry, 'checkpoint_load': load, 'relaunch': launch}

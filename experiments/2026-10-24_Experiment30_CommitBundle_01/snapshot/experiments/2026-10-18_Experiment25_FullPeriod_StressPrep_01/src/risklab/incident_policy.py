"""XID response based on code, causal recurrence and explicit diagnostics.

No XID31 hardware fraction or mandatory XID43 GPU-reset duration is inferred.
References: NVIDIA XID catalog and NVSentinel Health Events Analyzer v1.24.
"""
from __future__ import annotations
from collections import defaultdict, deque

NVIDIA_XID = 'https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html'
NVIDIA_HEA = 'https://docs.nvidia.com/nvsentinel/v1.24.0/configuration/health-events-analyzer/'


def classify_incident(event, history, config):
    codes = set(event['parsed_xid_codes'])
    if not codes:
        raise ValueError('An incident must retain a valid observed XID code')
    result = {'action': 'record_only', 'reason': 'Unclassified code; diagnostic evidence required',
              'source': NVIDIA_XID, 'codes': sorted(codes),
              'scope': str(event.get('incident_scope') or 'unknown')}
    diagnostic = str(event.get('diagnostic_result', 'unknown')).lower()
    hardware_confirmed = event.get('hardware_confirmed', False)
    if hardware_confirmed is True or codes & {48, 79, 95}:
        return {**result, 'action': 'hardware_recovery', 'reason': 'Hardware recovery instruction or explicit diagnostic evidence'}
    if 31 in codes:
        now = float(event['time'])
        window = float(config.get('xid31_window_seconds', 86400))
        prior = history.setdefault(str(event['gpu_uid']), deque())
        while prior and prior[0][0] < now-window:
            prior.popleft()
        episode_id = str(event['episode_id'])
        if all(eid != episode_id for _, eid in prior):
            prior.append((now, episode_id))
        count = len(prior)
        if count >= int(config.get('xid31_diagnostic_count', 2)) or diagnostic in ('pass','fail'):
            return {**result, 'action': 'diagnose', 'reason': 'Repeated XID31 on same GPU within past window',
                    'source': NVIDIA_HEA, 'recurrence_count': count, 'diagnostic_result': diagnostic}
        return {**result, 'action': 'app_restart', 'reason': 'XID31: restart app, investigate app/CUDA; hardware cause unconfirmed', 'recurrence_count': count}
    if codes <= {13, 43, 94}:
        return {**result, 'action': 'app_restart', 'reason': 'Application/session recovery; no GPU maintenance by code alone'}
    return result


def incident_defaults(config=None):
    return {'unmapped_app_mode': 'record_only', 'max_app_retries': 3,
            'retry_cap_source': 'Explicit simulator policy assumption, not a measured success limit',
            'backoff_multiplier': 1, 'max_retry_wait_seconds': 86400,
            'xid31_window_seconds': 86400, 'xid31_diagnostic_count': 2,
            'duration_semantics': 'retry_gap_total', 'durations': {}, **(config or {})}

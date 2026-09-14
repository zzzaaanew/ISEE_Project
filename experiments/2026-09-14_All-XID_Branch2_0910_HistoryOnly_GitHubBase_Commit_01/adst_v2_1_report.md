# [ADST v2.1] All-XID Pooled Validation + Batch ADWIN 보고서

## Material Passport
- **Status**: COMPLETED
- **Runner**: `run_sliding_adst_v2_1_pooled_adwin.py`
- **Target**: All-XID unified onset, 24-hour horizon
- **Branch 1**: 1h·6h·24h telemetry parallel ensemble
- **Branch 2**: 2026-09-10 History-only Historical Logistic (xid_count_30d, days_since_xid, no recency)
- **Split**: chronological Sliding Training, 36-hour purge, fixed terminal held-out test

## Terminal held-out mean
- Fused PR-AUC: `0.061371`
- Branch 1 PR-AUC: `0.022086`
- Branch 2 PR-AUC: `0.061371`
- Fused normalized PR-AUC: `0.051693`
- B2 positive-weighted PR-AUC: `0.135048`
- Fused ROC-AUC: `0.614443`
- Recall@100: `20.12%`
- Lift@100: `4.009x`

## Selector and ADWIN
- Final branch settings are selected from pooled validation feedback over the latest six development origins.
- Lambda is selected only after the pooled selected Branch 1 and Branch 2 predictions are fixed.
- ADWIN input is label-matured 30-minute validation loss, prediction disagreement, and sampled prevalence.
- Effective ADWIN trigger cycles: `0`
- Feedback batches recorded in final state: `0`
- Terminal held-out labels are used only for final reporting, never for selection or drift state.

## Scope and limitations
- v2.1 is a controlled methodology experiment; improvement is judged on the same held-out population as v1/v2.
- Pooled validation reduces dependence on one recent validation slice but may respond more slowly to abrupt drift.
- Held-out interval: `2023-08-05T06:45:00+00:00` ~ `2023-08-18T00:00:00+00:00`
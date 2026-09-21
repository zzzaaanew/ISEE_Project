# Branch 1 Performance-Triggered Dynamic Window 보고서

## Material Passport
- **Status**: COMPLETED
- **Runner**: `run_sliding_adst_branch1_performance_window.py`
- **Target**: All-XID unified onset, 24-hour horizon
- **Branch 1**: 1h·6h·24h telemetry parallel ensemble
- **Branch 2**: strict History-only, fixed 7-day window, no recency
- **Selector**: Branch 1 validation-performance controller; ADWIN disabled
- **Split**: chronological Sliding Training, 36-hour purge, fixed terminal held-out test

## Terminal held-out mean
- Fused PR-AUC: `0.045797`
- Branch 1 PR-AUC: `0.017877`
- Branch 2 PR-AUC: `0.061371`
- Fused normalized PR-AUC: `0.035757`
- Fused ROC-AUC: `0.738804`
- Recall@100: `20.47%`
- Lift@100: `4.078x`
- B2 positive-weighted PR-AUC: `0.135048`

## Dynamic window behavior
- Branch 1 confirmed window changes: `0`
- Candidate windows: `[7, 14, 21]` days
- Fixed Branch 1 half-life: `7` days
- Confirmation requirement: `2` checkpoints
- Terminal interval: `2023-08-05T06:45:00+00:00` ~ `2023-08-18T00:00:00+00:00`
- Held-out labels were used only for final reporting, never for window selection.

## Interpretation boundary
- This experiment tests whether Branch 1 benefits from validation-performance-based training-window changes.
- Branch 2 is a fixed control and is not dynamically adapted.
- The terminal held-out model is fitted once after final selection; this is not test-time adaptive retraining.
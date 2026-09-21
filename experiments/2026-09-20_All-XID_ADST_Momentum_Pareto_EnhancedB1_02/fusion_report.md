# All-XID Enhanced Branch 1 + History-only Branch 2 Pareto Fusion

## Material Passport
- **Status**: COMPLETED
- **Policy**: `adst`
- **Target**: All-XID `[13, 31, 43, 45, 94]`, 24-hour horizon
- **Split**: Sliding Training, 36-hour purge, common terminal held-out 20%
- **Evaluation**: full GPU-time scores → Pareto Fusion → Top-100 ranking
- **Branch 2**: strict History-only (`xid_count_30d`, `days_since_xid`)
- **ADWIN**: not used; previous ADWIN experiments remain historical artifacts

## Branch 1 feature contract
- Existing 1/6/24-hour telemetry inputs retained.
- Cross-Metric features: `thermal_efficiency, power_temp_ratio, util_fb_coupling, power_per_util, temp_fb_divergence, thermal_headroom, power_headroom`
- Topology features: `0`; source `disabled`
- Excluded from Branch 1: `xid_count_30d`, `days_since_xid`, XID filters, calendar features, and learned GNN layers.

## Pareto limitations corrected
- The GitHub post-hoc analyzer operates on an existing Top-100 tape; this runner computes Pareto weights for every evaluated GPU-time first.
- Rankings are recomputed after Fusion, so GPUs previously outside Top-100 can enter the tape.
- History counts use the decision-time minus 10-minute unavailable-data cutoff.
- Alpha is fixed at the code-faithful `1.0`; held-out labels are not used for tuning.

## Terminal held-out mean
- **Fusion PR-AUC**: `0.028738`
- **Branch 1 PR-AUC**: `0.020690`
- **Branch 2 PR-AUC**: `0.061371`
- **Fusion ROC-AUC**: `0.702817`
- **Recall@100**: `11.30%`
- **Lift@100**: `2.252x`
- **Mean Pareto lambda**: `0.9009`
- **Evaluated GPUs per decision time**: `1992`
- **Held-out cycles**: `13`

## Output
- `fusion_metrics.csv`
- `fusion_risk_tape.parquet`
- `fusion_selection_history.csv`
- `momentum_selection_trace.csv`
- `experiment_manifest.json`
- `checkpoints/`
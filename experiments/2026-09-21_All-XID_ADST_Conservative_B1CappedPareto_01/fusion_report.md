# Conservative Momentum-ADST + B1-shrunk Pareto Fusion

## Material Passport
- **Status**: COMPLETED
- **Goal IDs**: G-004, G-007, G-008
- **Target**: All-XID `[13, 31, 43, 45, 94]`, 24-hour horizon
- **Split**: time-ordered Sliding Training, 36-hour purge, common terminal held-out 20%
- **Branch 2**: strict History-only (`xid_count_30d`, `days_since_xid`)
- **Primary evaluation**: full GPU-time scores → B1-capped Pareto Fusion → Top-100

## Conservative ADST changes
- Momentum beta: `0.85`
- Confidence threshold: `0.7`
- Performance-drop trigger: `0.2`
- Quick validation: latest `2` blocks, median PR-AUC
- Window-change cooldown: `2` origins
- 7-day promotion guard: `+0.005` absolute and `+5%` relative requirement
- The lightweight Logistic proxy is retained in this run to isolate setting changes; architecture-matched ADST proxy remains a separate ablation.

## Fusion variants
- Primary: `pareto_cap_0.6` with effective Lambda `min(Pareto Lambda, 0.6)`
- Controls: current Pareto, fixed Lambda 0.6, B2-only, B1-only
- No terminal held-out label was used to select a variant.

## Primary held-out result
- **Fusion PR-AUC**: `0.029970`
- **Branch 1 PR-AUC**: `0.020690`
- **Branch 2 PR-AUC**: `0.061371`
- **Fusion ROC-AUC**: `0.705016`
- **Recall@100**: `13.06%`
- **Lift@100**: `2.601x`

## Output
- `fusion_metrics.csv` (primary capped Pareto tape metrics)
- `fusion_variant_metrics.csv` (all variants by held-out cycle)
- `fusion_variant_summary.csv` (variant means/medians)
- `fusion_risk_tape.parquet` (primary Top-100 tape)
- `momentum_selection_trace.csv`
- `experiment_manifest.json`
- `checkpoints/`
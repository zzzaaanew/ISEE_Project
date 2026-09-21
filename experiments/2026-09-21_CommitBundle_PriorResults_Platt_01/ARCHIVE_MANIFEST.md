# Experiment Commit Bundle

- Bundle date: 2026-09-21
- Goal IDs: G-004, G-007, G-008
- Repository branch: `experiment/2026-09-21-report-faithful-pareto-momentum-stateful`
- Baseline history: `54a12b6` (`experiment: add report-faithful Pareto momentum fusion`)
- Purpose: preserve the prior All-XID experiment trail together with the Platt Calibration ablation.

## Included result runs

- `2026-09-12_All-XID_GitHubBase_Reproduction`
- `2026-09-12_All-XID_GitHubBase_Reproduction_Rerun_01`
- `2026-09-13_All-XID_ADST_v2_BranchSpecific_Recency_RobustLambda`
- `2026-09-14_All-XID_ADST_v2_1_PooledValidation_BatchADWIN_Rerun_01`
- `2026-09-14_All-XID_Branch1_PerformanceDynamicWindow_01`
- `2026-09-14_All-XID_Branch1_PerformanceDynamicWindow_Rerun_01`
- `2026-09-14_All-XID_Branch2_0910_HistoryOnly_GitHubBase_Rerun_02`
- `2026-09-18_All-XID_Fusion_DiversifiedB1_HistoryB2_01`
- `2026-09-19_All-XID_ADST_Momentum_Pareto_EnhancedB1_01`
- `2026-09-19_All-XID_Sliding_Pareto_EnhancedB1_01`
- `2026-09-20_All-XID_ADST_Momentum_Pareto_EnhancedB1_02`
- `2026-09-21_All-XID_ADST_Conservative_B1CappedPareto_01`
- `2026-09-21_All-XID_ReportFaithful_ParetoMomentumCascade_Platt_01`
- `2026-09-21_Repro_B1_Diversified_Cascade_01`

## Included experiment runners

- `ML/run_adst_conservative_b1_shrink_fusion.py`
- `ML/run_platt_calibration_fusion.py`
- `ML/run_platt_calibration_fusion_resume.py`
- `ML/run_report_faithful_platt_calibration_fusion.py`
- `ML/run_sliding_adst_branch1_performance_window.py`
- `ML/run_sliding_adst_v2_1_pooled_adwin_rerun.py`

## Exclusions

- Raw telemetry/source data, caches, virtual environments, and secrets.
- Incomplete checkpoint-only directories and empty result directories.
- Operational stdout/stderr/supervisor/retry logs from the 2026-09-20 Sliding run.
- Local project documents outside this Git repository.

CSV and Parquet artifacts are intentionally included only under the listed experiment directories for reproducibility of the archived results; no raw input data is included.

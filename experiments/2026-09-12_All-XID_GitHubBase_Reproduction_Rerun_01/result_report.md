# All-XID GitHub Base 재현 실험 최종 결과 보고서

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-13
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: `2026-09-12-all-xid-github-base-reproduction-rerun-01`
- **Type**: training
- **Status**: completed
- **Command**: `python ML/run_bidirectional_adst_fusion.py --cadence-hours 24 --negative-ratio 10 --test-stride-bins 6 --output-dir experiments/2026-09-12_All-XID_GitHubBase_Reproduction_Rerun_01/results`
- **Working Directory**: `ISEE_Project_GitHubBase`
- **Git Commit**: `a8cb41a96db04496f088b6ee1c73ecd062a8ffa0`
- **Target**: All-XID; loaded codes `[13, 31, 43, 45, 94]`
- **Schedule**: 63 daily walk-forward origins, 30-day warmup, 24-hour retraining cadence
- **Risk Tape**: 302,400 rows

### Reproduction Settings

- Training candidates: 7/14/21 days
- Observation candidates: 1/6/24 hours
- Test stride: 30 minutes
- Negative sampling: 1:10, maximum 200,000 negatives
- Seed argument: 20260905
- Primary metric: PR-AUC
- Secondary metrics: ROC-AUC, Recall@100, Lift@100

## Overall Results

| Metric | Mean | Median | Std. dev. | 95% descriptive CI |
|---|---:|---:|---:|---:|
| Fused PR-AUC | 0.041135 | 0.017578 | 0.073360 | 0.023020–0.059250 |
| Branch 1 PR-AUC | 0.022118 | 0.015886 | 0.022900 | 0.016463–0.027772 |
| Branch 2 PR-AUC | 0.040677 | 0.009431 | 0.094639 | 0.017307–0.064047 |
| Fused ROC-AUC | 0.680936 | 0.677748 | 0.149995 | 0.643897–0.717975 |
| Recall@100 | 20.701% | 14.432% | 19.890% | 15.790–25.613% |
| Lift@100 | 4.124x | 2.875x | 3.962x | 3.145–5.102x |

The rounded overall values match the GitHub-committed report: fused PR-AUC `0.0411`, Branch 1 `0.0221`, Branch 2 `0.0407`, ROC-AUC `0.6809`, Recall@100 `20.7%`, and Lift@100 `4.12x`.

## Window Selection

- `L_train`: 7d `21` times, 14d `19` times, 21d `23` times
- `L_obs`: 1h `21` times, 6h `19` times, 24h `23` times

No single training or observation window dominated. This supports treating the window choice as a variable to analyze, but does not by itself establish that the ADST selector is stable or causally responsible for the performance.

## Branch and Fusion Analysis

- Fused PR-AUC exceeded Branch 1 in `36/63` origins (`57.1%`); mean difference `+0.019017`.
- Fused PR-AUC exceeded Branch 2 in `45/63` origins (`71.4%`); mean difference `+0.000458`.
- Fused PR-AUC exceeded the better of Branch 1 and Branch 2 in only `23/63` origins (`36.5%`); mean difference `-0.005291`.
- Therefore, the overall fused mean is higher than Branch 1 mainly because Branch 2 is strong, but equal-weight fusion does not consistently outperform the best individual branch. The GitHub script's blanket claim of simultaneous improvement is not supported by this origin-level comparison.

## Temporal Analysis

| Period | Origins | Mean test positives | Fused PR-AUC | Fused ROC-AUC | Recall@100 | Lift@100 |
|---|---:|---:|---:|---:|---:|---:|
| Early origins 1–31 | 31 | 827.5 | 0.030807 | 0.653797 | 17.405% | 3.467x |
| Late origins 32–63 | 32 | 852.4 | 0.051140 | 0.707227 | 23.895% | 4.760x |

The late-period mean fused PR-AUC is about `66.0%` higher, while mean test-positive count is only about `3.0%` higher. The median fused PR-AUC slightly decreases from `0.017578` to `0.017039`; therefore the late-period mean increase is driven by concentrated high-performing/high-positive episodes rather than a uniform improvement across every origin. This is evidence of temporal heterogeneity, not proof that ADST alone caused the increase.

## Reproducibility Assessment

- **Aggregate reproduction**: GitHub report values reproduced to the published four-decimal rounding.
- **Exact deterministic reproduction**: not verified. The script fixes NumPy and scikit-learn randomness but does not call `torch.manual_seed`; the first four-origin partial run and this rerun produced different CNN-dependent scores.
- **Assessment**: `PARTIALLY_REPRODUCIBLE` relative to the earlier partial run; full aggregate output is complete, but row-level exact reproducibility requires explicit PyTorch and data-loader seed control.

## Methodology Audit Against Project Decisions

- Branch 2 currently includes `hour_of_day`, `day_of_week`, and `is_weekend`; it is not yet the project's confirmed History-only Branch 2.
- `PURGE_NS` is declared, but the observed train/validation construction does not insert an explicit 36-hour gap before validation; the project-aligned follow-up must implement this separately.
- The run is a GitHub-base reproduction, not the final project-compliant Sliding Training/History-only ADST experiment.
- XID31+43 was not used as a restricted target; the loaded All-XID target includes codes 31 and 43 together with 13, 45, and 94.

## Output Files

- `results/bidirectional_adst_metrics.csv`
- `results/bidirectional_adst_metrics_derived.csv`
- `results/analysis_summary.json`
- `results/bidirectional_adst_risk_tape.parquet`
- `results/bidirectional_adst_report.md`
- `run.log`
- `environment.txt`

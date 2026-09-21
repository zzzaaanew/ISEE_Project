# All-XID GitHub Base 재현 실험 부분 결과 보고서

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-12
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: `2026-09-12-all-xid-github-base-reproduction`
- **Type**: training
- **Status**: timeout
- **Command**: `python ML/run_bidirectional_adst_fusion.py --cadence-hours 24 --negative-ratio 10 --test-stride-bins 6 --output-dir experiments/2026-09-12_All-XID_GitHubBase_Reproduction/results`
- **Working Directory**: `ISEE_Project_GitHubBase`
- **Git Commit**: `a8cb41a96db04496f088b6ee1c73ecd062a8ffa0`
- **Git Branch**: `migration/all-xid-github-base`
- **Target**: All-XID; 로드된 XID codes `[13, 31, 43, 45, 94]`
- **Schedule**: 63개 daily walk-forward origin, 30일 warmup
- **Completed**: 4/63 origins; 5번째 origin validation 선택 중 hard timeout으로 중단
- **Exit**: 30분 hard timeout에 따른 프로세스 중단

### Reproduction Settings

- Training horizon candidates: 7/14/21일
- Observation horizon candidates: 1/6/24시간
- Retraining cadence: 24시간
- Test stride: 5분 bin 6개 = 30분
- Negative sampling: positive 대비 10배, 최대 200,000개
- Seed: 20260905
- Primary metric: PR-AUC
- Secondary metrics: ROC-AUC, Recall@100, Lift@100

### Completed Origin Metrics

| Origin | Selected window | Validation PR-AUC | Fused PR-AUC | Branch 1 PR-AUC | Branch 2 PR-AUC | Fused ROC-AUC | Recall@100 | Lift@100 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7d / 6h | 0.2914 | 0.0193 | 0.0104 | 0.0233 | 0.3881 | 4.8% | 0.96x |
| 2 | 14d / 6h | 0.3035 | 0.0063 | 0.0052 | 0.0079 | 0.4393 | 8.9% | 1.77x |
| 3 | 21d / 6h | 0.2168 | 0.0106 | 0.0112 | 0.0042 | 0.7658 | 11.4% | 2.27x |
| 4 | 21d / 24h | 0.1495 | 0.0177 | 0.0184 | 0.0129 | 0.4437 | 14.6% | 2.90x |

### Partial Summary

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| Validation PR-AUC | 0.240300 | 0.149500 | 0.303500 |
| Fused PR-AUC | 0.013475 | 0.006300 | 0.019300 |
| Branch 1 PR-AUC | 0.011300 | 0.005200 | 0.018400 |
| Branch 2 PR-AUC | 0.012075 | 0.004200 | 0.023300 |
| Fused ROC-AUC | 0.509225 | 0.388100 | 0.765800 |
| Recall@100 | 9.925% | 4.800% | 14.600% |
| Lift@100 | 1.975x | 0.960x | 2.900x |

Window selection among completed origins was `L_train={7d: 1, 14d: 1, 21d: 2}`, `L_obs={1h: 0, 6h: 3, 24h: 1}`. This is descriptive only because four origins are insufficient to establish a stable selection policy.

## Interpretation

1. The fused PR-AUC exceeded the partial Branch 1 and Branch 2 means, but the difference is based on four origins and cannot be treated as a final improvement claim.
2. Fused ROC-AUC varied from 0.3881 to 0.7658 and averaged 0.5092, showing strong temporal heterogeneity in the early origins.
3. Lift@100 was above 1 for three of four origins; origin 1 was below random-ranking lift at 0.96x. The observed ranking benefit is therefore not yet stable.
4. The selected observation window was 6h in three of four completed origins. This is an early signal for the later ADST comparison, not a confirmed optimum.

## Reproduction Limitations and Methodology Audit

- The full run did not complete, so the GitHub report cannot be verified against a full local rerun. Verification status is `UNVERIFIED`.
- The script writes its final metrics CSV and risk tape only after all origins finish; therefore this run produced no final `bidirectional_adst_metrics.csv` or Parquet risk tape. The preserved log was parsed into `partial_metrics.csv` and `partial_metrics.json`.
- The current GitHub script's Branch 2 includes `hour_of_day`, `day_of_week`, and `is_weekend`; this does not yet satisfy the project's confirmed History-only Branch 2 contract.
- `PURGE_NS` is declared in the script, but the observed validation/training construction uses the immediately preceding validation window without an explicit 36-hour gap. This must be corrected in the project-aligned follow-up experiment rather than silently attributed to this reproduction.
- The 30-minute hard timeout was reached after approximately four complete origins; CPU-only execution and repeated feature extraction/model fitting caused excessive runtime and memory pressure.

## Anomalies Detected

- `OUTPUT_STALL` advisory during long feature/model-fitting sections; the process remained alive and the log later advanced.
- `RESOURCE_ANOMALY` advisory: process working set grew to approximately 4.45 GB during execution.
- `HARD_TIMEOUT`: mandatory stop after the 30-minute limit.

## Output Files

- `run.log`: raw execution log
- `partial_metrics.csv`: four completed origin metrics parsed from the log
- `partial_metrics.json`: structured partial summary and settings
- `environment.txt`: commit, branch, Python and package versions

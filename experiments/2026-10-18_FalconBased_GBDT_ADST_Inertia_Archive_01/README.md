# Falcon-based AcmeTrace experiments — 2026-10-18 archive

## Scope

This new experiment folder preserves the executed Falcon-style comparison work without changing any earlier experiment archive.

1. **Falcon45 dual-branch experiment:** telemetry-only 45-feature XGBoost CUDA B1; ST7/14/21 and mature-feedback ADST; XID-history-only Logistic B2 with ST21; KL logit Inertia before Fusion; uncapped Pareto and E19-inspired residual Fusion.
2. **Five-minute continuation:** the frozen ST21 + Inertia B1 setting selected by the predeclared positive-result gate. Fresh five-minute features, lagged states, risk scores and Top-100; no new GBDT/LR/Fusion fitting. The final ranked predictor is B1, not a newly selected Fusion.
3. **Earlier Falcon143 single-branch adaptation:** separate source and results; do not confuse its feature contract with Falcon45 or treat either as a complete reproduction of the Falcon paper.

All current experiments use All-XID 24h inclusive-bin labels, a 36h purge, a nominal 10min feature cutoff (latest telemetry 15min lag), 1,992 GPUs and seed 20260905. The 64-origin period is development data, not an independent terminal test set.

## Verified results

AP is the mean of origin-level average precision (zero-positive origin AP=0); Recall@100 is pooled positive GPU-time recall. Hits are GPU-time rows, not unique hardware failures.

| Run / predictor | Mean AP | Recall@100 | Hits | Coverage |
| --- | ---: | ---: | ---: | --- |
| Original E19 B1 reference | 0.017886578 | 11.5214% | 6100 | 30min / 64 origins |
| Original E19 dual-inertia reference | 0.033064138 | 16.0638% | 8505 | 30min / 64 origins |
| Earlier GPU-retrained E19 dual inertia | 0.032668930 | 15.8164% | 8374 | 30min / 64 origins |
| Falcon45 B1 ST21 + Inertia | 0.022381456 | 11.6914% | 6190 | 30min / 64 origins |
| Falcon45 B1 ST14 + Inertia | 0.021281981 | 12.2844% | 6504 | 30min / 64 origins |
| Falcon45 ST7 + Inertia + E19-inspired Fusion | 0.032751388 | 16.3509% | 8657 | 30min / 64 origins |
| Falcon45 B1 ST21 + Inertia, frozen five-minute replay | 0.022004021 | 11.8316% | 37531 | 5min / 64 origins |

The 30min run passed SCORE_ARTIFACTS_VERIFIED: 6,081,576 rows / 3,053 queries, 52,945 positive rows, 192 CUDA B1 fits and 64 B2 fits. The five-minute run passed FIVE_MINUTE_SCORE_ARTIFACTS_VERIFIED: 36,479,496 rows / 18,313 queries. Different cadences have different positive-row denominators and cannot be compared as identical query populations.

The gate baseline in the gpu-opt directory is the preserved original CPU-reference score tape (AP 0.033064138), not the separate GPU-retrained run (0.032668930). Both are archived separately here to prevent a directory-name-based misinterpretation.

## Interpretation

- Inertia improved B1 AP and pooled Recall under all four Falcon45 window policies. ST21 has the highest B1 AP; ST14 has the highest B1 Recall.
- The new ADST did not beat fixed ST21 in standalone B1. An ADST benefit is not established.
- The best new Fusion has slightly lower AP and higher Recall than original E19. B1 improvement alone does not establish final-pipeline superiority.
- E19-inspired residual has higher aggregate AP than uncapped Pareto in this run, with important time variability and onset-coverage/turnover trade-offs. Paired descriptive block-bootstrap outputs are included; no independent-test or multiplicity-adjusted superiority claim is made.
- Original E19 calibrates current B1/B2 and lagged B1/B2 states separately (up to 12 residual basis terms). This experiment compresses B1 current/state before a two-input, six-term residual and omits B2 Inertia. It is not a feature-only ablation of original E19.
- Supervisor status replacement failed with Windows PermissionError/WinError5 at 02:08 KST. Training/Fusion independently finished at 03:22. Recovery audited existing outputs without retraining, then resumed the conditional five-minute chain. Historical failure evidence is retained.

## Layout and use

- FalconStyle_GBDT_ADST_Inertia/: exact executed source snapshots, vendored feature/math sources, historical config/policy and execution contract.
- Falcon_AcmeTrace_ST_Top100/: earlier single-branch adaptation source, dependency pins and environment evidence.
- results/falcon45_30m and results/falcon45_5m: final audits, metrics, selection, fit/inference traces and compact fitted Fusion parameters.
- results/falcon143_30m: earlier adaptation evidence and report-recovery history.
- results/baselines: original E19 and GPU-retrained E19, clearly separated.
- results/recovery_evidence and results/stopped_5m_handoff: failure and recovery provenance.
- 실험 명세/ and 실험 기록/: specification and cumulative Markdown/HTML research note as archived snapshots.
- inputs/: raw input identities, required cache inventory, omitted binary artifact hashes and replay limitations.
- ARCHIVE_MANIFEST.json: per-file archive hashes and original source locations. Copied source files are byte-identical; the archive was not retrained during publication.

Pinned Python environment: Python 3.12; see Falcon_AcmeTrace_ST_Top100/requirements.txt and versions/requirements-lock.txt (shared environment includes XGBoost 2.1.4). Prepare the external inputs and update a new config for your machine; see inputs/README.md. Example entry point:

```text
python FalconStyle_GBDT_ADST_Inertia/ML/run_experiment.py --config /absolute/path/to/new-config.json --output /absolute/path/to/new-run --limit-origins 64
```

Full Parquet score tapes, Top-100 tapes, raw telemetry, caches, virtual environments and large trained checkpoints are not committed. The source local runs remain unchanged. Historical stored audit status is evidence of the completed run, not a claim that missing binary artifacts were reproduced from this compact checkout.

# Experiment 12 — B2-anchored candidate-union reranking

Status: exploratory saved-score replay, not a final prospective model result. Goals: G-007 and G-008. Archived on 2026-10-04 from the local 2026-10-01 Experiment 12 candidate-union run.

## Evaluation contract

- All-XID onset, 24-hour horizon, 10-minute feature cutoff, 36-hour purge.
- 30-day warm-up followed by 64 rolling origins and 30-minute decisions across 1,992 GPUs.
- The reranker fits on the previous 7 mature OOF days and selects on the following 7 mature OOF days; current-origin labels are excluded. It falls back to the exact B2 rank when evidence is insufficient.
- Input Branch scores were saved before this replay. The source B1 ADST controller may have used labels before maturity; the reranker cannot repair that upstream concern.

## Archived aggregate result

Across 6,081,576 GPU-time rows and 52,945 positive GPU-times, B2 Top-100 hit 8,335 positives (pooled Recall@100 0.157428); candidate-union hit 8,411 (0.158863), a difference of 76 hits. Pooled positive-query NDCG@100 was 0.079820 for B2 and 0.080190 for candidate-union. The 64 origins had 15 wins, 46 ties, and 3 losses versus B2. This is a small exploratory improvement, not established superiority. Repeated GPU-time positives must not be interpreted as distinct failure episodes.

`summary.json` and `run_contract.json` are copied aggregate evidence from the completed local run. The scripts under `ML/` are source snapshots with only their repository-root path resolution changed for this archive location. Accordingly, the archived code bytes differ from the original code SHA in `run_contract.json`; the original run files remain untouched.

## Reproduction prerequisites and exclusions

The scripts require the local Experiment 12 `full_population/score_part_*.parquet`, the local Top-100 validation metrics, `outputs/branch1/cache/grid_meta.npz`, and `../data/xid_onsets_metadata.parquet`. These inputs are **not** in this Git archive. Running only from a fresh Git checkout is therefore impossible. The script's `--output` argument must point to a new, nonexistent local output folder. No raw telemetry, per-GPU risk tape, XID ledger, generated CSV, or 64 Parquet rank parts are included in this commit.

The next conservative swap experiment is specified separately in `../2026-10-04_Experiment13_ConservativeSwap_Preparation_01/PROTOCOL.md`; it has not been run.

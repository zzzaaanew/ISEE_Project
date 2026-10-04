# Experiment 13 preparation — conservative swap decision

Status: protocol frozen for an exploratory follow-up; no Experiment 13 data run or result has been produced. Goals: G-007 and G-008. Prepared 2026-10-04.

## Question and fixed inputs

Can a stricter B2-anchored replacement rule reduce harmful Top-100 displacements without losing the candidate-union replay's possible gain? Compare against exact B2 Top-100 and the completed Experiment 12 candidate-union rule. Reuse only the 64 saved B1/B2 full-population score parts from the same Experiment 12 run, the original 5-minute All-XID onset ledger, and the same 1,992-GPU grid. The previous replay had 52,945 positive GPU-times among 6,081,576 rows (prevalence about 0.87%); verify this denominator again before comparing. No Branch retraining or BLOX operation is in this protocol.

## Time and label contract

Use the existing 30-day warm-up followed by 64 daily rolling origins, 30-minute decision cadence, 24-hour binary onset horizon, 10-minute feature cutoff, and 36-hour purge. At origin time `t`, only OOF queries with score time at most `t - 36 hours` are eligible for feedback; their 24-hour labels must be mature. Use the last 14 eligible days: earlier 7 days for the same standardized L2 logistic meta-ranker, later 7 days for rule selection. Require at least 250 decision queries in each slice; otherwise return B2 unchanged. Do not use current-origin labels for fitting, threshold choice, or fallback.

## Conservative rule, fixed before the next replay

- Retain the Experiment 12 candidate union grid `M in {150,250,400}` for B2 and `N in {50,100,200}` for B1. Candidate generation remains B2 top M union B1 top N.
- Compare only B2 tail incumbents to eligible B1-supported challengers. Limit replacements to **5 per decision time**, never 10.
- Require the learned challenger risk to be at least **1.5 times** the learned incumbent risk. The B1-rank-better-than-B2-rank condition remains mandatory.
- A candidate `(M,N)` is permitted only if the mature validation slice has strictly more Top-100 hits than B2 and non-inferior summed NDCG. Additionally, a day-block bootstrap on the seven validation days, 2,000 resamples with seed `20260905`, must have a strictly positive 5th percentile of the hit difference. This bootstrap is an origin-level **gate**, not a significance claim about future performance.
- If no candidate passes all gates, emit the exact B2 ordering and `K=0`. Never relax the gates because the next origin or the full 64-origin aggregate looks unfavorable.

Tie-breaking among passing candidates should be deterministic: larger validation hit gain, then larger validation NDCG gain, then smaller `M+N`, then smaller `M`, then smaller `N`. Stable GPU ordering and the source code hash must be recorded in the eventual run contract. If the strict gate yields no active origins, report that outcome rather than changing this protocol post hoc.

## Required evaluation when separately approved to run

Report positive count and prevalence; pooled GPU-time Recall@100 and positive-query NDCG@100 as primary metrics; origin-macro versions, hits gained/displaced, actual swaps, active/fallback origins, worst-origin deltas, Precision@100, Lift@100, and PR-AUC as secondary diagnostics. Use the same 64 origins and 3,053 decision times for all three methods. Verify label and B2 baseline parity with the previous archive, no feedback after the maturity cutoff, no more than five swaps per time, exact B2 parity in every fallback, deterministic rerun under seed `20260905`, and no alteration of source scores. A future implementation should write to a new local output directory and stop on any contract violation.

The exploratory pass criterion is pooled Top-100 hits strictly above B2, pooled positive-query NDCG non-inferior to B2, and fewer losing origins than the completed candidate-union replay. Report the complete origin-level distribution and an origin-block bootstrap interval even if these criteria are met; a non-positive interval means the gain is not robustly established. If the strict gate activates no origins, report a valid B2 fallback rather than a failed run or a reason to loosen the rule.

This protocol is motivated by already-observed Experiment 12 losses; a replay on the same 64 origins is **post-hoc exploratory** even with past-only fitting. Independent validation needs a later untouched period, causal replay of Branch scores (especially B1 ADST feedback), and episode-level failure capture. No final superiority claim should be made from this saved-score ablation alone.

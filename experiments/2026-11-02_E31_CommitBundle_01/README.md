# E31 archive: four-arm ML comparison and performance analysis

Archived 2026-11-02 KST for IISE 2027 abstract preparation. Goals: G-004, G-007, G-008.
User approved archiving and committing/pushing to `zzzaaanew/ISEE_Project:main`.
Parent main: `1877e09f81d47753fccbb55ba9acb4fb8372b2ed`.

## Read first

- [Korean performance report](reports/2026-11-02_E31_성능분석보고서.md)
- [Standalone HTML report](reports/2026-11-02_E31_성능분석보고서.html)
- [Original summary metrics](run/metrics_summary.csv)
- [Original per-origin metrics](run/metrics_by_origin.csv)
- [Post-hoc diagnostics](run/analysis/2026-11-02_performance_report_01/analysis.json)
- [Frozen execution contract](run/SPEC.md)
- [Execution audit](run/final_validation.json)
- [File provenance and exclusions](ARCHIVE_MANIFEST.json)

## What completed

| Arm | Branch 1 features | Branch 1 training | Fresh B1 fits |
|---|---|---|---:|
| A | basic30 | sliding 21 days | 63 |
| B | Falcon-inspired45 | sliding 21 days | 63 |
| C | basic30 | ADST 7/14/21 days | 63 |
| D | Falcon-inspired45 | ADST 7/14/21 days | 63 |

Shared Branch 2: LR + GBDT, sliding 21 days, 126 fresh classifier fits.
Total 378 classifier fits across 63 rolling origins and 1,992 GPUs.
35,913,768 actual five-minute GPU-time raw score rows were produced locally.
Branch-only, inertia, Fusion without inertia and main Fusion were evaluated at
5-minute and 30-minute cadence for development, terminal and full periods.
D-only cap/inertia sensitivities are development-only. These score transformations
and evaluation slices are not additional independently trained runs or seeds.
The final 14-calendar-day terminal period is retrospective prequential, previously
inspected data, not a pristine unseen independent test; full includes terminal.

## Interpretation limits

Equal-weight block AP is sensitive to a terminal block with only one eligible
five-minute timestamp. Weighted and block-exclusion figures are post-hoc diagnostics,
not replacement primary metrics or a new model selection. No consistent D superiority,
Fusion benefit, cap 0.60 optimality or ADST compute saving is established.
Inertia substantially reduces ranking churn while reducing unique onset capture.
No new E30 scheduling/operational benefit is claimed.

## Preservation and reproduction

`run/` is an evidence snapshot of the original run, including frozen source copies,
configuration, fit/ADST traces, compact metrics and validation, initial failure and
resume_01 evidence. All copied run files are byte-identical to their local originals.
Only report links were adapted; original local reports and experiment remain intact.
Project/meeting records are dated snapshots; links to older unarchived local experiments
inside historical records may require the original workspace.

Large telemetry, raw GPU-time score tapes, feature/label caches, fitted binary objects,
virtual environments, temporary files and unrelated experiments are excluded.
The run was not repeated to make this archive. No new training or inference occurred.

The archived code retains original `__file__`-relative workspace assumptions for
provenance and is not a self-contained runnable package. To reproduce, restore
`run/` to `<workspace>/ISEE_Project_GitHubBase/experiments/2026-11-02_ADST_Falcon_Holdout_01/`,
provide the separately held `<workspace>/DATA` inputs described by input_manifest.json,
and use the recorded environment and original run README. Start a new clean run under
an approved experiment plan; do not reuse an already-started supervisor attempt.
Do not run `finish_handoff.py` inside this archive; it targets the original project records.

Frozen manifests contain original paths, timestamps and hashes, including hashes of
excluded large files. Their historical paths were intentionally not rewritten.
The original frozen local Blox head records execution provenance only; the archive
commit adds files on main without merging Blox history.

# Validation protocols and comparability

Every reported score needs an artifact identity, fitting/teacher/initializer ancestry, sequence membership, mask, inference precision and selection history. A short name such as “valid” does not supply these fields.

## Fixed evaluation contract

Use [`wnn.gmetric`](../../../src/wnn/gmetric.py) and the [official metric reference](../methods/COMPETITION_METRIC.md): pooled global weighted Pearson, targets and predictions clipped to ±2, weights equal to the clipped target magnitude, then mean over the two targets. Do not substitute a mean of per-sequence correlations. Blend raw predictions before the official clipping step.

Direct metadata checks found 212,140,000 train rows (10,607 sequences), 37,460,000 valid rows (1,873 sequences), and 3,528,273 scored valid rows (9.4188%). Each sequence has 20,000 rows. Two mask copies had identical counts; rowwise identity was not established by that count check. [Data receipts](../reviews/audit-2026-10-17/data_metadata.json).

## Panels in the existing research

| Name | Membership / purpose | What it cannot establish |
|---|---|---|
| creative strict425 | 425 valid sequences, 123 inferred chains, actual scoring mask | Reused development panel; not Main F0 or untouched holdout |
| creative vfold467 | 467 sequences before the strict chain filter | Not directly comparable with strict425 |
| pure860 | Two strict folds, 860 sequences / 250 inferred chains | Different membership and reused selection evidence |
| train282 | 282 training sequences / 191 chains with a propensity mask | Diagnostic mask and teacher ancestry require qualification |
| Main F0 | 468 groups / 141 components | Not creative fold0; repeated exposure remains after new seeds |
| Main F1 | 457 groups / 149 components | Not a substitute for cross-fold validation |
| Main F3 comparison | 505 groups / 145 components in the cited teacher comparison | Different recipes and precision can confound blends |
| Full supplied valid | 1,873 sequences, official mask | In-sample when final models fit train+valid; unknown teacher overlap also matters |
| Public leaderboard | Hidden organizer set, displayed rounded scores | Not private performance, significance, or a calibrated local-to-public conversion |

Chain/component grouping is inferred from available sequence relationships; it is not an organizer-provided chronological identifier or proof that every leakage path is removed. Final all-data checkpoints have already seen panels that were held out during earlier recipe development.

## Separate the uncertainty being measured

- A component bootstrap conditions on the selected models, data and exposed panel. It does not include training-seed or teacher-cohort variation.
- Two student seeds using one teacher cohort do not independently replicate the teachers.
- A fixed ensemble score differs from a mean of individual model scores.
- An interval crossing zero is an inconclusive comparison at that scope, not a rejection of every related mechanism.
- A local delta is never added to a public score. Low-LR continuation already supplied a counterexample: a positive strict425 change accompanied .6680 public versus the earlier .6682 package.

## Status and known interpretation errors

| Record | Correct interpretation |
|---|---|
| A final checkpoint has a high valid score | Establish fitting/teacher ancestry first; all-data final grades are diagnostics |
| A cloud job says SUCCESS | Inspect the scientific process and terminal result; earlier GDN jobs failed before fitting |
| A folder, recipe, ZIP or draft PR exists | Preparation is distinct from fit/export/evaluation/submission |
| A cache contains finite arrays with the right shape | This does not validate fit ancestry or the feature configuration used to produce it |
| Early trajectory-average README says pending | Later receipts already contain exported/graded seed3/4 averages; both conditional intervals cross zero |
| Train group 528 is named “corrupt” in old code | The current audited file decodes and has finite floating values; historical recipes still exclude it |

The external creative `predict_any_env.py` failed to restore the training `ZERO_COLS` setting for zero-i1 predictions. This was found by static source comparison; the cache was not regenerated. A different live `predict_run.py` restoring that setting does not repair the former file. Do not turn the affected cache into a scientific negative.

Follow the [October 17 audit](../reviews/REPOSITORY_AUDIT_20261017.md), [Main quality tradeoff](../../campaign_2026-10-15/student_quality_tradeoff_r1/README.md), [architecture frontier](../../campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md), and [historical protocol record](../archive/2026-10-12/pages/01-data-and-validation.md) for source-qualified comparisons.

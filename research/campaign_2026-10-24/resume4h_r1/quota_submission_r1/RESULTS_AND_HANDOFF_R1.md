---
kind: result
status: complete
created_utc: 2026-10-24T12:52:02Z
updated_utc: 2026-10-24T12:52:02Z
hypothesis_ids: [H-20261024-tl3-h390-quota-r1]
experiment_ids: [E-20261024-tl3-h390-quota-r1]
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/QUOTA_GRADE_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/CROSS_REPLAY_GATE_R1.json
  - submissions/quota_20261024/BUILD_RESULTS_R1.json
---

# Two checked exploratory quota submissions

## Decision

For the user's explicit request to use submission quota, try the fixed H390
S3/S4 replacement first, then the S3 replacement. The pair has higher point
scores in both sampled strata and both targets, at greater runtime cost.
The pair-minus-S3 intervals span zero in both strata; this ranking is a
descriptive choice, not an established generalization advantage. Neither ZIP
has been uploaded by this task, and no expected public gain is assigned.

1. [TL3 with H390 S3/S4](../../../../submissions/quota_20261024/tl3_h390_s3_s4_20261024.zip): 16,161,940 bytes; SHA-256 `8bf614168d97d50b58557db3a1f5befa16f4912b5b8b1fbf609b087f62f96be2`.
2. [TL3 with H390 S3](../../../../submissions/quota_20261024/tl3_h390_s3_20261024.zip): 12,537,972 bytes; SHA-256 `56ebf18d26187541e539a31078a50dcbeb698e395ca0ecde5e1174aa2e929ded`.

## Frozen endpoints and evidence

Both retain the original TL3 teachers TL5/TL6 at 0.33 each. S3 replaces the
original student at 0.34; the pair uses S3 and S4 at 0.17 each. All coefficients,
ZIP bytes and the sample were fixed before scoring. These are distinct
all-data H390 checkpoints, with no new fit or coefficient search. H390 is an
experiment identifier: these students are H256×2, not hidden width 390.

The actual packaged native callback was replayed on 64 complete sequences,
32 distinct connected components from each of F0 and F2, selected by a frozen
hash rule. All packages passed ABI inspection, finite required outputs and an
A–B–A bitwise reset check. All ordered sequence, feature and need-mask hashes
match across the three replays. The scoring adapter uses the unchanged
official metric and mask, with source-qualified target lookup and stable
pre/post prediction and target hashes.

| Stratum | TL3 reference | S3 replacement | S3/S4 replacement |
|---|---:|---:|---:|
| F0 sample, 56,531 scored rows | 0.650211369 | 0.653724440 | 0.653953547 |
| F2 sample, 59,723 scored rows | 0.711752922 | 0.711723610 | 0.711958371 |

| Fixed contrast | F0 delta [conditional 95% interval] | F2 delta [conditional 95% interval] |
|---|---|---|
| S3 − TL3 | +0.003513071 [+0.000154701, +0.006747088] | −0.000029312 [−0.002148803, +0.001920418] |
| S3/S4 − TL3 | +0.003742177 [+0.000723176, +0.006701043] | +0.000205449 [−0.001943129, +0.002102199] |
| S3/S4 − S3 | +0.000229106 [−0.000495898, +0.000956909] | +0.000234761 [−0.000251038, +0.000834608] |

The equal-weight F0/F2 mean deltas are +0.001741880 for S3 and +0.001973813
for the pair. They are descriptive averages of two sampled strata, not pooled
significance tests. The full [grade](QUOTA_GRADE_R1.json), SHA-256
`a67c728bc08b0256dd7eb0785393461491a27fb1f61325801746397c21c6e56a`,
retains every target-specific value and fixed contrast.

## Runtime and execution

| Package | Total replay seconds | Callback seconds over 1.28M rows |
|---|---:|---:|
| TL3 | 144.858 | 128.720 |
| S3 replacement | 150.550 | 133.059 |
| S3/S4 replacement | 176.348 | 159.779 |

The S3 and pair callbacks cost about 3.4% and 24.1% more than the reference on
this host and sample. No full competition-runtime guarantee follows. Native
execution exercised the library selected by this host; other packaged ISA
variants were inspected statically, not independently replayed. The original
callback/native library bytes were retained.

The grade completed in 3.545 seconds under one core and a 400,000,000-byte
worker cap. Peak memory was 399,998,976 bytes, with reclaim events but no OOM
or swap. This completed-run evidence leaves very little worker memory headroom.
The baseline resource-snapshot error is separately preserved and excluded
from runtime/resource conclusions in its [erratum](baseline_RESOURCE_ERRATUM_R1.json).

## Limits and what the result can change

H390 S3/S4 were trained on both supplied train and validation data. TL3's
retained teachers have unresolved fitting ancestry. All validation has prior
development exposure. **F0/F2 here are sampling strata, not clean holdouts.**
Bootstrap intervals resample only the 32 components within each stratum,
conditional on the fixed fitted models; they omit training, selection and
public-transfer uncertainty. Neither the positive F0 result nor the smaller
F2 difference can be added to the historical public TL3 score of 0.66832.

This supports delivering two technically checked exploratory candidates under
the user's quota instruction. It does not establish a new public best or make
these fitted diagnostics a substitute for cross-fitted endpoint evaluation.
Any later public score should be recorded as transfer evidence without tuning
coefficients against it.

## Source records

- [Build inventory and provenance](../../../../submissions/quota_20261024/README.md), [sample](SELECTION_R1.json), and [admitted native plan](REPLAY_PLAN_R1.json).
- [Cross-replay input gate](CROSS_REPLAY_GATE_R1.json), [grade recipe](GRADE_PACKAGES_RECIPE_DRAFT_R1.json), [admission](GRADE_ADMISSION_R1.json), and [execution](GRADE_EXECUTION_R1.json).
- [Independent replay review](INDEPENDENT_REPLAY_REVIEW_R1.md), [independent grader review](GRADE_PACKAGES_INDEPENDENT_REVIEW_R1.md), and [independent scalar result review](QUOTA_GRADE_RESULT_REVIEW_R1.md).

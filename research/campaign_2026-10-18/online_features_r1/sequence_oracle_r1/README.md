# Early-to-late expert preference diagnostic

This exposed-F0 analysis asks whether the sequence-specific preference between
the frozen TL3 and R3 prediction caches persists across time. It uses all 468
fold-0 validation groups and their 141 connected components. Those labels and
both endpoints have been inspected repeatedly; TL3 training ancestry is
unknown, and R3 was selected using exposed F0. The panel is not clean CV.

For each sequence and target, the diagnostic chooses TL3 or R3 by the larger
official weighted-Pearson score on scored rows 99–4,999, breaks exact ties
toward TL3, and freezes that choice for rows 5,000–19,999. The resulting
`early_label_selector` is deliberately label-informed: the test callback has
no labels, so it is a persistence probe and cannot be submitted or called
online learning. The `late_hindsight_selector` chooses from the later labels
too; it is an in-sample hindsight reference, not a formal upper bound under the
globally pooled metric.

The eight-group memory-profile pass processes only eight F0 groups and emits no
scientific scores, but the compressed NPZ format causes the full validation
target and mask arrays to be materialized. The source is already part of the
repeatedly exposed validation corpus; this is disclosed as cache access, not a
new clean holdout or an all-fold model comparison. The profile measured the
actual allocation before admitting the full diagnostic.

Predeclared comparators are TL3, R3, and a fixed 75/25 TL3/R3 mixture. The
official `global_wp_clip2_v1` metric and mask are unchanged. Uncertainty for the
early-selector late score is summarized with a paired bootstrap over the 141
source components (10,000 replicates, seed 20261025), conditional on the fixed
selector choices and endpoints. This does not measure seed, training, clean-CV
or public-transfer uncertainty; the bootstrap tail fraction is descriptive, not
a p-value.

## Decision use

A substantial late advantage plus persistent expert choices would make a
feature-derived sequence router or a cross-fitted preference teacher a
reasonable B300 study. This packet alone cannot establish that input features
can predict the label-derived choices, nor that test labels mature in time for
online updates. Those are separate mechanisms and require a label-blind
predictor study or a verified feedback contract. An inconclusive or negative
point estimate does not close the broader online-learning direction.

## Result

The early selector scored 0.693871 WP on the late window, versus 0.697669 for
TL3 and 0.694457 for fixed 75/25. Its paired conditional component intervals
are [−0.005690, −0.001453] versus TL3 and [−0.002360, +0.001586] versus the
fixed blend. R3 scored 0.679802; the late-label hindsight selector scored
0.696703 and is not a formal pooled-metric bound. Early/late expert identity
agreement was 68.16%, below the descriptive independent-marginal expectations
of 70.16% and 68.64% for the two targets. See the canonical [result note](../../../../unified/validation/SEQUENCE_EXPERT_PREFERENCE_PERSISTENCE_20261018_R1.md)
and independent [result review](../../../../unified/reviews/SEQUENCE_ORACLE_RESULT_REVIEW_20261018_R1.md).

This does not meet the B300 router trigger. The label-informed selector has no
late advantage, and the marginal-adjusted agreement is not positive evidence
of persistence. No separate B300 draft follows from this experiment; this
does not close other online-learning or learned-state directions.

## Frozen artifacts

- [Allocation receipt](allocation.json): records the hypothesis, exact scope,
  limits and predeclared use of a positive or weak result.
- [Recipe](recipe.json): panel, source identities, selection rule, uncertainty
  procedure, interpretation limits and resource reservation.
- [Grader](grade_sequence_oracle.py) and
  [source verifier](sequence_oracle_sources.py): cache identity checks happen
  before target/mask arrays are opened; after those checks pass, the grader
  materializes the full target and mask arrays from the compressed archive.
- [Independent static review](../../../../unified/reviews/SEQUENCE_ORACLE_STATIC_REVIEW_20261018_R1.md): pre-grade PASS on source binding, window alignment, scoring and conditional uncertainty; records the reviewer's archive-loading caveat.
- [Profile receipt](profile_r1.json) and [result](result_r1.json): profile
  processed 8 groups without scores; final grade covers all 468 groups.
- [Independent result review](../../../../unified/reviews/SEQUENCE_ORACLE_RESULT_REVIEW_20261018_R1.md): verifies interpretation and resources, and adds the marginal-rate baseline.
- Runtime attempts live under the campaign's shared
  [runtime folder](../runtime/README.md); their receipts include cgroup peaks,
  OOM-counter deltas and exact commands.

The analysis does not train a model, alter an input feature, create a
submission, consume the GPU, or authorize new paid compute.

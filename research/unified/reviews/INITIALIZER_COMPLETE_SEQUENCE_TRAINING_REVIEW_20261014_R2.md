---
kind: review
status: complete
created_utc: 2026-10-14T02:50:11Z
updated_utc: 2026-10-14T02:50:11Z
hypothesis_ids: [H-STATE-INIT-20261014, H-MINGRU-TIMESCALES-20261014]
experiment_ids: [E-INITIALIZER-GRADIENTS-20261014-R1]
evidence_paths:
  - src/wnn/initializer_training.py
  - tests/test_initializer_training.py
  - research/campaign_2026-10-14/cpu_student_frontier_r1/INITIALIZER_GRADIENT_TESTS_R2.xml
  - research/campaign_2026-10-14/cpu_student_frontier_r1/RECURRENT_TIMESCALES_AND_INITIALIZER_GRADIENTS_R1.md
---

# Complete-sequence initializer helper follow-up review

## Question or claim

Did the helper and method note resolve the reviewed denominator-floor mismatch while preserving an explicit statement of the remaining full-sequence versus per-chunk objective difference?

## Mechanism and method

Read-only delta review of the helper, its synthetic test source, the R2 test XML, the method note, and the ordinary cached-distillation loss path. The tracked source matches commit `fb6e5e92` exactly. Source SHA256: `8244bab83bd613dd227eb727a692600954ecbf6c428bbbb82db3b92d83f30566`; test SHA256: `1f771794e76e9d2720bd0f7e5c24e9bf794d951d0713c6751f9bbc3809523dad`; method note SHA256: `336d6181588619910f5eef66786c0810450df7f6338154c2235905dbc8ecfe24`; R2 test receipt SHA256: `108801e379bfc977025e1d54277ad606047f2027712c44558e8043389310077e`. No Python/Torch import, test rerun, profile, or fit was performed under the active 100 MB support bound.

## Evidence and interpretation

The denominator is now `weights.sum().clamp_min(1e-8)`, matching the existing `wmse` chunk-loss floor in `scripts/train_rnn.py`. The sixth test scales scored weight mass from 4 to 0.004, below one but above the floor, and checks both loss and initializer-gradient invariance. The XML receipt records all six synthetic tests passing in 3.341 seconds. This resolves the specific floor discrepancy from the R1 review.

The method note now correctly says the helper normalizes once over complete-batch sequence weight mass while the existing trainer normalizes each chunk separately. That is a deliberate objective difference in addition to the one-update-per-complete-sequence budget; it is not a parity claim. The helper still begins each call with fresh hidden state, masks warmup rows 0–98, keeps the hidden graph across chunks, freezes the backbone, and requires the first chunk to include row 99, matching the prior source review.

## Limits and validity

The helper has no production caller in the current tree; `rg` finds it only at its definition and in these tests. It checks tensor shape and fixed-supervision `requires_grad`, but does not itself check finite targets, finite/nonnegative weights, or positive total weight mass. The method note says “The caller validates” those values, which is not yet supported by a call site. Treat those checks and a positive observed mass as required gates in the forthcoming actual-data B1 profile integration. Normalization source, target/propensity ancestry, resource peak, checkpoint reload, and fit quality remain unreviewed.

## Decision and next step

The denominator-floor correction and documentation delta pass. This clears only the prior source-level floor caveat for the planned actual-data resource profile; it does not admit a complete initializer fit or establish quality. Review the caller's supervision construction and positive-mass/finite-value checks with the sealed profile packet before interpreting its profile result.

## Source records

- [R1 independent helper review](INITIALIZER_COMPLETE_SEQUENCE_TRAINING_REVIEW_20261014_R1.md)
- [Helper](../../../src/wnn/initializer_training.py), [six synthetic tests](../../../tests/test_initializer_training.py), and [R2 XML receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/INITIALIZER_GRADIENT_TESTS_R2.xml)
- [Timescale and initializer-gradient method note](../../campaign_2026-10-14/cpu_student_frontier_r1/RECURRENT_TIMESCALES_AND_INITIALIZER_GRADIENTS_R1.md)

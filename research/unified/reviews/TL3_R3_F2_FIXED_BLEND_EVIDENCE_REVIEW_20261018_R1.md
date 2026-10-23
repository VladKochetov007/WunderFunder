# TL3/R3 fixed-mixture F2 evidence review, R1

## Scope and verdict

Independent read-only review of the completed fixed-endpoint comparison in
[`tl3_fixed_blend_f2_r1`](../../campaign_2026-10-18/online_features_r1/tl3_fixed_blend_f2_r1/README.md).
The reviewer inspected the frozen recipe and amendment, replay receipts,
preflights, corrected scorer, result, official metric and component-bootstrap
implementation. No rerun, payload read, or source edit was performed.

**Pass for the reported scores and their limited interpretation; no observed
scoring mismatch.** The scorer constructs the exact fixed 75/25 and 50/50
mixtures, clears warmup rows 0–98, and uses pooled `global_wp_clip2_v1` with the
paired source-component bootstrap. Reported scores and intervals match the
packet. The initial scorer failure and the first target/mask access are
disclosed; the grade-only repair did not change endpoints, split, metric,
coefficients, or bootstrap.

The review agrees that F2 is exposed, TL3 ancestry is unknown, and R3 was
selected on exposed F0. The intervals condition on four fixed prediction
caches and do not cover fit seeds, ancestry, selection, or public transfer.
All three fixed blends remain below TL3 on this panel. This does not establish
a new B300 direction or reject different students, teachers, or conditional
mixtures.

## Reproducibility caveat

The label-blind preflight hashes the local scorer/replay files and
`campaign_score.py`, but does not hash the imported official metric and
bootstrap modules, `src/wnn/gmetric.py` and `src/wnn/evaluation.py`. Their
current file SHA-256 values are respectively
`2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa` and
`96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e`.
These are post-hoc source identities, not inputs bound by the preflight. The
review inspected the appropriate implementations and found no mismatch, but
the original receipt does not cryptographically pin the complete scoring
dependency closure. Preserve this limit; do not rewrite the frozen preflight.

## Provenance

- Hypothesis/experiment: `H-TL3-R3-F2-COMPLEMENTARITY-20261018-R1` /
  `E-TL3-R3-F2-FIXED-MIXTURE-20261018-R1`.
- Recipe and amendment:
  [`recipe.json`](../../campaign_2026-10-18/online_features_r1/tl3_fixed_blend_f2_r1/recipe.json),
  [`recipe_gradefix_r1.json`](../../campaign_2026-10-18/online_features_r1/tl3_fixed_blend_f2_r1/recipe_gradefix_r1.json).
- Result:
  [`result_gradefix_r1.json`](../../campaign_2026-10-18/online_features_r1/tl3_fixed_blend_f2_r1/result_gradefix_r1.json).
- Reviewer: independent read-only evidence critic; no fit, data access, or
  changes.

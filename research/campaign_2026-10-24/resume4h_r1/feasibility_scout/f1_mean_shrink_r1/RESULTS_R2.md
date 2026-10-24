---
kind: result
status: complete
created_utc: 2026-10-24T12:47:10Z
updated_utc: 2026-10-24T12:47:10Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-20261024-online-f1-mean-shrink-r2]
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/F1_MEAN_SHRINK_GRADE_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/GRADE_EXECUTION_R2.json
---

# Fixed correction shrink is essentially flat on F1

## Question and method

Does the already frozen additional correction multiplier of 0.1 improve the saved
F1 mean-only A endpoint against its frozen parent? This follow-up was designed
after the F0 diagnostic. It uses the same float32 expression
`parent + float32(0.1) * (raw_A - parent)`, with no coefficient search or new
model fit. The official metric and score mask remain unchanged.

The comparison covers 457 valid sequences, 149 connected components and 812,167
official scored rows. The candidate readout used seed 1002 and 1,243 fitting
groups. Recursive membership checks cover 14 ancestry runs and 27 edges;
evaluated groups are excluded from these fold-specific fitting memberships.

## Measured results

| Endpoint | WP | Difference from parent |
|---|---:|---:|
| Frozen parent | 0.669922864 | — |
| Raw mean-only A | 0.668355451 | −0.001567413 |
| Fixed 0.1 correction | 0.669918167 | −0.000004696 |

The primary shrink-minus-parent conditional 95% component-bootstrap interval is
**[−0.000112863, +0.000109508]**. Its target-specific differences are
−0.000059315 for t0 and +0.000049923 for t1. The raw-minus-parent interval is
[−0.002605495, −0.000510080]; shrink-minus-raw is +0.001562717 with interval
[+0.000624836, +0.002487594]. All three frozen contrasts are retained.

The shrink endpoint improves the early row-99–499 diagnostic by +0.000643280,
but its other fixed position bands are negative. These are descriptive slices,
not a selected serving rule. The exact counts and target-specific results are
in the [grade](F1_MEAN_SHRINK_GRADE_R2.json), SHA-256
`334249364915c454b58a0faec600f0551186fe7c13e2671d8a9d62009bfbd6ea`.

## Interpretation and limits

Shrinking the correction removes most of the raw endpoint's damage on this
cell. It does not reproduce F0 A's +0.000131368 parent-relative point gain.
The two observations weaken a claim of dependable improvement from this fixed
scale; they do not prove equivalence, identify an optimal multiplier or reject
the broader online-state family. F1 has a different fold and fitting seed, so
these axes cannot be disentangled from two cells.

All evaluation has prior development exposure. This is a post-F0-designed
diagnostic, not fresh confirmation or completion of the historical 12-fit panel.
A retains continually updated means, forecasts and errors; it is not a
no-history control. The parent historically used FP16 AMP, while the cached
endpoint was exported on CPU in strict FP32. Historical teacher-sidecar fields
remain less explicit than current metadata; the review records that limitation.
No public-score boost or serving-ready online endpoint follows.

## Execution and next decision

After preserving the R1 producer-schema failure, the reviewed R2 metadata join
passes full label-free preflight in 1.349 seconds. The admitted grade completes
in **15.036 seconds**, with one core, peak 399,998,976 bytes under the
400,000,000-byte cap, zero OOM events and zero swap. Target and both prediction
digests remain unchanged after scoring. Memory-limit reclaim events are recorded;
the worker did not exceed its hard cap.

Keep the fixed parent as the reference. A useful next diagnostic separates
parent calibration, the shared A correction and the additional relationship
direction before spending on more fits. The new Astra reframe owns that proposal;
it does not yet establish the cause of either fold's result.

## Source records

- [Frozen R2 recipe](F1_MEAN_SHRINK_RECIPE_R2.json), [source manifest](F1_MEAN_SHRINK_SOURCE_MANIFEST_R2.json), and [independent protocol review](INDEPENDENT_REVIEW_R2.md).
- [Full preflight](F1_MEAN_SHRINK_PREFLIGHT_R2.json), [root admission](F1_MEAN_SHRINK_ADMISSION_R2.json), and [grade execution](GRADE_EXECUTION_R2.json).
- [F0 diagnostic](../correction_shrink_r1/RESULTS_R3.md) and [control interpretation](../CONTROL_INTERPRETATION_R1.md).

The [independent scalar result review](RESULT_REVIEW_R2.md) checks score arithmetic, recipe/source/preflight pins, coverage and target/prediction stability. It does not rerun arrays or establish generalization; protocol review preceded execution.

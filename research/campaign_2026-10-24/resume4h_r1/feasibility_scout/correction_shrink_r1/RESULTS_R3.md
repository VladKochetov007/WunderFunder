---
kind: result
status: complete
created_utc: 2026-10-24T11:57:29Z
updated_utc: 2026-10-24T11:57:29Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-20261024-online-f0-correction-shrink-r3]
evidence_paths: [CORRECTION_SHRINK_GRADE_R3.json, EXECUTION_R3.json]
---

# A smaller correction improves the saved F0 endpoints

## Question or claim

Does one fixed additional scale of 0.1 preserve useful information in the five unique online-head corrections? Yes in this exposed F0/seed1001 cell: every shrunken correction has a small positive parent-relative point difference. The mean-only control improves almost as much as the relationship variants. This supports checking correction scale on another cell; it does not establish a relationship-specific gain.

## Mechanism and method

The original model already uses a frozen parent plus a zero-initialized learned residual multiplied by 0.1. This diagnostic evaluates `parent + float32(0.1) * (exported_head - parent)` in FP32, then applies the unchanged official clipping and metric. The extra coefficient was fixed after seeing the original F0 result; there was no coefficient search, new fit or new model inference. All five unique endpoints receive the same coefficient. The omitted A/lag16 array is byte-identical to A/lag1.

“Mean-only A” refers to its slope-zero relation estimator. It still receives211 raw/event features and matured input movements, continuing mean forecasts and their errors. It is not a no-history or no-adaptation arm. The [control interpretation](../CONTROL_INTERPRETATION_R1.md) explains why A-versus-parent cannot separate online-statistic value from generic added readout capacity.

The frozen R3 protocol declares five primary shrink-versus-parent contrasts and thirteen secondary comparisons. It rechecks each raw endpoint's score in all five bands against the original grade before writing a result. The source/payload and recursive-ancestry checks are inherited from the reviewed F0 adapter; target hashes match before and after reading. Evaluation uses 468 sequences, 141 source-qualified dependency components and 808,863 scored rows.

## Evidence and interpretation

The frozen parent scores 0.6811541494 with `global_wp_clip2_v1`.

| Endpoint | Shrunken WP | Difference from parent | Conditional component 95% interval |
|---|---:|---:|---:|
| A, lag1 | 0.6812855176 | +0.0001313682 | [−0.00002324, +0.00030374] |
| B, lag1 | 0.6812850537 | +0.0001309044 | [−0.00002505, +0.00030233] |
| C, lag1 | 0.6812903669 | +0.0001362175 | [−0.00002004, +0.00030569] |
| B, lag16 | 0.6812844455 | +0.0001302961 | [−0.00002869, +0.00030414] |
| C, lag16 | 0.6812831482 | +0.0001289988 | [−0.00002902, +0.00030014] |

The original corrections lose about 0.000544–0.000626 against the same parent. Shrinking them changes their point differences to a narrow positive range. These are strongly related outputs, not five independent replications. The mean-only A gain is within 0.000005 of the largest observed gain. Continued relationship updates are therefore not necessary to obtain most of this diagnostic improvement.

The result is compatible with an overlarge correction, regularization effects, correlated prediction geometry or fitting/evaluation residual mismatch. It does not prove optimizer overshoot or identify the cause. The original fit-side parent predictions are in-sample, unlike the evaluated predictions; that disclosed mismatch remains a concrete follow-up hypothesis.

Execution took 48.8313 seconds with zero new fits. The 400,000,000-byte worker reached 399,998,976 bytes and recorded 6,836 memory-limit events, zero OOM and zero swap. Successful completion does not imply ample memory headroom.

## Limits and validity

The coefficient and question were chosen after the original F0 grade, and the same already-exposed cell is reused. The 2,000-draw component bootstrap is conditional on these fixed predictions; its intervals are not simultaneous bounds and do not measure training, fold or public-transfer variation. Every interval spans zero. Neither that fact nor the small positive differences alone settles the research mechanism or promotes a submission.

No missing cells were fitted. The original twelve-fit F0/F1 panel remains incomplete, and historical parent training precision/ancestry is unchanged. Large ancestral prediction caches were bound through their metadata and lineage rather than newly rehashed. No public score is inferred.

## Decision and next step

Prepare a separate fixed-coefficient check of the already-saved F1 mean-only head and its own frozen parent. Freeze its coefficient, metric, source-qualified ancestry and comparison before reading its labels. This tests whether the common-control direction transfers to another existing fold/seed cell; it will still be exposed evidence and will not disentangle fold from seed effects. Additional relationship fits need a decision-changing mechanism or complete-panel rationale.

## Source records

- [R3 recipe](CORRECTION_SHRINK_RECIPE_R3.json), [source manifest](CORRECTION_SHRINK_SOURCE_MANIFEST_R3.json), [independent pre-execution review](REVIEW_R3.md), and [root admission](CORRECTION_SHRINK_ADMISSION_R3.json).
- [Complete grade](CORRECTION_SHRINK_GRADE_R3.json), [execution receipt](EXECUTION_R3.json), and [original F0 result](../RESULTS_R2.md).
- [Independent scalar result review](RESULT_REVIEW_R3.md) reproduces all125 original raw score fields and checks the18 contrasts; it does not re-open prediction or target arrays.
- [Live notebook and exported plots](../../eda/README.md). The notebook also offers descriptive position slices for the official mean score and each target.

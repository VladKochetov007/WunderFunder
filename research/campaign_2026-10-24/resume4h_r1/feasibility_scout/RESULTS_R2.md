---
kind: result
status: complete
created_utc: 2026-10-24T11:16:48Z
updated_utc: 2026-10-24T11:16:48Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-20261024-online-f0-exploratory-r2]
evidence_paths: [F0_EXPLORATORY_GRADE_R2.json, F0_GRADE_EXECUTION_R2.json]
---

# Saved F0 relationship heads do not improve their frozen parent

## Question or claim

Do the six completed F0 heads support continued relationship updates, or early-frozen relationships, over a mean-only control and the frozen parent? The newly recovered grade finds small, mixed relationship increments and a larger negative point difference between every trained correction and its parent. This is one exposed fold and fitting seed, not the missing full-panel result.

## Mechanism and method

Arm A uses means with slope forced to zero. B updates slope/intercept through row 98 inclusive, then freezes them. C continues updating them. Both lags 1 and 16 use the same F0/seed1001 cell and frozen parent; the two A prediction arrays are byte-identical and are not independent replications.

The existing architecture already freezes the parent and adds a zero-initialized correction: `parent + 0.1 * learned_residual`. Therefore the observed loss is from the trained correction; this result does not identify a missing residual connection or missing zero initialization.

The separate, preregistered F0 protocol grades seven existing fixed prediction arrays with `global_wp_clip2_v1`, original scored-row mask, clipping at ±2 and absolute clipped-label weights. It evaluates 468 sequences, 141 dependency components and 808,863 scored rows. Source/payload preflight checks 27 runtime and 15 historical grader pins, candidate payloads, and ten recursive ancestry nodes with 27 edges. Source-qualified valid fitting/evaluation overlap is zero. Large teacher prediction caches are identified by their pinned metadata and lineage; their payloads were not rehashed. Target archive hashes agree before and after scoring.

## Evidence and interpretation

| Endpoint | Official WP | Difference from parent |
|---|---:|---:|
| Frozen parent | 0.6811541494 | — |
| A, either lag | 0.6805967662 | −0.0005573832 |
| B, lag 1 | 0.6805564865 | −0.0005976629 |
| C, lag 1 | 0.6806097840 | −0.0005443654 |
| B, lag 16 | 0.6805295732 | −0.0006245762 |
| C, lag 16 | 0.6805285842 | −0.0006255652 |

| Prespecified primary contrast | Point difference | Conditional component 95% interval |
|---|---:|---:|
| C − B, lag 1 | +0.00005330 | [−0.00003841, +0.00015966] |
| B − A, lag 1 | −0.00004028 | [−0.00012523, +0.00003790] |
| C − B, lag 16 | −0.00000099 | [−0.00014716, +0.00014316] |
| B − A, lag 16 | −0.00006719 | [−0.00017714, +0.00001743] |

All fifteen declared contrasts, both targets and fixed position-bin diagnostics are retained in the grade. The strongest candidate's parent difference has interval [−0.00196645, +0.00101260]. The intervals span zero and do not measure seed, fold or public-transfer variation. Their inclusion of zero is not a universal stop rule. Here, the main allocation consideration is that the point changes between relationship treatments are much smaller than the common correction-versus-parent loss.

The grade completed in 32.7095 seconds with zero new fits. Peak cgroup memory was 399,998,976 bytes under a 400,000,000-byte limit, with zero OOM, zero OOM kills and zero swap. There were 6,001 memory-limit events, so the successful execution had little memory headroom and substantial reclaim pressure.

## Limits and validity

F0/seed1001 was already exposed, and the historical parent has its declared training ancestry, including historical precision choices. Exported FP32 predictions and pinned producer code do not establish independent checkpoint-inference parity. The original F0/F1 twelve-fit recipe and grader remain unchanged and incomplete; this separate result does not silently waive their coverage requirement. No public score or submission improvement is inferred.

The original recipe already discloses that parent predictions on fitting data are in-sample, whereas evaluated predictions are out of fit. Residual distribution mismatch is therefore a concrete alternative explanation, alongside objective mismatch, regularization, optimization or ordinary fitting variation. None is established by this comparison. Mean-only A is also a learned correction, so C-versus-A is not a pure comparison against the untouched parent.

## Decision and next step

Retain the mechanism as uncertain, while reducing the priority of blindly repeating this exact correction recipe. A bounded fixed shrinkage diagnostic can distinguish an overlarge correction from a poor incremental direction without new fitting. It must be labeled as proposed after inspecting this grade and applied uniformly across all candidates. More expensive follow-up should address the fit-side residual or objective mismatch and retain matched controls and additional source-qualified cells.

## Source records

- [Frozen R2 protocol](F0_EXPLORATORY_RECIPE_R2.json), [source manifest](SOURCE_MANIFEST_R2.json), [root admission](F0_GRADE_ADMISSION_R2.json), [grade](F0_EXPLORATORY_GRADE_R2.json) and [execution](F0_GRADE_EXECUTION_R2.json).
- [Independent protocol review](../f0_grade_review/REVIEW_R2.md) and [independent scalar result review](../f0_grade_review/RESULT_REVIEW_R1.md); the reviewer did not rerun prediction or target payloads.
- [Live EDA notebook and plots](../eda/README.md), [historical panel](../../../experiment_drafts/online-relationships-20261022/README.md), and [ten-reviewer synthesis](../../../unified/reviews/astra_ultra_audit_20261023_r1/SYNTHESIS.md).

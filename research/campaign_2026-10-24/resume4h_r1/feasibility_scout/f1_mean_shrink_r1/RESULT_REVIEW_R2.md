---
kind: review
status: complete
created_utc: 2026-10-24T12:53:29Z
updated_utc: 2026-10-24T12:53:29Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-20261024-online-f1-mean-shrink-r2]
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/F1_MEAN_SHRINK_GRADE_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/F1_MEAN_SHRINK_RECIPE_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/F1_MEAN_SHRINK_PREFLIGHT_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/F1_MEAN_SHRINK_ADMISSION_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/GRADE_EXECUTION_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/CORRECTION_SHRINK_GRADE_R3.json
---

# Independent scalar review — fixed F1 correction shrink R2

## Scope and identity

Reviewed the scalar grade output and its recipe, preflight, admission, and execution receipt. Grade SHA-256 is `334249364915c454b58a0faec600f0551186fe7c13e2671d8a9d62009bfbd6ea`; recipe SHA-256 is `3bfe93978314a6448754467a6f2106220a4f51ca0d87e4fb0130cb2bad74e243`; source manifest SHA-256 is `287371e5a8bc42c38a37d32388e70e2c9125ac98a97b44e4125643161d07e7ee`; preflight SHA-256 is `c07a2e502e0865d25075a5b1ac4001bfab185d6343b32a619b9d36ea0ababe9b`. Admission binds those inputs and authorizes this fixed target-quality read. I did not open prediction or target arrays, rerun the scorer, or execute any imports.

## Arithmetic and coverage

The endpoint arithmetic is consistent. Parent WP is 0.669922863710469; raw A is 0.668355450641337; fixed-shrink A is 0.669918167489120. Therefore raw A−parent is −0.001567413069132, shrink−parent is −0.000004696221349, and shrink−raw A is +0.001562716847783. The primary shrink−parent difference also equals the mean of its target-specific deltas: (−0.000059315055454 + 0.000049922612756)/2 = −0.000004696221349.

The primary paired component-bootstrap interval is [−0.000112862668, +0.000109507555], over 149 components and 457 sequences. It spans zero and is conditional on the fixed endpoints. The raw A loss versus parent has interval [−0.002605495080, −0.000510079631]; shrink versus raw A is positive, [0.000624835585, 0.002487593797]. This shows the fixed shrink removes most of the observed raw loss; it does not show that shrink improves the parent.

Coverage is 457 fold-1 evaluation groups, 149 connected components and 812,167 official scored rows, against 1,243 candidate fit groups with zero source-qualified overlap. The recipe uses fold 1 / seed 1002, no new fit, and one primary plus two secondary fixed contrasts. The official metric, clipping, mask and target-magnitude weights are unchanged. Candidate and parent prediction hashes match before and after scoring. The target-cache expected, pre-score and post-score hashes all match at `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`.

## Execution and limits

The execution receipt reports success in 15.036 seconds under the admitted 400,000,000-byte, one-core, zero-swap worker; peak memory was 399,998,976 bytes, with zero OOM and zero swap. The run used no new fit. Historical parent training used FP16 AMP while the cached parent endpoint was evaluated in CPU strict FP32. The candidate is a mean-only A readout whose means continue updating; it is neither a no-history nor a raw-only control.

This was designed after the exposed F0 shrink result. F0's corresponding fixed-shrink point delta was +0.000131368228 with a conditional interval [−0.000023240, +0.000303742], which also spans zero. F1 is essentially flat and does not reproduce that positive point estimate. Because the follow-up was designed after F0 and both cells are exposed development evaluation, these results do not establish an independent or stable scale effect. They do not complete the original 12-fit panel and do not support a serving, clean-CV, or public-score claim.

No arithmetic, pin, target-stability, or coverage inconsistency was found in this scalar review. The folder README had stale pre-grade language; it now links this review and records the completed admitted grade. The reviewed result itself is not a new quality grade.

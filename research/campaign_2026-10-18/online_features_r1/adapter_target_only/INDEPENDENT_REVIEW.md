# Independent grade: target-only adapter

## Finding

The frozen target-only checkpoint scores **0.680145482657** on the 468-sequence, 141-component F0 panel. It is below the R2 full-feature adapter used as the recipe's primary control (**0.680449904025**, delta **−0.000304421368**) and below the inherited parent (**0.681226513222**, delta **−0.001081030565**). The observed direction contradicts the recipe's positive prediction versus R2 for this one-seed arm. It does not establish that geometry features or the objective mechanism are generally ineffective.

## Identity and comparison checks

- [Pre-score identity freeze](independent_grade_frozen_inputs.json) binds the fold, target archive, scorer sources, recipe, candidate/control/parent checkpoints, predictions, sidecars and run provenance. The [grading receipt](independent_grade_receipt.json) binds the exact command, grader, output and hashes.
- Candidate `model_ep0.pt` is SHA-256 `1cacafc098480f6575f25c13004529399c25202f1013ee93023008f65ab0e800`; its 468-group FP32 prediction cache is `b12cc6a2b42306b1deba9881093f5185c0176ad94cbf6a25570dee5b2260f8b1`. Both hashes match candidate provenance/sidecar/score records.
- Matched R2 checkpoint `model_ep0.pt` is `f03785c3dbc364d3d12a0ad0cbe1969a50a95bdc3a95d1f4bd0f4758f654df19`; its prediction cache is `b86574111adb85cbd1a4de564f4a91f3efc2d455c20ff932312eafbae1a21f0d`. Candidate and R2 sidecars have identical ordered F0 groups and evaluation-membership hash. Their full run argument dictionaries differ only in artifact name and adapter objective (`target-only` for the candidate versus the R2 inherited default); they share seed, optimizer/schedule, sampler, data/split, feature layout, FP32 precision and initializer SHA.
- The parent checkpoint SHA is `78357eca2aec188e227544d46887935265720c35e99a1437269b952d01eb88c2`. Its FP32 cache covers 1,873 valid groups; the grader aligned its selected F0 rows by source group ID. The parent result exactly reproduces the recipe reference score.
- All three saved `fit_row_groups.valid` memberships contain 1,405 groups and are disjoint from F0's 468 `evaluation.valid` groups. Candidate and R2 fit memberships are identical by source. Train and valid parquet row-group numbers are separate namespaces and must not be unioned for this overlap check.
- A preliminary metadata check incorrectly unioned train and valid row-group IDs, producing a 463-group apparent overlap. This was a grader audit bug, not a candidate leak. Its original false flag remains in the frozen pre-score JSON; the corrected per-source result is in [independent_grade_result.json](independent_grade_result.json). That preliminary summary also treated the parent's absent `score.json` as a failed prediction binding; all three prediction byte hashes do match their sidecars, and the parent score was recomputed independently.

## Recomputed scores and uncertainty

The independent grader used the unchanged `global_wp_clip2_v1` `GlobalAccumulator`, the exact F0 row order, the stored target/mask cache, the official first-scored-row boundary of 99, clipping at ±2 and the label-magnitude weights. It scored 808,863 selected rows. Candidate and R2 WP/T0/T1 exactly match their existing `score.json` records; the parent WP matches the recipe's stored parent reference exactly.

Conditional paired component-bootstrap 95% intervals for fixed predictions are:

- Candidate minus R2: **−0.00030442**, interval **[−0.00053839, −0.00009132]**.
- Candidate minus parent: **−0.00108103**, interval **[−0.00181516, −0.00044532]**.

These intervals resample the 141 components for these fixed endpoints. They do not account for fitting-seed variation, adaptive selection, teacher ancestry uncertainty, domain transfer or public-score uncertainty. F0 is exposed and adaptively reused; this is a directional result for the tested recipe, not untouched confirmation.

## Concrete unresolved check

Old-head bitwise invariance is **unverified from saved artifacts**. The objective-fidelity receipt is an AST/code-path check, the integration receipt predates fit admission, and the training log/provenance do not record a tensorwise comparison against the parent. In accordance with the bounded review assignment, no Torch import or checkpoint tensor load was performed. The checkpoint identity is verified, but it does not prove which tensors changed.

No custom TL3/R3 grader code was reused. The small independent driver uses only the canonical metric and paired-component bootstrap implementations. The two earlier invocations failed on a grader path-construction bug before writing output; the corrected invocation completed under the recorded resource cap. No training, GPU work or public-gain claim is part of this review.

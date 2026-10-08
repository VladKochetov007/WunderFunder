# H232 fixed-reference blend grader: read-only review

Review date: 2026-10-08 UTC  
Disposition: no blocking source or membership defect found; two nonblocking admission and metadata-coherence observations are recorded below.

## Scope and limits

This review covered the frozen R2 preregistration, `grade_fixed_reference_blends.py`, the H218 metric helper, and the metadata-only producer/membership preflight. The reviewed hashes are recorded in the paired JSON receipt. The candidate prediction artifact was not read, no score was computed, and this review does not authorize grading or approve a result. No grader, preregistration, or receipt was changed.

## Checks

- The R2 grader binds the expected preregistration digest and the declared artifact set, including candidate checkpoint, prediction and sidecar identities; the expected run, split, fixed epoch, FP32 export, and source provenance fields are checked against the preregistration.
- The producer preflight binds the actual producer provenance and split. A bounded replay of the shared membership validator returned 10,535 fit-train groups, 1,405 fit-valid groups, and the 468 fold-0 evaluation groups (141 components), with no fit/evaluation group or component overlap. The fold digest matched the frozen fold-0 manifest.
- The scoring path uses `global_wp_clip2_v1`: the fixed prediction/scored mask, target and prediction clipping at ±2, absolute clipped-target row weights, and pooled target-wise correlations followed by their arithmetic mean. The grader evaluates the fixed declared alpha values and reports all preregistered blends rather than selecting a public winner.
- The fixed prediction comparison and component bootstrap are descriptive for the already exposed fold-0 panel. The preregistration states that this is not a public-transfer estimate and that the candidate/reference recipes are not matched in objective, initialization, or architecture.

## Observations

1. The grader checks the root admission's `fit_completed` and `export_completed` assertions and hashes the bound evidence files, but it does not itself parse the raw fit/export terminal receipts to validate their terminal status, exit code, attempt identity, and checkpoint linkage. This is not a scoring-code blocker because those semantics belong to the separate root artifact-admission boundary; root admission should verify them before authorizing a grade.
2. The grader compares its alpha list with `fixed_weights_on_candidate`; the preregistration also has the redundant `comparison_design.blend_weights_on_candidate` field. Those lists agree in R2. Checking both fields would guard future preregistration drift, but the current packet is internally consistent.

## Conclusion

No blocking flaw was found in the reviewed frozen R2 metric, fixed-weight comparison, or producer-membership binding. This is a source-and-metadata review only. The independent grader remains responsible for the actual score, and the controller remains responsible for validating terminal evidence and authorizing that run.

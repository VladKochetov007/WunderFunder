# Kalman innovation-gain GRU draft: read-only implementation review

**Review scope:** code and planned-comparison audit of the code-only B300 draft at `experiment/kalman-innovation-gain-gru-b300`. The independent reviewer did not edit files, inspect competition payloads, run code, fit models, profile a GPU, or validate scores. This is not empirical evidence.

## Findings

- **Causal/chunk state: no defect found by inspection.** The observer carries its last two encoded measurements and a saturated row counter between chunks. The local-linear prior handles the first rows and one-row chunks consistently. The standard training and prediction paths reset recurrent state for each sequence. This does not establish full-sequence/chunked numerical parity or serving integration.
- **Precision and provenance: code path matches the draft description.** `--precision fp32` selects the strict FP32 path; observer/core source hashes are recorded in checkpoint provenance, prediction sidecars include the prediction-driver hash, and cache construction verifies the observer identity. This is source inspection, not a run receipt.
- **Control interpretation: narrow by design.** `context_only` removes innovation only from the gain network's conditioning input. Both observer arms still use innovation in the gain-weighted correction and recurrent residual path. The primary contrast therefore tests innovation-conditioned gain against context-only gain, not innovation versus no innovation.
- **Student-cache blocker:** [`recipe.json`](../../experiment_drafts/kalman_innovation_gain_gru_b300/recipe.json) leaves `student_cache.per_cell_source_registry` null. The four cache source run IDs, checkpoint hashes, epochs, coefficients and clean-fold ancestry must be frozen and checked against producer manifests before any student cache or student fit. Until then the student half is not runnable.
- **Required prefit checks:** run and record a synthetic prefix/chunk/reset/batch-isolation check, then measure the actual H512 B300 profile before admitting the matrix. Eight B300s are the intended target, not a measured or reserved allocation.

## Disposition

The code-only draft is suitable for teammate review if these limits remain explicit. The missing registry and prefit checks block empirical student work, not publication of the draft proposal. No quality, runtime, submission, or public-transfer claim is supported by this review.

## Provenance

Read-only critique from the independent `mft_innovation_strategy` role, using the experiment branch's implementation and plan at the time of review. Supporting artifacts: [experiment draft](../../experiment_drafts/kalman_innovation_gain_gru_b300/README.md), [PR body](../../experiment_drafts/kalman_innovation_gain_gru_b300/PR.md), and [recipe](../../experiment_drafts/kalman_innovation_gain_gru_b300/recipe.json).

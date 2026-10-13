---
kind: incident
status: inconclusive
created_utc: 2026-10-13T19:16:15Z
updated_utc: 2026-10-13T19:16:15Z
hypothesis_ids: [H-QAT-A8-20261013]
experiment_ids: [E-QAT-A8-BACKEND-20261013-R1]
evidence_paths:
  - ACTIVATION_ACTUAL_GPU_R2.json
  - ACTIVATION_ACTUAL_GPU_R2_a8.log
  - ACTIVATION_ACTUAL_GPU_R2_no_a8.log
  - ../../campaign_2026-10-06/bandit_continuation/runtime/attempts/qat-a8-actual-first-chunk-r2.json
---

# Activation-QAT actual-profile wrapper invariant — 2026-10-13

## Question or claim

Why did the root-admitted actual-data scan profile terminate as a backend
failure even though both compiled recurrent modes completed their bounded
optimizer updates?

## Mechanism and method

The preserved R2 attempt used immutable source revision
`a0e3a4f699cef424027ee9a2360bbe682e313aea`. It ran the frozen CP5
`recurrent_w4` actual trainer path in two modes, with one actual B16 first
batch reused for two updates per mode and only the first 1000 causal rows
forwarded. The FP32-activation scan is the matched control for recurrent A8.
This is a compiled backend/profile smoke, not a fit or quality comparison.

## Evidence and interpretation

The controller attempt `qat-a8-actual-first-chunk-r2` ended with exit status 1
after 35.292 seconds. Both child trainer processes returned 0, compiled one
graph, completed two finite updates, and emitted all profile/resource records.
The A8 mode reported 9.437 seconds of first-forward compilation, 2.951 seconds
of first-backward compilation, and 3.256 GB allocated / 4.131 GB reserved
CUDA peaks. The no-A8 control reported 5.855 / 1.397 seconds and 3.206 GB /
4.081 GB peaks. Their child-process peak RSS values were 3.051 GB and
3.012 GB. Each update had a valid pre-update represented-weight cache identity
and cleared the cache after the optimizer step. Each mode's represented-weight
fingerprints changed between its first two forwards, and post-update
parameters were finite.

The wrapper nevertheless marked both modes `backend_failure`: every recorded
latent parameter `_version` counter remained 2 before and after each step, and
the wrapper required the counter to advance. The frozen profile uses fused
Adam. The direct child evidence distinguishes unchanged PyTorch version
counters from a failed update: optimizer counts, finite gradients/weights and
changed represented tensors all passed. Thus the first failure is an invalid
wrapper invariant, not a Dynamo/Inductor exception or a QAT quality result.

The isolated successor replaces version advancement as the update proof with
SHA-256 fingerprints of the selected raw latent QAT tensors immediately before
and after every bounded optimizer step. The receipt keeps `_version` values as
diagnostics. The wrapper now requires matching fingerprint key sets and stable
shape/dtype, a real raw-tensor value change in each update, valid cache identity
before the step, cache invalidation after it, changed represented weights,
finite update evidence and paired provenance. This retains a direct parameter
change check instead of weakening it to optimizer counters or quantized-cache
changes alone. Fingerprinting runs only in the explicit profile path; ordinary
training behavior is unchanged.

## Limits and validity

The immutable original receipt remains the execution record. The root attempt
failed at wrapper validation and is not retroactively marked successful. The
timings and peaks are observed from the attempted actual-shape profile, but the
paired backend receipt is invalid until the corrected invariant is independently
reviewed and a fresh root-owned attempt completes. No model-quality metric,
long fit, serving result, or activation-QAT gain was measured.

## Decision and next step

Keep the R2 attempt as a wrapper-validation incident. Before any successor
profile, obtain independent review of the source fix. Then run the focused
profile-validator regression tests and a fresh two-mode actual first-chunk
profile under root-owned resource admission. Require both modes to compile one
graph, complete two finite updates, prove raw latent changes and QAT cache
invalidation for each update, change represented weights, and pass every
paired checkpoint/membership/normalization/cache/source/sample-order check.
Any backend exception or missing fingerprint remains a backend failure; no
quality conclusion follows from a profile pass.

## Source records

- [Actual profile receipt](ACTIVATION_ACTUAL_GPU_R2.json) and its two mode
  logs preserve the raw first attempt.
- [Controller attempt](../../campaign_2026-10-06/bandit_continuation/runtime/attempts/qat-a8-actual-first-chunk-r2.json)
  records terminal exit status and bounded resource observation.
- [Profile-validator regression tests](../../../tests/test_profile_activation_qat.py)
  prove a changed raw latent fingerprint passes even when `_version` is
  unchanged, and reject unchanged or malformed fingerprint evidence. They pass
  without importing Torch. Root owns source review and any new profile admission.

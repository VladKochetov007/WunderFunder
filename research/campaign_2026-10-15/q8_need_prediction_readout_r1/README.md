# Opt-in Q8 readout skipping on unrequested rows

## Proposal

The native Python scorer currently calls `advance()` before inspecting
`need_prediction`, so it computes each model's readout even on rows whose
prediction is discarded. In `FastRnn::step_dyn_impl`, the input projection and
all recurrent-cell GEMVs, gate updates, and recurrent Q8 quantization precede
the readout boundary. The readout comprises only the two head GEMVs, head
activation, and output GEMV. The root's source accounting estimates the four
original member heads at about 0.658M of 4.020M matrix coefficients; skipping
them on unrequested rows could reduce work without changing recurrence.

Implement this as an explicit opt-in `PyEnsemblePredictor` constructor flag,
defaulting to `false`, for the strict `Precision::Q8Vnni` path only. Construction
must reject any actual-precision fallback or unsupported mode (including a
budget governor or shared-backbone mode) rather than silently selecting another
kernel. On unrequested callbacks, continue raw-row sanitization, causal feature
and normalization updates, initializer handling, every member's recurrent
state transition, and recurrent activation quantization; return before the
readout matrices. Requested callbacks and the original `step()`/default
`predict()` path keep their current computation. No time-based policy or weight,
coefficient, preprocessing, model, or quantization change is included.

## Validation plan and limits

In an isolated successor of this worktree, compare the opt-in callback against
the unchanged default callback using the same original four Q8-VNNI weights and
coefficients. Exercise complete 20,000-row sequences with all-true, all-false,
last-row-only, and mixed request masks; sequence changes and explicit reset;
and require bitwise equality for every requested prediction. A native step test
should also compare recurrent state/code/scale after skipped rows against the
ordinary step. Verify the strict constructor assertion rejects portable and
fallback precision. Benchmarking may report head work avoided; it does not by
itself establish end-to-end scorer speedup or prediction-quality gain.

This proposal does not alter or import fitted packages and does not authorize
an upload. Build/test work remains subject to a fresh aggregate CPU/RAM check
while the independent C/G/F/W fit delivery continues.

## R2 source triage

The [R2 source-grounded triage](TRIAGE_R2.md) closes this proposal for the
current callback workload before implementation. Only warm-up rows 0–98 have
`need_prediction=false`; using the original four-member coefficient estimate,
readout skipping can avoid at most about0.081% of per-sequence matrix work. The
separate `is_scored` mask is unavailable to this callback, so no larger skip
fraction is claimed. No implementation or benchmark was started.

## Source basis

- `native/src/python.rs`: `PyEnsemblePredictor::predict` and `advance`.
- `native/src/rnn_fast.rs`: `FastRnn::step_dyn_impl` and Q8-VNNI dispatch.
- `native/src/rnn_fast.rs` tests and callback-level Python tests: parity/reset
  validation points.

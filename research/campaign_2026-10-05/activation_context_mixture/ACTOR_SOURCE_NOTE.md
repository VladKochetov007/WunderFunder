# E484 actor source receipt

This source packet implements the registered six-arm E484 comparison for
seeds 271834 and 271835. It changes only the activation after the existing
readout hidden linear: GELU, fixed 50/50 mixture, learned static gate, or a
zero-initialized per-channel gate conditioned on recurrent state, activated
input projection, or current readout hidden state. The mixed 64-channel
representation passes through the unchanged final output linear.

The dynamic gate uses the registered non-centered channel RMS normalization
`c / sqrt(mean(c**2, channel) + 1e-6)` independently for each sample and time
step, then `sigmoid(a * normalized_context)`. Each learned gate adds exactly
64 zero-valued parameters without consuming initialization RNG. The six arms
share the same E474 initialization and seed-specific row-group permutation;
all five mixture arms begin with the same function. The training recipe keeps
E470 weighted loss and fit-only cohort, native E474 GRU, AdamW, B4 by 256-row
chunks, carried and detached state, 474 optimizer updates, and the frozen
telemetry steps.

`fit_context_mixture.py` binds the E470, E474, E477, `wnn.nn`, and
`wnn.resources` source hashes, plus the registered design, critiques, dataset,
and cohort hashes. The fit path writes each checkpoint and arm JSON receipt
before proceeding, appends and flushes one JSONL receipt per completed arm,
and emits the aggregate manifest only after all twelve fits complete. It
captures IEEE precision flags inside the E474 precision context. The separate
`gpu_probe.py` exposes an all-arm B4×256 actual-device update probe; this
packet has not launched it or any full fit.

The synthetic suite verifies parameter counts and shared bytes, RNG neutrality
of the zero gate, identical initial mixture functions, formula and gradients
for all learned gates, future-row causality, batch-element isolation, and
exact E477 prediction/state/gradient parity for GELU and fixed-mixture
anchors in GRU training mode, followed by an identical active-clipping AdamW
step on both implementations. The synthetic parity objective deliberately
scales gradients above the clip threshold; it is not the fit objective. The
bounded CPU preflight at `preflight_cpu_v4/fit_manifest.json`
performed one real fit-only update per arm under a 1-CPU/2-GiB cap. Its receipt
records twelve finite, changed cells, complete AdamW parameter groups, shared
orders within each seed, sixteen registered contrast instances, and
`test_rows_loaded: false`. This is a preflight receipt only; it is not a CUDA
probe, full fit, or result grade.

Earlier `preflight_cpu_v1` through `preflight_cpu_v3` folders are preserved as
superseded development receipts. V2's contrast list was corrected to match the
registered eight contrasts per seed before v3. V4 additionally binds the
training-mode parity and clipped-step check; no full-fit model recipe changed.

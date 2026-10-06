# E-493 / H-121 handoff

The fixed readout transfer is implemented and CPU-validated. The model keeps
the historical GELU default; `fixed_mix` applies equal GELU and LeakyReLU with
fixed slope 0.01 at the readout. It adds no parameters or RNG draws. ONNX and
native packaging reject this research-only readout until a separate port and
deployment review are complete.

The existing bounded receipt records 28 passing treatment/control tests in
3.141 seconds, including archived GELU parity through outputs, gradients and an
optimizer update, initialization and RNG identity, checkpoint inheritance,
and export/package refusal. Current packet source hashes match the working
files. Its ledger packet remains a proposal; the shared root ledger has not
been updated by this actor.

The screen has one fresh control shared with E-494 for each seed/fold. It
compares control against `fixed_mix` at seed 3/fold 0, with seed 4/fold 0 and
seed 3/fold 1 reserved for confirmation. Each endpoint is the fixed 50/50
parent-epoch-3 and head-epoch-0 prediction mixture; parent scores are retained
separately. No fit, prediction export or score has been produced.

The sealed shared-control full-shape resource probe finished successfully as
`night-e493-control-fullshape-probe-v1`: 82.162 charged seconds, 7,687,917,568
peak host bytes and 3,611,295,744 peak observed device bytes; exit 0 and no new
OOM counter. This was a short resource profile, not a full fit or a scientific
result. The fixed-mix candidate probe, independent source review and controller
fit admission remain outstanding. The two other planned probes belong to
E-494. All 26 candidate/control/prediction drafts in this directory remain
unsealed. Their six endpoint configurations are fixed across the three
seed/fold blocks and parent/complete readouts.

Next gate: complete the independent candidate-source review, seal and run its
fresh full-shape probe within the registered resource limits, then let the
controller decide whether the 12 parent fits and 12 head fits can be admitted.
The current evidence makes no predictive, public-score or deployment claim.

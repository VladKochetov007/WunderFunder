# E479: learned bounded recurrent activation shape

Registered before implementation or scoring. This is a controlled mechanism
experiment, not a public-score prediction. Original fixed-shape results guide
the question but cannot select a best checkpoint on the evaluation panel.

For each of two GRU layers, use `u=tanh(v)` and
`n=u+alpha_layer*sin(pi*u)`. Learned amplitudes are
`alpha_layer=0.2*tanh(raw_layer)`; they start at either zero or0.1.
The inverse initializer for0.1 is `atanh(0.5)`. The bound gives
`dn/du >= 1-0.2*pi > 0` and fixes both endpoints. This proves a property of
the scalar candidate map; it does not prove stability of the complete GRU.
Reset/update gates and the reset-after hidden-bias convention remain unchanged.

Four arms: fixed0, fixed0.1, learned-per-layer starting0, and learned-per-layer
starting0.1. Cross all four with AdamW and MuonAdamW, at seeds271828 and271829:
**16 fixed fits**, with no pruning from pilot scores. Retain all16 scores and
all registered contrasts. Use the E478 H64, two-layer topology, 42 causal
inputs, concatenated recurrent/embedding head and the same24 fit/16 exposed
test components. Fixed arms have61,058 parameters; learned arms have61,060.
The two extra scalar parameters are explicit, not a matched-capacity claim.
Each seed shares exactly the original base weights, identity normalization
buffers and shuffled fit order across every arm. External normalization is
computed from the24 fitting groups only. No test rows enter training.

Fit one complete schedule, batch4, chunk256,474 updates. Both optimizers use
LR0.001 and WD0.01 for original weights, Adam betas(0.9,0.999). Muon selects
exactly the four stacked GRU matrices with three gate blocks each. The learned
shape parameters use a separate AdamW group with LR0.001 and **zero decay**.
They must remain differentiable tensor arguments to the candidate scan;
`.item()`, conversion to Python float, or detachment inside training is invalid.
No new RNG draw may shift common base-weight initialization.

Train-only telemetry at updates0,46,236,473: amplitudes for both layers,
raw-parameter gradients, raw-parameter updates, original four parameter-family
RMS statistics, loss, and state ranges. Verify nonzero alpha gradients and an
actual alpha update in the tiny probe. At alpha0 the modified recurrence must
match the original baseline in outputs/state, input and parameter gradients,
and a clipped optimizer step under common FP32 policy. Fixed-versus-learned
same-initial-alpha forward parity must pass before training. Test causality,
carried/chunk/reset state, strict optimizer partition, finite updated tensors,
scalar candidate bounds and parameter counts. Preserve any failures.

Primary contrasts, separately within each seed and optimizer:

1. learned-start0 minus fixed0;
2. learned-start0.1 minus fixed0.1;
3. fixed0.1 minus fixed0;
4. learned-start0.1 minus learned-start0.

Report all16 paired effects plus8 optimizer contrasts (Muon minus Adam for
each shape/seed) and4 learned-versus-fixed by optimizer interactions jointly
resampling the same16 components. All28 comparisons remain exploratory;
component intervals condition on fitted weights and do not cover seed/search
uncertainty. Do not pool seed CIs or promise a public gain.

Mechanism hypothesis: learning a separate candidate slope redistribution per
layer adapts to depth and optimizer, whereas a universal fixed alpha imposes
the same response everywhere. Fixed controls and both initializations separate
learning from merely starting with the earlier preferred shape. The hypothesis
weakens if learned amplitudes do not move, gains reverse across seeds, or the
same-initial fixed control ties or wins. Learned tensor graph integrity is a
validity gate rather than a scientific negative.

Admission: independently criticized source, bounded CPU checks, representative
real-device finite-update/parity probe, then a frozen source/data contract and
full fits. One local GPU worker,2 CPU,6GiB worker RAM,2GiB allocator and3GB
observed stop; actual aggregate12 CPU/30 decimalGB hard RAM/27GB soft/no swap.
Approximate reserve30GPU-job-minutes, charged against the existing local block
and80-device-GPU-hour campaign ceiling. No new budget. Admission deadline
07:00UTC /10:00Kyiv October5. Stop new admissions then; admitted workers drain.

# E481: matched central slope and endpoints, different candidate curvature

Registered before implementation, probe, fitting or scoring. This follows the
bounded-candidate observations without assuming the sine itself is necessary.

Write `u=tanh(v)`. Compare three candidate maps:

1. baseline: `n=u`;
2. sine: `n=u+0.1*sin(pi*u)`;
3. cubic: `n=u+0.1*pi*u*(1-u*u)`.

Sine and cubic have the same value and derivative at zero and the same fixed
endpoints at ±1. They redistribute slope differently away from zero. Their
endpoint derivatives differ, so this is not a control matching every aspect
of the function. Both maps remain bounded and strictly monotone on [-1,1].
For the cubic, the derivative is `1+0.1*pi*(1-3*u*u)`, with minimum
`1-0.2*pi > 0`. For the sine, the minimum is `1-0.1*pi > 0`.
These scalar bounds do not prove stable gradients through a complete GRU.

The sine argument here is limited to [-pi,pi]. This candidate does not expose
an unbounded periodic feature coordinate. A positive result alone would not
establish a periodic inductive bias. The cubic is also an inference-cost lead:
its candidate correction uses multiplications rather than a transcendental
sine. Native scorer timing and numerical parity remain separate future gates.

Cross all three maps with AdamW and MuonAdamW at seeds271828/271829/271830:
18 full fits, no pruning. Use E478's H64 two-layer concatenated embedding/state
readout, 61,058 parameters, 42 inputs, original sigmoid gates and reset-after
hidden candidate bias. Common baseline initialization and fit order per seed,
fit-only external normalization and identity internal buffers. Exactly the
same24 fit and16 exposed evaluation components as E478; no public forecast.

Use B4, chunk256, one epoch,474 updates, constantLR0.001, WD0.01,
Adam betas(0.9,0.999), global gradient clipping1.0. Muon only on the same four
stacked GRU matrices, three gate blocks. Keep the original IEEE recurrent
precision controls; Muon's inherited CUDA BF16 orthogonalization is unchanged.
No added learned parameters, dropout, checkpoint selection or test feedback.

Source checks: alpha-zero output/state/input and parameter-gradient parity,
clipped AdamW and Muon update parity, scalar derivative bounds and endpoint
checks, carried/chunk/reset state, prefix causality, optimizer partition,
common initialization/order and actual finite parameter-changing B4×256 probe
for every cell. Independent critic precedes a resource-bound CUDA probe;
independent scorer follows frozen complete fits. Preserve all failures.

Report all18 official global/t0/t1 scores. Within each seed/optimizer, report
the three pairwise effects: sine-minus-baseline, cubic-minus-baseline,
cubic-minus-sine (18 effects total). Add the nine Muon-minus-Adam contrasts and
three jointly resampled curvature-by-optimizer interactions, for30 contrasts.
Use2000 draws of the same numeric-sorted16 connected components, joint
multiplicities for interactions, and no pooled seed CI or best-arm selection.
Export raw selected predictions/labels/masks/IDs and all contrasts.

If cubic matches or beats sine consistently, sine-specific curvature is not
required at this scale. If sine beats cubic, the difference is a candidate
shape result, still not proof of a periodic market process. Direction changes
across optimizers or seeds are evidence to retain, not reasons to discard cells.

Charge against existing80-device-GPU-hour and local24-hour block budgets.
One local GPU worker,2CPU/6GiB RAM,2GiB allocator/3GB observed stop, actual
aggregate12CPU/30decimalGB hard/27GB soft RAM/no swap. Reserve30GPU-job-minutes.
Admission closes07:00UTC/10:00Kyiv October5; admitted work drains gracefully.

# Blind specification critique: E489

Reviewed `E489_DESIGN.md` at SHA-256
`0859e6ac528b850452183afc46ab28358cac13c97ace5cb0350e413c2b9a7090`
before inspecting any implementation. The question is coherent and useful:
whether the registered bounded recurrent activation's effect changes during
continued optimization under the newly explicit FP32 Muon orthogonalization
recipe. The design correctly says this cannot retrospectively validate the
failed E485 BF16 run or make a general public-performance claim.

The experimental counts reconcile. Three fresh seeds times two activation
arms gives six uninterrupted trajectories; four epochs at 474 updates gives
1,896 updates per trajectory and 11,376 total full-fit updates. Epoch 1, 2,
and 4 snapshots yield 18 snapshots. For each seed the three saved epochs
produce nine sine-minus-alpha0 contrasts; two predeclared within-seed
duration interactions yield six more, for 15 total. The interaction formulas
should be written explicitly in the machine-readable registration as
`(delta_epoch2 - delta_epoch1)` and `(delta_epoch4 - delta_epoch1)`, where
`delta_epoch = WP(alpha0.1, epoch) - WP(alpha0, epoch)`. Keep all contrasts
and do not pool seeds or select epochs from evaluation.

Paired initialization, sample order, fit-only normalization, sequence-state
carry/detach, reset boundaries and continuous optimizer moments are specified
well. A probe of each of the six seed/arm cells for two successive chunks is
appropriate for checking both fresh and continued state/optimizer behavior;
for each seed, alpha0 must be compared with the native reference at forward,
state, input/parameter gradient and clipped-update levels. The alpha0.1 arm
must separately show finite gradients/state/parameters and a changed update.
Do not infer full-fit stability merely from the alpha0 parity control.

Before an actor starts, make two provenance contracts explicit in a
machine-readable preregistration addendum: (1) record exact immutable source
hashes for the H128 topology/candidate transform and the E487 FP32 Muon
implementation the new packet is permitted to reuse; state the candidate
equation and bounded alpha values there, rather than depending only on a
historical experiment name; (2) make E487's independent saved-delta arithmetic
pass and the no-evaluation-use receipts for seeds 271836–271838 hard launch
preconditions. The seed-exposure statement is a factual receipt gate, not
just design prose. The new optimizer is a distinct treatment, so preserve all
three seeds and both arms even if the expected-shape parity passes.

The stated projection is internally consistent: measure synchronous update
times for both successive chunks in all six cells, then use the maximum times
11,376 updates, a 1.5 safety factor, 120 seconds fixed overhead and actual
probe elapsed time. Require the result to be at most 7,200 seconds before
full-fit admission, and retain the separate 600-second probe limit, 2 GiB
allocator cap and 3 GB physical-device stop. This projection is deliberately
conservative but should be computed from the slower of the two chunks per
cell, not an average. Keep allocation/setup, snapshot writes, teardown and
monitor overhead inside the registered margin/cap. The exact 07:00 UTC
admission cutoff applies to new full-fit admission; already admitted jobs can
finish under their sealed timeout.

One telemetry choice deserves a short clarification before freeze: the
registered fixed steps end at 473, 947 and 1895, corresponding to the end of
epochs 1, 2 and 4; there is no epoch-3 boundary sample. This is consistent
with the saved-snapshot epochs but should be stated as intentional so later
readers do not mistake it for a missing trajectory update. The six points
remain fixed unless the registration is amended before implementation.

**Disposition: GO for implementation after the provenance/exposure
preconditions above are recorded.** No implementation or data was inspected
for this blind critique. The actual E487 saved-delta result, all-three-seed
alpha0 parity, 12-cell CUDA probe, bounded timing projection and source review
remain independent admission gates; this critique admits no GPU work or fit.

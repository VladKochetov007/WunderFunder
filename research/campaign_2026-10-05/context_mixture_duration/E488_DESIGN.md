# E488: causal mixture gates across training duration

Registered before implementation. E484 found very small, seed-dependent
dynamic-versus-static gate effects after one epoch. Learned coefficients
start at zero; a one-epoch result need not characterize the trained family.
This study asks whether longer optimization changes the difference, without
changing the gate, optimizer, data or model capacity within a matched arm.

Use the exact six-head E484 architecture but retain these four arms only:
`gelu`, `fixed_mix`, `static_mix`, `recurrent_mix`. Their equations, per-row
noncentered context RMS normalization (epsilon1e-6), LeakyReLU slope0.01,
sigmoid gates and zero coefficient initialization are unchanged. H64,
two-layer native GRU,42 causal features, no dropout; GELU/fixed anchors have
61058 parameters and learned gate arms61122. Capacity differences remain a
limitation of learned-versus-fixed contrasts. Context-dependent versus static
heads have equal parameter counts, not equal function classes.

Run 12 continuous trajectories at fresh seeds271839/271840/271841, four arms
per seed. Initialize the common trunk/head bytes once per seed, clone into
all four arms; adding zero gate parameters must consume no RNG. All mixture
arms agree at initialization; pure GELU intentionally differs. Train three
complete epochs with AdamW LR0.001, betas(.9,.999), weight decay0.01, global
gradient clip1.0, FP32 IEEE/no AMP. No teacher, EMA, scheduling or fine-tune.
Each epoch has474 updates (six batches×79 chunks), B4×256; state carries and
detaches between chunks and resets between batches and epochs. Shuffle the
canonical24 fit-role positions with `default_rng(seed+epoch_index).shuffle`
on `np.arange(24)`. Optimizer moments continue across all three epochs.
Save epoch1 and epoch3 checkpoints immediately, with row-order/update/source
and normalization receipts. These are24 snapshots from12 fits, not24 fits.

Predeclared scores use the same16 historically exposed components, only
after every trajectory finishes and fit/probe/source receipts validate.
For each seed and each saved epoch retain four GlobalWP contrasts:
recurrent-minus-static, recurrent-minus-fixed, static-minus-fixed, and
fixed-minus-GELU. Also retain the interaction
`(recurrent_3-static_3)-(recurrent_1-static_1)`. This is27 global contrasts,
with t0/t1 scores and target-wise counterparts retained. Use2000 common
component-bootstrap draws at seed271839, numeric-sorted component IDs;
no seed pooling, checkpoint selection, or recipe adaptation from evaluation.

The inventory is exactly12 trajectories/24 saved snapshots/27
global contrasts. The fixed count controls should reject any extra/missing
arm, step, snapshot or contrast rather than reinterpret partial completion.
Report sampled coefficient RMS/gate range/preclip-gradient RMS/head update
RMS at fixed updates0,46,236,473,947,1421. These describe training, without
being fitted or selected from evaluation labels.

Require independent blind design critique, then independent source review,
fit-only CPU checks and an actual B4×256 CUDA probe. Probe must match native
E484 reference output/state/input/all-parameter gradients and clipping-active
AdamW parameter updates for GELU and fixed mixture, exact on the same device.
One2CPU6GiB GPU worker,2GiB allocator/3GB physical stop, parent12CPU/30decimal
GB/no swap; admit at most900GPU seconds including probe and full trajectories
from existing budgets, with measured projection and overhead margin. Stop
new admissions at07:00UTC. Already admitted trajectories finish gracefully.

No new-market or public prediction claim follows from this reused panel.
Positive later-epoch gate effects would support a duration interaction for
this recipe; a null cannot reject mixtures universally. This is independent
of E485's failed custom-recurrent Muon step: the native recurrence and AdamW
remain unchanged here. No production module changes or provider publishing
are authorized by this design alone.

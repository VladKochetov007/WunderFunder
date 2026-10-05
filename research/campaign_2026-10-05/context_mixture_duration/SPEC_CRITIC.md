# Blind specification critique: E488 duration study

Reviewed `E488_DESIGN.md` at SHA-256
`3764a095504af6da66a4faa908a47ff5dd3f94266249d5bea7c07cf611148824` before
inspecting any implementation. The preregistered question is well scoped:
whether the small one-epoch recurrent-versus-static context-gate difference
changes after continued optimization under the same native GRU, AdamW recipe,
cohort, and arm definitions. It does not claim that epoch 3 is converged or
that a null rules out context mixtures generally.

The paired design is coherent. All four arms within each seed need the exact
same trunk/head initialization and sample order. Allocating the two learned
gate vectors must consume no RNG; assert byte equality of their common initial
parameters and equality among the three mixture-arm outputs at initialization
(the pure GELU anchor is intentionally different). Keep AdamW moments alive
across epochs, while resetting recurrent state at each epoch boundary because
the sequence order changes. Carry and detach state only across the 79 chunks
of one sequence batch, then reset before the next batch. Verify the exact 474
updates per epoch and fixed 0-based snapshot update indices 0, 46, 236, 473,
947, and 1421.

Before code freeze, make two inherited-data conventions explicit in the
machine-readable source contract: (1) epoch numbering is zero-based, so each
shuffle uses `seed + epoch_index` for indices 0, 1, 2; and (2) every arm/epoch
uses the identical E484 train-only normalization derived from the canonical
24 fit groups, with the E484 model buffers kept at their identity values if
normalization is external. Record the normalizer and canonical cohort hashes;
do not refit statistics per arm or per epoch. Bind the exact E484 model,
optimizer helper, source, data, and cohort identities rather than relying on
directory names.

The registered contrasts are count-consistent: four global comparisons per
seed at each of two epochs (24), plus one within-seed difference-in-differences
per seed (3), for 27 total. Preserve every contrast and all target-wise and
per-sequence counterparts. Use one sorted component-ID list and one common
set of 2,000 bootstrap component draws for every arm and both epochs, then
recompute each WP contrast on each draw. Do not bootstrap arms separately,
pool the three seeds, or pick an epoch by its test score. The interaction is
specifically `(recurrent_3 - static_3) - (recurrent_1 - static_1)` within each
seed; it measures this recipe's duration interaction, not a general causal
effect of recurrence.

Fit-only and device admission should fail closed on any missing/extra arm,
seed, update, snapshot, or changed input. In addition to exact native parity
for GELU and fixed-mixture output/state/input/parameter gradients and the
clipping-active AdamW step, run one actual-shape finite/changed-update probe
for all four arms. That catches a learned-gate graph or clipping issue that
the two reference arms cannot exercise. Keep all four arms on the same B4 x
256 batches, optimizer groups, seed order, and precision policy. Check finite
loss, gradients, parameters, optimizer state, and recurrent state at every
update; retain per-arm order and chunk-start telemetry. For the six registered
diagnostic updates, report preclip norms, clipping factors, gate ranges and
coefficient RMS, and head-update RMS. These are training diagnostics only.

The 900-second ceiling is plausible only after a measured full-trajectory
projection. Record probe duration separately, estimate all 12 trajectories
from actual H64/B4x256 throughput with overhead margin, and do not begin a
trajectory if the admitted budget cannot finish all 12. A failed probe must
not be repaired by silently reducing steps, batches, arms, or epochs. Stop new
admissions at the registered cutoff and let already admitted runs finish
gracefully.

**Disposition:** GO for independent source implementation and fit-only CPU
checks after the normalization, epoch-index, and telemetry contracts above
are explicit. CUDA admission remains contingent on independent source review,
the actual-shape probe, all-arm finite/update checks, and a measured budget
projection. The result can support only a duration interaction in this fixed
recipe; it cannot support public prediction claims or a claim that mixtures
are universally closed or open.

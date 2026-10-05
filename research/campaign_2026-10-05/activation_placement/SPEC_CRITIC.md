# Blind specification critique: E474 activation placement

This critique uses the frozen `activation_optimizer/ACTIVATION_NEXT_DESIGN.md`
and the existing `src/wnn/nn.py` / E470 cohort code. It was written before
reviewing a new `activation_placement/` implementation. It does not authorize a
fit or held-out result selection.

## Scope and design correction

The six arms per seed isolate the site of one fixed pointwise shape on the
incumbent H64 topology: GELU/GELU baseline; sine or linear addition at the
embedding only; sine or linear addition at the head only; and sine addition at
both. Use two paired seeds, 271828 and 271829, the fixed AdamW recipe, 24 fit
and 16 already-exposed test components, 474 updates per fit, and no per-arm
trainable activation coefficients. This answers a narrow localization and
sine-curvature question at H64. The exposed components are exploratory only.

There is a trainable-parameter-count typo in the design: the repository's
`RNNModel(42, hidden=64, layers=2, dropout=0, n_out=2, cell="gru",
enc="linear")` has **61,058** trainable parameters, not 61,826. I independently
counted the live `RNNModel`: input projection 2,752; two-layer GRU 49,920;
concatenation head 8,386. Its H256 count of 932,354 in the design is correct.
Treat 61,058 as the H64 invariant and require it for every arm; parameter-free
activation substitutions must not change it. The RNNModel's `mean` and `std`
are buffers, not trainable parameters.

## Architecture and initialization invariants

Construct the actual `wnn.nn.RNNModel` with 42 inputs, hidden width 64, two
`nn.GRU` layers, zero recurrent dropout, linear encoder, and two outputs. Keep
its parameter names and forward path: normalized `x` enters `inp`; GELU makes
`z`; the ordinary fused GRU produces `o`; concatenate **`[o, z]`** in that
order; then apply `Linear(128,64)`, the single head hidden activation, and
`Linear(64,2)`. Embedding-only changes affect both the GRU input and the
unchanged skip `z`, as intended. Head-only changes only the activation between
the two head linear layers. Do not replace or unroll the GRU, reorder the
concatenation, change either linear layer, or substitute the E470 width-32
residual-head model.

For each seed, instantiate one GELU/GELU baseline with that seed, clone its
entire state dict (including the zero/one normalization buffers), and load the
identical tensors into every arm before replacing only parameter-free
activation modules. Assert exact equality for every trainable parameter,
buffer, and parameter name across the six initial models. Resetting a seed
independently before constructing each differently structured model is weaker
than cloning the one baseline, because module construction can consume RNG
differently. Parameter counts, key sets, initialization, normalization, and
forward parity at all-GELU must be tested before fitting.

E470 supplies the 42 causal features already normalized and clipped using
fit-only cohort statistics. `RNNModel` defaults to mean zero and std one, so
the bounded harness must preserve those neutral buffers and must not apply the
same normalization a second time. Verify its exported input tensor is the
same frozen E470 tensor by identity/hash or exact fit-only reconstruction.
Targets are checked finite before clipping to `[-2, 2]`; training uses E470's
absolute-target weighted MSE with the `need_prediction` training mask. The
official evaluation selection is the conjunction
`need_prediction & is_scored`. Do not substitute one mask for the other.

State and update bookkeeping must match the fixed recipe: shuffle the exact
24 fit row groups once per seed and reuse the same order for all six arms;
batch four sequences; begin each batch with zero GRU state; carry state over
the 79 contiguous 256-row chunks and detach after each optimizer step. This is
6 batches × 79 chunks = exactly 474 optimizer updates per arm. At evaluation,
start each test sequence with fresh state, carry it over every chunk, and
score only its official selected rows. State shape must be `(2, B, 64)`.
No test rows may affect normalization, loss, ordering, scheduling, or
calibration. Keep sequence/component provenance with each export.

## Activation, optimization, and backend controls

Implement exactly `GELU(x) + 0.1*sin(x)` and `GELU(x) + 0.1*x`; both have
derivative 0.6 at zero and no learned coefficients. GELU stays unchanged at
unmodified sites. The input sine also changes the skip-path values, so its
effect is the full embedding-site intervention, not an isolated recurrent
input path. The head sine is the hidden readout site. These formulas should
match at training and scoring. Test formulas on positive/negative, near-zero,
and large finite inputs, finite gradients, output shape, exact GELU baseline
parity, and parameter-free status.

All arms use AdamW `lr=0.001`, `weight_decay=0.01`, betas `(0.9,0.999)`,
global norm clipping 1.0, constant LR, no dropout, and one epoch. Keep the
same optimizer construction and identical shuffled batches per arm. A common
nominal LR is not a match of effective functional step size: changed
activations alter gradient values, clipping frequency, Adam moments and
parameter-update RMS. Capture the preregistered gradient and per-family update
diagnostics, but do not use them to retune within these 12 fits. The design is
not a tuned optimizer comparison and its constant LR is not the full RNNModel
OneCycle recipe.

Run every arm under one explicitly recorded FP32 precision policy: TF32
disabled for CUDA matmul and cuDNN, cuDNN enabled, highest FP32 matmul
precision, autocast disabled, and FP32 inputs/parameters. This standard-GRU
study is numerically distinct from E471's custom recurrent cell, but the
common policy also prevents backend settings from varying by arm. Do not
compare its scores as exact replicas of historical E470 results that used
cuDNN TF32. Preserve the same policy from production-shape probe through all
fits and evaluation.

## Scoring and statistical interaction

Use the official pooled `global_wp_clip2_v1` calculation: clip selected raw
targets and predictions at ±2 only after raw-finite checks; compute weighted
Pearson separately for `t0` and `t1` over all selected rows pooled across
components; report their mean as WP. Keep both per-target values and selected
row counts. Recompute scores from exports rather than trusting training
summaries.

For uncertainty, resample the 16 component identities in paired fashion and
recompute each arm's pooled, two-target GlobalWP inside each bootstrap draw.
Then compute paired differences. Do not average sequence-level WPs or treat
rows as independent: Pearson is nonlinear, and a mean of component WPs is not
the official pooled score. For each seed, report `sine_embed - base`,
`slope_embed - base`, `sine_embed - slope_embed`, analogous head contrasts,
`sine_both - base`, and the predeclared interaction

`WP(sine_both) - WP(sine_embed) - WP(sine_head) + WP(base)`.

Compute that interaction in each paired component-bootstrap draw from the four
pooled-correlation estimates, using the same sampled component multiplicities
for every arm. Do not derive it from unweighted per-component correlations or
from independently bootstrapped arms. Keep seed-specific tables and intervals;
two training seeds do not justify pooling into a seed-generalization interval.
Retain every preregistered contrast and both seed results; no exposed-test
winner selection or multiplicity-free significance claim.

## Falsifiers and failure cases

- Reject a purported sine-specific effect if localized sine does not
  consistently beat its matched-slope arm at the same site across both seeds,
  or if its component pattern is not directionally coherent. Parallel sine
  and slope changes versus baseline point toward local slope/scale changes,
  not periodic curvature. A single positive seed or marginal conditional
  interval does not reverse that conclusion.
- If an arm changes any parameter count/key/value at initialization, uses a
  different fit order, optimizer state, schedule, precision flag, feature
  normalization, state boundary, or scoring mask, its paired contrast is
  invalid. Fail closed; do not fix it by post-hoc seed or LR selection.
- Reject incomplete/nonfinite rows before clipping, duplicate or overlapping
  components, missing selected targets, different sequence segmentation,
  stale source/data hashes, or any arm not completing exactly 474 updates.
- Reject a nonfinite loss/gradient/update/prediction, a no-op optimizer step,
  or evaluation without resetting state per sequence. CUDA memory, throughput,
  and timeout must be measured on the actual two-layer H64 B4×256 topology
  before any full matrix; E471's different width-one custom-cell timings are
  not a capacity estimate.
- Do not infer H256, full 211-feature, OneCycle, dropout, deployment, or public
  benchmark effects. This screen changes the activated embedding/readout
  sites together with their induced gradients; it cannot establish a
  universal activation mechanism.

## Admission checklist before implementation review

Require a CPU test proving exact cloned initialization and all-GELU forward
parity against `RNNModel`, exact count 61,058, equal `state_dict` keys and
shapes across all six arms, formula/gradient tests, no hidden stochastic layers,
and one finite real fit-only optimizer update. Then independently review the
sealed source, data/cohort identities, scorer/paired-bootstrap implementation,
resource watchdogs, and a parent-run CUDA B4×256 probe under the common FP32
policy. This document is a specification critique only; the implementation
and every result require a separate review.

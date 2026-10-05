# E478: recurrent candidate transfer to incumbent H64 topology

**Design frozen before implementation.** Source-only stage: no CUDA probe, full
fit, or test-group read is authorized here. Any later score comparison is
exploratory because the fixed 16 test components were exposed previously.

## Question and fixed matrix

E471/E475's bounded candidate-activation comparison reported a positive
`alpha=+0.1` versus `alpha=0` direction for Muon across three fixed seeds on a
smaller one-layer architecture. E478 asks whether that candidate-level
optimization effect transfers to the actual incumbent recurrent layout:
42-input linear projection with GELU, two H64 GRU layers, concatenated
`[recurrent, embedding]` readout and GELU MLP head. Compare exactly two fixed
values `alpha in {0.0, +0.1}` at each of seeds `271828, 271829, 271830`, all
using the same MuonAdamW recipe. This is six fits, not an optimizer comparison
or a candidate search.

## Exact recurrence and parameter layout

Keep the reset and update gates as sigmoid and retain PyTorch GRU gate order
`reset, update, candidate`, both input and hidden biases, and PyTorch's
reset-after-hidden-affine placement:

```text
r = sigmoid(i_r + h_r)
z = sigmoid(i_z + h_z)
u = tanh(i_n + r * h_n)
n = u + alpha * sin(pi * u)
h_next = (1 - z) * n + z * h
```

Here `h_n = W_hn h + b_hn`, so both hidden candidate weight and bias remain
inside the reset multiplication exactly as in `nn.GRU`. Apply the existing
TorchScript E471 single-layer scan successively to both layers, feeding layer
0 outputs to layer 1. Keep `nn.GRU(64,64,num_layers=2,batch_first=True,
dropout=0)` only as the registered parameter container. Preserve every
`RNNModel` `state_dict` key, shape, and initialization; recurrent state shape
is `[2,B,64]`. The custom module's forward must match ordinary `RNNModel` at
alpha zero within the tolerances below.

The candidate is bounded: on `u in [-1,1]`, it fixes both endpoints and has
derivative `1 + alpha*pi*cos(pi*u) >= 1 - 0.1*pi > 0`. Thus it is monotone,
maps into `[-1,1]`, and convex GRU state updates preserve the hidden-state
bound. This is a bounded local candidate-shape change; it does not establish
long-horizon gradient stability.

## Initialization, optimizer, and fit recipe

For each seed, instantiate one actual
`RNNModel(42, hidden=64, layers=2, dropout=0.0, n_out=2)` and clone its full
state dictionary to both arms. Require exact parameter names, shape, values,
buffers, and 61,058 parameter count. Alpha zero is the custom-equation/backend
control; it may not reuse fused `nn.GRU` for the candidate fit.

Use the E470 frozen cohort and feature path, loading the 24 fit groups only.
Use the same NumPy-seeded 24-group order for both alphas at each seed. Train
with B=4, chunk size 256, state reset at each batch and carried/detached across
the 79 sequence chunks, one epoch, exactly 474 updates, FP32 inputs and
parameters, no dropout and the E474 common IEEE FP32 policy. Keep the unchanged
weighted-MSE and `need_prediction` mask. Use MuonAdamW only: Muon on exactly
the four two-layer GRU matrices (`weight_ih_l0`, `weight_hh_l0`, `weight_ih_l1`,
`weight_hh_l1`) with three gate blocks each, AdamW on all remaining input,
head, and bias parameters; lr 0.001, weight decay 0.01, betas (0.9,0.999),
constant LR and global gradient norm clip 1.0. Verify the Muon/AdamW groups are
a disjoint exact cover of trainable parameters. Do not add an Adam-only arm.

At zero-based updates 0, 46, 236, and 473, reuse E474's fixed fit-only
preclip-gradient RMS and post-update parameter-delta RMS telemetry for `inp`,
`gru`, and `head`. Telemetry is descriptive and cannot adapt the run.

## Predeclared parity thresholds and tests

Before any GPU probe, the alpha-zero custom model must pass ordinary
`RNNModel` comparisons on CPU and at actual CUDA B4x256 under the common full
IEEE FP32 policy. Use fixed acceptance tolerances:

- output/state: `atol=2e-5`, `rtol=2e-4`
- input/parameter gradients: `atol=5e-5`, `rtol=5e-4`
- matched Muon optimizer parameter values after one clipped step:
  `atol=1e-4`, `rtol=1e-3`

These thresholds inherit E471's previously passing GPU forward/gradient
budget, with a predeclared modest allowance for the extra stacked layer and
the nonlinear upstream embedding/head reductions. They are fixed before E478
observations; do not widen them after a failed probe. If the actual-topology
parity check fails, stop and report the failure without fitting.

CPU tests additionally check exact cloned initialization/state keys, the
61,058 count, finite candidate values/gradients and `[-1,1]` bound, future
perturbation causality, full-sequence versus carried-state chunk parity,
state reset, and finite one-update gradients/parameters for both alphas. The
fit-only CPU preflight performs one B4x256 update for each of the six
seed/alpha arms. The parent-owned production-shape probe performs actual-data
alpha-zero output/state/input+parameter-gradient/one-step parity, then one
actual-data update per fixed fit arm. Neither probe reads test rows or scores.

## Analysis and limitation

Root-owned grading will retain all six fits and compare `alpha=+0.1` against
`alpha=0` paired within each seed, with paired component bootstrap on pooled
`global_wp_clip2_v1` if exposed results are inspected. Do not pool seeds into a
seed-generalization interval, select a model by exposed test results, or
claim public-score improvement. This screen tests transfer from the earlier
32-wide tanh-embedding pilot to the H64, two-layer GELU incumbent topology; it
does not establish performance at H256 or on unseen components.

# Blind specification critique: learnable bounded GRU candidate scale

**Decision: design is mathematically coherent, with implementation gates below.**
This note is based on the proposed design only. I have not inspected an
implementation, opened data, run a fit, or launched GPU work.

## Candidate map and bounds

For each GRU layer, let `u=tanh(v)` be the ordinary candidate preactivation
after the reset gate, and use

```text
alpha_l = 0.2 * tanh(raw_alpha_l)
n_l     = u + alpha_l * sin(pi*u)
h_next  = (1-z)*n_l + z*h
```

The two trainable scalars are layer-specific and shared over batch and time.
Since `u` is in `[-1,1]`, the candidate map fixes both endpoints. Its derivative
is `1 + alpha_l*pi*cos(pi*u)`, bounded below by `1 - 0.2*pi`, approximately
`0.371681`, so it is strictly increasing and stays in `[-1,1]`. The usual
convex GRU state update preserves the hidden-state bound for bounded initial
states. This local monotonicity argument says nothing about products of
Jacobians over long sequences; do not describe it as long-horizon gradient
stability.

The four candidate conditions should mean exactly:

| Condition | Per-layer coefficient during training | Trainable candidate parameters |
|---|---|---:|
| fixed zero | `alpha_l = 0` | 0 |
| fixed positive | `alpha_l = 0.1` | 0 |
| learned from zero | `alpha_l = 0.2*tanh(raw_l)`, `raw_l=0` initially | 2 |
| learned from positive | same parameterization, `raw_l=atanh(0.5)` initially | 2 |

`atanh(0.5)` is about `0.549306`, which makes the learned condition start at
effective `alpha=0.1`. Record both raw and effective layer values. State clearly
that learned arms contain 61,060 trainable parameters, while fixed arms contain
61,058. The paired function at initialization does not make the learned and
fixed arms capacity-matched, and this design cannot support an equal-capacity
claim.

## Differentiability and optimizer partition

The candidate scan must preserve autograd through both layer coefficients.
Pass scalar `Tensor` coefficients into the TorchScript recurrence and multiply
the tensor by `sin(pi*u)`. Do not convert coefficients to Python floats with
`.item()`, `float(...)`, `tolist()`, or a detached/clamped copy in the forward
path. Any of these silently removes the alpha gradient or creates a graph break.
The raw parameter should be a two-element FP32 `nn.Parameter`; compute the
bounded coefficients using tensor operations on each forward pass. Add a test
that alpha gradients are present, finite, and nonzero on a fixed synthetic
batch for both initializations. A finite forward alone does not show the
parameters actually learn.

Preserve the incumbent base model's exact initialized tensors, state-dict keys,
input normalization buffers, and shuffled row-group order for every arm of a
seed. Construct and save the ordinary 61,058-parameter `RNNModel` state first;
attach the two raw alpha values deterministically after that initialization so
the extra parameters do not consume or shift the RNG stream used for the base
weights. All four conditions for one seed start from those same base bytes.

Keep the fixed controls at 61,058 parameters. Do not add frozen dummy parameters
to imitate the adaptive arm. For learned arms, put `raw_alpha` in a separate
AdamW parameter group with the same `lr=1e-3` and betas but `weight_decay=0`;
decoupled decay on the unconstrained raw coordinate would change alpha even
when its data gradient is zero. For Muon, the existing selector should still
apply Muon to exactly the four two-layer GRU matrices with three gate blocks.
The input projection, head, biases, and both alpha scalars must use AdamW. Split
the alpha scalars out of the generic AdamW remainder into their explicit
zero-decay group. Verify by parameter identity that groups are disjoint and
cover every trainable parameter exactly once in both optimizer families.
Keep the base `MuonAdamW` group settings, momentum, weight decay, and orthogonal
update unchanged; adding alpha must not accidentally widen Muon's scope.

The full matrix has four candidate conditions × two optimizer families × two
fixed seeds = 16 fits. Use the same one-pass 24-fit-group order within each
seed, B=4, 256-row chunks, state reset at each batch, and state carry/detach
across the 79 chunks. Keep exactly 474 updates, unchanged weighted-MSE and
`need_prediction` mask, fixed clipping, and the common IEEE FP32 policy. Do not
use learned-alpha traces to change the schedule or stop early. Log initial
effective alpha before update 0, then a clearly defined post-update value at
the fixed telemetry indices 0, 46, 236, and 473; record the raw coordinates as
well so saturation or a stalled parameter is visible.

## Required falsifiers before fit admission

- At `alpha=0`, the independent candidate scan must match ordinary stacked
  `nn.GRU`/`RNNModel` output, both final layer states, input gradients,
  parameter gradients, and the matched clipped optimizer step at actual
  B4×256 under the same FP32 backend policy. Use already fixed parity tolerances;
  do not raise them after seeing a discrepancy.
- Initial learned-at-zero output and states must match fixed zero. Initial
  learned-at-0.1 output and states must match fixed 0.1. The corresponding base
  tensors and data order must remain identical.
- The scripted scan must retain the custom `Tensor` coefficient gradient
  through both layers. Compare analytic/autograd gradients to a finite
  difference on a small CPU fixture and assert both per-layer values receive
  finite nonzero gradients. Test without `.item()` or detach in the recurrence.
- Synthetic tests must verify coefficient bounds over a dense raw-alpha grid,
  candidate range and monotonicity on `u∈[-1,1]`, finite outputs/gradients,
  future-row isolation, reset versus carried state, and equivalence of a full
  sequence to contiguous chunks with carried state. Confirm state shape
  `[2,B,64]` and reset for each new four-sequence batch.
- Run one CPU B4×256 optimizer step for every seed/condition/optimizer cell.
  Check finite loss, gradients, raw/effective alpha, and parameters; verify the
  new alpha group is included by global clipping and has zero weight decay.
- An actual-shape GPU probe must recheck alpha-zero parity, both adaptive
  initializations, effective-alpha gradients, one step in each of the 16 cells,
  finite recurrent states, parameter changes, memory bounds, and teardown.
  A failure in alpha-zero parity is a stop condition, not a reason to loosen
  tolerance or proceed with the adaptive candidates.

## Statistical interpretation

The proposed registered contrast matrix is coherent and should remain exact:
the four listed condition contrasts within each seed and optimizer (16 total),
Muon minus AdamW for each of the four conditions within each seed (8), and the
learned-versus-fixed interaction across optimizer for each initialization and
seed (4). For those four interactions, define them explicitly as
`(learn-start0 - fixed0)_Muon - (learn-start0 - fixed0)_AdamW` and
`(learn-start0.1 - fixed0.1)_Muon - (learn-start0.1 - fixed0.1)_AdamW`, once per
seed. Do not add or remove contrasts after seeing scores. Equal nominal learning
rates do not imply equal effective parameter movement, so keep optimizer
contrasts descriptive and conditional on these exact recipes.

For exposed scoring, align all 16 model predictions on the same 16 components
and selected rows. Use paired component resampling and recompute pooled
`global_wp_clip2_v1` for every arm and every resample, including each
difference-of-differences. Report each seed separately; do not pool two
optimization seeds into a generalization interval, choose a winner on exposed
components, or claim public-score improvement. The learned-alpha trace is a
fit diagnostic, not a test-set selection rule.

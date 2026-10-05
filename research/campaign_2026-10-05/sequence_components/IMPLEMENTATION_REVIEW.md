# Independent implementation review: seven-arm sequence screen

Reviewer: Luna. Scope is `components.py`, `fit_sequence.py`, `tests.py`, and
the frozen cohort. No source changes or fits were made. Synthetic checks ran on
CPU with one thread and a 2 GiB guard.

## Findings that affect what the seven-arm result can claim

**Attention is confounded with added capacity.** `gru_attention_mlp` has 9,148
parameters versus 8,235 in `gru_mlp`; `ssm_attention_spline` has 3,899 versus
2,978 in `ssm_learned_spline`. The attention arm adds Q/K/V projections and
also widens the readout from 32 to 40. There is no same-family, same-width,
same-parameter pointwise/no-memory branch, so a gain cannot be attributed to
history retrieval. Before interpreting an attention effect, add a pointwise
8-dimensional branch with a matched parameter budget and the same 40-wide
readout for each memory family, or label these attention arms exploratory and
do not claim an attention-specific win. This keeps the attention hypothesis
alive while making its mechanism falsifiable.

**The GRU and diagonal-SSM arms are not capacity matched.** The core sizes
computed from the current constructors are 8,235 (`gru_mlp`) versus 2,987
(`ssm_learned_mlp`), about 2.76× apart. The SSM is intentionally much simpler
than the GRU at width 32, so equal width is not equal parameter count. The
fixed/learned SSM contrast is a clean narrow ablation (1,056 vs 1,088 params
in the memory module), and the MLP/spline head comparisons are close, but a
GRU-versus-SSM score difference is a joint architecture-and-capacity result.
Report counts and avoid a capacity-independent claim unless a matched control
is added.

**The spline head has dead knots under the proven hidden-state bounds.**
`AdditiveSpline` uses knots `[-3,-2,-1,0,1,2,3]` and clamps outside its domain.
Its input is the concatenated `memory` states: GRU outputs are tanh-bounded,
SSM state is a convex combination of tanh-bounded inputs, and attention output
is a positive normalized average of tanh-bounded values. I verified for all
seven composed arms that spline inputs remain in `[-1,1]`. Thus the outer four
knots (`-3,-2,2,3`) receive no basis activation or gradient. Nominal correction
counts are close—457 vs 448 parameters at width 32, 561 vs 560 at width 40—but
the spline's effective fitted basis is much smaller than its nominal count.
Use a fit-only hidden-state range/quantile to freeze a knot grid over actual
support, and save per-interval occupancy, or explicitly report this as a
three-active-knot additive piecewise-linear ablation. It is accurately called
KAN-inspired/PWL in code; it is not the original multi-layer cubic-B-spline
KAN. Its endpoint clamping prevents extrapolation instability here, but knots
and basis occupancy should be tested on the held-out cohort before claiming a
nonlinear-function advantage.

**Training uses a per-chunk, per-target normalized loss.** At lines 201–203,
each 256-row sequence minibatch is divided by its own target weight mass before
averaging the two sides. This implements an average of chunk-level weighted
MSEs, not the single empirical weighted-MSE objective over all fit rows. Since
chunks stay in chronological order and their absolute-target mass varies, a
low-weight chunk gets the same loss normalization as a high-weight chunk; it
also rebalances target 0 and target 1 independently inside each update. This is
a valid deliberate chunk-normalized surrogate, but the report must name it. If
the arm is intended to train the global WP-optimal mean objective, accumulate
weighted numerators/normalizers on a fixed fit population (or otherwise make
temporal chunks an unbiased sampling scheme) and keep the exact same rule for
all arms. A tiny falsifier: chunk A with total target weight 1 and unit squared
error, chunk B with weight 100 and zero error gives current normalized loss
`0.5` but global weighted loss `1/101 ≈ 0.0099`.

## Correctness checks that passed

- `causal_features` uses the current 20 observed columns, current-minus-previous
  differences (with row 0 compared to itself), a current event bit from
  `i0_p0/i0_p11`, and event age accumulated only through the current index.
  `SequenceModel` has no future-row operation. Full-composition prefix
  perturbations on every one of the seven arms left all earlier predictions
  bit-identical.
- The attention recurrence writes the current row before its read. This is
  causal because the current feature vector is available at prediction time.
  Shared positive decay and positive softmax keys make the key/value ratio a
  normalized average; its outputs are bounded by the tanh value range. This is
  a causal recurrent kernel-attention/fast-weight component, not a Transformer
  block, though it has a globally learned rather than input-selective decay.
- The SSM recurrence is diagonal `decay * previous + (1-decay) * tanh(Wx)`;
  its state remains in `[-1,1]` from zero initialization, and learned decays
  stay between 0.5 and 0.9999. It is a compact learned/fixed EMA-bank model,
  not a full S4 or Mamba implementation. `fixed_decay` and `learn_decay` start
  from the same log-spaced values.
- `load_cohort` fits feature normalization on fit groups only. Training uses
  `need_prediction` and does not filter fit rows by `is_scored`; that is
  explicit in the prior EDA design. Evaluation uses `need_prediction &
  is_scored` on the destination row, and concatenates rows before calculating
  the global target WP. This is the correct held-out scoring mask. The saved
  frozen cohort was independently checked earlier: selected fit/test group,
  class and component identities matched cache metadata; fit/test components
  do not overlap. This fit runner rechecks only fit-versus-test component
  overlap, so it should also assert unique selected row groups and unique
  selected components, and include the cohort-manifest hash in its receipt.
- The existing CPU tests passed (`sequence component checks passed`). An
  additional guarded CPU test confirmed composed-arm prefix causality and
  bounded spline inputs. No real data arrays, GPU, or training runs were used.

## Truncated-BPTT and state/update interpretation

`run_arm` processes each 20,000-row sequence batch in 256-row chunks, detaches
the recurrent state, and steps AdamW after each chunk. This is ordinary
truncated-BPTT with online chunk updates, not a code correctness error; the
campaign's main trainer uses the same basic cadence. It does mean each carried
state was produced under the preceding chunk's weights, and training updates
parameters about 79 times while processing a sequence group, whereas inference
uses one fixed checkpoint. The mismatch can affect different recurrence
families differently. Keep the same fixed schedule across arms and disclose
the limitation rather than silently giving one method fewer optimizer steps.
This one-epoch, 24-fit-sequence probe is a method screen only, with no fresh
generalization or 0.7000 forecast claim.

State `None` is reset at each new minibatch and each held-out sequence; state is
carried within a sequence. Warm-up inputs are processed, with zero objective
weight before `need_prediction` becomes true. The chunked-vs-full tests exercise
the component states, and the extra composed-model test checked prefix
perturbations. These controls support causal execution but do not remove the
truncated-BPTT mismatch.

## Recommended admission status

No causal-feature, state-bound, scoring-mask, or chunk-state bug was found that
by itself invalidates a small exploratory run. Before sealing results as a
comparison of mechanisms, either add the matched pointwise attention-capacity
control and make spline knots active on fit support, or predeclare those arms
as capacity/support-confounded screens. Clarify the per-chunk normalized loss
as the actual objective or change it uniformly across arms. Preserve the
current shared chunk/update schedule, log exact parameter counts and training
steps, and add cohort uniqueness/hash receipts. These are focused controls;
they do not reject attention, diagonal state models, or KAN-inspired functions.

## Addendum: rereview after root's corrections

The updated ten-arm source now lands the main review fixes:

- GRU and SSM each have attention and static-pointwise branch pairs. Both use
  the same 40-dimensional prediction input; CPU parameter counts are
  9,148/9,142 (GRU attention/static) and 3,900/3,894 (SSM attention/static).
  The six-parameter mismatch is negligible and the static branch is a genuine
  current-row-only control. This comparison isolates recurrent fast-weight
  retrieval from an added current-feature MLP reasonably well.
- Spline knots now span `[-1,1]`, the supported range of all three recurrent
  outputs and the tanh-bounded static control. The former four guaranteed-dead
  outer knots are no longer dead by construction. Per-bin occupancy on fit and
  held-out predictions remains useful evidence because actual activations may
  occupy only part of that range.
- `load_cohort` rejects duplicate selected row groups and components, checks
  20,000-row sequences and finite clipped targets, and the result receipt now
  contains the frozen cohort hash. The scalar score uses the same centered
  moment functions as the official `global_wp_clip2_v1` scorer.

I reran a one-thread, 2 GiB CPU synthetic check over all ten composed arms.
Future perturbations left prior outputs bit-identical; outputs were finite;
spline input features stayed within `[-1,1]`; both attention/static pairs had
the same widened head and near-equal parameter counts. On synthetic clipped,
zero-weight and constant-target cases, the local score wrapper matched
`utils.weighted_pearson` to `<1e-12`. The existing bounded component suite had
also passed in the previous review.

Residual interpretation limits, not launch-blocking code defects: the
GRU-versus-SSM core remains a large parameter-count comparison (8,235 versus
2,987 in the corresponding basic MLP arms); the SSM recurrence is not matched
to a GRU on parameters. `ssm_attention_spline` has no static-pointwise-spline
partner, so treat that particular attention×spline combination as exploratory;
the attention main effect should come from its matched MLP pairs, while the
spline main effect comes from the non-attention SSM head pair. The per-chunk,
per-target normalized weighted-MSE surrogate and carried-state truncated-BPTT
cadence remain as preregistered, shared training choices. The carried state
was produced with the immediately preceding chunk's weights; that is the
disclosed training/inference-state mismatch, not a forward-causality error.

The rereview does not inspect GPU-fit outcomes. Subject to the frozen v2
admission receipt and the preregistered scope, the updated source is fit for
the bounded ten-arm mechanism screen with these residual comparison limits.

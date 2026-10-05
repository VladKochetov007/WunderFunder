# E467: memory and spline components, fixed small-cohort screen

The user requested linear state-space models, attention without a Transformer,
and KAN-like components with useful combinations. These are ten new causal
models, not a decision that a previous architecture result closes a family.
This is a cheap mechanics screen, not an estimate of expected public score.

Use the already frozen target-mechanics cohort: 24 fitting and 16 scoring
sequences, distinct feature-graph components between these roles. This cohort
was previously exposed and its labels were used for earlier diagnostic scores.
Cached label-derived class metadata was used to stratify selection only; it
is never a prediction input. No claim of fresh independent validation follows.

All models see the same 20 observed columns, causal one-row changes, observed
event flag and log event age. Standardization uses fitting sequences only.
Every sequence is processed from its first row; fitting uses need_prediction,
scoring uses need_prediction AND is_scored. No target goes into recurrent state.
Prediction heads receive causal representations of observed inputs only.

The initial eight-arm GPU shape probe passed before fitting. Independent
criticism added two near-parameter-matched static branch controls; scientific
fitting therefore has ten arms, with a fresh v2 probe and content identity.

Fixed arms, seed 271828, width 32:

| Arm | What it tests |
|---|---|
| gru_mlp | Small ordinary recurrent control |
| gru_spline | Additive seven-knot spline edges over the same GRU and anchor |
| ssm_fixed_mlp | Fixed stable spectrum, nonlinear input embedding |
| ssm_learned_mlp | Learned stable diagonal decay versus the fixed spectrum |
| ssm_learned_spline | Spline head over the same learned memory |
| gru_attention_mlp | Small fixed-state associative memory alongside the GRU |
| gru_pointwise_mlp | Eight-dimensional current-row branch, same widened head, six fewer branch parameters |
| ssm_pointwise_mlp | Same static branch control on diagonal memory |
| ssm_attention_mlp | Associative memory alongside diagonal memory |
| ssm_attention_spline | Combined memory plus spline, with both immediate controls |

This spline is a piecewise-linear KAN-inspired component, not a reproduction of
the original cubic KAN. Knots cover [-1,1], the bounded state range; outside
this range edge contributions clamp and the ordinary linear anchor remains.
Heads are zero residuals at initialization. Within a given memory family,
spline/MLP head parameter counts differ by at most nine parameters. GRU and SSM
whole-model counts differ substantially and are reported; a between-family
comparison cannot attribute an effect to architecture at equal capacity.
Attention comparisons against the pointwise branches hold head width and
branch capacity within six parameters. A constant-key recurrence is still
needed to distinguish content retrieval from ordinary learned smoothing.

Train one full epoch with AdamW lr .001, decay .01, batches of four full
20,000-row sequences, sequential chunks of 256, gradient clipping at 1. There
are six full sequence batches and 79 chunks each: 474 optimizer updates per arm.
The recurrent state carries between chunks with its gradient detached. As in
ordinary carried-state TBPTT, a weight update can leave the carried state
computed under older weights. This is a training/inference distribution
limitation, especially for learned long-lived state. It is shared across this
screen; no claim that it reproduces a fixed-weight full-sequence gradient.
For the next long-memory attribution test compare fixed-weight sequence
accumulation at a separately declared, sufficient optimizer-update budget.

The loss averages the two per-target absolute-clipped-label weighted MSEs,
normalizing each target by its current chunk's mass. Predictions remain
unclipped in this training loss; the unchanged official centered scorer clips
both labels and predictions. No auxiliary loss, tuning or early stopping.
Evaluate only the final checkpoint. Save all arms, raw selected predictions,
parameter counts and component/sequence scores; no winner-only reporting.

CPU invariant checks precede a GPU probe using actual B4/chunk256 and an
optimizer step for every arm. The probe is separate from scientific fitting.
One local GPU worker, 2 GiB allocator, 3 decimal GB observed-device stop,
6 GiB worker RAM, 2 CPU quota, 3600 s total fit/evaluation cap, no worker swap.
The parent 36 GiB /12 CPU /5 GB GPU limits still apply. All attempts are
content-bound through the existing guarded runner; existing frozen runner and
grader remain unchanged. An incomplete or nonfinite run is a failed attempt,
never a completed negative scientific result.

Initial CPU smoke v1 failed before data or GPU access because a direct script
could not import root utils. Version 2 uses the existing installed scorer's
centered moments and passed all eight initial actual-shape forward/backward probes.
Preserve this import failure; the numerical scorer was not modified.

Scientific interpretations stay local: a weak or null spline result only tests
this head/basis/budget; a weak diagonal spectrum does not reject selective SSMs,
complex-valued LRU, oscillatory state, or an SSM branch on the strong GRU.
Likewise, bounded additive attention does not test delta-rule overwrite or a
causal event-ring retrieval module. Further work needs a concrete nuanced
hypothesis, a matched control and its own bounded contract.

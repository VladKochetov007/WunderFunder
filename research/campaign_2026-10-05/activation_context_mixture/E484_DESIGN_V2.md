# E484: causal context-conditioned head activation mixtures

**Status:** registered design before implementation, probes, or fit admission.
This follows E477’s mixed two-seed result: the fixed same-channel 50/50
GELU/LeakyReLU mixture changed global WP by −0.00575 for seed 271828 and
+0.00526 for seed 271829. Those 16 previously exposed components do not select
a winner or motivate discarding mixtures.

## Question

Does a causal context vector help choose between GELU and LeakyReLU at each
readout time step, beyond a fixed or learned-static per-channel mixture? Keep
the H64 two-layer GRU, 42-feature input, concatenation order, head linears,
weighted-MSE objective, fit-only normalization, 24 fit components, B4×256
traversal, 474 updates, AdamW recipe, and dropout=0 unchanged.

Let `h_t` be the existing 64-wide head preactivation, `r_t` the current GRU
output, and `z_t = GELU(inp(x_t_normalized))` the existing activated 64-wide
encoding used as recurrent input and projection skip (exact GELU, existing
linear encoder, no new projection). For any 64-vector
`c_t`, define `c̃_t[j] = c_t[j] / sqrt((1/64) Σ_k c_t[k]^2 + 1e−6)`: divide by
the row RMS with epsilon `1e−6`, do not subtract the row mean, and do not learn
normalization parameters. The dynamic variants use
`g_t = sigmoid(a ⊙ c̃_t)` and
`y_t = g_t ⊙ GELU(h_t) + (1−g_t) ⊙ LeakyReLU_0.01(h_t)`, where `c_t` is
respectively `r_t`, `z_t`, or `h_t`. Insert the mixture after the existing
head hidden linear and before its activation; keep the final linear layer
unchanged. Since `r_t`, `z_t`, and `h_t` use only present/past inputs, the gate
uses no future rows. Initialize each 64-vector `a` to zero, so
all three dynamic arms start at the same 50/50 mixture. The matched learned
static control uses `g = sigmoid(b)` with one 64-vector `b=0`. This gives every
learned-gate arm the same 64 added parameters and the same initial function,
without consuming extra initialization RNG. Retain the exact GELU and fixed
50/50 mixture as zero-added-parameter references. The learned-static and
dynamic arms each have 61,122 parameters; GELU and fixed-mixture anchors each
retain 61,058.

This is a gated activation / GLU-family reparameterization, not a novel
architecture claim. The causal question is whether routing from recurrent
state adds useful information compared with a learned static gate, local
projection context, or the head preactivation itself. Equal parameter counts
alone do not match optimization geometry: zero-start `a` receives
context-scaled gradients while static `b` does not. That difference is the
intended treatment. The three dynamic sources otherwise share the same gate
formula, context normalization, initialization, parameter count, fit order,
and optimizer. This normalization removes each context vector's overall
amplitude, so the proposal tests routing from its signed relative channel
pattern, not routing from state magnitude.

## Fixed budget and arms

Use two new seeds, 271834 and 271835, with six arms each (12 fits maximum):
GELU; fixed 50/50 mixture; learned-static per-channel gate; gate from `r_t`;
gate from `z_t`; gate from `h_t`. Reuse one baseline initialization and one
fit-group permutation per seed across all six arms. The new gate vectors are
zero-initialized without advancing the RNG. Preserve all cells and report the
two seeds separately. No arm selection, dropout sweep, extra regularizer, or
schedule adjustment is allowed.

The primary contrasts per seed are recurrent-context minus learned-static,
recurrent-context minus input-projection-context, and recurrent-context minus
head-self-context; report both target metrics and global WP with paired
component bootstrap intervals over the same 16 exposed components. GELU and
the fixed mixture are descriptive anchors. The outcome can support only a
small mechanism comparison: a recurrent-gate advantage over the static gate
and both same-size context controls would be consistent with state-dependent
routing; an advantage over the static gate alone would not identify recurrence
as the useful context source. No result on this panel estimates public score.

Before preregistration: review gate scaling/stability and source construction;
verify exact parameter counts, shared initial bytes/RNG, full-sequence carry,
future-row isolation for predictions and gates, finite gate values/gradients,
and synthetic formula/gradient tests. Use native PyTorch/cuDNN recurrence and
at most 12 small fits. Require a fit-only actual B4×256 GPU probe and independent
design/source review before any fit admission. Keep existing local resource
guards and campaign budget unchanged.


## Controller contract clarification

The mixed 64-vector is an activated hidden readout, not the two-target output;
apply the unchanged final linear to it. Routing changes the representation
of history already available to every arm; it does not add new information
or establish a new architecture. Separate gate sources also change gradient
paths, so their ranking cannot identify information content alone.

Retain all three primary contrasts per seed. Also register static-minus-fixed,
recurrent-minus-fixed, projection-minus-fixed, self-minus-fixed and
fixed-minus-GELU: eight global contrasts per seed,16 total. Recompute global
correlation jointly with2000 numeric-sorted component draws; report both
target scores and every contrast separately per seed, without seed pooling.
Counts:12 complete fits,474 updates each. These new initializations do not
create fresh market validation. Preserve the raw B4x256 receipts, optimizer
groups, fit-normalizer inputs and whole-sequence carry. Gate coefficients
remain zero-initialized; no RNG draw for gate parameters.

Before implementation, an independent design critic must challenge causality,
gate normalization, initialization and fair controls. Source review and CPU
fit-only/synthetic tests precede a separately admitted CUDA probe. Test
prefix/future-row isolation and per-row sample independence, all six initial
functions (all mixture arms initially identical), gradients reaching gates,
finite coefficients, and exact baseline output/state/gradient parity.
An independently implemented grader checks completed source/attempt/fit/
checkpoint provenance and rejects nonfinite raw outputs before clipping.

Existing budget only:reserve at most1800 GPU seconds. One local GPU worker,
2CPU/6GiB RAM,2GiB allocator/3GB observed GPU stop, actual aggregate
12CPU/30decimalGB/no swap. E483 has queue priority; no concurrent local GPU
workers. New admissions close07:00UTC/10:00Kyiv; admitted jobs then drain.
No expectation of0.7000 public score follows from this exposed pilot.


## Version2 clarification before implementation

This version responds to the independent blind critic's projection ambiguity.
The registered version1 and its critique remain preserved. All five mixture
functions (fixed/static/recurrent/projection/self) initially agree; the pure
GELU anchor deliberately differs. Shared parameter bytes and RNG are identical
across all six arms; functional equality applies only to the five mixture arms.
Compare GELU's output/state/gradients with its own frozen GELU reference,
and compare each mixture arm with the frozen fixed-mixture reference at zero
gate coefficients. A nonzero-gate synthetic case must exercise real causality
and batch-isolation properties; zero gates alone could hide bad context logic.
Capture active precision flags while still inside the IEEE context, before
restoration. No dropout or optimizer/scheduler factor is added by this revision.

# R11: fixed-step initial-gradient importance sampling

## Question and scope

On the accepted, exposed R3 panel, does one fixed initial-gradient importance
sampler change final pooled official WP relative to uniform row sampling after
the same small number of optimizer updates? This is a conditional, fixed-step
predictive screen for H-NF19-17. It is not a test of equal-wall-clock
efficiency, lower total cost, a new validation fold, or a full training recipe.
The norm prepass has real cost and will be timed separately. No R7 score or
endpoint is an R11 baseline or tuning signal.

Use the accepted R3 sample at
`B/research_tmp/nf19-sample-r3-v2`, sample SHA256
`b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64`,
membership SHA256
`e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217`, and
exposed canonical F0 split
`c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7`.
The fixed panel contains 64 fit components, 31,808 fit rows selected at stride
40 after the source warmup, and 32 evaluation components with 59,436 original
scored rows. Fit and evaluation components are disjoint within the accepted
sample; neither set is a fresh holdout. Do not resample, repartition, tune,
early-stop, or choose a result using evaluation scores.

## Shared representation and initialization

Reuse the frozen R7 complete-history feature path: raw 112 inputs plus the
canonical signed-delta EMA with half-life 32, assembled over each complete
causal sequence before selecting stride or scored rows. The fixed input width
is 224. Reuse the R7 FP64-reduction mean and standard deviation fitted only on
the 31,808 fitting rows; scales below `1e-6` map to one. Apply this same
normalizer to both arms and return FP32 features. Evaluation rows do not enter
normalization, sampling probabilities, training, or stopping decisions.

Both arms use the same `Linear(224,64) -> GELU -> Linear(64,2)` mean network
and an identical, zero-initialized `Linear(64,2)` variance head that is never
used. Create one initial state with seed 19041 and clone it byte-for-byte into
both arms. Each arm gets a fresh CPU AdamW optimizer with learning rate 0.001,
betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`, no fused/foreach
kernel, FP32, and global preclip norm threshold 1.0. The variance head must
remain gradient-free and unchanged in both arms.

## Objective and fixed sampling distributions

Let `N=31,808`, and for fitting row `i` and target `t` let
`w_it = abs(clip(y_it,-2,2))`. The original fit selection mask is already
applied when those rows are collected. Define the fixed per-row mean weight
sum

```text
mu_w = (1/N) * sum_i sum_t w_it
l_i(theta) = sum_t w_it * (mean_it(theta) - clip(y_it,-2,2))^2 / mu_w
```

Thus `(1/N) * sum_i l_i(theta)` is exactly the pooled two-target clipped,
absolute-label-weighted MSE. `mu_w` is the mean row-sum weight; the equivalent
global denominator is `D=N*mu_w`, not `mu_w` itself. Fail before fitting if
`mu_w` is nonpositive or nonfinite.

At the common initial state `theta0`, compute one closed-form, real-valued
joint per-example gradient-norm estimate `g_i` for the fixed row contribution
`l_i`. Evaluate the formula in FP64 from the stored FP32 features and
parameters. This is not bit-identical to the FP32 autograd machine gradient;
the tiny autograd check establishes agreement within a fixed tolerance, not
machine parity. The probabilities use this fixed norm estimate as a sampling
score. Importance correction, rather than exactness of the score, establishes
the marginal preclip unbiasedness. Include the shared trunk and both output
targets together in the norm; do not calculate separate target norms and add
them. The unused variance head contributes no gradient. Use the exact GELU derivative
`Phi(a) + a*phi(a)` and FP64 arithmetic for the deterministic analytic
prepass, starting from the FP32 features and FP32 parameter values. For
`a=W1*x+b1`, `h=GELU(a)`, `m=W2*h+b2`,
`u_t=2*w_t*(m_t-clip(y_t))/mu_w`, and
`delta=(u @ W2) * GELU'(a)`, the per-row squared norm is

```text
||u||^2 * (||h||^2 + 1) + ||delta||^2 * (||x||^2 + 1)
```

This uses the Frobenius norm identity for each outer-product weight gradient
and avoids materializing a per-row parameter-gradient tensor. Exclude AdamW
weight decay, as `g_i` describes the data-loss gradient at `theta0`. Reject
nonfinite norms or a nonpositive total norm.

Uniform arm: sample rows with replacement using `p_i=1/N`. Importance arm:

```text
p_i = 0.5/N + 0.5*g_i/sum_j(g_j)
```

The 50% uniform mixture guarantees support and implies the importance
correction `1/(N*p_i)` is at most 2. For each draw, multiply `l_i(theta)` by
that correction and take the batch mean. Both arms use the same fixed
31,808-row feature/target table and the same generated uniform variates from
seed 19042; map those variates through each arm's own inverse CDF. The
different sampled row sequences are intentional and this coupling does not
make them same-example paired.

At any fixed `theta`, the corrected importance minibatch gradient is unbiased
for the gradient of the globally normalized pooled WMSE before clipping:

```text
E_p[ grad(l_i(theta)/(N*p_i)) ] = (1/N) * sum_i grad(l_i(theta))
```

This does not imply unbiased clipped gradients, AdamW updates, training
trajectories, or final models. Report the fixed `theta0` norm-only second
moment `sum_i ||grad l_i||^2/(N^2*p_i)` for importance sampling and
`mean_i ||grad l_i||^2` for uniform only as an uncentered second-moment
diagnostic. Their ratio is not a gradient-variance ratio or a full-training
variance result.

## Draw and update budget

Use 12 draw rounds of `N` with-replacement examples per arm. Split every round
into 62 batches of 512 plus a final batch of 64. This is 63 optimizer updates
per round, 756 attempted and required successful updates per arm, and 381,696
draws per arm. Call these 12 draw rounds, not 12 epochs of unique-row
coverage. Do not refresh `g_i` or `p_i`; no adaptive sampling, schedule,
mixture, seed, architecture, endpoint, or correction search is allowed.

Record per-arm attempted/successful/nonfinite counts, objective values,
preclip gradient norms, global-clip incidence, fixed learning rate, round
progress, and actual wall/CPU time. Record the norm prepass time separately,
along with probability/correction ranges and effective sample size. If either
arm has fewer than 756 successful updates, write a failed/incomplete attempt
and make no predictive comparison. Never convert nonfinite updates into
skipped batches or endpoint substitutions.

## Outcome and interpretation

The sole primary contrast is final canonical pooled `global_wp_clip2_v1`
`importance - uniform` over all 59,436 original scored evaluation rows. Report
both target scores, per-component scores and fixed prediction drop-one
contrasts as sensitivity summaries, not independent replications. Fit
objective and preclip/clipping diagnostics describe the tested trajectories;
they do not select an endpoint. Archive only the two final model states and
shared fit-only normalizer in a create-only NPZ on physical B scratch. Keep
arrays and model parameters off the Mac; compact JSON contains scalar
statistics, counts, hashes, and archive metadata only.

An improvement supports only a conditional final-WP difference after this
fixed update/example budget on the exposed R3 panel. It does not establish
lower total compute, equal-wall-clock advantage, card-17 paper replication,
generalization to a fresh holdout, or public-score improvement. If the
importance arm does not improve, close this exact fixed initial-gradient
sampler on this panel; that does not reject every importance-sampling method.

## Bounded execution and checks

Root owns data/model admission, staging, execution, capture, and publication.
The actor performs only synthetic CPU checks on the Mac. The single actual
unit is bounded to 0.5 CPU, 1.4 GB memory-high, 1.5 GB maximum, and 600
seconds, with CUDA hidden, Torch/BLAS threads set to one, and a 256 MiB memory
reserve. Before the first model step, the runtime must confirm that
`/proc/self/cgroup` identifies the exact R11 service under
`wnn-campaign.slice`; do not fall back to a root or host cgroup counter.

Before source/data imports, the CLI verifies the exact sample and preparation
manifest bytes, all reviewed local/canonical source pins, and accepted R3
sample/membership identities. The runtime import inventory must resolve to
those pinned files. The source and test freeze includes a B-shaped import and
read-sentinel smoke, analytic-norm versus tiny-autograd agreement,
fixed-denominator unbiasedness on a deterministic fixture, probability and
correction invariants, common-uniform determinism, identical initialization,
and official scorer pooling. No actual panel arrays, model fit, accelerator,
or provider execution is part of the local checks.

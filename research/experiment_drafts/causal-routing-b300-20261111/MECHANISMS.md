# Mechanisms, controls and decision rules

## SR-A: signed interaction between attention maps

Hypothesis: a positive weighted average of historical values retains nuisance
context that a signed combination can suppress. The experiment tests this in a
causal market-data Transformer; it does not assume language-model gains transfer.

Keep each canonical `qkv` projection, 12 heads, head width 64, RoPE, context 1024
and FP32 streaming caches. After the original RoPE, split Q and K into two
32-coordinate halves. Each half defines one softmax map; both maps use the same
full 64-coordinate V. Pad the half-Q/K back to width 64 and multiply both Q and K
by 2**0.25 before the unchanged backend, preserving the intended 1/sqrt(32) scale.
This is an interface adaptation, not a new attention kernel. The two halves inherit
different bands of the original RoPE frequencies; this is frozen in both arms.

For outputs U1 and U2, use:

- Additive control: `0.5 * RMSNorm(U1 + 0.5 * U2)`.
- Differential treatment: `0.5 * RMSNorm(U1 - 0.5 * U2)`.

RMSNorm is per head, affine-free, with epsilon 1e-6 and FP32 reduction. Both arms
have identical parameters and make two attention calls. Their common normalization
reduces a simple amplitude confound. The separate plain anchors identify whether
either construction helps the existing model. No trainable lambda, new FFN,
additional memory, normalization sweep or auxiliary objective is introduced.

The inspiration is the difference of softmax maps in
[Differential Transformer](https://arxiv.org/abs/2410.05258v1). This packet is a
restricted adaptation: it keeps the repository's QKV layout and RoPE, fixes lambda
at 0.5 and uses matched additive controls. It does not reproduce the paper's full
architecture or claim its reported results.

Primary contrast: differential minus plain WP. Mechanism contrast: differential
minus additive WP. A gain only against additive, with no improvement over plain,
does not justify replacing the teacher. Equal performance between the two-map arms
weakens the signed-interaction explanation. Numerical instability, causality failure
or an inability to preserve cache/reload semantics blocks scientific fitting.

## SR-F: row-conditioned routing inside the FFN

Hypothesis: a single fixed combination of FFN features handles different market
states poorly. Routing between existing feature partitions may represent different
local regimes while preserving one shared causal Transformer.

Split the existing 3072 FFN activations into two 1536-wide partitions. Apply the
corresponding column blocks of the existing output projection to define E1 and E2.
Do not add expert matrices. Scale each activation partition by `2 * gi`, then call
the original output projection once. This implements
`2 * (g1 * E1 + g2 * E2) + output_bias`, including the bias exactly once. With
`g1=g2=0.5`, the projection receives the original activation tensor; the server
must still measure floating-point agreement through the complete model.

Each block adds a two-row router matrix and two biases, initialized to zero:

- Static control: router input is a zero tensor; only the per-layer gate bias can
  learn an effective preference. The matrix remains in the graph with zero gradient.
- Conditional treatment: router input is the current row after the existing
  feedforward LayerNorm. Softmax is evaluated in FP32 at fixed temperature 1.

Both evaluate both expert partitions for every row. There is no top-k routing,
token dropping, load-balancing loss, cross-token pooling or sparse-dispatch
optimization. The equal parameter inventory does not imply equal effective routing
capacity: that difference is the treatment. The canonical optimizer assigns the
new router matrix to its existing AdamW matrix group and biases to its existing
no-decay group; QKV and FFN matrices retain their Muon roles.

Input-dependent expert selection is motivated by
[Shazeer et al.](https://arxiv.org/abs/1701.06538), but this is a dense two-partition
adapter, not that sparse architecture. In particular, the all-token mixing in
[Soft MoE](https://arxiv.org/abs/2308.00951) is not used, because an unrestricted
version would violate causal streaming. Neither reference establishes a gain here.

Primary contrast: conditional minus plain WP. Mechanism contrast: conditional minus
static WP. If both gates improve equally, row dependence is not supported. If the
conditional gate collapses to a constant, the proposed regime explanation is weak.
Older prefix-specialist routing's negative evidence remains relevant counterevidence,
not a result for this different intervention.

## Frozen comparison and uncertainty

For all arms: complete 10,325 fitting sequences, unchanged train-only normalization,
four epochs, fixed final E4 endpoint, two B300 GPUs, H768/L24, context 1024,
chunk 256, batch 128 per rank, accumulation 4, BF16/BF16x9, native Muon/AdamW, LR 3e-4,
weight decay 0.01, clip 1, 5% warmup, cosine floor 0.1 and all-epoch retention.
All arms start from scratch and retain source-qualified membership. No validation
labels enter inputs, normalization, state or initialization. Scores use all 1,873
official validation sequences and 3,528,273 eligible rows with the unchanged scorer.

Review every seed and both targets. A mean paired WP increase of at least 0.0005,
both seeds positive, and no unexplained target regression worse than 0.001 is a
descriptive allocation heuristic, not a significance or automatic submission rule.
A mixed result remains inconclusive; an interval crossing zero does not close the
mechanism. The official split is exposed development data. Confirmation requires
a separately frozen allocation and documented exposure; relabeling the same data
as a new fold would not restore independence.

Canonical controls are deliberately shared between the two studies and must be
reported as the same observations. Do not count them twice as independent evidence.
Model quality and actual downstream student/blend benefit are distinct outcomes.

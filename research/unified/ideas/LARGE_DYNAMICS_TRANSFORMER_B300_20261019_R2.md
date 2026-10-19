---
kind: idea
status: proposed
created_utc: 2026-10-19
hypothesis_ids:
  - H-LARGE-DYNAMICS-TEACHER-20261019-R2
experiment_ids:
  - E-LARGE-DYNAMICS-TEACHER-SCALE-20261019-R2
evidence_paths:
  - research/unified/reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md
  - docs/data_overview.md
  - src/wnn/gmetric.py
---

# Large causal dynamics teachers on eight B300s

## Decision and scope

Build a strong numerical causal Transformer, then measure the quality that can
actually transfer into a GRU. The first flagship scale is about **1.65B parameters**,
with a **0.68B reference** and an eligible **3.17B extension**. Start with reliable
row-level attention, then investigate long-context hierarchical attention. Spend
compute on representation, optimization, context and student transfer; parameter
count alone is not the objective. A heterogeneous teacher ensemble is also eligible.

This is a design and draft handoff, not a launch or a runnable distributed trainer.
No B300 profile, new fit, improved student, public-score forecast or numerical
speedup is claimed. The [new policy](../../../specs/adr/0043-fast-screens-and-transformer-teacher-scale.md)
budgets 4–6 TOTAL new fits per initial idea comparison, not per arm. Frozen older
CP30 packets remain historical; do not change their completed-run interpretation.
The local October20 plan remains governed by its existing budget and start.

Strict FP32, no autocast and no TF32 remain binding. Keeping FP32 model weights
while computing attention or Muon orthogonalization in BF16 would violate this
contract. A future precision change would be an explicit new experiment.

The old [AR teacher proposal](CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md)
supplies earlier equations and source history. This R2 supersedes its prospective
replication count and makes the larger-model/optimization choices concrete; it
does not rewrite its frozen recipes or claim its proposed trainer already exists.

## Latest user source: TradeFM and observable transition likelihood

The user supplied [TradeFM](https://arxiv.org/abs/2602.23784) after the initial
scale design. Its event-modeling mechanism motivates our proposed
[observable-event adaptation](https://github.com/VladKochetov007/WunderFunder/blob/experiment/large-dynamics-teacher-b300-r2/research/experiment_drafts/large-dynamics-teacher-b300-r2/TRADEFM_ADAPTATION.md):
retain raw rows and supervised targets, add next-change survival and mark losses,
and test them at fixed0.68B capacity before spending on the whole scale ladder.
This comparison costs four new fits; the0.68B/1.65B capacity screen is a separate
four-fit allocation. The source's distributional generation results do not
establish weighted-Pearson or GRU transfer. Anonymous columns do not justify
copying price/volume/timestamp formulas. The addendum preserves the actual
source inspection, our censoring/causality adaptation and independent scout review.

## What the B300 results tell us

The [partial B300 review](../reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md)
compares GRUs, not Transformers. Width beat depth slightly in the R01 recipe;
the tested geometry bundle was negative; age/innovation had no paired scores.
None establishes a Transformer ceiling or an effective Transformer feature set.
In particular, do not insert the entire geometry bank into the default teacher.
Retain it as a distinct representation branch if a mechanism warrants revisiting.

Anchor quality to the strongest available source-qualified recurrent teacher,
not only to a weak labels-only GRU. Reuse its exact clean fold predictions where
available. TL3 is a useful complete-endpoint comparator, but its unknown fitting
ancestry must remain explicit; a replay is not automatically clean CV. A win
between two new Transformers does not establish that either improves that anchor.

## Architecture and numeric embeddings

Use one complete observation row per temporal token. Preserve all 112 raw values
and the canonical 99 event features as the reference input, with fitting-only
feature scales. Compare against the identical raw-only encoder if event features
prove redundant. Never turn 112 feature columns into 112 extra time positions.
Sequence IDs and scoring flags are not model inputs.

The representation ladder is deliberately separable:

1. **Dense reference:** an MLP from the ordered 211 values to the model width.
2. **Grouped numerical encoder:** use the supplied 13 groups (six per instrument
   plus additional features), attaching each event channel through a reviewed
   semantic registry. Per-group small MLPs produce 128-dimensional representations;
   concatenate and project into the temporal width, retaining a direct raw-row
   residual projection. Do not add this on top of a changed attention/loss and
   then attribute the whole effect to grouping.
3. **Richer scalar basis:** fitting-only piecewise-linear numeric embeddings,
   initially 8–16 bins per selected channel, retain the raw scalar and an explicit
   missing/nonfinite indicator where required by the data contract. Periodic
   scalar embeddings are an alternative basis, not evidence of a periodic process.
4. **Local dynamics branch:** causal lag differences at 1/4/16/64 rows, log1p age
   since observable updates, or a small left-padded dilated convolution. Keep raw
   values. Compare one branch at a time; full-sequence statistics and centered
   smoothing are invalid for early outputs. Rolling features are optional tools,
   not a requirement that large models must consume a large manual feature bank.

A learned causal GRU/filter stem pretrained on observable changes is another
input-state candidate: feed its state alongside the raw row, then fine-tune
jointly or freeze it as an explicit arm. Keep every preprocessing/pretraining
ancestor fold-clean. This transfers the previous-competition learned-feature
idea without assuming that its reported gains apply to our task.

Fixed column identity and instrument/group identity matter. The names do not
certify physical price units, true order-book depth adjacency or actual time
between rows. Use row order and observable event ages; do not invent timestamps.

An optional dual-clock positional arm assigns some heads row-order RoPE and
others the cumulative count of observable i0/i1 events. It may distinguish long
quiet stretches from genuine state transitions. Counts are prefix-only and reset
per sequence; this is our proposed event-time mechanism, not an established gain.

The baseline backbone uses pre-RMSNorm, RoPE, SwiGLU, full multi-head attention
with 64-dimensional heads, residual dropout 0.05 and attention dropout 0. QK L2
normalization with a learned per-head score scale initialized to sqrt(64) is a
stability candidate. If used, it replaces the usual dot-product scaling; do not
apply both scalings accidentally. A query-dependent sigmoid gate after attention
is another small, separately testable mechanism for suppressing noisy writes.
Neither QKNorm nor gating is established to improve this dataset.

Initialize residual output projections with a depth-aware small scale; record
the exact rule. No claim of muP or automatic width-independent LR transfer is
made. Use separate target heads for t0/t1 on a shared state. Avoid choosing an
MoE solely to inflate total parameters: routing, load balancing and active compute
would create additional mechanisms before the dense baseline is understood.

## Context, attention and causal long memory

The first scalable reference is dense causal attention over 4096 rows, with a
2048-row left prefix and a 2048-row scored core. Prefix rows provide context but
are not duplicated in the fitting loss. Early cores use all available prefix
with padding masked. At export, reproduce these exact window/position semantics;
do not compare a short-window teacher cache with an unrelated full-history model.
This reference has bounded context, not the whole 20k sequence.

For the full-history challenger, preserve a fast row stream and a slow history:

- Fast stream: causal local attention over 512 or 1024 rows; retain per-row outputs.
- Slow stream: summarize each COMPLETED32-row block from input embeddings, using
  pooled and last-row information, then apply a smaller causal Transformer over
  the at-most 625 completed blocks in a 20k sequence.
- Every fourth fast block may cross-attend to slow history. A row at position t
  can see a slow token only when that token's block end is <=t. Slow self-attention
  is causal. Build summaries from input embeddings, not from later fast states
  that themselves depend on the slow branch; avoid a circular state definition.
- No row earlier in a block may see its whole-block summary. At the completion
  row, that summary may include the current row. Define zero-length history,
  partial blocks, positions and sequence reset explicitly.

This preserves an exact high-resolution recent path and a compressed older path.
It changes the model and may discard useful older detail; compare it against the
dense reference and measure age/activity regimes. The slow branch and cross
projections add parameters beyond the trunk table below. Plain completed-block
pooling before a per-row prediction without visibility masks would leak future.

Use FP32 math SDPA as a correctness reference. Profile an explicitly IEEE-FP32
FlexAttention/Triton implementation with forward AND backward checks for the
selected masks, shapes and PyTorch build. Current FlexAttention/FA4 documentation
does not certify that every backend supports our dtype and gradient path. Record
the actual selected backend and dtype. Avoid materializing dense 20k masks for a
sparse design. Activation checkpointing reduces retained activations, not the
quadratic work of dense full attention. A GQA variant may reduce KV cost, but
full MHA is the quality reference and GQA does not eliminate attention pair work.

Full20k dense attention remains a legitimate teacher arm if a verified tiled
FP32 kernel and real profile make it economical. Do not assume a sparse or
compressed teacher is equivalent to full-history exact attention.

## Training signals that reward dynamics

Use supervised regression as the main anchor. A proposed explicit target policy
is c(y)=clip(y,-2,2), weight 0.05+abs(c(y)), and a separately normalized weighted
mean squared error for each target. The0.05 floor and training clipping are
recipe choices, not the official metric definition. Reuse the successful canonical
target policy instead when doing a faithful baseline comparison; do not change
target treatment and attribute the result entirely to architecture.

Add future-observation learning as a declared auxiliary arm:

- Predict normalized observable changes x[t+h]-x[t] at h=1,4,16,64,256.
- Predict whether selected feature groups update within those horizons. This
  provides a signal beyond copying persistent values; report no-change and
  change-conditioned errors to detect a trivial persistence solution.
- Future values appear ONLY as fitting targets, never inside the input embedding,
  normalization, patch, memory token or augmentation available to output t.
- Mask unavailable horizons, padding and cross-sequence targets. Fit all scales
  and event-balancing constants on fitting components only. For a selected crop,
  load a target-only future tail or mask its missing horizons explicitly.
- Start with Huber auxiliary losses normalized over their valid elements and a
  fixed coefficient 0.05. Measure shared-backbone gradient contributions before
  the long fit; freeze the coefficient rather than silently rebalancing it later.
  Each head and horizon needs its own declared denominator.

Initially train supervision plus the admitted auxiliary losses jointly. A
pretrain-then-finetune sibling may use only each outer fold's fitting components,
with fresh or reset optimizer state specified at the transition. Self-supervised
training on the outer fold would expose that fold even without target labels.
The all-data final teacher is a later production artifact, not a held-out score.

The optional output AR head follows the old proposal: yhat[t]=a[t]yhat[t-1]+
(1-a[t])u[t], a[t]<0.99, initial a=0.10, model feedback only. It can be evaluated
with a verified differentiable affine scan without serial Transformer calls.
Never feed true lagged competition targets; their future horizon is undisclosed.
AR is an ablation, because persistence can also blur abrupt changes. Keep the
plain head reference. Record carried-state truncation/optimizer staleness if
training crosses cores with detached state.

## Scale and memory: what fits

NVIDIA's DGX B300 specification lists 288 GB per GPU; the receiving operator must
verify actual visible memory, exclusivity and NVLink topology. Eight devices do
not make one 2.3-TB allocation under DDP: each rank holds a full model replica.

| Trunk | Layers | Width | Heads | SwiGLU width | Trunk parameters | FP32 state arithmetic,12–16 bytes/parameter |
|---|---:|---:|---:|---:|---:|---:|
| Smaller reference | 16 | 1024 | 16 | 2816 | 205.5M | 2.47–3.29 GB |
| Large reference | 24 | 1536 | 24 | 4096 | 679.5M | 8.15–10.87 GB |
| First flagship | 32 | 2048 | 32 | 5632 | 1.644B | 19.73–26.31 GB |
| Scale extension | 40 | 2560 | 40 | 6912 | 3.172B | 38.06–50.75 GB |
| Further capacity arm | 48 | 3072 | 48 | 8192 | 5.436B | 65.23–86.97 GB |

These are analytical trunk counts, 4Ld²+3Ld*d_ff, before input encoders, norms,
heads, slow stream and cross projections. Full counts must come from the model.
The low state figure counts FP32 weights/gradients/Muon momentum; the conservative
high figure uses two optimizer moments. Neither is total peak memory or a fit
certificate. Include orthogonalization workspace, DDP buckets, allocator reserve,
temporary optimizer arrays, checkpoint/export buffers and activations. A gradient
bucket view can avoid some duplication but must be measured.

At one sequence,32 heads and 20k rows, a single FP32 dense score map costs 51.2 GB;
at 24 heads it costs 38.4 GB. Backward and softmax temporaries add more. DDP does
not divide this allocation by eight. Bigger batch/context, not parameter states
alone, is likely the admission constraint for these scales. The executable
[capacity calculator](https://github.com/VladKochetov007/WunderFunder/blob/experiment/large-dynamics-teacher-b300-r2/research/experiment_drafts/large-dynamics-teacher-b300-r2/capacity.py)
prints these state/map formulas and conditional step-time tables; it does not
estimate activation peaks or certify memory safety.

Choose0.68B and 1.65B for the first real quality contrast. Retain3.17B when useful
underfitting, a better objective or improved transfer offers a concrete reason.
Do not reject scale from one noisy screen, but do not assume a 5B teacher will
generalize better from the same correlated dataset. Hundreds of millions of rows
are not hundreds of millions of independent regimes.

## Muon, AdamW and schedule

Use Muon on an explicit allowlist of hidden attention/FFN matrices. Keep numeric
input encoders, learned embeddings, task readouts, norm scales and biases in
AdamW groups even when a parameter is 2D. Every trainable parameter belongs to
exactly one group. Keep Q/K/V as separately defined matrices for orthogonalization;
fusing them into a tall QKV matrix changes the Muon update geometry.

| Setting | Proposed starting recipe, not a measured optimum |
|---|---|
| Muon convention | `match_rms_adamw`, shape factor 0.2*sqrt(max(rows,cols)) |
| Base peak LR,0.68B | 1e-4; targeted alternatives 0.5× and 2× |
| Base peak LR,1.65B | 7.5e-5; targeted alternatives 0.5× and 2× |
| Base peak LR,3.17B | 5e-5 starting prior; retune rather than assume a law |
| Momentum / Nesterov | 0.95 / true |
| Orthogonalization | 5 Newton–Schulz iterations, all intermediates FP32 |
| AdamW groups | same base LR initially, betas(0.9,0.95), epsilon 1e-8 |
| Weight decay | 0.05 on backbone matrices;0.01 on input/readout matrices;0 on biases/norms |
| Clipping | global preclip norm logged, threshold 1.0, once per update before either optimizer |
| Reference optimizer | AdamW on all matched groups, same base LR/schedule, separately declared contrast |

The base LR above is NOT the original-Muon LR0.02 convention. Pin the shape
adjustment; do not mix these recipes. In the referenced convention decay uses
the unadjusted base LR; make that semantic explicit. Audit the installed source:
current upstream PyTorch Muon casts Newton–Schulz work to BF16, so its unmodified
implementation does not satisfy this plan. A source-pinned FP32 variant needs
finite-update, dtype, state restoration and actual-step-cost checks. B300's
advertised low-precision throughput is not available merely by using Muon.

Use WSD for an extendable quality branch. Anchor warmup at 1500 successful updates
for the initial 24-epoch,65,536-new-row/update scenario below. Use that SAME warmup
for the 12/24/48-epoch descendants; this retains a shared stable prefix. For each
frozen horizon branch from the stable checkpoint at 80% of that horizon, then
cosine-decay over the final 20% to 0.1× peak. Preserve the uncooled stable state for
an admitted extension. A cosine-only schedule at matched horizon is a separate
comparison; neither scheduler is established as best for this task.

Advance both optimizers and scheduler exactly once per successful update. Save
loss components, LR per group, preclip norms, clipping/nonfinite counts, update
norms, throughput and checkpoint identities. Never select the best outer-fold
epoch retrospectively and present it as a fixed terminal comparison. EMA/SWA
of cooled checkpoints is a declared candidate and must be evaluated, not an
assumed gain. Warmup and LR must be rebudgeted if global batch changes.

## Eight-card execution

Begin with ordinary synchronous NCCL DDP, one process per device. Model states
at these scales are plausibly far below per-card capacity; DDP is for throughput,
not pooling HBM. Compare one 8-rank job with two 4-rank jobs at the SAME global
row batch and objective. Independent4-rank fits may deliver more useful completed
comparisons; use measured time-to-graded-endpoint, not utilization percentage.

For the 4096-window/2048-core reference,8 ranks ×2 windows/rank ×2 accumulation
microsteps =65,536 NEW core rows/update, before boundary masks. The processed
input count is about twice the new-row count because prefix is recomputed.
With4 ranks use 4 windows/rank and 2 microsteps for the same nominal batch, subject
to an actual profile. Whole20k-sequence training at 1 sequence/rank on 8 ranks
instead gives about 160k raw rows/update and a different masked eligible count;
do not reuse the 65,536-step/LR arithmetic for it.

For target k and update window, let W_k be the all-reduced sum of fitting weights
over ALL ranks and accumulation microsteps. Each rank's microstep loss is
R/2 * sum_k(local_weighted_error_sum_k / W_k), under default DDP gradient averaging.
Accumulate those terms without another division by accumulation count. Determine
denominators from data/masks without retaining every forward graph. Normalize
auxiliary objectives separately. `no_sync` encloses forward AND backward for
nonfinal microsteps. Handle zero denominators coherently across ranks.

Keep each live sequence on one rank where recurrent/slow state is used; reset
all state, positions and feature priors correctly. Avoid sampler duplication;
use zero-weight padded work to equalize collective counts when necessary. Compute
the official score from pooled sufficient statistics or aligned prediction banks,
not an average of rank correlations. Save RNG/sampler/sequence offsets as well
as model, optimizer and scheduler states.

Use pinned software and bounded resident data staging. Do not reread/decompress
Parquet every microstep or copy all data once per rank. Compile fixed-shape work
after correctness checks and report compile time separately. First use replicated
Muon for a simple exact reference. If its orthogonalization is a measured
bottleneck, an owner-per-WHOLE-matrix implementation can distribute that work
and broadcast identical updates; splitting a matrix into arbitrary shards changes
Muon. FSDP needs complete-matrix handling before orthogonalization and is not a
free drop-in replacement. None of this optimizer-distribution code is implemented
by this design packet.

The supplied [gradient-delay paper](https://arxiv.org/abs/2606.30634) concerns
asynchronous pipeline training and Muon's robustness to one-step delay. It does
not establish a quality gain for synchronous DDP on this regression problem.
Pipeline parallelism and error-feedback updates are a future engineering branch
if a much larger model genuinely needs stages; they are not prerequisites for
the first 1.65B teacher, whose state fits analytically on a single replica.

## Duration and how to obtain a credible ETA

The data contract contains 249.6M supplied rows. Exact fitting membership and
eligible masks set N; use **N=200M only as an arithmetic scenario**. At65,536
new rows/update,12/24/48 visits through that fitting population need approximately
36,622 /73,243 /146,485 updates. A complete-sequence batcher may change the exact
count; record actual attempted and successful steps and row visits.

| Measured whole-job seconds/update, hypothetical until profiled | 12 epochs | 24 epochs | 48 epochs |
|---:|---:|---:|---:|
| 0.5 | 5.1h | 10.2h | 20.3h |
| 1 | 10.2h | 20.3h | 40.7h |
| 2 | 20.3h | 40.7h | 81.4h |
| 5 | 50.9h | 101.7h | 203.5h |

These are conditional examples, not predicted B300 speeds. An8-rank fit consumes
eight GPU-hours per wall hour. Add data/setup/compilation, evaluation, checkpoints,
teacher export and GRU training; a 4-fit screen multiplies actual fit costs.
Measure both processed and eligible-new rows/second and real checkpoint/evaluation
overhead. Forward/backward cost scales approximately with 6P*processed_tokens plus
attention pair work; context overlap and quadratic attention make parameter-only
FLOP estimates incomplete. Muon and communication add further work.

Proposed engineering measurement:20 warmup plus 50 timed successful updates at
each admitted actual shape, including optimizer and synchronization; preserve
dtype/backend, p50/p95 step time, device peaks, host charge, I/O and NCCL time.
Test checkpoints and resume before a multi-day fit. Profiles need an explicit
bounded device/time reservation and are not free or completed scientific fits.
Choose12 epochs as the first cooled endpoint, then 24/48 when quality, learning
curves or transfer warrants continuation. Extra epochs reuse data, not new regimes.

## Small fitting panels and the GRU endpoint

Freeze each contrast separately; the list below is a portfolio, not one funded
Cartesian grid. Initial example cells are F0/seed 1001 and F2/seed 1002, subject to
ownership. Two arms ×two cells =four fits. A third declared cell gives six fits.
This cannot estimate fold and fitting-seed effects independently or prove significance.

1. **Capacity recipes:**0.68B versus 1.65B at matched context, data, nominal row
   budget and fixed cooled horizon. Scale-specific LRs are explicit recipe
   differences; do not call this an isolated architecture effect. Four new fits.
2. **Optimizer:**selected scale's matched AdamW versus FP32 Muon. Reuse exact
   clean Muon endpoints only if EVERYTHING matches; otherwise this is another
   four-fit screen. Do not count an LR pilot as independent confirmation.
3. **Representation/dynamics:**one grouped encoder or auxiliary-loss addition
   versus its exact reference, then full-history attention as a distinct arm.
   Each admission has its own 4–6-fit inventory; not all are automatically funded.
4. **Transfer:**freeze the chosen teacher targets and compare same-initializer
   GRU label-only/reference-teacher versus new-teacher KD on matched cells. Four
   new student fits if teacher caches already exist. Count teacher fitting/export
   costs separately; a teacher screen alone cannot establish teacher→GRU value.

Start regression KD at alpha 0.8 (labels 0.2/KD0.8), weighted with separately
declared denominators and preserved teacher amplitude. Alpha0.5/0.9 are alternatives,
not an assumed universal optimum. Use the exact same fold-clean student parent,
normalization, features and optimizer schedule in the pair. A cold GRU and a
warm student test different questions. No logits/temperature KL is needed for
two continuous outputs. A teacher that consumes privileged future information
cannot be repaired by distilling it into a causal student.

If transfer saturates, test an intermediate larger GRU/compact Transformer,
projected latent-state supervision or a teacher mixture as explicit alternatives.
Rank them by the final GRU and complete served blend, not teacher accuracy alone.
An assistant may help capacity mismatch but is not guaranteed. Periodically
distill cooled teacher checkpoints; a later teacher may transfer worse despite
better standalone fit. Do not extrapolate dataset-independent scaling laws.

Teacher fitting, self-supervised pretraining, learned embedding knots, auxiliary
scales, initialization and teacher mixtures must exclude each evaluated component.
Fitting-data teacher targets may be in-sample; label them. For final all-data
production fitting, freeze recipe/horizon from development and retain its ancestry.
All existing folds remain exposed. Prediction bootstraps cover fixed-fit component
sensitivity, not model selection, seed variability or public transfer.

## Readiness, independent criticism and next implementation

The [draft packet](https://github.com/VladKochetov007/WunderFunder/blob/experiment/large-dynamics-teacher-b300-r2/research/experiment_drafts/large-dynamics-teacher-b300-r2/README.md)
contains a recipe, fit inventory and analytical calculator. It is launch_ready=false.
Reuse canonical dataset/input/objective/checkpoint/reporting components; do not
copy the monolithic trainer or put Transformer/DDP logic in scripts/train_rnn.py.
Needed components are a causal teacher factory, explicit-mask attention backend,
FP32 Muon adapter, distributed update engine and source-bound target exporter.

Before any actual fit, verify future-perturbation/prefix invariance, per-sequence
reset, completed-block visibility, sequential-versus-scan AR gradients if used,
single-rank-versus-DDP weighted updates with unequal masks/accumulation, actual
producer checkpoint/cache schemas, dtype internals and full resume. Then profile
actual B300 shapes. No source-only check establishes these numerical properties.

Independent Luna/max scout and specification critic completed read-only reviews
for this task. Their shared findings were causal patch/summary visibility,
per-matrix optimizer semantics, finite-data risk, the absence of runtime evidence
and explicit student cost. The scout's phrase 4–6 matched runs was corrected in
synthesis to the user's4–6 TOTAL new fits. Their public findings are preserved in
the [review receipt](../reviews/LARGE_TEACHER_DESIGN_REVIEW_20261019_R2.md).
The controller's synthesis is a proposal, not independent experimental approval.

## Source records

[Primary-source register](../sources/TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md)
records the actual paper/docs lookups, API caveats and transfer limits. Related
H/E ledger entries register this unrun scale/dynamics design. No supplied data,
weights or external pretrained payloads were imported for this task.

Published handoff: [draft PR #31](https://github.com/VladKochetov007/WunderFunder/pull/31) targets research-team. Its publication receipt names the initial source/base commits and recipe hashes. No B300 fit is admitted.

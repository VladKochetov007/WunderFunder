---
kind: method
status: proposed
created_utc: 2026-10-13T23:55:12Z
updated_utc: 2026-10-13T23:55:12Z
hypothesis_ids: [H-CPU-POLICY-20261014, H-STUDENT-LADDER-20261014, H-STATE-INIT-20261014]
experiment_ids: []
evidence_paths: [PLAN.md, SHAPE_ACCOUNTING_R1.csv]
---

# Theory that can guide experiments while CPU submissions wait

## Question or claim

Which architectural changes have a plausible route to better quality per causal callback, and which measurements distinguish them? This is a source-grounded analysis and set of falsifiable hypotheses, not a measured frontier or a runtime certificate. The user explicitly asks for theory and independent work while target-host uploads are pending.

## Mechanism and method

### Depth, width and physical working sets

For the current model in `src/wnn/nn.py`, with211 input features, a width-H projection, L homogeneous GRU layers, a2H→H→2 readout and no auxiliary outputs, the dense matrix coefficient count is

`P(H,L) = 211H + 6LH² + 2H² + 2H = (6L+2)H² + 213H`.

The terms count the encoder, three input and three recurrent gates per layer, and readout. Biases, normalization, scales, SIMD padding and retained native arrays are excluded. [Derived accounting](SHAPE_ACCOUNTING_R1.csv) gives578,816;556,992;972,032;2,206,208;2,146,176;3,413,120;3,779,072;4,109,504 coefficients in the plan's order. This verifies its approximate counts, without certifying runtime. Extra distillation/auxiliary heads must be accounted for in the actual exported artifact.

H512×2 has4× the control's quadratic term, but only2× its input projection and final2-output layer. The2H→H first head layer scales4×. Compared with four separately invoked H256×2 members, it can also remove repeated dispatch, activation quantization and some nonlinear work. It loses independent-fitting diversity and creates larger individual matrices. Those tradeoffs are why the matched student/ensemble comparison needs both an official grade and a complete callback measurement.

The current native Q8 convention uses Q8 recurrent and first-head weights, I16 input projection and FP32 last-head weights. Under that convention, unpadded logical matrix bytes are `(6L+2)H² + 430H`:1,027,584bytes for a regularH256×2. This reproduces the [existing actual-shape accounting](../../campaign_2026-10-13/team_models_r1/LOGICAL_Q8_WEIGHT_TRAFFIC_R1.json). It is near a1MiB cache even before scales and scratch. Changing only ZIP storage does not change these active bytes.

AMD documents1MB private L2 and256-bit internal paths behind AVX512 on Zen4. These facts motivate comparisons; they do not establish a256-bit winner or prove which working set resides in a shared competition container. Register pressure, load reuse, row padding and the dependent recurrent path matter. [AMD architecture white paper](https://www.amd.com/content/dam/amd/en/documents/products/epyc/4th-gen-epyc-processor-architecture-white-paper.pdf), pp.8–9, retrievedOctober14Europe/Kyiv.

The cheapest discriminating evidence is therefore nested: matrix cost, complete cell cost, then complete callback cost including features/reset/Python. Repeated original controls bracket machine variation. Record CPU and wall time independently, since throttling and host load can change wall time. The matched competition probe is still required for a target-host claim.

The depth/width approximation literature establishes expressivity results for particular feed-forward function classes, rather than an optimum for a finite trained GRU on this task. Its transferable lesson is to avoid treating representability as a prediction of trainability or sample efficiency. [Lu et al.,2017](https://arxiv.org/abs/1709.02540). The one-epoch ladder is a fixed-budget screen; differing convergence rates justify longer matched finalist fits and an exploration slot, not a universal shallow/deep conclusion.

### Why integer scheduling can be exact and fusion can fail

For a K-term unsigned-byte by signed-byte dot product, the conservative magnitude bound is `255×127×K`. At K=1280 this is41,452,800, below signed32-bit range. Reordering non-saturating integer accumulation can therefore preserve the mathematical sum for the declared ladder. It does not automatically preserve float epilogues, rounding, biases or nonlinear functions. Compare actual quantized codes/scales and output bits.

Input and previous-hidden vectors have different dynamic scales. Fusing their quantization into a single concatenated vector changes the represented arithmetic. A valid exact fusion retains each scale and each bias placement; it also reads the entire old hidden vector before publishing a new state. Correct matrix tests alone cannot detect all layer-order or state-aliasing bugs. Full sequences, row99 behavior and reset replay are necessary.

### Genuine W4: capacity benefit requires compute evidence

Three kernels must share one represented W4 code/scale tensor: expanded bytes, in-kernel nibble decode, and bit-plane lookup. The current expanded-byte path has transport savings without a smaller recurrent execution matrix. In-kernel decode saves weight traffic but adds decode/shuffle work. A lookup kernel pays a per-activation table construction cost that must be amortized over output rows.

T-MAC constructs tables of partial activation sums indexed by low-bit weight patterns, then tiles lookup/accumulation; its documented layout and table-storage tradeoffs are concrete starting points. Its LLM benchmarks do not establish a speedup for our smaller recurrent matrices. [T-MAC,2025revision](https://arxiv.org/html/2407.00088v2), §3. A useful screen measures table construction and end-to-end matvec separately across192–640 widths, using identical represented weights. Promotion requires the largerW4 student's complete quality/runtime point to beat an appropriate smallerW8 point; storage size alone is insufficient.

### Initial-state information and its decay

Write the deployed GRU update as `h_t = F(x_t,h_(t-1))`. A small initial perturbation evolves approximately as `δh_t = J_t … J_(s+1) δh_s`, where eachJ is the full state Jacobian. The update-gate value alone is not a valid contraction estimate, since the reset gate and candidate also depend on old state. Use measured finite perturbations on fitting sequences; an autograd Jacobian-vector product is an optional separate diagnostic.

A constant or first-row state traverses99 warmup updates before first scoring. A residual injected after row98 avoids those99 transitions, which creates a specific testable difference. Contextual RNNs provide precedent for end-to-end learned input-conditioned state, including first-element context, but do not establish usefulness for this market sequence or permit future context. [Wenke and Fleming,2019](https://arxiv.org/html/1902.03455v1), §§1.1.2–2.

For the specified prefix MLP,633 inputs represent mean/std/last over211 normalized features. A64-unit hidden layer producingLH residual coordinates has `633×64 + 64×LH + 64 + LH` parameters:73,856 forH256×2, or295,424FP32 bytes before metadata. The one-time dense work is73,280multiply-accumulates, about3.664 per row amortized over20,000 rows. Prefix-statistics collection and normalizer/preprocessing work remain separate measured costs. A cheap one-time operation can still have no useful predictive effect.

Measure hidden perturbation norms and prediction changes separately in the frozen position bins. A large hidden change in a readout-null direction is not a useful prediction change. Grade the official global metric plus position-restricted diagnostics; do not add bin correlations as though they sum to the global score. Full reset clears both statistics and the one-time flag. Only rows0–98 enter the prefix vector; row99 cannot enter it before injection.

Truncated BPTT introduces another important distinction: with chunk1000, gradients from later chunks do not reach the initializer through a detached state. Joint training under that rule optimizes a truncated objective even if effects persist longer. Record the gradient horizon, keep it matched, and test initializer-only adaptation against the same frozen backbone. If measured useful influence is brief, persistent prefix conditioning is a new arm with a different per-row cost, not proof that the initializer succeeded.

### Cheaper recurrence and allocation

minGRU removes hidden-to-gate/candidate dependencies, leaving input-conditioned diagonal recurrence. The paper gives roughly2IH matrix coefficients per layer versus3H(I+H) for GRU and allows a parallel training scan. [Feng et al.,2024](https://arxiv.org/html/2410.01201v3), §3.1. Its inference recurrence still needs sequential state updates in our callback. The simplified transition may lose useful hidden-conditioned decisions; evaluate as a trained student with the same teacher rather than as an exact GRU substitution. The plan's separate ReLU-candidate GRU still has sigmoid gates and dense recurrent dependencies and must not be relabeled minGRU.

For a stacked3H×H gate matrix, a rank-r factorization costs4Hr coefficients instead of3H², with an arithmetic break-even at `r < 0.75H`, before extra intermediate quantization and dispatch. Rank selection, factor sharing and nonlinear sensitivity determine whether it helps. Structured/shared low-rank RNNs have precedent in speech models, not direct evidence for this competition. [Lu,Sindhwani,Sainath,2016](https://arxiv.org/abs/1604.02594). The first screen should compare exact dense reconstruction against the factorized deployed path to separate approximation loss from execution changes.

Sharing a two-branch encoder saves only211H projection coefficients at equal widths; each independent recurrent transition still needs its own gates/state. Exact shared-backbone serving requires matching tensors and preprocessing, not matching architecture names. Shared-encoder training and independently fitted ensemble diversity are separate experiments.

## Evidence and interpretation

The shape formulas are derived from current source and reproduce historical logical byte accounting. Literature provides mechanisms and boundary conditions. None of these sources establishes a new competition-quality gain, a target CPU capacity range, a best architecture, or a winning initial state.

## Limits and validity

All local folds remain exposed. Native arithmetic, source ancestry, equal fitting membership and teacher version are mandatory controls. Hardware theory and synthetic matrix results remain provisional until complete serving measurements. Historical final team weights retain fitted/unknown validation overlap and are suitable for engine equivalence, not clean student selection.

## Decision and next step

Continue the matched eight-shape screen; retain explicit cheaper-recurrence and initializer branches independent of portal timing. After local profiles, prioritize equal-cost comparisons and measure the two leading students under stronger frozen teacher versions. Preserve unknown numerical priors and all failed attempts. Use the [live frontier](FRONTIER.md) for actual completed stages.

## Source records

[Complete plan](PLAN.md), [contract](CONTRACT.json), [independent specification review](../../unified/reviews/CPU_STUDENT_FRONTIER_SPEC_REVIEW_20261014_R1.md), [previous EPYC evidence](../../campaign_2026-10-13/team_models_r1/EPYC_RUNTIME_REFLECTION_R1.md). Online sources above were opened in this session. The original AMD URL timed out; the alternate official AMD white-paper URL succeeded. Official competition pages opened as client-rendered shells; current local documentation and user-supplied premises remain the environment evidence pending a rendered refresh.

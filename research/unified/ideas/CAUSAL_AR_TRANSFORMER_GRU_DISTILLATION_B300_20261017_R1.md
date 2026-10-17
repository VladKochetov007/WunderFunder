---
kind: idea
status: proposed
created_utc: 2026-10-17T11:11:30Z
updated_utc: 2026-10-17T11:21:03Z
hypothesis_ids:
  - H-TRANSFORMER-CAUSAL-AR-20261017
  - H-TRANSFORMER-INNOVATION-AUX-20261017
  - H-TRANSFORMER-AR-GRU-KD-20261017
  - H-TRANSFORMER-QUALITY-SCALE-20261017
experiment_ids:
  - E-TRANSFORMER-AR-FACTORIAL-20261017-R1
  - E-TRANSFORMER-AR-GRU-KD-20261017-R1
  - E-TRANSFORMER-QUALITY-SCALE-20261017-R1
evidence_paths:
  - docs/data_overview.md
  - src/wnn/gmetric.py
  - research/RESEARCH_METHODS.md
---

# Causal autoregressive Transformer teachers, GRU distillation, and eight-B300 training

## Question or claim

Can a causal Transformer with a learned autoregressive output term produce a better teacher and, more importantly, a better deterministic GRU student for root Wunder Connectome Challenge 2?

This is a detailed design proposal requested by the user. The eight NVIDIA B300 GPUs are user-reported planned resources; no access, topology, software stack, memory profile, throughput, fit, or score was verified for that cluster. Writing this plan does not reopen the paused local research campaign or authorize a remote launch. The user wants complete training and variation across fitting seeds and source-qualified folds, with quantization deferred until submission engineering.

Latest user steering: “focus on best possible quality, we have a lot of compute.” This supersedes treating a small model or a short training budget as the preferred endpoint. The 86M model and six-epoch run remain matched references. Primary quality exploration expands to roughly 207M and 682M teachers, a 1.65B scale arm, longer/full-sequence context, 12/24/48-epoch training comparisons, and heterogeneous teacher ensembles, followed through actual GRU distillation. Section 19 defines this quality-first scope. The user has requested a research note, not an actual cluster launch in this turn.

The recommendation is a numerical-input causal decoder Transformer with a small learned AR(1) readout outside the attention backbone. An auxiliary future-observation prediction objective is a separate, combinable autoregressive-learning arm. Neither a causal mask alone nor an auxiliary forecasting head should be misreported as feedback of past predictions.

All hyperparameters below are proposed starting settings, not established competition optima. All runtime numbers are arithmetic scenarios, not B300 measurements.

## Mechanism and method

### 1. What is known about this task

The [local data contract](../../../docs/data_overview.md) specifies 112 numerical features, 20,000 ordered rows per sequence, two future-price-related targets, and causal row-by-row prediction. Train has 10,607 sequences / 212,140,000 rows; supplied validation has 1,873 sequences / 37,460,000 rows. Actual fitting row counts depend on the frozen component-disjoint fold manifest and any permitted-data exclusions. The local reader documents corrupt train row group 528; do not silently include it or silently repair the canonical file.

The first 99 rows update model state but require no prediction. All later rows require outputs, although only the hidden scoring subset contributes to the official score. Sequence IDs are identifiers, not chronological information. There is no usable target-history feed at inference.

Crucially, the previous label y[t-1] describes future price movement relative to row t-1. Its value cannot be assumed observed at row t. The undisclosed target horizon also prevents inventing a label-maturation delay. Feeding y[t-1] during fitting and substituting a model output at inference would introduce unavailable information and a train/serve mismatch.

The frozen official metric remains [global_wp_clip2_v1](../../../src/wnn/gmetric.py): predictions and labels clipped at ±2, original clipped-label magnitude weights, pooled scored rows, mean over the two target correlations. Training weights and training-target clipping are distinct recipe choices.

### 2. Proposed backbone

| Component | 86M matched reference; larger quality variants in Section 19 |
|---|---|
| Numerical token | One complete 112-feature row |
| Input mapping | MLP 112 → 768 → 768; fixed feature identities |
| Blocks | 12 causal decoder blocks, pre-RMSNorm |
| Hidden width | 768 |
| Attention heads | 12 heads, head dimension 64; ordinary multi-head attention |
| Feed-forward | SwiGLU, intermediate width 2048 |
| Position | RoPE using actual row positions within the sequence |
| Output branches | Two-value innovation proposal u[t], two-value AR gate a[t] |
| Residual dropout | 0.05 initially |
| Attention dropout | 0 initially; any change is recorded |
| Nominal input window | Up to 4096 rows per training chunk, including left context |
| Training precision | Strict FP32 reference; see the explicit precision distinction below |
| Research model size | About 86M parameters including input/head overhead; implementation must count exactly |

The approximate block count is 12 × (4 × 768² + 3 × 768 × 2048), about 84.9M before embeddings, readouts, and normalization parameters. This is our parameter arithmetic, not a measured model inventory.

The decoder/normalization/feed-forward ingredients follow the mechanisms in [S17]. A proposed scale ladder is S: 8 layers / width 512 / SwiGLU 1536, about 28M parameters; M: 12 / 768 / 2048, about 86M; L: 16 / 1024 / 2816, about 207M. Compare them at both fixed compute and declared full training horizons. Scaling parameter count and context together cannot identify which change helped. These optional arms are not extra admitted spending.

A grouped-input sibling uses separate MLPs for the supplied 13 feature groups, concatenated and projected into one temporal token. It does not expand each row into 112 temporal tokens. Column indices are identifiers, not guaranteed order-book depth; do not assume adjacent indices are adjacent price levels.

Use frozen fitting-only feature-wise mean/standard-deviation normalization initially, flooring standard deviations at 1e-3. Record exact statistics and feature order. A whole-sequence mean/variance computed using rows after t would leak into an early prediction. Dynamic normalization, RevIN, robust scales, feature clipping, and hand-engineered clocks are separate representation arms. Pair candidate/control by explicitly copying the common initial backbone and input/head weights before adding the zero-initialized gate branch, so extra parameter initialization does not inadvertently change all common weights or the data-order RNG.

RoPE is a starting positional mechanism [S1]. A row index represents event order, not physical elapsed time. Any additional clock or age feature must come from observables available at that row.

### 3. Add the autoregressive term without serializing the Transformer

For each target independently, compute from the causal Transformer representation:

$$
u_t = g_u(H_t), \qquad
a_t = \rho_{\max}\,\operatorname{sigmoid}(g_a(H_t)),
$$

$$
\widehat y_t = a_t \odot \widehat y_{t-1}
             + (1-a_t)\odot u_t.
$$

Here H[t] depends only on observable inputs up to t. The feedback is the model's own previous output, never a true target. The recurrence operates on two scalars, one per target.

Proposed initialization:

- rho_max = 0.99, with a separate 0.95 cap arm only if needed.
- Initialize gate-head weights to zero.
- Initialize gate-head bias to logit(0.10 / 0.99), so every initial gate is 0.10.
- Reset the two-value AR state to zero at the start of each sequence.
- Update it on every row, including all 99 warmup rows and unscored rows.
- Do not clip the AR state to the official scoring range during training by default.
- The exact non-AR control fixes a[t] = 0, yielding yhat[t] = u[t].

This is a learned input-dependent persistence term. It can preserve a previous estimate in quiet periods and refresh it on informative inputs. It can also oversmooth fast changes, create lag, or learn a redundant state; those are explicit failure hypotheses. Fixed output smoothing and this jointly trained gate are different recipes, and neither is presumed beneficial.

The gate's instantaneous derivative with respect to the previous output, holding the current observable-derived backbone output fixed, is a[t]. Thus each one-step feedback coefficient is bounded below 1. This does not establish a generalization improvement, good calibration, or a universal numerical-stability guarantee for the entire learned network.

Keep the feedback OUTSIDE the Transformer input. If yesterday's predicted output is injected into today's Transformer token, a complete teacher rollout requires sequential backbone calls, and the intended time-parallel training path is lost.

For the proposed readout, let b[t] = (1-a[t])u[t]. Affine recurrence pairs compose associatively:

$$
(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\ b_2+a_2b_1).
$$

Consequently, training can compute all H[t], u[t], and a[t] in parallel and evaluate the small recurrence with a differentiable prefix scan. This algebra is our derivation. Gated linear-attention literature [S9] supplies a related hardware-aware recurrence precedent, not evidence for this particular readout.

A production scan must have a verified backward implementation. Do not assume an experimental scan API supports autograd or all requested dtypes. Avoid formulations that divide by a long cumulative product: products can underflow over 20,000 rows. A sequential FP32 recurrence is the correctness reference; a blocked associative scan is the optimization candidate. A CPU/Python loop over all rows is not an acceptable throughput assumption.

### 4. A separate auxiliary autoregressive learning term

Add an optional head predicting future changes in observable features:

$$
\Delta x_{t,h}=x_{t+h}-x_t,\quad h\in\{1,8\},
$$

$$
\mathcal L_{\mathrm{aux}}
=\operatorname{mean}_{t,h,j}\,
\operatorname{Huber}\left(
\frac{\widehat{\Delta x}_{t,h,j}-\Delta x_{t,h,j}}{s_j}
\right).
$$

The scale s[j] is fitted on fitting groups only, floored away from zero, and frozen for evaluation. Exclude cross-sequence pairs and out-of-range horizons. Predict the designated observable feature columns, excluding sequence identity, scoring flags, and target columns.

Future observations are auxiliary TARGETS during fitting, never inputs available to the representation at t. This distinction must survive tokenization, normalization, augmentation, caching, and batching.

Start with beta_aux = 0.05 only after defining both supervised and auxiliary terms as normalized means. Log their separate losses and shared-backbone gradient norms: a numerical coefficient alone does not fix their relative gradient contribution. Zero changes may dominate this task; report change/no-change diagnostics. Reweighting changed rows or adding an update-indicator objective is another arm, not an invisible repair.

The next-observation auxiliary task and learned output recurrence answer different questions. Neither requires a language-model softmax or discrete vocabulary.

### 5. Train four teacher variants as a factorial comparison

| Arm | Learned output AR | Observable-change auxiliary loss |
|---|---:|---:|
| T0 | No | No |
| T1 | Yes | No |
| T2 | No | Yes |
| T3 | Yes | Yes |

Keep the backbone, fitting groups, target treatment, optimizer, update horizon, and checkpoint-selection rule matched. Record tiny parameter-count differences.

Primary contrasts are T1−T0, T2−T0, and interaction T3−T1−T2+T0. The interaction is measured on the declared score endpoint; no gain is assumed additive. Evaluate both raw u[t] and final yhat[t] diagnostically, but freeze the selected served/teacher output before comparison outcomes.

### 6. Supervised objective and target semantics

A concrete proposed baseline uses:

$$
c(y)=\operatorname{clip}(y,-2,2),\qquad
w(y)=0.05+|c(y)|,
$$

$$
\mathcal L_{\mathrm{sup}}
=\frac12\sum_{k=0}^1
\frac{\sum_t m_t w(y_{t,k})
(\widehat y_{t,k}-c(y_{t,k}))^2}
{\sum_t m_t w(y_{t,k})}.
$$

The training mask m excludes warmup, padding, repeated prefix rows, and rows outside declared fitting membership. It does not pretend the hidden test scoring mask is observed. The available fitting-source scoring-mask/propensity policy must be declared; the primary simple recipe above uses all otherwise eligible fitting rows and no newly learned propensity model.

This proposes CLIPPED training targets explicitly. It is not mandated by the official metric. A raw-target or inherited successful-teacher-loss arm is valid if separately registered and used for both AR/control. Do not attribute an inherited-recipe comparison entirely to architecture when loss or training membership changes.

Teacher total loss is L_sup + beta_aux L_aux. Evaluation always calls the official scorer. Batch correlation is not the official pooled score and is not the default teacher training loss in this proposal.

### 7. Chunking, state, and context are part of the architecture

Proposed starting layout: 1024 new core rows plus up to 3072 preceding raw rows, giving an input length up to 4096. Enumerate nonoverlapping cores in sequence order so every eligible fitting row is a supervised core row once per epoch. Pad partial final cores and mask padding.

Within each window, attention is causal. Core rows have approximately 3072–4095 preceding rows available once away from the sequence start, rather than an identical 4096-row rolling history for every output. This is an anchored overlapping-window model. Do not label it an exact full-sequence or exact rolling-window Transformer.

Recompute prefix representations under current weights. Only core outputs advance the persistent AR state; replayed prefix rows must not update that state twice. At the first window of a sequence, include the initial rows and perform warmup state updates. Carry the two-value output state across cores; detach it at the declared training TBPTT boundary. This truncates the recurrence gradient across cores and must be recorded.

Contiguous sequence slots stay on their assigned rank until completion. An optimizer update between cores means the carried training state was formed under slightly older weights; that is a declared truncated-training approximation. Evaluation/export uses fixed weights for the complete sequence and has no such optimizer-state drift.

Teacher-target export must use the same anchored window definition and row positions. Changing window anchors, replacing it with a different rolling KV cache, or resetting state at each chunk changes the model and must be measured. Merely carrying a KV cache across all layers is not automatically equivalent to recomputing the raw prefix.

Long-context arms change the prefix/window lengths explicitly. Transformer-XL-style detached hidden memory [S2] is a separate architecture/training arm. A per-layer sliding attention width can yield a larger stacked receptive field; do not confuse it with a hard bound on total input history.

## Distillation into GRU

### 8. Why this can be distilled, and what is not guaranteed

After freezing teacher weights and using its own recurrent outputs, the teacher is a deterministic causal function of x[0:t]. Its AR state contains no privileged target information. A GRU can be trained on that function without receiving the Transformer, its attention states, or its previous teacher output at inference.

This establishes information compatibility, not representational equivalence. Finite student capacity, different memory, optimization, or a teacher's brittle fine-scale behavior can prevent successful transfer. Teacher-score improvement is therefore insufficient for selecting the teacher. The final endpoint is the trained student's official score and its contribution to the frozen submission blend. Distillation-capacity literature motivates this separation [S6]; it does not guarantee transfer here.

Export continuous two-value teacher predictions with dropout disabled, complete sequence state/reset, and pinned precision/window semantics. Use FP32 output storage as the reference; two outputs for all 249.6M supplied rows occupy about 1.997 decimal GB before IDs/provenance. This is arithmetic, not a measured cache format.

### 9. Two student architectures

S0 is the matched existing GRU architecture with a direct two-value head. It approximates the teacher's final yhat[t] through its own recurrent state.

S1 uses the same GRU backbone plus the same two-value learned AR readout. It adds explicit output recurrence to the student's state. At deployment, the GRU needs only its hidden state and two previous predictions; u[t] and a[t] come from its own representation. Gate initialization, sequence reset, and warmup updates follow the teacher's contract. Its actual package/runtime/ABI are later submission-engineering measurements.

For both students, primary targets are the teacher's final FREE-RUNNING yhat[t]. Do not train the student using the teacher's previous prediction as a feedback input and then replace it with its own previous prediction at deployment. That distribution mismatch is unnecessary. True previous targets are unavailable as inputs in both teacher and student.

Scheduled sampling [S4] discusses a related train/inference feedback mismatch in sequence generation, but does not make lagged hidden future-price labels valid features in this competition. The primary plan uses free-running predictions throughout and does not introduce scheduled sampling.

### 10. Student objective and controls

Use the same explicitly declared supervised target treatment for every student arm:

$$
\mathcal L_S=(1-\alpha)\mathcal L_{\mathrm{sup}}(S,y)
+\alpha\,\mathcal L_{\mathrm{KD}}(S,\widehat y^T),
$$

where KD is weighted squared regression with a declared normalization. Initial default alpha = 0.9; alpha = 0.5 is a separate full-training comparison. The teacher has two continuous outputs, so temperature-softmax KL is not the primary KD loss.

For the primary KD term, use the same label-derived weights and eligible fitting rows as the supervised term, with denominators calculated separately per target. Using teacher confidence, clipping teacher targets, or rescaling teacher amplitude is a declared variant. Preserve original teacher amplitude unless the recipe explicitly compares calibration.

Minimum matched controls:

1. Label-only GRU, alpha = 0.
2. Strong GRU-teacher → GRU, alpha = 0.9.
3. T0 Transformer → GRU.
4. T1 Transformer with output AR → GRU.
5. Fixed 50/50 GRU/selected-Transformer teacher mixture → GRU, if admitted.
6. For the selected teacher, S0 direct head versus S1 explicit AR head.

Warm-start students share the same exact initializer within each paired comparison. The label-only control receives the same optimization budget. Different teacher seeds, shared parents, and weight exports do not constitute independent training repetitions.

Optional hidden-state alignment requires learned projections and an additional matched objective; there is no natural coordinate correspondence between a 768-wide attention representation and a GRU state. It is not required for the first KD recipe.

### 11. Fitting ancestry

For each evaluation fold, every teacher, initializer, normalizer, auxiliary scale, and any fitted teacher mixture must exclude that fold's evaluation components across BOTH Parquet sources. Retain source-qualified group identities. The official team assets are reusable, but unknown/all-data fitting ancestry cannot support a clean local comparison.

Teacher targets on fitting data may be in-sample teacher outputs; record that exposure. Out-of-fold targets within the fitting partition are a separate, more expensive design. Evaluation predictions must still come from an evaluation-excluding teacher/student ancestry.

All existing local folds have prior exposure. Additional seeds and cross-fold runs measure robustness; they do not create a pristine holdout. Final all-data teachers/students are separate fitting artifacts and cannot be graded as held-out models on the data they fitted.

## Optimizers, schedules, and duration

### 12. Teacher: concrete reference recipe

| Setting | Proposed initial value |
|---|---|
| Optimizer | AdamW [S5] |
| Peak LR, model M | 3e-4 |
| LR comparison | 1e-4, 3e-4, 6e-4 as separately budgeted complete runs |
| Betas | (0.9, 0.95) |
| Epsilon | 1e-8 |
| Weight decay | 0.05 for backbone/input/readout weight matrices |
| No weight decay | Biases, norm scales, gate biases |
| Global gradient norm clip | 1.0, after DDP gradient synchronization |
| Reference training horizon | 6 full fitting epochs; quality horizons 12/24/48 in Section 19 |
| Warmup | Linear for max(100, ceil(0.02 × planned updates)) successful updates |
| Reference schedule | Cosine from peak to 0.1 × peak over remaining planned updates |
| Checkpoints | Every 0.5 fitting epoch and at the terminal horizon |
| EMA/SWA | Disabled in the primary architecture comparison |

The LR/betas/decay choices are engineering priors, not prescriptions established by the cited optimizer paper. Do not multiply LR by eight solely because there are eight GPUs. Keep the global batch fixed when comparing world sizes; if the global batch changes, update count and optimization also change.

Six epochs is a reference full-training budget, not a claim that every model has converged or the desired final quality horizon. Quality-first 12/24/48-epoch arms answer whether a recipe is undertrained. Give each arm its own planned schedule from the start, or a declared stable-checkpoint/cooldown design. Appending six epochs after a completed six-epoch cosine is a continuation experiment, not an equivalent 12-epoch run.

A WSD alternative uses 2% warmup, 78% constant LR, and a 20% decay to 0.1 × peak [S7]. Preserve stable-phase checkpoints for separately declared cooldown branches. Checkpoint reuse saves physical work but does not make branches independent repetitions. Do not compare a stable-phase checkpoint against a fully cooled cosine endpoint and call that a scheduler effect.

Muon [S8] is an optimizer arm after a faithful AdamW control exists. A candidate uses Muon only on eligible hidden two-dimensional matrices, momentum 0.95 and five Newton–Schulz steps; input/output/norm/bias parameters remain in explicitly listed AdamW groups. A tentative Muon LR of 0.02 is meaningful only after pinning the implementation's shape/update-RMS convention. AdamW and Muon LR numbers are not directly comparable. The paper's language-model efficiency findings do not predict competition speed or score.

### 13. Student: distinguish cold training from warm continuation

| Setting | Cold GRU student | Warm student from exact strong parent |
|---|---|---|
| Initial budget | 8 full fitting epochs | 2 full fitting epochs |
| AdamW peak LR | 1e-3 | 1e-4 |
| Betas | (0.9, 0.999) | (0.9, 0.999) |
| Weight decay | 0.01 on declared matrices | 0.01 on declared matrices |
| Gradient clip | 1.0 | 1.0 |
| Warmup | max(100, 2% planned updates) | max(50, 2% planned updates), capped below total |
| Schedule | Cosine to 10% peak | Cosine to 10% peak |
| KD alpha | 0.9 initial; 0.5 separate arm | Same |
| State | Carry across contiguous chunks; declared TBPTT detach | Same |

These are proposed alternative recipes, not an instruction to overwrite an inherited Muon/propensity/distillation recipe while claiming exact replication. Keep an inherited-optimizer reference when changing the optimizer. More warm continuation is not automatically useful; the [existing quality-tradeoff record](QUALITY_BOTTLENECK_AND_STUDENT_TRADEOFFS_20261015_R1.md) documents a relevant degradation risk.

Carry state through full sequences; do not train shuffled isolated rows. Freeze the GRU chunk length from a measured existing recipe initially, and record it independently of the teacher's input-window length. Evaluate student quality versus training progress and teacher-imitation error separately.

### 14. B300 precision and kernel policy

The user asked for full precision and no quantization. This note's admitted-by-design reference is strict FP32: FP32 parameters/forward/backward/state, no autocast, and no silently enabled TF32 matrix products. Record the actual backend controls because library defaults can change.

BF16 mixed-precision training is distinct from INT8/INT4 submission quantization, but it is also distinct from strict FP32. If the user's intended meaning of full precision is simply no quantization, BF16 with FP32 master parameters, loss reductions, optimizer state, and AR state is an explicit optional recipe. It is not silently substituted here. Document operation-level autocast behavior and verify the actual fused backend.

Official FlashAttention documentation distinguishes dtype/hardware support [S12]. FlashAttention-4 targets Hopper/Blackwell [S13], but its headline throughput does not certify our exact B300, dropout, backward, shape, dtype, or causal-mask path. A fast BF16 attention result is not a strict-FP32 runtime estimate. For FP32, profile the actual selected SDPA/math/other supported backend and its peak workspace. Disabling a fused path may materially change time and memory.

Eight physical cards do not imply a single contiguous GPU allocation. Inventory device memory, topology, driver/CUDA/PyTorch/NCCL/kernel versions, power policy, host RAM, CPU limits, and physical scratch before actual admission. Do not launch local laptop fits or inherit the remote cluster's resources into local support tasks.

### 15. DDP recipe and exact loss normalization

Use one process per GPU, a complete model replica per rank, and NCCL for GPU gradient communication [S10, S16]. Start with ordinary DDP; these model sizes do not by themselves justify model sharding, tensor parallelism, or FSDP. Activation memory and throughput still require real profiles.

At 86M parameters, FP32 weights, gradients, and two Adam moment arrays alone account for approximately 16 bytes/parameter, or 1.38 decimal GB per replica. This excludes activations, attention workspace, communication buffers, temporary optimizer arrays and allocator overhead; it is not a complete admission bound. The attention/window implementation can dominate the memory footprint.

Candidate launch shape, illustrative only:

    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
      train_transformer_teacher.py --config <frozen-recipe.json>

No such new trainer or runnable command is implemented by this note. A multi-node deployment needs a real rendezvous/network configuration; the single-node command is not a claim about the user's topology.

Candidate global batch target: 65,536 NEW core rows/update. For 1024-row cores, world size 8, microbatch 8 sequence slots/rank, accumulation 1 gives 8 × 8 × 1024. Equivalent example: microbatch 2/rank with accumulation 4. These are target shapes to profile, not safe capacity certificates.

Use no_sync on the first accumulation microsteps, enclosing BOTH forward and backward [S10, S11]. Synchronize on the last microstep, clip once, step once, and advance the scheduler once per successful optimizer update.

For weighted supervision, let A[r,j,k] be the weighted squared-error SUM on rank r, microstep j, target k. Let W[k] be the globally summed effective weight across all ranks and microsteps in the current optimizer update. Because DDP averages rank gradients, use:

$$
L_{r,j}=\frac{R}{2}\sum_{k=0}^{1}\frac{A_{r,j,k}}{W_k},
$$

where R is world size. Accumulating these local backward contributions and then allowing DDP averaging produces the gradient of the globally normalized per-target loss, provided weights/masks are fixed data-derived quantities. Do not additionally divide by accumulation count: W already spans the whole update. Auxiliary/KD terms need their own correctly defined denominators.

This is our derived implementation contract. Uniform equal-count, unweighted batches can use simpler average-loss rules, but unequal scoring weights, padding, and partial chunks make naive means-of-local-means incorrect. All ranks must participate in collectives in the same order. If W[k] is zero, handle that target coherently on all ranks.

Precompute the per-update global denominators from the planned microbatch labels/masks before backward, or use a mathematically equivalent gradient-rescaling implementation. Do not retain every forward graph merely to discover denominators later. This formula assumes default DDP gradient averaging; a custom communication hook that changes reduction semantics needs its own derivation.

DDP/data requirements:

- Deterministically shard sequence slots and keep each active sequence on one rank.
- Shuffle sequences per epoch; never shuffle rows within a live sequence.
- Set the distributed sampler epoch where that sampler is used; preserve its actual seed/state [S14].
- Avoid sampler-padding duplication of effective training/evaluation examples. Explicit zero-weight padding can equalize the final distributed step.
- Keep dummy ranks graph-connected and participating in collective operations.
- Prefix/warmup/padding rows update only their valid state semantics and never inflate the supervised denominator.
- Do not split one recurrent sequence across ranks without an implemented, audited state handoff.
- Start with find_unused_parameters false only if every trainable branch really participates; the auxiliary-disabled model must not leave unhandled unused heads.
- Enable static_graph only after the implementation satisfies its contract, not because the config name sounds static.
- Pin rank-specific RNG, data order, accumulation position, per-slot sequence IDs, next row offsets, and carried states in resumable checkpoints.
- Record nondeterministic training kernels. Deployment determinism and repeatable reset remain separate requirements.

For validation, reduce the official sufficient statistics over all ranks and compute the final correlation once. Averaging rank-local correlations is not the official global metric. Avoid duplicated evaluation rows from a generic padded DistributedSampler.

### 16. Choosing how to use all eight GPUs

Compare one run with 8-way DDP, two runs with 4-way DDP, and eight independent single-GPU runs at matched GLOBAL batch and training horizon. This is a throughput/resource profile, not a scientific quality ranking. Avoid filling the queue with unprofiled jobs.

For relatively small models, parallel independent fits may finish more scientific endpoints than one eight-way job. Long-context FP32 work may favor more resources per run. The actual winner depends on host input bandwidth, attention backend, communication overlap, batch shape, and GPU utilization.

The allocation objective is completed, evaluated teacher/student artifacts per cluster-hour. A high GPU-utilization percentage or large tokens/s count that includes repeated context is not sufficient. Include target export and student training in the budget.

### 17. How long should it train?

Freeze exact fitting membership, then count eligible supervised core rows N. With global batch B_new approximately 65,536, the rough update count is E × N / B_new. Padding and the complete-sequence schedule give the exact count.

For an ARITHMETIC EXAMPLE with N = 200M fitting rows:

- About 3,052 updates per epoch.
- Six teacher epochs: about 18,311 updates and 1.2B new supervised row visits.
- About 367 warmup updates under the proposed 2% rule.
- Eight cold-student epochs: 1.6B new row visits.
- Two warm-student epochs: 400M new row visits.

Do not apply language-model compute-optimal token ratios to claim that correlated market rows are independent text tokens. Repeated epochs revisit the same supplied sequences.

Let q be MEASURED end-to-end new eligible core rows/second for one entire training job, including backward, optimizer, synchronization, input stalls, and the chosen AR scan. Then:

$$
T_{\mathrm{fit}}=EN/q,\qquad
T_{\mathrm{delivery}}=T_{\mathrm{setup}}+T_{\mathrm{fit}}
+T_{\mathrm{validation}}+T_{\mathrm{export}}+T_{\mathrm{student}}.
$$

| Hypothetical job throughput q | One 200M-row epoch | Six teacher epochs | GPU-hours for six epochs if using 8 GPUs |
|---:|---:|---:|---:|
| 20,000 new rows/s | 2.78 h | 16.67 h | 133.3 |
| 50,000 new rows/s | 1.11 h | 6.67 h | 53.3 |
| 100,000 new rows/s | 0.56 h | 3.33 h | 26.7 |

These are conditional arithmetic scenarios, not predictions that B300 achieves any of these rates. An eight-GPU rate cannot be reused for a single-GPU job. Prefix recomputation processes roughly four input rows per new core row away from sequence boundaries; record both rates.

Proposed future profile: compile/setup recorded separately, then 50 warmup and 100 measured optimizer updates on actual fitting data with the actual objective. Record FP32 and any separately authorized BF16 paths independently. Measure peak allocated/reserved GPU memory, charged host memory, new/processed rows per second, step percentiles, communication fraction, reader stalls, scan cost, export rate, and validation time. This is a proposed engineering profile, not one performed in this task.

For 30 paired repeats per recipe, multiply by actual run count. For example, 30 × 53.3 GPU-hours is about 1,600 GPU-hours for teacher fits alone at the middle hypothetical eight-GPU rate. Controls, students, LR/optimizer variants, and exports add real costs; they are not included for free.

## Evidence and interpretation

### 18. Evaluation and replication contract

The primary outcome is official student score versus its matched control, and then its fixed-mixture contribution to the complete endpoint. Teacher score, KD regression error, and auxiliary loss are explanatory diagnostics.

The user proposed about 30 runs per experiment. The project already has four source-qualified folds; do not invent a fifth pristine holdout. A concrete 30-run design is 8/8/7/7 fitting seeds across the four existing folds, using the same allocation and paired seeds for candidate/control. Report fold means with equal fold weighting plus the full per-run table. A balanced 32-run design is an alternative budget, not an automatic extra allocation.

Use full declared fitting membership and full training horizons for admitted scientific recipes. The brief hardware/correctness profile is not a small-data scientific rejection gate. Do not imply that every listed optional arm is already funded for 30 runs; freeze the selected arm set and total GPU-hours first.

Keep training-seed variation separate from paired component resampling of frozen predictions and from public transfer. Source components recur across folds and fitting sets overlap; 30 runs are not 30 independent datasets. Preserve exposure and the entire tried configuration set.

Freeze checkpoint selection before outcomes. Use the declared terminal horizon for each comparison: six epochs for the reference, or the predeclared cooled 12/24/48-epoch quality endpoint. Retain intermediate learning curves diagnostically. If a checkpoint-selection policy is used later, apply it symmetrically and account for that selection. Public leaderboard feedback does not tune weights or replace local selection.

Report by target, fold, seed, sequence position, observable activity/change regimes, gate values, prediction lag, and relevant source components. In particular, diagnose early warmup recovery and abrupt-change rows for AR oversmoothing.

There are no new empirical results in this note. No efficacy, public-score forecast, scaling law, distillation success, DDP speedup, or strict-FP32 B300 throughput has been demonstrated.

## Quality-first expansion under the latest user instruction

### 19. Larger teachers, longer training, and complete context

The latest instruction prioritizes best achievable final student quality. The reference configurations make comparisons interpretable; they are not capacity ceilings or terminal training limits.

| Role | Layers | Width | Heads × head dimension | SwiGLU width | Approximate parameters | Initial AdamW peak LR |
|---|---:|---:|---:|---:|---:|---:|
| Matched reference M | 12 | 768 | 12 × 64 | 2048 | 86M | 3e-4 |
| Primary L | 16 | 1024 | 16 × 64 | 2816 | 207M | 2e-4 |
| Primary XL | 24 | 1536 | 24 × 64 | 4096 | 682M | 1e-4 |
| Further scale XXL | 32 | 2048 | 32 × 64 | 5632 | 1.65B | 7.5e-5 |

The larger-model LRs are starting priors, not a discovered scaling law. Budget matched factor-of-two LR alternatives for serious contenders. Count parameters in the actual implementation. A larger model remains eligible when a smaller, differently optimized recipe is weaker.

Use WSD as the expandable quality route, with an initial warmup followed by a stable main branch and separately declared final cooldown branches. Target 12, 24 and 48 complete fitting epochs. For horizon E, branch near 0.8E epochs and cool down for the final 0.2E; retain the stable branch for an admitted extension. Count the initial warmup as part of that path. Branches share ancestry and are correlated.

For a shared quality branch, freeze warmup to max(100, ceil(0.02 × U24)) successful updates, where U24 is the exact planned update count for 24 epochs at that branch's fixed global batch. Reuse that SAME warmup for its 12/24/48-epoch descendants; independently changing warmup to 2% of each horizon would invalidate a shared training prefix. The cooled branch decays to 10% of peak LR over its final 20% horizon. Save optimizer, scheduler, RNG and data-order state at every branch point.

Compare a fixed-horizon cosine recipe at matched membership, global batch and horizon. Evaluate cooled endpoints, not merely plateau checkpoints. Six epochs is a useful reference; it is not a quality stopping rule.

A long extension may overfit repeated components, drift from a useful initializer or transfer worse into GRU. Continue an uncertain direction when learning dynamics or a distinct optimization/regularization intervention justifies it. Preserve a demonstrated degradation of an exact continuation instead of repeating the identical recipe solely because compute is available.

Student quality arms extend cold training to 8/16/32 epochs and warm continuation to 2/4/8 epochs, with their own declared schedules. The existing two-epoch warm and eight-epoch cold recipes remain controls. When longer warm continuation degrades, use the fixed-parent anchor or teacher/objective changes as explicit comparisons.

Expand context to 4096, 8192, 16,384 and the full 20,000-row sequence. A full-sequence dense causal-attention arm processes each sequence once, with all warmup rows and the complete AR scan; it removes overlapping-window anchor effects and permits gradients across the complete sequence. Activation checkpointing and, after profiling if needed, context/sequence parallelism are implementation options. No memory or speed certificate is assumed for this shape.

Full-sequence batches may change the global eligible-row count relative to the 1024-core reference. Freeze the real batch and optimizer-update counts. A simultaneous context/batch/schedule change is a new recipe rather than an isolated context effect. A persistent-memory Transformer is another architecture arm, not an assumed equivalent of dense full-context attention.

Compute-optimal and data-constrained language-model papers [S20] motivate measuring scale and data reuse jointly. They do not establish exponents or token ratios here. Repeated visits to the same market sequences are not new independent data; finite data alone also does not prove that a small teacher is optimal.

### 20. Heterogeneous teacher ensembles and compression stages

Train strong GRU and Transformer families with source-clean fold counterparts. Compare equal-weight ensembles across families, contexts and fitting seeds, then predeclared family-balanced mixtures. Multiple exports of one checkpoint are not independent diversity.

Carry expensive teachers through actual compression routes:

- Selected Transformer directly into the deployment-sized GRU.
- Fixed GRU/Transformer teacher ensemble into GRU.
- Larger research GRU students, such as H512×2, H768×2 and H512×4, beside H256×2.
- Large Transformer ensemble into a larger GRU assistant, then into a smaller GRU.
- An intermediate Transformer assistant as a distinct bridge if warranted.

Teacher-assistant distillation [S18] and ensemble compression [S19] motivate these paths. Their reported task results do not guarantee benefit here. Compare every multistage route against direct KD and record its complete cost.

An optional family-supervision student has one final two-value head plus fitting-only auxiliary heads for individual teacher families. Train the final head on the frozen family-balanced target and auxiliary heads on family predictions, with a normalized additional loss. Auxiliary heads may be discarded at serving if they never feed recurrent state. This is a distinct family-distillation objective rather than an invisible change to scalar KD.

Keep cold and warm student lineages. A teacher that transfers poorly into one narrow student can still be useful to a larger student, an assistant or a mixture. Conversely, a higher teacher score alone does not justify replacing a better final student.

Research teachers and assistants may exceed final submission limits in parameter count, ensemble size, context and runtime. Measure the eventual compression/blend path against actual competition limits. Quantization remains deferred.

### 21. Allocation and stopping follow quality

Favor properly trained L/XL teachers, substantial long-context work, heterogeneous ensembles and student transfer, while keeping an XXL scale branch and independent mechanisms active. Do not allocate all eight cards indefinitely to repeating the small reference.

The eight cards may serve one large distributed job or several complete fits, depending on measured memory, topology and communication. Completed fits/hour is a resource measurement; lower throughput alone does not reject a materially stronger full-context teacher. Independent runs can also buy quality through robust selection and teacher diversity.

Use the requested approximately 30 paired fold/seed runs for admitted scientific comparisons, including large-model routes when allocated. Optional Cartesian combinations are not automatically funded. A short hardware/correctness profile does not become a small-model scientific kill gate.

Promote using reliable local student/complete-endpoint comparisons. Public confirmation remains separate. An interval crossing zero is not an automatic stop; valuable uncertain mechanisms remain eligible for another bounded full-training comparison.

Reprofile every model/context/precision/DDP shape. In the earlier illustrative table, 12/24/48 epochs cost 2×/4×/8× the six-epoch fit only if throughput genuinely stays unchanged. An 86M rate cannot estimate the 1.65B full-context arm. Include teacher families, replication, exports, assistants, students and final all-data fits in the total allocation.

This note invents no new user GPU-hour ceiling. A future launch still needs verified cluster access and a concrete operational allocation for the chosen batch; this task remains documentation.


## Limits and validity

The AR term may learn nothing beyond the Transformer/GRU hidden state. It may improve fit but hurt the clipped global metric, lag fast events, or transfer poorly to a constrained student. A teacher can improve while its student degrades. A mixed teacher can help despite a weaker individual member; measure that complete path.

Different input encoders, raw/clipped targets, propensity weighting, warm parents, context semantics, and precision are material recipe changes. Unmatched comparisons cannot isolate the autoregressive mechanism.

Full-sequence replay with fixed weights is required for teacher targets and student evaluation. Associative scan algebra does not establish numerical equivalence at every dtype or a working autograd kernel. Future-observation auxiliary labels are legal fitting targets only when they cannot enter the forward input or preprocessing state at the predicted row.

Compute availability is user-reported. No cluster admission, paid spending, software installation, or job launch is performed by this document. Local CPU/scratch limits continue to apply to support work.

## Decision and next step

Before any future scientific fit, the implementation must establish causal future-perturbation invariance, full-20k sequential-versus-scan output and gradient agreement, correct initial/warmup/sequence reset, no double update of AR state on replayed prefixes, teacher export/reload repeatability, and single-process-versus-DDP weighted-update agreement with unequal masks and padding. Test numerical equivalence with dropout disabled and a fixed reference dtype. These are specified future engineering checks, not checks performed for this documentation task.

Prepare matched T0/T1 teacher pairs at the reference and primary L/XL scales, with T2/T3 as explicit auxiliary/interaction siblings. AdamW is the reference optimizer; quality-first 12/24/48-epoch WSD routes, larger/full-context teachers, Muon comparisons and heterogeneous ensembles remain active proposals under Section 19. The six-epoch warmup/cosine model is a control rather than the final capacity/horizon preference. Keep strict FP32 as the reference until numerical policy is explicitly resolved; no quantization is proposed.

Before a future fit: freeze component manifests, architecture/loss/precision configs, scan/window/export semantics, exact optimizer groups, checkpoint policy, seed/fold matrix, GPU-hour budget, and the actual cluster profile. Then own each fit through checkpoints, complete causal predictions, official evaluation, and matched student training.

The scientific deliverable is the comparison of trained students, including failed transfer. A trained teacher alone is an intermediate artifact.

## Source records

Primary sources were opened or searched on 2026-10-17. External evidence concerns other tasks; proposed settings, equations specific to this head, and runtime arithmetic above are our design choices.

| ID | Primary source | What it supports; local limitation |
|---|---|---|
| S1 | [Su et al., RoFormer / RoPE](https://arxiv.org/abs/2104.09864) | Positional mechanism; not a market-model quality result. |
| S2 | [Dai et al., Transformer-XL](https://arxiv.org/abs/1901.02860) | Segment recurrence precedent; not equivalence to our overlapping windows. |
| S3 | [Salinas et al., DeepAR](https://arxiv.org/abs/1704.04110) | Autoregressive forecasting precedent; its observed-series setup does not license feeding our hidden future-price labels. |
| S4 | [Bengio et al., Scheduled Sampling](https://arxiv.org/abs/1506.03099) | Feedback-distribution mismatch; not a remedy that authorizes unavailable inputs. |
| S5 | [Loshchilov and Hutter, Decoupled Weight Decay](https://arxiv.org/abs/1711.05101) | AdamW basis; our LR/decay settings remain proposals. |
| S6 | [Cho and Hariharan, On the Efficacy of Knowledge Distillation](https://arxiv.org/abs/1910.01348) | Teacher/student capacity and optimization considerations in another task. |
| S7 | [Understanding Warmup-Stable-Decay Learning Rates](https://arxiv.org/abs/2410.05192) | WSD analysis; does not establish the best schedule here. |
| S8 | [Liu et al., Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) | Matrix-optimizer/update-scaling precedent; no local efficiency forecast. |
| S9 | [Yang et al., Gated Linear Attention](https://arxiv.org/abs/2312.06635) | Hardware-aware gated recurrence precedent; not validation of our two-scalar output AR. |
| S10 | [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) | Replica/gradient communication and no_sync contract; our weighted-loss formula is derived separately. |
| S10b | [PyTorch DDP design notes](https://docs.pytorch.org/docs/stable/notes/ddp) | Gradient-reduction and graph-participation implementation context; exact installed-version configuration still needs review. |
| S11 | [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | Accumulation synchronization guidance; cluster speed remains unmeasured. |
| S12 | [Official FlashAttention repository](https://github.com/Dao-AILab/flash-attention) | Current kernel/dtype/hardware interfaces; installation and local compatibility not verified. |
| S13 | [FlashAttention-4 paper](https://arxiv.org/abs/2603.05451) | Blackwell-oriented kernel design; no claimed runtime for this recipe. |
| S14 | [PyTorch data / DistributedSampler](https://docs.pytorch.org/docs/stable/data.html) | Sampler interface reference; this session's web rendering redirected, so exact installed-version behavior needs inspection before implementation. |
| S15 | [PatchTST](https://arxiv.org/abs/2211.14730), [iTransformer](https://arxiv.org/abs/2310.06625) | Alternative patch/channel/variate tokenizations from the preceding architecture discussion; not selected as the primary event-wise causal backbone. |
| S16 | [Li et al., PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704) | Primary DDP systems paper: model replicas, gradient communication, and computation/communication overlap; its historical scaling results are not B300 forecasts. |
| S17 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | Primary attention, gated feed-forward, and RMSNorm mechanisms; their reported task gains do not establish a preferred market-model configuration. |
| S18 | [Mirzadeh et al., Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393) | Intermediate-capacity distillation precedent from vision; no demonstrated benefit for our GRU compression stages. |
| S19 | [Hinton et al., Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | Ensemble-to-student compression precedent; continuous regression objectives here are separately specified. |
| S20 | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) | Reasons to measure scale and repeated-data duration; their empirical laws are not imported as market-data exponents or capacity limits. |

The explicitly invoked arxiv-search skill was read at /home/vlad/src/agent-system/arxiv-search/SKILL.md. Its Bash search implementation ran successfully with the broad query “autoregressive transformer time series distillation” (four results), which returned mostly tangential material, and then “DeepAR” (two results), which returned the original DeepAR paper and an unrelated application. Only the verified original DeepAR source was used; search ranking was not treated as evidence of relevance. Other sources were checked directly through primary arXiv/PyTorch/official repository pages.

Navigation and ledger links: [ideas](README.md), [sources](../sources/README.md), [hypotheses](../../hypotheses.jsonl), [experiments](../../experiments.jsonl), [research methods](../../RESEARCH_METHODS.md). No model weights, source datasets, or private credentials are embedded.


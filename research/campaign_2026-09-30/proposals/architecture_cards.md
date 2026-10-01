# Independent architecture commitment — 2026-09-30

Proposer: `/root/architecture_proposals_0930`. These cards were written without viewing competing proposals. Status: proposed, not implemented or tested. Only this Markdown and its companion JSON were added; no model/trainer edits, GPU work, or heavy CPU work were performed. Commitment means these artifacts are persisted before synthesis; it does not mean a Git commit over the shared dirty worktree.

## Contract and evidence

Root Challenge 2 only. Target remains expected public Global WP >=0.6660, with public incumbent 0.6622. This is verification tier C: clean local chain-component validation is useful, but cannot establish transfer to the public period. The September 30 handoff reports that the official metric ambiguity was resolved: pooled clipped weighted Pearson is primary; block WP remains a diagnostic. Use the controller's current contract if it supersedes the older skill text.

No broad training until the clean since/since controls finish. Total campaign budget is 8 GPUh, approximately 6.57 GPUh remaining at assignment; these cards request no immediate spend. One GPU job, aggregate <=12 CPU workers, <=36 GiB RAM and <=5 GB observed VRAM. Reserve confirmation and final fit budget before screening. An uploaded ZIP must have fixed members/weights, <=20 MB, <=52 minute projected one-vCPU runtime, sequence resets, deterministic output, and scorer parity.

Inspected local evidence:

- `src/wnn/nn.py`: normalized input -> GELU projection -> two h128 GRUs -> nonlinear head over recurrent and immediate representations. Feature-token attention is **within a row**, not temporal attention; its native path is unsupported.
- `scripts/train_rnn.py:376-429`: `forward_chunk` accepts carried state, `detach_state` handles a tensor or a flat tuple, and `predict` resets at sequence-batch starts. New designs must return a flat tensor tuple; nested states require an explicit compatible change. The trainer resets `h` for each independent batch and carries/detaches it across TBPTT chunks.
- `native/src/rnn_fast.rs`: allocation-free aligned state and weights; existing kernels support GRU, not the proposed states. A native port is required for every serious survivor.
- `native/src/python.rs`: resetting a submission resets features, online normalization, and each model. All new memory, counters, router state and scratch must be included.
- `research/experiments.jsonl:E-056`: event-clock GRU 0.6305 vs row GRU 0.6310; adding another event recurrent compression has already failed.
- `E-168`: convolution/MLP encoders had residual correlation >=0.993; score alone is weak evidence of ensemble diversity.
- `E-267`: EMA6 public 0.6612/0.6616, below incumbent. `E-269`: merged GRU addition negative at all tested weights against the public pair, though that comparison is asymmetric OOF. `E-270`: clean bilinear minus pair -0.005687, and minus additive -0.000500; a richer frozen nonlinear head is not a promising default.
- `research/campaign_2026-09-29/logs/c20260929_since_s3_f0.log`: 0.26M parameters, batch32/chunk2000; four training epochs took 240.4+263.5+287.0+294.1=1085 seconds (18.1 min), excluding loading and other jobs. Existing parent job records show observed VRAM about 4.56 GB, so an additive full-size backbone risks crossing the ceiling.
- Existing `pair_states/*fit*` data select scored fit rows; they are **not** a dense causal timeline. They can support the sparse-readout microprobe below, but cannot train temporal memory by pretending adjacent cached rows are adjacent time steps.

All timings below are engineering estimates, **not measurements**. FLOP counts cannot predict Python or AVX2 latency accurately. Screens use one newly frozen configuration each, not a weight sweep.

## A. Normalized decayed content memory

**Family:** content-addressed recurrent linear attention. **Lens:** re-encode history as a bounded associative memory and change time by event-conditioned writes.

Replace the second GRU with a four-head memory after an h96 first GRU. Each head has key dimension 8 and value dimension 16. Its state is a matrix `S_j[8,16]` plus normalizer `Z_j[8]`. Two heads write on every row with half-lives 64 and 512 rows; two write only on i0 price events with half-lives 2 and 16 events. For positive feature map `phi=ELU+1`:

```text
w_t = 1 (row heads), or causal i0-price-event flag (event heads)
a_t = 2^(-w_t/half_life)
S_t = a_t*S_(t-1) + w_t*phi(k_t) outer v_t
Z_t = a_t*Z_(t-1) + w_t*phi(k_t)
r_t = (phi(q_t)^T S_t) / max(phi(q_t)^T Z_t, 1e-6)
p_t = head(concat(h_first_t, z_t, gate_t*r_t))
```

Use a bounded gate and FP32 `S,Z`; if no event has occurred, event retrieval is zero. Current-row insertion is causal and is also used in every control. This is a normalized linear-attention variant inspired by the [Linear Transformers paper](https://proceedings.mlr.press/v119/katharopoulos20a.html) and decay-based recurrent/chunkwise [RetNet](https://arxiv.org/html/2307.08621v4). It is not a claim to reproduce RetNet, whose signed retention and normalizations differ.

**Mechanism and observable.** A GRU stores mixed history in one nonlinear vector. Content memory lets the current book/trade context select relevant prior state without repeatedly overwriting it. Predicted gains should concentrate on repeated book configurations and quiet rows following informative events. The content arm must improve exact WP over a decay-matched memory with no content addressing; otherwise the apparent gain is merely smoothing or another head.

**Minimal matched falsifier.** One clean fold, same train components, order, seeds, loss, EMA recipe, auxiliary targets, fixed final checkpoint. Compare (i) learned query/key content memory with (ii) position-only memory with identical head count, state dimensions, half-lives, value maps, output maps and update count. For (ii), replace q/k content input with learned constant vectors plus causal age/event flags; keep identical q/k projection tensors and execute their maps on position codes so parameters and work stay matched. Include the completed clean pair as the anchor, not as the mechanism-specific control. Architecture/control use the same parameter budget within 5% and identical training rows/optimizer updates. Fixed four-epoch checkpoint; equal GPU cap 0.35h per arm. If either runs out of cap, stop and mark screen inconclusive rather than compare unequal updates.

**Estimate.** h96 GRU1 + memory/head approximately 0.10–0.16M parameters; recurrent memory 544 FP32 floats =2176 bytes/sequence plus h96. About 80–130k MAC/row including projection/readout; native single member plausibly 0.5–0.9 incumbent single-member cost. Paired GPU screen reserve 0.7h. Microblocks of 64 rows can use causal batched attention within each microblock and carried linear state between microblocks; the 2000-row TBPTT boundary remains identical to the control. Never allocate full 20,000² attention or a state tensor for every sequence row. Estimated 2–4.8 GB VRAM with EMA and batch32; profile before authorizing a fit. A Python timestep loop can be slower by orders of magnitude and is a hard implementation risk. Native port needs ELU, small outer products, denominator protection and carried FP32 memory; moderate complexity, no external package required.

**Reset/parity.** Reset h, S, Z and event/write counters only on a sequence boundary; carry at every chunk boundary. No label, future quote, scored-mask or sequence-identity writes. Check full sequence vs chunk sizes 1/17/64/2000, quiet sequence with zero events, first-row event, long event-free gaps, and repeated sequence reset. Zero events must not generate NaNs or query-dependent arbitrary memory.

**Complementarity.** Content selection differs from seed diversity and may add conditional information. Predeclare blend `0.8*clean_pair + 0.2*candidate`; evaluate without fitting the blend weight. Report candidate alone, blend, weighted residual covariance, per-class effects and event-gap effects. Low residual correlation by itself is insufficient: an inaccurate member can be diverse.

**Reject.** Content-minus-position <=0; exact blend delta <+0.0005; a gain solely on event rows or unscored-mask stress; class delta <-0.0005; instability in a denominator/gap; native mismatch; or projected package time >52 min. Failure here rules out this small content-memory recipe, not all attention.

## B. Structured multi-timescale state with shared nonlinear readout

**Family:** diagonal/block-diagonal structured state model. **Lens:** factor history into explicit dynamical modes instead of asking the GRU to learn retention gates.

Replace both GRUs with two residual blocks at width 96. Each block has 64 real scalar decay modes and 32 real 2x2 rotation-decay modes (128 state floats total). State dynamics are linear and stable:

```text
u_t = GELU(W_in*x_normalized_t)
s_t = A*s_(t-1) + B*u_t
o_t = C*s_t + D*u_t
u_next_t = u_t + gate(u_t)*GELU(o_t)
p_t = head(concat(u_final_t, original_projection_t))
```

`A` is scalar `rho` or `rho*R(theta)` per mode with `0<rho<1`. Initialize half-lives on a fixed log grid 2,8,32,128,512,2048 rows and a small deterministic phase grid. Constrain learned half-lives to [1,8192]. Normalize input injection by a mode-dependent bounded factor such as `sqrt(1-rho²)`; do not let near-unit modes explode. This is an S4D/LRU-inspired engineering design, not full S4 or HiPPO. [S4D](https://arxiv.org/abs/2206.11893) emphasizes that initialization matters; [LRU](https://arxiv.org/abs/2303.06349) studies linear/diagonal recurrence, parameterization and normalization; [S5](https://arxiv.org/abs/2208.04933) provides a parallel-scan implementation rationale. Their benchmark successes do not establish usefulness for this task.

**Mechanism and observable.** Explicit slow modes preserve weak activity/order-flow evidence through quiet rows while fast modes track fresh changes. Rotation modes can distinguish a decaying change from a reversal. Unlike E-056, this changes the state transition class and does not add an event GRU. Predicted improvement must exist on all required rows, with a positive long-gap effect; residual persistence alone is not justification because E-238 shows overlapping future outcomes create much of that persistence.

**Minimal matched falsifier / best first executable screen.** Two identical models with the same parameter tensors, blocks, real state representation, initial projection, seed and updates: B-multi uses the log-grid initial half-lives; B-single initializes all modes to half-life32. Both can learn radii in the same [1,8192] range, so this tests the multi-timescale initialization/optimization claim. Freeze phases for this first screen to isolate decay-scale representation. Fixed four epochs, matching the completed clean controls' data, loss/auxiliary/EMA settings and epoch policy. One fold only, 0.30 GPUh per arm maximum, no broad search. Initial 32-sequence batch memory/profile must fit <=5GB observed; lower a batch only by declaring identical accumulation for both arms. Require at least 20% of B-multi modes to retain learned half-life>128 and a positive long-gap slice, otherwise the stated mechanism failed even if score rose.

A still cheaper CPU falsifier can fit a train-only ridge correction using **fixed causal state modes**, compared with the identical readout over half-life32 modes. This tests whether the chosen history statistics contain additional information; it does not validate the end-to-end model. Use dense raw sequence timelines and streaming sufficient statistics for weighted ridge, fixed regularization and one fit/eval pass. It needs no GPU and O(state+feature²) working memory; estimate 20–60 min at one thread for a bounded 256-fit-sequence/full-clean-fold probe after data-loading measurement. Do not use scored-only caches as a time series or interpret a null fixed-projection probe as disproof of learned modes.

**Estimate.** Two width96/state128 blocks approximately 0.12–0.20M parameters, ~90–140k MAC/row, 256 FP32 state floats=1024 bytes/sequence. Batch32/chunk2000 FP32 stored state alone is about66 MB before autograd copies; expect 2–4.8GB total observed VRAM with inputs/EMA, not measured. Paired optimized scan/convolution screen reserve 0.6h; naive Python stepping is unacceptable. Because transitions are input-independent within a block, use a chunk convolution plus the analytically propagated incoming state, or an associative scan. Validate equality to the simple scalar recurrence before spending on fit. Native port is the easiest of these three: real 2x2 updates, matrix-vector projections, bounded scalar activation; no attention cache, complex runtime or third-party dependency.

**Reset/parity.** Flat tuple `(state_block1,state_block2)`; detach each across TBPTT, never zero at chunks. All states zero at each independent sequence. Prefix causality, partition invariance, long constant inputs, impulse decay and stationary state magnitude are decisive checks. Learnable rho/phase serialize explicitly; evaluate in FP32 to avoid silent loss of slow modes under FP16.

**Complementarity.** Its linear memory bias is farther from GRU seeds than another width/depth choice. Predeclare the same fixed 80/20 blend and compare both B-multi and B-single blends. A useful low-cost third member could fit the runtime budget, but this requires a one-vCPU measurement; no extrapolation from parameter count is a deployment claim.

**Reject.** Multi fails to beat single on exact WP; blend <+0.0005; no sustained long modes or all gain confined to source-like chains; regression <-0.0005 for any market class; state explosion, scan/stream mismatch, or runtime >52 min. If both arms beat the GRU equally, retain the general state-model family as an observation but reject the specific multi-timescale claim.

## C. Causal sparse mixture of state-conditioned operators

**Family:** sparse mixture / conditional computation. **Lens:** factor conflicting conditional regimes into operators without adaptive ensemble membership.

One h96 shared GRU layer feeds four small expert states (32 real decay modes each). Every expert state receives and updates from current shared representation on **every row**, so an unselected expert is never stale. A causal router from `(h_t, current event flags, log gap, activity prefix)` picks exactly one expert readout (top1), and a universal head remains:

```text
for every expert e: s_e,t = rho_e*s_e,t-1 + B_e*h_t
r_t = argmax(router(h_t, causal_context_t)), stable lower-index tie rule
p_t = universal(h_t,z_t) + selected_router_weight_t * expert_r_t(s_r_t,h_t)
```

All four expert identities, weights and update rules are fixed for the whole submission. Only the causal per-row gate depends on input; no wall clock, host performance, sequence ID or mask. Selected expert readout is a low-rank linear map, not four full GRUs. Initialize universal path from a compatible one-layer control or train all arms from scratch; do not load incompatible two-layer weights silently. Use train-only occupancy balancing, top1 training as well as inference, no dropped rows and no inference capacity limiter. The [Switch Transformers paper](https://jmlr.org/papers/v23/21-0998.html) supports sparse parameter selection and highlights stability concerns; it supplies no evidence that small financial experts help here.

**Mechanism and observable.** Distinct states/operators can represent incompatible direction responses in high-activity, stale, reversal and inter-book lead contexts. Routing conditions the transition/readout interpretation rather than reweighting finished incumbent scalar outputs. Expected evidence is specialized held-out behavior that transfers across chain components. Explicit clocks do not guarantee meaningful regimes.

**Minimal matched falsifier.** Train learned causal top1 router vs deterministic input-hash router using the same h96 backbone, four expert state tensors, top1 readout cost, initialization, optimizer updates and balancing target. Hash only quantized current/prefix features, never row IDs, chain or sequence identity. Fix salt before data inspection. Additional universal-only control allocates saved router/expert work to a low-rank shared operator, matched active FLOPs; it may have fewer total stored parameters, so report the difference rather than pretend sparse parameter count is free. Candidate must beat both routing control and universal-only behavior. Four fixed epochs; <=0.35GPUh per learned/hash arm. Fixed 80/20 blend with the clean pair.

**Lowest-cost microprobe, before temporal implementation.** Existing clean paired-state caches can fit four **linear ridge experts** on a single causal per-member hidden vector, routed by four train-only frozen gates. Compare same experts under hash gates and one universal ridge residual; regularization and row weights frozen. Read cached rows independently, preserving causal hidden-state provenance. This costs no new backbone evaluation and uses <=1 CPU thread, approximately 2–10 min and <=2GB working memory if streamed. It is an information/specialization falsifier, not a successful architecture screen: E-270 makes overfit a serious risk, and adding four linear heads alone does not implement the proposed independent expert state. Freeze gates without evaluation labels/masks; score on complete eval rows with the existing grader. Stop the small readout lineage if it repeats E-270 losses, while not claiming to disprove end-to-end experts.

**Estimate.** About0.11–0.18M parameters, ~90–150k MAC/row including all four cheap state updates and one readout; shared h96 plus 128 expert state floats=896 bytes/sequence. Proposed paired GPU reserve0.7h. Grouping rows by selected expert introduces training overhead; dense evaluation of all heads may fit GPU training but invalidates claimed active FLOP equality unless used by both arms. Estimated2–4.8GB VRAM with batch32/EMA, not measured. Native port moderate: one supported GRU plus new linear states/router and selected low-rank readout, allocation-free. A top1 route discontinuity creates a harder parity problem than A/B.

**Reset/parity.** Reset shared h, all expert states and causal prefix activity. Never reset only selected expert. Carry all states across chunks. Audit router decisions across Python streaming/native AVX2/v2/v3; stable tie-breaking alone does not prevent floating-point route changes near a tie. Record logit margins; any observed route divergence requires exact router arithmetic or a deterministic feature-threshold router before promotion. No adaptive dropping of experts under load.

**Complementarity.** Could produce context-dependent residuals rather than global seed noise. Predeclare blend; inspect expert occupancy and residual gains in each chain component, including new-source contexts. Correlation plus score determines usefulness; occupancy is a training diagnostic, not a route-selection target tuned on validation.

**Reject.** Learned router <=hash; universal-only matches/exceeds it; fewer than two experts receive >=10% scored validation mass; routes mainly predict source identity; gains vanish after chain exclusion; any class <-0.0005; blend <+0.0005; routes differ between allowed native builds; or fixed full model exceeds runtime.

## Execution recommendation

Finish exact controls, then run B-multi versus B-single as the first full architecture screen. It has the clearest matched falsifier, easiest native deployment and bounded state. Spend at most0.6 GPUh on the pair, with a profile/abort checkpoint before fitting. A is the next choice if B loses or cannot preserve long modes; C is a higher-overfit arm and should face the cached linear microprobe first.

Do not spend all approximately6.57 remaining GPUh on these cards. Proposed maximum first-fold architecture screen spend is2.0GPUh across the three pairs; only survivors earn another fold/seed. Preserve at least1.0GPUh for final fit/package reserve plus independent confirmation. A positive local +0.0005 screen is a continuation gate and is far below the roughly+0.0038 public gain needed for the campaign target. No paper, low residual correlation, or proxy delta clears the expectation gate.

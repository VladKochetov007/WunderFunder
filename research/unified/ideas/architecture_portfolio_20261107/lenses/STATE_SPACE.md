# State-space and compressed-memory architecture scout

Scope: independent design note for the root Challenge 2 two-output causal per-timestep regressor. This note was written from primary papers and official code, not from task results. Root owns integration and any later experiment admission.

## Corrected task shape and validation

A 7 November read-only source correction from root supersedes the stale dimensions in the portfolio brief: each complete sequence has 20,000 timesteps and 112 raw numeric features; 2,000 is the TBPTT chunk length, not the sequence length. There are 99 derived event features, so the full representation has 211 channels total when those optional features are included. The official train/valid boundary is 10,325 / 1,873 whole sequences. train.parquet has no is_scored column; the frozen train loss mask is all timesteps after index 99 unless a separate train-only policy was prespecified before fitting. The official is_scored mask is for validation/evaluation only (3,528,273 valid scored rows under the qualified manifest). Keep all context timesteps, including the initial unscored prefix, in the causal scan. Do not infer dates or sort seq_ix.

The decision policy is the accepted [ADR0044 chronological protocol](../../../../../specs/adr/0044-chronological-primary-validation.md): fit on official train, select on complete later official valid, and keep valid out of normalizer, teacher, initializer, calibration, or blend-weight ancestry. Valid is development-exposed. Any result below is a proposal; no task-specific fit or hardware measurement is claimed here.

## Ranked candidates

### 1. Mamba-2 / SSD residual stack — first screen

**Concrete block.** Train-only standardize each of the 112 raw numeric channels with fixed per-channel statistics. Project to d=96; if the 99 derived event features pass a feature-availability audit, project them through a separate train-only-normalized stem and add a learned gated event residual. Use six pre-norm residual Mamba-2 blocks with expand=2, headdim=32, d_state=16 (32 only as a later fixed alternative), and d_conv=4, then a featurewise norm and a linear 96 -> 2 head. No bidirectional pass, global pooling, time-axis normalization, or additional attention in this first screen. Approximate block parameter count follows the official implementation's ~3 * expand * d_model^2: at d=96, six blocks are about 0.33M; add roughly 0.011M for the numeric stem, 0.010M if the 99-event stem is present, plus small norms/head and state/convolution parameters. Expect approximately 0.35M without events and 0.36M with them; verify exact counts from the frozen config.

**Why it ranks first.** Dao and Gu derive SSD as a link between selective SSMs and structured semiseparable matrices. Their paper reports a 2–8x faster core than Mamba-1 and competitive language-model results; that is a paper claim about their tested language workloads, not a speed or quality claim for this dataset. The [official Mamba repository](https://github.com/state-spaces/mamba) includes Mamba2, a minimal SSD listing, and a one-step inference path that keeps convolution and SSM states. The compressed state and existing step interface make it the most direct fit to a streaming CPU student among the candidates here. [Paper](https://arxiv.org/abs/2405.21060).

**Project hypothesis.** Input-conditioned decay/write dynamics may keep useful signal across 20k steps while rapidly adapting after local regime changes. Its per-step state is nheads x headdim x d_state plus the d_conv history; for this config, about 6*32*16 + 192*4 = 3,840 floats/layer, roughly 90 KB across six layers in FP32. This is a state-size calculation, not a throughput estimate.

**Teacher/student path.** First test the d=96, L=6 student directly against the frozen train-only objective and preprocessing. If that student shows a repeatable validation advantage but appears capacity-limited, a train-only d=192 or 256, L=8 Mamba-2 teacher is a reasonable next arm; distill only on official train sequences, then serve the small student. Do not spend the teacher cost in the first discriminating screen.

### 2. Gated DeltaNet — targeted-write comparison

**Concrete block.** Use two pre-norm gated-delta blocks at d=128, four heads with key/value head dimension 32, expand_v=1, short causal convolution size 4, output gate enabled, and a SwiGLU feed-forward ratio 2. Use chunk mode for training and the recurrent update for single-step serving. Add a separate optional 99-feature event stem to the 112-raw-feature stem and a 128 -> 2 head. A rough estimate is 2 * (about 5 d^2 for mixer projections + 6 d^2 for a ratio-2 SwiGLU), about 0.36M before stems/norms and roughly 0.39–0.40M total with both stems. Verify against the exact implementation/config; do not assume the paper's larger language-model defaults.

**Evidence and fit.** The [Gated Delta Networks paper](https://arxiv.org/abs/2412.06464) proposes a decay gate for erasure and a delta update for targeted memory correction, with chunkwise parallel training. It reports advantages over Mamba-2 on several language/retrieval benchmarks, but its own results are not evidence for this tabular time-series task. The official [Flash Linear Attention layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py) has chunk training, a fused recurrent mode, optional short-convolution state, cached recurrent state, and final-state handling. The recurrent state for the proposed width is four 32 x 32 matrices (4,096 floats) per layer plus short-convolution histories.

**Project hypothesis.** A delta update could help preserve and correct a small number of persistent latent quantities, while the forget gate suppresses stale state. It is second because this matrix-state update and its CPU implementation are more involved than Mamba-2's diagonal SSD recurrence, and no matching CPU cost has been measured.

**Teacher/student path.** Start with the compact two-block model. Only if it beats the frozen validation anchors should a train-only teacher widen to d=256, 6–8 blocks, retaining the same head and input ancestry; distill to the compact model using train-only per-timestep predictions. Keep teacher state resets and chunk boundaries identical to the student.

### 3. xLSTM mLSTM stack — matrix-memory fallback

**Concrete block.** A compact test is four pre-norm mLSTM blocks at embedding_dim=96, four heads of width 24, convolution width 4, with no sLSTM block in the first portable CPU attempt; finish with 96 -> 2. Use the official chunkwise training path and native/recurrent step implementation for CPU serving. A projection-count estimate is P ~= L * (5–9) * d^2 + stem/head; for L=4,d=96 this is about 0.18–0.33M before the stem/head. Allow approximately 0.2–0.4M total, then record the exact count from the actual module. This range is intentionally broad because block config and gate/FFN choices change the count.

**Evidence and fit.** The [xLSTM paper](https://arxiv.org/abs/2405.04517) introduces exponential gates, sLSTM scalar memory/mixing, and an mLSTM matrix memory with a parallelizable covariance update. Its experiments are language-model and synthetic state-tracking/associative-recall tasks. The [official NX-AI repository](https://github.com/NX-AI/xlstm) exposes block-stack configs; the official [mLSTM kernel library](https://github.com/NX-AI/mlstm_kernels) documents chunkwise, parallel, and recurrent-step implementations. The repository recommends native implementations on other hardware, but does not establish CPU latency for this competition package.

**Project hypothesis.** Matrix memory may retain associations between event-like feature patterns and targets that a diagonal SSM forgets. This is the lowest-ranked proposal because each layer carries a matrix state, numerical stabilization and exact CPU parity need attention, and the paper's reported workloads differ substantially from this task.

**Teacher/student path.** First test a small mLSTM student directly. If useful, make a wider d=192, 6-block train-only mLSTM teacher, then distill into the four-block student. Do not introduce sLSTM mixing until an mLSTM-only comparison shows that compressed matrix memory helps; otherwise it would confound the first test.

## Causal and sequence-state traps

- Treat each full 20,000-step sequence as one independent stream. Zero every SSM, matrix-memory, and convolution cache at sequence start. When training in ten ordered 2,000-step TBPTT chunks, carry the state from one chunk to the next; a chunk boundary is not a reset. If state is detached to cap activation memory, document the truncated gradient horizon. Evaluation and deployment must scan the full 20,000 steps with the same state recurrence.
- Preserve token order inside every sequence. Batching/shuffling whole sequences is okay; random row splits, shuffled timestep order, or splits by seq_ix are not.
- Keep local convolutions strictly left-padded/causal. Do not use symmetric same padding, centered rolling windows, reverse scans, or bidirectional training.
- Normalize features only across channels at each timestep, or use train-fitted fixed channel statistics. Never compute per-sequence/time-axis means, variances, ranks, or clipping thresholds from future positions. Fit every scale and event encoding on train only.
- Process the first 99 unscored rows as context. Since train.parquet has no is_scored column, use the frozen train target/loss mask over all timesteps after index 99, unless a separate train-only policy was prespecified before fitting. The official is_scored mask is for validation/evaluation only. Do not substitute the validation mask into training or reset at either boundary.
- For the optional 99 derived event features (in addition to the 112 raw features), confirm each feature is available at that timestep before using it. Avoid event aggregations or joins that incorporate future rows. Keep missingness handling and any event normalization train-only.
- Use FP32 recurrent accumulators for initial parity. Mamba's official repository warns that selective-state dynamics can be precision-sensitive. FLA/xLSTM chunk kernels must return the same carried state as their recurrent step path; compare outputs across chunked/full scans before relying on chunk training.

## Cheapest discriminating screen

1. Freeze one Mamba-2 recipe and two seeds: six blocks, d=96, expand 2, head width 32, state size 16, causal conv 4; no teacher, no valid-derived preprocessing, same approved objective and optimizer schedule as the selected train-only student control.
2. Train the two candidate seeds on the 10,325 official train sequences as complete chronological streams, chunked only for TBPTT at 2,000 with carried state and the frozen train mask (all timesteps after index 99 unless a separate train policy was prespecified). Evaluate each on all 1,873 official validation sequences and the exact official is_scored mask. Regrade the frozen current TL3 and fixed TL3/LSTM validation predictions as the no-fit anchors; do not learn blend weights on valid.
3. If root cannot verify matching frozen control predictions/recipes, use two fresh matched GRU control seeds as well (four fits total). Keep each model's predictions and global/target metrics separate. The official pooled clipped weighted Pearson remains the primary endpoint; no proxy replaces it.
4. A useful result is repeatable direction across the two candidate seeds and a better full-valid endpoint than the fixed anchors, with a viable exact-reset CPU step path. Mixed seed direction or a gain that depends on valid-tuned blending falsifies promotion. If the candidate is competitive but not clearly better, stop before teacher fitting and ask root to decide whether a second family is worth the compute.

No batch size, runtime, memory ceiling, throughput, or expected score gain is proposed from paper results. The first actual-device profile and resource admission belong to root under a later scoped execution packet. If admitted, keep the first comparison small: two B300s for the two candidate seeds; use existing frozen predictions for controls. A fresh matched control cohort or teacher needs a separate allocation decision. No new compute is authorized by this note.

## Evidence ledger

| Item | Verified in paper/code | Still a project hypothesis or unknown |
|---|---|---|
| Mamba-2 | SSD relation and the paper's language-model comparisons; official module exposes Mamba2, SSD reference, causal convolution cache, and recurrent SSM cache. | Whether selective decay/write improves the official two-target metric; CPU full-set latency; stable best width/state. |
| Gated DeltaNet | Gated delta rule, chunkwise training algorithm, and reported language/retrieval results; official FLA code exposes chunk and recurrent modes with cached state. | Whether exact-write memory helps these numeric sequences; whether a native CPU step path is practical at 20k x 1,873 validation points. |
| xLSTM mLSTM | Matrix-memory recurrence, parallel/chunkwise kernels, and recurrent-step kernel exist in official sources. | Whether its richer matrix state helps this task; CPU latency and numerical parity in the final package. |
| Data scale | Root's 7 Nov read-only correction: 20k timesteps, 112 raw features plus 99 derived event features (211 total), 2k TBPTT chunks; train.parquet lacks is_scored; official train mask is all after 99 unless a separate train policy was prespecified; is_scored is for valid/evaluation. | What memory horizon the labels actually reward; whether all derived event features are causal; unseen public/private transfer. |

**Source boundary.** The papers establish proposed mechanisms and their reported benchmark results. The fit rationale, ranks, widths/depths, parameter estimates, teacher paths, and likely advantages above are design hypotheses for this task. None is evidence of a score improvement, runtime, or public/private transfer.

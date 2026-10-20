# Plateau review: implementation and research plan

Status: accepted; first-comparison code integrated, prefit checks pending. Recorded 19 October 2026, Kyiv.
Owner: root/controller. This is the canonical continuation plan for this cycle.

## Resume here after compaction

1. Read this file, [current state](../current-state.md), and the campaign
   [frontier](../../research/campaign_2026-10-20/plateau_review_r1/FRONTIER.md).
2. Inspect Git changes, live workers, the newest authority/stop receipts, and
   measured resource/quota usage. Do not duplicate an existing job.
3. Continue the first unfinished deliverable below. Update its evidence path and
   the frontier before handing off; distinguish implementation from completed fits.
4. Preserve this accepted plan. Record consequential changes as dated amendments
   with their reason; do not silently replace the comparison after seeing scores.

## Authority and timing

The user requested that the external review and research guideline inform the
next research cycle, then supplied `claude-plan.txt`, expanded the allocation to
**15 local GPU-hours plus all available Kaggle quota in parallel**, instructed
us to implement, and explicitly requested this durable record.

- Prepare implementation and source records now. The agreed experiment start is
  **20 October 2026, Europe/Kyiv**; this is not a claim that jobs are scheduled.
- The latest instruction waives the older **30 paired cells per arm** requirement
  for this local/Kaggle cycle. Use fast, matched exploration and targeted follow-up.
  Full-data fits are allowed within the budget; this is not a screens-only limit.
- This scoped authorization succeeds the 18 October stop for this cycle. Keep
  that historical stop and its drained-job receipts intact. Unrelated campaigns
  and another agent's reserved Kaggle capacity do not automatically reopen.
- No paid rental, Vast job, top-up, or billing change. B300 work remains a teammate
  handoff when a large direction warrants it. No new queue manager.
- Existing authorization permits Git commits/pushes and experiment draft PRs to
  `research-team`. It does not imply an external competition upload.

## Goal and working hypothesis

Find a useful representation or learning change beyond repeated short
continuations of the same GRU parent. Start with a controlled question:
**does i1 hurt, help through existing inputs, or become useful when its temporal
state is supplied explicitly?** A weaker standalone model may still improve a
fixed blend or teacher target; measure that contribution separately.

Use official local evaluation to choose directions. Leaderboard feedback confirms
transfer. Do not add local deltas to public scores. The reported TL3 public score
0.66832 is a dated reference, not the clean control for this experiment.

## Compute and admission

| Resource | Bound / policy |
|---|---|
| Local GPU budget | 15 GPU job-hours total for this cycle, including GPU profiles, fits, and prediction/export; count failed attempts too |
| Local concurrency | One GPU worker; two aggregate CPU cores including support |
| Local memory | 5,000,000,000 bytes aggregate task RAM, no swap; 5 GB observed VRAM ceiling |
| Scratch | Physical disk `/home/vlad/development/wunder/research_tmp`; never RAM-backed `/tmp` or `/dev/shm` |
| Kaggle | Current verified free quota after other-agent reservations; parallel with local work; provider concurrency/session limits apply |
| Precision | Strict FP32; autocast and TF32 off; no quantization during research |

Soft local allocation: 2h initial screens/profiles, 8h deeper or full fits, 3h
targeted replication/mechanism work, 2h prediction/export/recovery. These are
allocation guides within 15h, not promises to complete every optional stage.
Reserve a complete comparison including exports before admitting its treatment.

The prior H512 R4 profile measured 2.71 GB RAM and 0.91/1.29 GB CUDA
allocated/reserved, with about 1.53 seconds per batch of eight complete sequences.
Its roughly 1.9h/full-fit estimate is a projection, not an observed full fit.
The new 156-channel block must be profiled at actual shapes. A complete full pair
may need about 3.8h before exports; a second pair may exceed the optional 3h block.

Reuse the existing resource controller, GPU lock, `scripts/limited`, and delivery
helpers. Set explicit memory/CPU bounds; the wrapper's default is too large.
Budget parent cgroup charges, active worker reservation and support memory
without double-counting. Serialize heavy local staging/upload/download work.
Do not retry an OOM without diagnosing its cause and changing the bounded attempt.

## First comparison: three equally wide models

All arms have **367 input slots = 112 raw + 99 standard event + 156 history**.
Use identical model initialization, component membership, sequence order,
normalization, optimization schedule, and fixed checkpoint rule.

| Arm | Inputs available to the model | Question |
|---|---|---|
| A: i0 + additional | Mask every raw and derived i1 channel after preprocessing; zero all history slots | Is the current i1 representation harmful or unnecessary at this fidelity? |
| B: standard control | Standard 112 raw + 99 events; zero all history slots | Matched reference |
| C: i1 history | B plus 156 causal history channels | Does explicit temporal state make i1 more useful? |

The mask must remove i1 raw values **and** their price deltas, event flags,
since-event counters and any later derived channels. Use semantic feature names
or an explicit reviewed registry. Zeroing raw i1 before event construction is
insufficient. Apply the mask on all assembly paths and after normalization so
disabled slots stay exactly zero. Fit shared preprocessing on the same fitting
membership before masking. The supported first CLI scope is i1; a general
arbitrary-group masking framework is not required for this comparison.

This removes explicitly i1-derived channels. Retained i0/additional values may
still be correlated with i1; A is a named-input ablation, not proof that the
model has no indirect information about the other instrument.

### Causal i1 history definition to freeze in the recipe

For each of the 52 i1 channels (raw positions 52–103), append three values:

1. Residual from the **previous** causal EMA with half-life 32.
2. Residual from the **previous** causal EMA with half-life 512.
3. `log1p(sqrt(EMA_512(delta_x**2)) / fitting_scale)`.

Use `alpha(h) = 1 - 2**(-1/h)`. Initialize each mean with the first observation,
the previous observation with that row, and change energy with zero. Compute
the residual before updating its mean; the change-energy update may include
the current observed delta. First-row history outputs are zero. Reset all state
between sequences, carry it across chunks, and detach only where the declared
training protocol calls for it. The per-channel scale is the RMS of first
differences within fitting sequences, floored at 1e-4. Reuse the deterministic
bounded fitting sample supported by the prior fitter; serialize its exact
source-qualified members, estimator denominator and scale with the checkpoint.
Do not describe sampled fitting statistics as statistics of every fitting row.

Keep raw values available in C. Do not assume anonymous channels are physical
prices or depths. Reuse existing causal-feature primitives where appropriate;
the scientific novelty here is the matched i1-specific comparison, not EMA itself.

### Common first fitting recipe

- Cold GRU H512 × 1, seed 1001, three epochs, fixed epoch index 2.
- AdamW, LR 0.001, weight decay 0.01, OneCycle, gradient clipping 1.
- Batch 8, TBPTT chunk 2000, full sequence state carried between chunks.
- Label-only training: no teacher, cached KD, warm initializer, auxiliary loss,
  quantization, or per-arm checkpoint selection.
- First fitting fidelity: 10% of fitting **components**, preserving every
  sequence/row/history of selected components. Freeze exact membership and order.
- Evaluate the same complete exposed F0 panel for A/B/C. Kaggle starts with the
  full-fitting F2 B control once source, quota and matched-treatment reservation
  are ready. F2 is also exposed.
- Freeze remaining CLI defaults explicitly before fitting. Profile first. If
  batch 8 does not fit, revise the batch size for all arms before recipe freeze.

## Sequence and decision rules

1. Prepare source archive, assumption audit and the matched implementation.
2. Freeze membership/source/config and run resource/causality/restore checks.
3. Run local A and B, then C; grade each completed comparison without waiting for
   a broad matrix. Start the independent Kaggle F2 control when its dependencies
   and October 20 timing are satisfied.
4. Advance the more informative A or C versus B to broader/full fitting. Preserve
   both negative and uncertain results. If both lose at 10%, consider a bounded
   C/B 30% pair to assess screen fidelity before shelving this representation.
5. Use saved predictions to evaluate fixed **0.9 B + 0.1 candidate** mixtures.
   Do not search coefficients or use leaderboard outcomes to tune them.
6. Add a fitting seed or fold only when the completed evidence and remaining
   budget support that question. Do not spend the entire budget on one noisy gain.
7. Package a new endpoint only when locally supported and actually checked for
   deterministic serving, size and runtime. No new ZIP is promised by this plan.

A confidence interval crossing zero does not close a direction. Classify success,
uncertainty and failure by effect, component/target/position consistency, matched
blend contribution, training stability, cost and expected information value.
Predeclare a practically useful effect in each card before fitting; do not
invent an effect-size threshold after the score arrives.

## Measurement and scientific controls

- Official metric: unchanged `global_wp_clip2_v1`; `need_prediction & is_scored`,
  clipping ±2, original clipped-label magnitude weights. Label clipping during
  training is a separately recorded setting.
- Reuse `wnn.gmetric.wp_sums/wp_from_sums` and
  `wnn.evaluation.paired_component_moments/paired_component_bootstrap`.
  The latter moments are **centered Welford moments**, not additive raw sums;
  use their matching `centered=True` merge/bootstrap path.
- Pair on source-qualified dependency components. Keep fit-seed uncertainty
  separate from conditional component bootstrap intervals. All four existing
  folds have prior exposure; repartitioning or Kaggle hardware creates no holdout.
- Report pooled/per-target scores, four fixed position bins, official-mask
  coverage and auxiliary all-`need_prediction` diagnostics, clearly distinguished.
- Record preclip gradient norm, LR, training/validation progress, source/config
  hashes, fit/evaluation membership, normalization and teacher/initializer ancestry,
  epoch checkpoints, predictions, failure receipts and measured resource cost.
- Independent criticism checks input masking, future leakage, group membership,
  checkpoint restoration, metric and claimed evidence strength before promotion.
- Historical controls can be reused only with exact compatible source, fitting
  membership, scale fitting, optimizer/schedule, seed and checkpoint selection.

## What existing evidence changes

- Tiny geometry and frozen-parent Kalman screens do not answer this cold H512
  comparison. Existing causal_moments/innovation modules should be reused where
  possible, not presented as new discoveries.
- Chunk-length sweeps and inference-reset sensitivity already exist. Detachment
  length is not available forward history; do not conflate those mechanisms.
- Cached KD hard-codes WMSE even at alpha zero; `--loss` does not select a different
  objective in that branch. Loss comparisons require a verified label-only path.
- E204 showed approximately balanced target mass/loss. Per-head loss balancing
  needs new diagnostic evidence. Existing Huber/correlation negatives narrow
  those recipes and discourage a generic loss grid.
- Fixed TL3/R3 mixtures already lost to TL3 on exposed F0 and F2. TL3 ancestry is
  unknown. Do not turn those caches into clean controls or repeat blend tuning.
- PRs #22/#23 are proposals, not implemented runnable cold-fit adapters.

## Deferred questions, admitted only when informative

1. **Gradient horizon:** detach 500 versus 2000 while holding the optimizer update
   at 2000 and carrying full forward state. Existing macro-update code requires
   cached KD; label-only support must be implemented before a cold-fit claim.
2. **Transfer:** determine whether a teacher representation gain survives transfer
   to identical cold students, with matched teacher ancestry and fitting groups.
3. **History representation / interactions:** only deepen timescales, separate
   encoders or input interactions after the first three arms reveal a useful fork.
4. **Loss / sampling:** use target/gradient/coverage diagnostics to select one
   discriminating comparison; avoid implementing the whole imported grid first.
5. **Large models:** a justified larger direction gets a code-bearing English
   draft PR for B300 teammates. Existing proposals do not imply completed fits.

## Implementation and durable deliverables

- [x] Archive `new.txt`, `research.txt`, `claude-plan.txt`, and `new-meta-review.txt` verbatim with source
  hashes, English evidence summaries and descriptive unified README entries.
- [x] Record the scoped CP30 override and October 20 start in
  ADR0042, AGENTS.md, current-state and campaign navigation.
- [x] Save the machine-readable compute contract and first F0 fitting membership.
- [ ] Freeze exact execution recipes against completed implementation and profiles.
- [x] Implement a focused history component, semantic i1 mask and CLI/config,
  normalization, checkpoint, prediction and source-provenance integration. Reuse
  `src/wnn/training/`; do not grow `scripts/train_rnn.py` or copy a trainer.
- [ ] Perform bounded full/chunk/streaming, prefix-invariance, zero-influence and
  checkpoint-restore checks required by the recipe; Ruff and training-structure
  checks distinguish static correctness from measured performance.
- [x] Prepare exact local recipes, component membership and fit/predict/score
  commands using the existing trainer and finite-step runner; see the campaign
  command index. Code is committed and statically reviewed, not numerically checked.
- [ ] Prepare the sealed Kaggle bundle after actual-device and paired-budget
  checks. The October 19 read-only snapshot found about 5.02h gross GPU quota;
  refresh it and other-agent reservations at admission. Code-only stages are not
  completed experiments.
- [ ] Run admitted jobs from October 20; collect checkpoints, predictions and
  per-phase receipts; independently grade, index and update the research tree.
- [ ] Commit/push coherent documentation, implementation and results separately.
  Publish experiment handoffs as drafts against `research-team` when warranted.

The [campaign frontier](../../research/campaign_2026-10-20/plateau_review_r1/FRONTIER.md)
holds mutable progress, actual worker IDs, spent budget and the next owner/action.
This spec holds the accepted comparison and allocation. Never substitute a chat
summary for either record.

## Amendment 1 — October 19: student/transfer meta-review

The user supplied `new-meta-review.txt` after plan acceptance and asked to think
about it while continuing this plan. Its [source audit and integration](../../research/unified/reviews/META_REVIEW_RECONCILIATION_20261019_R1.md)
preserve the useful competing explanations and correct stale/mismatched evidence.

- **First A/B/C remains unchanged.** The new review does not invalidate it.
- **Frozen-parent anchoring becomes a conditional second-wave comparison.**
  Historical external-T3 C was trained in FP16, so it is not a reusable control
  for this strict-FP32 cycle. The direct-parent profile reached 5.85 GB RAM.
  Require a new matched FP32 S/C pair, exact target/calibration audit, a revised
  resource profile and reservation for the complete pair. Do not reuse six-epoch
  alpha-zero results as a two-epoch control. No extra GPU budget is implied.
- **Collect existing width/depth evidence before duplicating it.** Research-team
  already has a verified six-epoch H512×1 F3 terminal checkpoint in a newer study;
  the paired grade was pending at the audited revision. Check its owner/state
  before any repeat. Only add a matched H256×2 B-input arm if it answers a distinct
  question and leaves the first comparisons/exports funded.
- **Combination path:** if causal i1 history helps a teacher, compare its cache
  with a baseline teacher cache in identical cold students. A later baseline/history
  × own-parent/T3 interaction is conditional on the parent comparisons earning
  follow-up; do not add their effects or automatically run a four-cell grid.
- **Kaggle availability is measured, not assumed:** the October 19 read-only
  [snapshot](../../research/unified/operations/PLATEAU_KAGGLE_READINESS_20261019_R1.md)
  found about 5.02 GPU quota-hours, zero provider reservations and no active kernels
  in the checked recent set. Recheck quota, device and other-agent ownership at
  admission. This may limit full-data paired follow-ups.

The imported review's quoted four-fold H256→H384 deltas remain unverified. Its
claimed teacher/students gains are historical evidence with known exposure,
not results from this cycle. The first 10% F0 membership is now frozen with the
existing subset builder: 315 complete fitting components, 1,118 train and 118
valid fitting sequences; the full 468-sequence F0 evaluation remains unchanged.

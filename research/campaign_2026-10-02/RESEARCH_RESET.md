# Research reset (2026-10-02 21:30): discovery before implementation

Design-only pass. Queue paused (`research/campaign_2026-10-01/QUEUE_PAUSE`); the screen that was already running
(`c20261002_muon_f0`, epoch-2 Muon screen) finishes, nothing else launches.

## 0. The observation that challenges the current bottleneck story

Current story (OPT_REPORT verdict, repeated in my summaries): "the optimizer is fine, so label noise is the binding limit."
Three measurements say the remaining error is still largely *reducible* and sits in correlated rows:

- E-152 learning curve (fold 0, pre-distillation GRU128 recipe, one seed): 25 / 50 / 100 % of training sequences →
  0.6675 / 0.6690 / 0.6724. The second doubling is worth +0.0034, larger than the first (+0.0015): no saturation.
- OPT_REPORT F1: training loss falls 5.5 % ± 1.4 % (se, 11 student runs) across the first re-exposure (epoch 0→1)
  and ~0 within epoch 1. Sequence-specific fitting starts at the second pass.
- E-197: the since-weight (which *down-weights* rows deep in quiet runs) improved WP most on exactly those rows (rows
  129+ before the next change +0.0061; class 3, 33–128 rows +0.007). E-320/E-323: the α 0.7 distillation gain also sat
  in rows ≥ 32 before the next move. Two different variance reducers act on the same rows.

Competing explanations of the plateau:

| | explanation | predicts |
|---|---|---|
| H1 | irreducible label noise: the student is near the Bayes risk for these inputs | student data slope ≈ 0, width ≈ 0, averaging gain ≈ 0 |
| H2 | estimation variance from correlated labels: every row of a run shares one label, so the effective sample is runs/sequences, not rows | student data slope ≫ seed SE; gains from any variance reducer (data, bagging, reweighting duplicated rows, memorisation-breaking views); variance reducers substitute for each other |
| H3 | representation / inductive-bias limit | small data slope, positive width/structure response |

E-152 does not settle this for the student: distillation is itself a variance reducer, so the distilled student's
slope is unknown. Smallest separating test (spec C1): the student recipe at 50 % vs 100 % of training sequences with
two seeds each, plus one GRU384 arm, on one fold. H1: Δdata ≈ 0 and Δwidth ≈ 0; H2: Δdata ≥ ~0.002; H3: Δdata small,
Δwidth > 0.001.

## 1. Facts rebuilt

### Baseline (exact artifacts)
`submissions/c20261002_D256_i16/submission.zip` (sha256 c493f042…0e32, public 0.6641, 36m21s): one trunk
`c20261001_final_d256_s3@ep3` with merged heads `|c20261001_final_d256_s3_gcorrh@ep0`, int16 engine build
`research/campaign_2026-10-01/engine/native_build` (sha e8b1f871…). Training: `research/campaign_2026-10-01/jobs/final_d256.sh 3`
(GRU256×2, α 0.5, teachers ab67_final_since_s3/s4, train + dev valid, holdout excluded, 4 epochs, checkpoint ep3,
head-only gcorr FT). Unverified: whether a GPU re-run reproduces the checkpoint bit-for-bit (cuDNN nondeterminism).

### Status reconciliation
- Completed and scored (this campaign day): E-326 α-pair dilution; E-327 teacher memorisation (inconclusive);
  E-328 α 0.7 fold 1; E-329 α 0.85; E-330/337 teacher ensemble folds; E-331 H-059; E-332 weight soup; E-333 finals on
  holdout; E-334 DTW/piecewise round (CPU); E-335 class-routed heads; E-336 q16x engine; E-339 TE final holdout;
  E-340 public α 0.7 (0.6635/0.6639); E-341/342 bag holdout; E-343 LB probe; E-344 residual notebook;
  E-345 LSTM192 (2 folds); E-346/347 TCN (2 folds).
- Interrupted / failed to run: TE seed-4 final (GPU-cap kill, then OOM); stopped-bootstrap folds (OOM at end of epoch 0,
  twice) — the stopped-bootstrap mechanism has **no data yet**; engine-l2 (agent stopped mid-edit, diff saved, not built).
- Running: `c20261002_muon_f0` (finishes; then pause holds).
- Queued, paused, never started: muon_f1, final_te07_s4r2, sbd05r2_f0/f1, chunk1000_f0/f1, chrono64_f0/f1.
  Twoclock smoke passed (282 s); its fold screens are not queued.
- Implemented, untested on data: `--arch twoclock`, `--distill-boot-h`, optim.py flags, TCN streaming reference.

### Implementation facts (read from `scripts/train_rnn.py` this pass)
- Row weight w = |y| · 1[t ≥ 99] · (π̂(x)² + 0.03) · max(0.25, 1/(1 + since_i0px/32)); π̂ = learned propensity of
  `is_scored` from causal inputs; for valid sequences with the true mask, the π̂ term becomes c·(is_scored + 0.03) with
  c preserving those sequences' weight mass (`true_mask_weights`).
- Per chunk c (32 sequences × 2000 rows): L_c = Σ_i w_i (p_i − t_i)² / Σ_i w_i, one optimizer step per chunk,
  t = (1−α) y + α · mean(teachers). Aux heads (struct, next) at 0.3. TBPTT 2000, state carried detached.
- AdamW 1e-3 OneCycle (β1 cycled), wd 0.01 on all tensors, global grad clip 1.0, EMA 0.999 (also mean-teacher
  consistency), 4 epochs, checkpoint ep3, then 1-epoch head-only gcorr FT.

### Audit of search-closing claims

| claim | evidence | narrowest supported conclusion | cheapest observation that could change it |
|---|---|---|---|
| input representation exhausted | E-334 cross-fitted ridge/HGB/kNN/DTW probes on OOF residuals ≈ 0; older input additions negative | the *tested hand-made features* add no probe-detectable signal to the residual of a model that already sees raw codes. Whether the trunk *uses* the raw inputs it has (by class × run position) was never measured | permutation-usage map of input groups (card 8, CPU) |
| stable training ⇒ label noise binds | no spikes, AMP skips 0.04 %, train loss falls while valid plateaus | the optimizer is not the bottleneck *for this objective*; the plateau is a generalisation gap of unknown kind | C1 |
| a failed config rejects the family | TCN C128×12 (one size, GRU-tuned lr/schedule, RF 4096 rows), LSTM192 (forget-gate bias 0), dual-clock (freeze confound) | each rejects that configuration under that recipe. TCN's −0.0044 is large enough that small tuning is unlikely to close it at that size; LSTM's −0.0009 is within reach of init/schedule fixes | a predeclared 3-point lr/init budget per family before closing it |
| residual autocorrelation is exploitable / irrelevant | segment residual ACF lag-1 0.18–0.49, PACF ≈ 0 after lag 1–2; labels mature after 30–48 events | correlated at lags shorter than maturity, so label feedback cannot use it. Exploitation through state (a persistent latent error) is untested. My E-344 reading ("label-overlap mechanics") is an interpretation: an overlap-only MA(q) process has an ACF that cuts off and a PACF that decays; we saw the opposite, which is more AR(1)-like | ACF of targets vs residuals and ACF at lags ≥ maturity (CPU); state probe (card 3) |
| two seeds establish seed noise | public 0.6635 vs 0.6639 | two draws give σ̂ ≈ 0.0003 with a 95 % interval for σ of roughly [0.0001, 0.009] (χ² with 1 df). Local: M01 paired width deltas over 3 seeds sd 0.00057 | 2–3 seeds of one recipe on one fold (inside C1) |
| arm gains stack | bandit index treats them as additive | no evidence for additivity; evidence against: α 0.5 + 0.7 pair −0.0009 vs α 0.7 alone; LSTM diversity offset by weakness; the TE fold gain vanished in the final | 2×2 factorial (C3) |

## 2. Alternative explanations by dimension

- **Observation and encoding.** Inputs are anonymised discrete codes (22-level price/volume codes) normalised globally;
  the target lives in tick units fixed per class. Justified invariance: level shift of i0 prices (the target is
  defined from i0 quote moves). Not justified: bid/ask mirror (not identifiable from anonymised codes). Risk: global
  normalisation of codes across classes with very different tick scales makes the same physical state look different
  per class.
- **Time and evidence.** Event = a6 lower-endpoint row (i0 quote move), 17.5 % of rows. Between events trades, size
  codes and i1 moves still arrive; trades back-load before the next move (E-334 H3). Three different quantities:
  elapsed time since the event (causal), remaining time (retrospective only; any weight built on it changes the
  optimum, see §6), label availability (maturity 30–48 events). The twoclock design summarises an interval by the mean
  row state, which discards within-interval ordering.
- **Learning and memory.** Gradients reach ≤ 2000 rows back. State carries further but receives no gradient for it.
  Existing evidence (E-175, older conv-GRU): resetting state every 4000 rows changed WP by −0.00001, so that model used
  ≤ 4000 rows of history. The head carries ~90 % of the gradient norm, so the global clip (active on 87 % of warmup
  steps) rescales trunk updates by the head's norm.
- **Objective and sampling.** x-measurable multipliers (propensity, since) leave the pointwise optimum unchanged but
  reallocate finite capacity and change estimator variance (§6). Per-chunk self-normalisation makes the effective weight
  w_i / W_c, where W_c depends on other rows' |y|. Within a run every row repeats one label realisation, so a run of m
  rows contributes m² times one row's noise variance to the gradient.
- **Generalisation.** Reversals (α: fold 0 +0.0023 → fold 1 −0.0007 → holdout +0.0004 → public −0.0004; TE: folds +0.0011
  → holdout −0.0008) fit seed noise of 0.0004–0.0007 per run plus fold composition, final-vs-fold data differences and
  holdout reuse. The LB probe (E-343) shows holdout-selected models over-predicted by 4–13e-4. Twelve public points
  cannot identify the hidden distribution (LOO 11e-4).
- **Computation and composition.** A GRU256 member is memory-bound (2.2 MB of int16 weights streamed per row); the
  frontier favours L2-resident models (≤ ~1.2 MB). Members help a bag only at equal strength. Gains that act on the
  same rows (since-weight, α, stopped bootstrap) should substitute rather than add.

## 3. Hypothesis cards

Duplicates removed: H-059, stopped bootstrap, RB-weight, LSTM/TCN/twoclock/optimizer arms are implemented or tested and
not re-proposed. Cards 1, 2, 3, 6, 7, 8 change the objective, training views, learning signal or experimental design.

### Card 1 — runs are clusters: re-derive the row weighting for distilled students
- **Observation:** E-197 (since-weight gains concentrated deep in quiet runs) and E-320/323 (α gain in the same rows);
  all rows of a run share one label.
- **Challenges:** "|y|-weighted per-row MSE is the right estimator because its pointwise optimum matches the metric."
- **Mechanism:** a run of m rows contributes its label noise m times; with exchangeable within-cluster correlation ρ the
  minimum-variance working weight is ≈ 1/(1 + (m−1)ρ) (GEE, Liang & Zeger 1986). The since-weight is a positional
  proxy tuned on a non-distilled GRU128 (E-182, +0.0008) and inherited unchanged by every student. Distillation
  already shrinks the noise of deep rows, so the proxy may now be mis-tuned in either direction.
- **Analogy (cluster-correlated regression / GEE):** cluster = target run, ρ = within-run correlation of label noise
  (≈ 1), m = run length. Where it breaks: m is future information at row t. Weighting by m changes the pointwise optimum
  whenever m carries information about y given x, so only x-measurable proxies (elapsed, class) are admissible. Also,
  the metric counts rows, not clusters, so efficiency weighting trades estimator variance against capacity spent where
  the metric counts.
- **Nearest prior:** E-182 (since τ 32, floor 0.25, GRU128, single seed). Non-equivalence: distilled students, and a
  weight curve derived from measured residual variance by (class, since bucket) instead of a two-parameter guess.
- **Signature:** gains in deep since buckets with flat early buckets; calibration slope unchanged.
- **Probe:** from D05 OOF predictions, estimate residual variance and within-run residual correlation by
  (class × since bucket) and derive the implied efficiency multiplier; compare it with the current curve (CPU).
  Positive control: synthetic clustered regression with known efficient weights, to show the estimator recovers them.
  Negative control: shuffled run ids (no clustering), where the implied curve should be flat.
- **Plausible failure:** after distillation the residual variance is no longer concentrated in deep rows.
- **Next decision:** + → C3, then a measured-curve weight; − → keep the incumbent weight and close the line; ? → C3
  decides anyway.

### Card 2 — memorisation at re-exposure: fresh views of the same sequences
- **Observation:** F1 epoch-boundary loss drop 5.5 %; E-152 slope; GRU192 and no-EMA runs lose in the low-lr tail;
  the 4th epoch adds +0.0004 for students.
- **Challenges:** "more epochs with EMA is the efficient use of fixed data."
- **Mechanism:** identical chunk boundaries and identical initial states every epoch let the trunk memorise
  sequence-specific trajectories. Random-origin training (each sequence starts at a random offset, with a burn-in
  excluded from the loss and chunk boundaries jittered per epoch) makes every epoch a different view without discarding
  labels.
- **Analogy (bagging / augmentation):** each epoch becomes a perturbed resample. Where it breaks: no new label
  realisations are created, so it can remove a memorisation shortcut but not the variance of finite runs.
- **Nearest prior:** row thinning (screens never ran: launch failures), inference-time periodic reset (different: it
  changes inference, not training). Non-equivalence: a training-time view change with inference unchanged.
- **Signature:** a smaller epoch-boundary loss drop, a smaller train/valid gap at ep3, and validation still rising at ep4.
- **Probe:** baseline vs random-origin on one fold; compare the logged epoch-boundary drop, and loss on 64 held-out
  training sequences scored along training (memorisation predicts a drop on seen sequences only). Positive control:
  a synthetic sequence task with memorisable noise. Negative control: shuffled labels, where both arms memorise.
- **Plausible failure:** the network memorises content regardless of alignment; burn-in costs labelled rows.
- **Next decision:** + → a 6-epoch random-origin final; − → close; ? → read together with C1's slope.

### Card 3 — sequence-level state from a training-only matured-label target
- **Observation:** oracle per-sequence scale +0.0061 (class 3 +0.0137) and per-sequence level of D +0.027 (non-causal
  bounds); post-hoc recalibration and DTW retrieval are null; residual ACF/PACF looks AR(1)-like.
- **Challenges:** "post-hoc nulls show the sequence regime is not recoverable."
- **Mechanism:** the loss rewards only the current target; nothing asks the state to track the sequence's realised
  gain. At training time the realised gain over matured rows (≤ t − M, M = maturity) is a legitimate auxiliary
  *target* (never an input). An aux head on it shapes the state to carry an online gain estimate.
- **Analogy (online system identification, RLS/Kalman parameter tracking):** θ_s = slowly varying per-sequence gain,
  observations = matured (x, y) pairs, filter state = GRU hidden. Where it breaks: the network cannot see matured labels,
  so it must infer the gain from x-history alone. The forgetting rate is unknown, and E-334 H2 found x-summaries useless
  for retrieving it. That is evidence against, unless the trunk extracts better features than those summaries.
- **Nearest prior:** matured-label feedback (labels as *inputs*; a leak, then null), post-hoc recalibration, DTW scale
  retrieval. Non-equivalence: representation shaping only; inference unchanged.
- **Signature (before any score gain):** the baseline state-probe R² for realised past gain is near 0; with the aux head,
  probe R² rises and the dispersion of per-sequence calibration slopes falls.
- **Probe:** stage A on CPU — export h_t for ~200 fold-0 sequences from the D05 parent; target = |y|-weighted realised
  gain over matured rows before t; sequence-cross-fitted ridge R², compared with ridge on x-summaries. Positive control:
  synthetic sequences with a planted gain recoverable from an x-feature (GRU with/without aux; the probe must detect it).
  Negative control: gains shuffled across sequences (R² ≈ 0).
- **Plausible failure:** the gain is not predictable from inputs, so no state can carry it.
- **Next decision:** + (deficit, and the synthetic control passes) → aux training on 2 folds × 2 seeds; − (state
  already matches the x-summary bound) → close; ? → stop.

### Card 4 — credit horizon vs memory horizon
- **Observation:** TBPTT 2000; E-175 reset-every-4000 rows cost −0.00001 (older model); earlier chunk experiments
  changed the update count (confounded, per the external audit).
- **Challenges:** "carrying state indefinitely gives long memory."
- **Mechanism:** dependencies longer than a chunk get no gradient, so the trunk stores only what pays within 2000 rows.
- **Nearest prior:** chunk 4000/500 runs. Non-equivalence: chunk 8000 with 4-way gradient accumulation (same optimizer
  steps and EMA cadence).
- **Signature:** a reset-at-lag-k curve that stays flat beyond 2000 for the baseline but drops for the long-credit arm.
- **Probe (CPU, no training):** WP of the D05 student when its state is reset at t − k for k ∈ {250, 500, 1000, 2000,
  4000}. Positive control: a synthetic task needing 5000-row memory (TBPTT 2000 fails, update-matched 8000 learns).
- **Next decision:** flat beyond ~2000 → run update-matched long TBPTT; flat beyond ~250 → long memory is irrelevant
  here, close; still rising at 4000 → the state already uses long history, close.

### Card 5 — head-dominated gradient under one global clip
- **Observation:** the head carries ~90 % of the gradient norm; clip 1.0 fires on 87 % of warmup steps, 3 % late.
- **Challenges:** "a single global clip and LR are neutral."
- **Mechanism:** the clip rescales trunk updates by the head's norm, slowing representation learning early.
  Likely small, since late steps are rarely clipped.
- **Probe:** relative update size ‖Δθ‖/‖θ‖ per module from the existing grad-stats instrumentation; per-module clip
  arm only if the trunk's relative update is ≪ the head's. Decision: close if the ratio is near 1 after warmup.

### Card 6 — self-normalised per-chunk loss
- **Derivation:** the step for chunk c uses w_i / W_c with W_c = Σ_{j∈c} w_j. If W_c is correlated with y_i given
  x_i (shared regime, class composition of the batch), the minimiser shifts; independently, it adds gradient noise.
  Self-normalised importance sampling has O(1/n) bias (Owen, *Monte Carlo theory, methods and examples*, ch. 9;
  recalled, not re-read in this pass).
- **Probe (CPU, data only):** the distribution of W_c over training batches and the binned covariance of 1/W_c with
  |y_i| y_i. Close if CV(W_c) < 5 %. Otherwise test a global (EMA) normaliser.

### Card 7 — the search is underpowered (experimental design)
- **Observation:** α and TE reversals; single-seed screens with |Δ| ≈ σ_seed·√2; holdout reused (E-343 bias).
- **Mechanism:** decisions at ~1 SE produce false discoveries and false rejections.
- **Change:** estimate σ_seed per recipe (C1), pair arms with common random numbers (same seed and data order),
  require 2 folds and a predeclared minimum detectable effect, and treat public pulls as the only confirmation.
- **Probe (CPU):** re-analyse existing arm pairs to get the fraction whose sign replicated across folds.

### Card 8 — input usage map
- **Observation:** E-334 probes were post-hoc; the streams agent found i1 features with raw corr 0.4–0.5 to yD but
  partial < 0.05, i.e. already used by the GRU. Unknown for trade-mark and size codes in deep quiet rows.
- **Challenges:** "the inputs are exhausted."
- **Mechanism:** an input group with content in class-3 deep rows but low usage would indicate an encoding or learning
  failure (e.g. global normalisation across classes).
- **Probe (CPU):** within-sequence permutation importance per input group × class × since bucket on fold-0 OOF.
  Positive control: a synthetic informative feature appended to a short synthetic set.
- **Next decision:** a high-content, low-usage group → an encoding change for that group; otherwise the claim narrows
  to "inputs used as far as this model class can".

### Ranking (subjective, visible)
By information per GPU-hour: C1 (decides between H1, H2 and H3, and powers every later screen) > card 3 stage A and
card 4 probe (CPU, decisive) > C3 (since × α) > card 2 > card 8 > card 6 > card 5. The order of the first three holds if
σ_seed is 0.001 instead of 0.0004 (CPU probes are unaffected; C1 then needs 3 seeds per cell).

## 4. Criticism for the three strongest cards

**Card 1 (cluster weighting).** For: deep-row gains from both the since-weight and α are what variance reduction
predicts. Strongest competitor: the since-weight helped by reallocating capacity toward early rows, where
signal-to-noise is high, not by reducing variance. Discriminating observation: variance reduction predicts gains in deep
buckets and substitution with α; reallocation predicts gains in early buckets and no interaction with α. C3 measures
both per bucket. Synthetic control: clustered regression where the efficient weights are known. Passing shows the
estimator works, not that the real data behave this way.

**Card 2 (memorisation).** For: the epoch-boundary drop and the low-lr-tail losses. Strongest competitor: the drop is
ordinary learning, made visible by the schedule and the EMA mean-teacher target changing between epochs. Discriminating
observation: loss on never-trained held-out training sequences across the same boundary. Memorisation predicts a drop
only on seen sequences.

**Card 3 (sequence state).** For: large oracle per-sequence gains and AR(1)-like residuals. Strongest competitor: the
gain is not inferable from inputs at all (E-334 H2 retrieval null), so neither state nor aux can carry it.
Discriminating observation: stage A compares state-probe R² with the x-summary R² on the same target. If both are ≈ 0,
the competitor wins; if state R² < x-summary R² > 0, the state is missing an inferable quantity and the aux test is
justified. Levels kept separate: detectable (synthetic control), learned (probe R² after aux), deployed (2 folds × 2
seeds, then one public pull).

Tuning budget: each new arm gets 3 predeclared lr points at matched epochs; report both the matched-intervention
comparison and the best-of-3. Ep2 screens are validated only for same-architecture recipe changes (OPT F2); optimizer,
architecture and curriculum arms run the full schedule.

## 5. Interaction experiment (evidence-based 2×2)

Since-weight {on: τ 32, floor 0.25 | off} × distillation α {0 | 0.5}. Evidence: since gains in deep rows (E-197),
α gains in deep rows (E-320/323), students inherited the since-weight untuned. Prediction under H2: substitution, i.e.
(since gain at α 0) > (since gain at α 0.5). Primary endpoint: class-3 WP on rows with since_i0px ≥ 32 (where both
effects live, which raises power); secondary: pooled and public-mix WP. Spec in C3.

## 6. Theory checked at the point of use

1. **WP optimum.** Weights |y|, m(x) = E[|y| | x], g(x) = E[|y| y | x] / m(x). Then Cov_w(p, y) = E[m(x)(p − p̄)(g − ȳ)]
   and Var_w(p) = E[m(x)(p − p̄)²]. By Cauchy–Schwarz in L²(m), WP is maximised by any p affine in g. This holds for
   unconstrained measurable p; the metric's clipping at ±2 means g should use clipped y.
2. **x-measurable multipliers.** Minimising E[c(x)|y|(p − t)²] gives p*(x) = E[c(x)|y| t | x] / E[c(x)|y| | x] = g(x)
   for t = y, so propensity and since weights do not move the pointwise optimum. The true-mask weights use is_scored,
   which is not in x: p* becomes the scored-row conditional, which matches the metric's selection. Any weight using run
   length or remaining time (future) moves p* unless it is conditionally independent of y given x.
3. **Distillation target.** With t = (1−α)y + αT(x): p* = (1−α)g(x) + αT(x). This is unbiased only if T = g; it trades
   teacher bias for variance (Menon et al., ICML 2021, *A statistical perspective on distillation*; recalled, not re-read
   in this pass).
4. **"RB weight preserves the minimiser" (lit-scout card): false as stated.** With weight m(x) and target y,
   p* = E[y | x] ≠ g(x). The consistent version uses weight m(x) and target |y|y/m(x): the expected gradient
   2(m p − E[|y|y | x]) equals that of |y|(p − y)², with less noise in the p-coefficient term.
5. **Stopped bootstrap.** Let M_s = E[y_r | F_s] for rows of run r. σ = t + h if the run outlives t + h, else ∞.
   σ is a stopping time because the run end is causally observed. Optional stopping for bounded σ then gives
   E[M_σ | F_t] = M_t. With a real teacher T ≠ M, the target's bias equals the teacher's error at t + h. "Unbiased" is
   therefore assumption-dependent: it needs a calibrated teacher.
6. **"Noise ceiling."** None is established. Oracle numbers are non-causal upper bounds, not Bayes risks.

## 7. Decisions

### A. Diagnosis of the search process
1. Decisions were taken at ~1 SE with single seeds; the α and TE reversals are what that produces (card 7).
2. The bottleneck story moved from "optimizer" to "label noise" without testing the reducible-variance alternative
   that E-152, F1 and E-197 point to.
3. Configuration failures were recorded as family verdicts (TCN, LSTM, dual-clock).
4. Inherited choices tuned on an older recipe (since-weight, epochs, clip) were never re-checked for distilled students.
5. The holdout was used for selection and is no longer confirmation.
6. Infrastructure failures (GPU cap, OOM) consumed a large share of the day's GPU time.

### B. Hypothesis cards
See §3 (8 cards, de-duplicated).

### C. Three prioritised specifications

**C1 — bottleneck diagnostic: student learning curve with seed replicates and a width arm.**
- Arms (fold 0, full 4-epoch schedule, checkpoint ep3, parents only):
  - S100 seed 4 (seed 3 exists: c20261001_d256_s3_f0);
  - S50 seeds 3 and 4 (`--train-seqs 5303`; valid-train rows kept as in E-152);
  - W384 seed 3 (`--hidden 384`, deployability ignored).
- Population: the 468 fold-0 evaluation sequences (an exploration set, already exposed); report pooled, per class and
  class-3 deep rows.
- Resources: ~2.8 GPU-h (S100 45 min, S50 2×23 min, W384 ~75 min), 8 CPU threads, ≤ 14 GB RAM per job, VRAM cap 4.6 GB.
- Stopping rule: none, it is a fixed design.
- Readout: Δdata = mean(S100) − mean(S50); σ_seed from the seed pairs; Δwidth = W384 − S100.
  - Δdata ≥ 0.002 (≥ 3 SE) → H2: prioritise cards 1 and 2 and C3.
  - Δdata ≤ 0.0005 and Δwidth ≥ 0.001 → H3: capacity/structure line (L2-resident wide models, engine work).
  - Both small → formulation cards 3, 4, 8 and stop recipe tuning.

**C2 — distinct mechanism: sequence-level state (card 3), staged.**
- Stage A (CPU, ~1 h, ≤ 4 GB): state export for 200 fold-0 sequences; ridge probes; synthetic positive and negative
  controls.
- Stage B (only if stage A shows a deficit): aux head on realised matured gain (weight 0.1, training-only target), folds
  0 and 1 × 2 seeds, full schedule, ~6 GPU-h, vs matched controls with the same seeds.
- Stop at stage A if state R² ≥ x-summary R², or if the synthetic control fails.
- Teaches whether the regime is inferable but not represented.

**C3 — interaction: since-weight × α (2×2).**
- Cells per fold: (on, 0.5) exists for seed 3; new cells (on, 0), (off, 0.5), (off, 0) for seed 3; folds 0 and 1.
- Cost: 6 runs × ~45 min ≈ 4.5 GPU-h.
- Primary endpoint: class-3 deep-row WP. The interaction contrast's SE is ≈ 2σ/√(2 folds) ≈ 0.0007 pooled at
  σ = 0.0005; deep-row effects in E-197 are ~0.006, which gives adequate power there, not pooled.
- Teaches whether variance reducers substitute (H2) and whether the students' inherited weight is mis-tuned.

### D. First experiment once execution is authorised
**C1.** It reuses an existing run, needs no new code (`--train-seqs` and `--hidden` exist), and its outcome redirects
the next ~20 GPU-hours. A steep slope makes this a sample-efficiency problem (weighting, fresh views, bagging). A flat
slope with a width response makes it a capacity/runtime-frontier problem. Flat on both says the current formulation is
exhausted, so effort should move to the learning-signal and representation probes. Its seed replicates also give the σ
that every later screen needs. Card 3 stage A and the card 4 reset probe can run on CPU alongside it.

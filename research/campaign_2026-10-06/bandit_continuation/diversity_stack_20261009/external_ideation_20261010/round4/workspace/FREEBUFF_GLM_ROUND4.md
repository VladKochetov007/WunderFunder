# Round 4 — eight long-run training-regime experiments (GLM answer)

Read: `METHOD_REVIEW.md`, `NOTES_ROUND3.md`, `learning_curve_6epochs.jsonl`. Everything below is written against the fold-1 hurdle of **0.670**, i.e. roughly +0.006 above the scratch-teacher mean (0.66642) and +0.0008 above the best current model (student `h230_f1_s3_ep3`, 0.66922). With seed sd 0.0010 and one-run A/B sd 0.0022 (`hpo/VALIDATION_PROTOCOL.md`), 0.670 is a real target, not a coin-flip over the noise floor — but anything claimed at +0.002 needs a second seed.

---

## 0. Diagnosis first: why epoch 3 is the cliff, and why your teammate disagrees

From `learning_curve_6epochs.jsonl` (fold 1, WP):

| snapshot | WP |
|---|---|
| ep2.248 (mid) | 0.66479 |
| ep2 end | 0.66342 |
| ep3.123 (mid) | 0.66628 |
| **ep3 end** | **0.66709** ← peak |
| ep4.123 (mid) | 0.66266 |
| ep4 end | 0.66194 |
| ep5.123 (mid) | 0.65054 |
| ep5 end | 0.64903 |

Three facts: (a) the peak is one snapshot wide, (b) mid-epoch snapshots are as good as epoch ends (so the number is schedule-driven, not "finished an epoch"-driven), (c) the fall accelerates (−0.005, −0.013). An epoch is one pass over ~11.9k sampled sequences, so by epoch 5 the model has seen each sequence six times **in identical row-aligned form**. With 1.04 M parameters and ~212 M training rows that fit in the capacity, this is textbook alignment memorisation — matching NOTES_ROUND3 ("data economy, not optimisation, limits the teacher") and the ≤±0.002 ceiling on every recipe knob tried so far.

Why the teammate's "more epochs helped" is not necessarily noise or error — the four live explanations, in order of my prior:

1. **They average.** EMA/SWA over many epochs converts a sharp single-point peak into a wide plateau; with averaging the curve of *averaged* checkpoints keeps rising past ep3 while the raw curve falls. Our HPO already found SWA in its best trial (35: muon_hidden, WSD-linear, dropout 0.16, SWA).
2. **They changed the objective** (distillation / label smoothing / blending), which caps how much label-alignment detail is worth memorising.
3. **They changed the data regime** (random crop windows, shuffling across streams, augmentation) so epochs are not six repeats of the same alignment.
4. **It's one-fold noise** (±0.002) on a non-frozen valid protocol.

Cheap diagnostic before spending GPU money: get the teammate's exact recipe. If it contains (1)–(3), that decides the experiment ordering for free.

Design rule used below: **every long run must change one of {what the model sees per step, what it is asked to predict, how much it is allowed to fit, what checkpoint is kept}**. A long run under the current recipe just retraces the ep4–ep5 collapse with more cosine.

---

## 1. The eight experiments

Conventions: Muon on GRU matrices + AdamW elsewhere, lr 1e-3 / wd 0.01 is the reference recipe; "WP" = fold-1 weighted Pearson; probabilities are P(final checkpoint config > 0.670 on fold 1), my judgment, σ≈±0.08.

### E1 — Stream-window epochs (random 2k-row windows from the 3,288 continuous streams) · **unconventional**

**Settings.** chunk 2000, batch 16 sequences/window-chunks, Muon/AdamW lr 1e-3, wd 0.02, one-cycle over **15 epochs**, one epoch = 7,460 steps (= same step count as the reference epoch) → 238,720 window draws per "epoch". Each step draws 16 **random contiguous 2000-row windows at random offsets** from train streams, *not* fixed 20k chunks. Loss masked on the first 99 rows of every window (never scored; also absorbs the zero-state mismatch — the 99 warm-up rows already do this job at serving, per NOTES_ROUND3 §1). Event features recomputed from the corrupted/re-cut input inside the trainer, as already happens. ~105k distinct windows exist; with stride-1 offsets the alignment seen by the model is effectively never repeated.

**Why it should work on this data.** It is the only experiment that removes the memorisation at its root: what the model currently memorises is *alignment* (which row of which chunk it is on), because an epoch repeats the identical 20k-row objects 3–6 times. Random phase breaks alignment identity while preserving every label. The lookahead ladder (§9 of METHOD_REVIEW) says the label is a smooth function of a long future path — the information per label is unchanged by re-cutting, so this is a pure variance/overfitting lever. It also converts "epoch" into "step budget", which is exactly what you need for 15–30-epoch runs.

**Failure signature (pre-registered).** By epoch 5–6: mid-epoch snapshot WP < 0.660 while train loss keeps falling below the reference run at equal steps → the re-cut is leaking (window-boundary state contamination or the label-run straddle at cut points); also check that predictions in the first 200 rows of windows are not systematically worse than elsewhere. Abort by epoch 8.

**P(beats 0.670): 30%.**

### E2 — Long born-again distilled run (the teammate hypothesis, made explicit)

**Settings.** Warm-start `--init <6-epoch run>@ep3`, targets = the existing four-family blend you already cache for fold 1 students, `--distill-alpha 0.7` (0.3 true / 0.7 teacher), AdamW **lr 4e-4** (4× the current student's 1e-4, since the run is long), wd 0.05, dropout 0.15, one-cycle over **12 epochs**, batch 16, chunk 2000, plus **SWA over epochs 8–12** evaluated as a separate checkpoint. Family heads on (`--distill-family-head --distill-family-weight 0.1`), unchanged.

**Why.** This is the one regime where *more epochs is already proven to help*: the distillation blend caps how much alignment detail pays, and the +0.0032 student gain came after only 3 short epochs at a tiny lr — the knob was never pushed. 12 epochs at lr 4e-4 with 0.3 true-label mass is a genuinely different objective, and SWA re-uses the plateau (the current peak is one snapshot wide; averaging widens it). It directly tests teammate-explanation (1)+(2).

**Failure signature.** Raw (non-SWA) snapshot WP monotonically decreases from epoch 4 onward and the SWA average ≤ raw peak − 0.0005 → the 0.3 true-label mass is still the memorisation channel; drop alpha to 0.15 before killing the run. Also flat: nothing beats 0.66922 by epoch 9 → distill saturation, stop at 10.

**P(beats 0.670): 35%.**

### E3 — Heavy-regularisation long run: low lr + wd 0.08 + dropout 0.25 + WSD

**Settings.** Muon on GRU (scope `rnn`), lr **4e-4**, Muon multiplier 1, wd **0.08**, **dropout 0.25** on encoder + head (cuDNN GRU can't take recurrent dropout — don't fight it), schedule **WSD**: 5% warmup, 60% stable, 35% linear decay to ~0 over **15 epochs**, batch 32, chunk 1000.

**Why.** This is the falsification run for the whole standard-sweep half of your plan. The 6-epoch curve says the constraint is capacity-vs-data; the only levers that side are throughput-limiting regularisation and a schedule that spends epochs 6–10 at constant lr *after* the memorisation point, then decays into an averaged minimum. HPO's best objective (0.66876, trial 35) already sits at exactly this quadrant (muon_hidden, WSD-linear, dropout 0.16, SWA) at width 384 — this run is it at width 256 for 15 epochs.

**Failure signature.** Fold-1 WP pinned at 0.660–0.663 from epoch 6 to 12 with train loss still descending → the regularisers cost more than the memorisation they prevent (underfit regime); if by epoch 10 max WP < 0.664, stop. Divergence/NaN in the first epoch → lr multiplier issue, don't relaunch blind.

**P(beats 0.670): 18%.**

### E4 — Averaging-isolated plateau run (SWA + EMA on the unmodified recipe, 12 epochs)

**Settings.** Exactly the default teacher recipe for 12 epochs but with a **constant lr 1e-3 through epoch 8, then cosine to 0 over epochs 9–12**; SWA collection from epoch 5 (one snapshot/epoch) + an EMA copy (decay 0.999) kept in parallel; evaluate raw, SWA and EMA checkpoints each epoch. Also add the ablation arm nobody has paid for yet: epochs 9–12 rerun with **pure AdamW**-only optimiser from the epoch-8 checkpoint at lr 3e-4 (1 h), so the "Muon vs AdamW on matrices" question is answered cheaply instead of with a full 15-epoch run.

**Why.** Cheapest experiment that turns your sharpest finding (peak = one snapshot) into a lever, and the only one that reads cleanly as a single-factor A/B on *checkpointing*, independent of everything else. If averaged checkpoints do nothing, three downstream ideas (E2's SWA, seed-merging R3-2, E3) lose their shared assumption and you save a whole line of work.

**Failure signature.** SWA/EMA ≤ raw peak − 0.0005 at every epoch → the late-epoch direction is drift, not noise-pooling; kill by epoch 8. Positive but sub-noise (< +0.001 still, after 3 seeds) → averaging is a stabiliser, not a gains source — report and move on.

**P(beats 0.670): 20%.**

### E5 — Multi-horizon auxiliary supervision (predict the future labels you already have) · **unconventional**

**Settings.** Default recipe, but the loss adds 4 aux heads with weight 0.1 each predicting **y shifted by k = 2, 4, 8, 16 rows** (labels for future rows exist during training — use them), chunk **4000**, batch 16, lr 1e-3, wd 0.02, one-cycle over **12 epochs**. Main head unchanged and is the only thing served.

**Why.** The lookahead ladder (0.680 @ +2 rows … 0.960 @ +512) shows the future label path is *highly* predictable — you're just not supervising it. Forcing the same trunk to predict y at several horizons makes it learn the price-move forecast the label actually is (your own §9 wording), instead of letting the causal head satisfy the loss with a weaker shortcut. This is the training-only analogue of the "lag-split target" (R3-1) but smoother: it adds information, it doesn't split the supervised rows. The aux parameters cost nothing at serving.

**Failure signature.** Main-head WP < 0.658 by epoch 4 → aux gradients dominate the shared trunk (halve aux weights once, then stop). Aux heads learn (their WP rises) while main stays ≤ 0.664 → forecast quality isn't transferring to the causal head; that result is still valuable — it would prune a whole hypothesis family (H-32x boundary/forecast ideas).

**P(beats 0.670): 25%.**

### E6 — Label-preserving augmentation: idle-run permutation + block dropout · **unconventional**

**Settings.** Default recipe, 15 epochs, one-cycle, plus two augmentations: (a) inside each training sequence, permute the row order within maximal runs where the target is row-to-row constant **and** neither instrument's price/trade block fires an event (label piecewise-constant, mean run 9.5 rows — these runs are informationally idle); p = 0.5 per sequence; (b) block-level input dropout: drop the whole 8-column extras block (104–111) or one volume block with p = 0.15 per sequence. **Recompute event features from the corrupted input** (counters/lags stay consistent); never reuse precomputed features. Dropout 0.1 as in default.

**Why.** Directly attacks the memorisation mechanism NOTES_ROUND3 identified ("epoch repeats the same sequences") without touching the label: the same information content arrives in thousands of different row orderings, so what gets memorised must be content, not alignment. This is a trainer change (R3-5, already queued), here given the long-run treatment it was designed for — augmentation gains only appear when the model would otherwise run out of data.

**Failure signature.** t1 WP falls > 0.004 relative to t0 within 3 epochs → the permutation is actually deleting label information (either the event detector misses real updates or the future book moved inside an "idle" run; the label is defined on the *future* book, so a constant-y run can still be label-critical). Check this first on a CPU probe of 2 epochs; if plateau WP ≤ 0.663 at epoch 10 → permutation diversity isn't reaching the trunk.

**P(beats 0.670): 18%.**

### E7 — Sample-starved class recovery: oversample long-age rows, class-3 weight 2.0

**Settings.** Default recipe, 12 epochs, one-cycle; oversample ×2 the windows/sequences containing class-3 (age > 16) runs; class-3 loss weight **2.0** (inside the HPO-space range 0.7–2.5, at the top); propensity weighting **power 1** instead of 2 with the same floor 0.03 (avoid double-downweighting the starved rows).

**Why.** Your own notes call class 3 "sample-starved" and the HPO space's biggest single-knob trial (27: class-3 weight 2.45, 3 layers, 0.66779) points the same way. A long run is the right setting: at 12 epochs the starved rows get 24 effective passes instead of 8. It's also the one experiment whose primary readout is *sub*-pooled: class-3 sign accuracy at age > 256.

**Failure signature.** Pooled WP < 0.665 from epoch 6 onward while class-3 sign accuracy is unchanged → the extra mass is coming out of generic fit; revert propensity power before killing. Primary falsifier (your own numbers): class-3 sign-acc ≥ +2 pts **and** pooled ≥ 0.667, else no-go by epoch 10.

**P(beats 0.670): 12%.**

### E8 — Muon on every 2-D weight + long WSD (the optimizer-scope question, given its best shot)

**Settings.** Muon scope on input projection + GRU + head, lr 7e-4 (input/head often prefer ~0.7× the GRU lr), Muon multiplier 0.7 for the non-recurrent matrices, wd 0.03, dropout 0.15, schedule: warmup 3 epochs → stable to epoch 10 → linear decay to 0 over **18 epochs** total, batch 32, chunk 1000.

**Why.** You're about to test the first rows of this matrix at 3–4 epochs where nothing can separate (everything moves ≤0.002 before ep3); the honest test needs the long-run regime where an optimizer difference can compound. Muon on the input projection specifically plausibly helps *this* data: 211 inputs of wildly different update rates and scales, where AdamW's per-coordinate adaptivity is also what lets the head over-specialise to alignment.

**Failure signature.** GradScaler skips > 1% of steps per epoch after warmup (Muon + full-precision normalization conflict on the input proj), or WP at epoch 6 < 0.659 → the scope change hurts; the answer to "Muon everywhere?" is then *final* for the space. Kill by epoch 10.

**P(beats 0.670): 10%.**

---

## 2. Coverage vs your planned sweep — and what to drop

| Planned axis | Status here |
|---|---|
| lower lr 3e-4–5e-4 | E3 (4e-4), E2 (4e-4 student) |
| wd 0.05–0.1 | E2, E3 |
| dropout 0.1–0.2 | E2 (0.15), E8 (0.15); E3 pushes to 0.25 as the stress case |
| AdamW vs Muon(GRU) vs Muon(all 2-D) | E8 is the full-scope run; the pure-AdamW arm is a 1 h ablation bolted onto E4, *not* a 15-epoch run — spending a long run on AdamW-only is the lowest-EV cell of your matrix and I would not pay 3 h for it |
| one-cycle vs WSD | E3, E8 (WSD); E1 (one-cycle long) — aim: WSD's stable phase is where a long run differs, one-cycle's late cosine is not |
| 15–20 epochs | all runs are 12–18 |

## 3. Sequencing and the single run I would start first

Budget: 8 runs ≈ 24–30 h. Order with information flow, not just EV:

1. **E2** → day 1
2. **E4** (averaging A/B; its verdict applies to E2/E3/R3-2) → day 1, can share serialization of E2
3. **E1** (root-cause test; its verdict re-frames every later run) → day 2
4. **E5**, then **E3** → day 2–3
5. **E6**, **E7**, **E8** → only if slots remain after re-prioritising on the first five.

**First run: E2, the long born-again distilled run.** Reasons: highest probability of actually crossing 0.670 (it starts from the strongest family of results you have — distillation already delivered +0.0032 — and it is the direct implementation of the most likely explanation of your teammate's "more epochs helped"); near-zero implementation risk (reuses the distill cache, student trainer, SWA machinery); and it outputs two decision-relevant facts at once — whether the 1e-4 student lr was leaving gains on the table, and whether averaged checkpoints matter — so even a miss shaves the search space for E1–E8.

One caution that applies to all of these: fold 1 has been the selection surface for a long time, and local +0.003 gains have not reliably transferred to public before (§7 of METHOD_REVIEW). Whatever wins here should be confirmed once on the reserved folds 2/3 before it changes the pipeline.

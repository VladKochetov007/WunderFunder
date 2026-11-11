# Loss/objective ideation and red team — 2026-10-25

## Part A — Ideas

### A1. Batch-mass normalization (already coded, zero regret)
**Change:** Normalize the loss denominator by a running EMA of per-batch Σw instead of the per-batch sum; optionally divide only the *gradient scale*, keep the reported loss per-batch.
**Why:** With w = |y|·ĝ, batch weight mass swings 0.1×–9.7×. In WMSE, gradient scale ∝ Σw, so effective LR whips 10× between batches — pure optimization noise, worst exactly on heavy-|y| batches (your weak class). Metric-optimal conditioning is unaffected (denominator is per-batch in both cases as an estimator); this targets variance, the one channel your Bayes-alignment note leaves open.
**Test:** 2 fits: F0, 2 seeds vs matched control. It is already coded — this is the cheapest discriminating experiment you own.
**Falsifier:** Paired delta ≤ 0 or within ±0.001 on both seeds.
**Risk:** Interacts with AdamW's per-parameter rescaling; may want LR ±20% in the same screen (2 extra fits only if the first pair is positive).

### A2. Tail weight-EMA / SWA of endpoint weights
**Change:** Maintain a Polyak EMA of weights over the last ~10% of training (decay 0.999); evaluate the EMA endpoint. Zero new fits: also re-evaluate existing checkpoints.
**Why:** Your last wave of "gains" is +0.0001–0.0004 with endpoint seed noise 0.0007–0.0014 — endpoint variance is currently a first-order term vs every claimed effect. Weight averaging directly attacks that, and WMSE landscapes late in training are nearly convex locally so EMA is close to free. Distillation with a=0.9 has a slowly moving target, so EMA is safe.
**Test:** 0 new fits. Evaluate EMA endpoints for your last 4–8 checkpoints across folds/seeds.
**Falsifier:** Mean delta across ≥4 (fold, seed) cells ≤ 0.
**Risk:** Nearly none; the only threat is that a fast late LR decay already removed most of the tail noise.

### A3. β-NLL with β≈0.5, detached variance
**Change:** Joint mean head + log-variance head; loss = Σ w·σ̂^(−2β)·(p−y)², σ̂ detached, β=0.5, keep w = |y|· propensity.
**Why:** Your measured NLL failure is diagnostic, not scientific: 100% clipped updates (mean clip factor 0.04) means the experiment never trained the objective — 1/σ² gradients through a global grad-clip annihilated them. β-NLL is the designed fix: β=0.5 restores WMSE-like gradient scales while still adaptively downweighting genuinely irreducible-noise rows — i.e., it shifts gradient mass toward rows where the conditional is estimable, which is the same "variance-limited" mechanism your brief bets on.
**Test:** 2 fits (β=0.5, F0 ×2 seeds); warm-start σ̂ from a 1-epoch variance head finetune to avoid the σ̂ head dragging early training.
**Falsifier:** Paired delta ≤ 0; also check the clip factor stayed moderate (if it clips again, the test was invalid, not the idea).
**Risk:** Variance head can latch onto the propensity-|y| correlation and just re-implement |y| weighting; monitor corr(σ̂², |y|).

### A4. Structural decomposition of the two targets (pre-analysis + 1–2 fits)
**Change:** Given "t0 = F − ask, t1 = bid − F for a future F", compute d = (t0−t1)/2 = F − mid and s = (t0+t1)/2 ≈ −spread/2. Check empirically how much of s is predictable from the 112 causal features (ridge/GBM on train+valid). If s has high per-row R², retrain the GRU on d only and reconstruct (p0, p1) = (d̂+ŝ, −d̂+ŝ).
**Why:** This halves the target variance the recurrent net must model and removes a structural component it currently wastes capacity approximating. The metric allows a global affine per target, so splicing s is fully legal. Note this is *not* the refuted sum-channel-zero regularizer (a hard prior that the sum is zero); it is a data-driven check of a stated structural relation.
**Test:** 0 fits for the predictability analysis (CPU); 1–2 fits only if ŝ beats your model's implied sum-channel accuracy by a clear margin.
**Falsifier:** s is already effectively captured by canonical inputs — reconstruction changes nothing; predictability analysis shows R²(ŝ) ≈ what the current p0+p1 head gets.
**Risk:** "Structurally like" may be inexact for anonymized targets; the −0.0016 sum-zero result hints the sum channel carries real signal, so only trust the measured R².

### A5. Gated two-head mixture for the high-|y| class
**Change:** Two output heads with a causal gate g(x) = sigmoid(MLP(features)); output p = (1−g)·p_A + g·p_B; same weighted loss.
**Why:** One mass class carries 43–48% of metric weight with WP ≈ 0.43 vs 0.74–0.81 elsewhere, while an oracle-branch readout reaches 0.968. That gap is a representation/branching problem, not an objective-constant problem: a single output must interpolate shapes for regimes with different conditional geometry. A mixture decouples them while remaining a pure function of causal inputs (metric-legal; per-*state* rescaling stays off).
**Test:** Before any fit, 0-fit check: AUC of predicting the heavy-class membership/|y| quantile from x. If the gate input is barely predictable (AUC ≲ 0.75), the whole idea is capped — spend the fits only if it clears. Then 2 fits (F0 ×2 seeds).
**Falsifier:** Heavy-class WP unchanged and/or gate collapses to one head (monitor g entropy). Also run one arm with a=0 (raw labels): at a=0.9 the teacher may already obliterate the regime distinction the gate needs.
**Risk:** Optimizing g under distillation is unstable; a learned mixture can degrade the 74–81% bulk that never needed help. Variant arm: instead of a mixture head, train a small *separate* heavy-decile specialist GRU (rows top-|y| decile, 5–10× upweight) and splice by the same gate — keeps the main endpoint untouched.

### A6. Per-target loss denominators
**Change:** Currently both targets are pooled into one Σw denominator; the metric averages two pooled correlations. Normalize each target's loss by its own mass.
**Why:** If t1 carries 2× the weight mass of t0, t0's effective gradient magnitude (hence effective LR) is silently halved — a pure optimization imbalance invisible to the Bayes-consistency argument, which only pins the *minimizer*, not the optimizer's per-target dynamics.
**Test:** 0–1 fits: implementable as a short finetune from an existing checkpoint (2–3k steps), which is a legitimate variance-ancillary screen; escalate to full fits only if positive.
**Falsifier:** Delta ≈ 0 (masses already similar — measure them first, it may resolve in 10 minutes of CPU).
**Risk:** Minimal; worst case negligible.

### A7. Small unweighted-MSE auxiliary term
**Change:** Loss = WMSE(|y|·ĝ) + λ·MSE(unweighted), λ ≈ 0.1–0.2, same outputs.
**Why:** The |y| weighting makes gradients from the heavy class dominate ∝ |y|, so common-regime representations get ~no gradient pressure even though the heavy class's 0.43 WP is plausibly a *representation* transfer problem: features learned only for rare rows are noisy estimates. A small unweighted term is an implicit prior/regularizer on the encoder that costs nothing at the optimum's minimizer (it shares the same conditional; minimizers coincide up to the affine family), so the Bayes note is untouched — it's another variance-reduction lever.
**Test:** 2 fits (λ=0.15, F0 ×2 seeds).
**Falsifier:** Paired delta ≤ 0.
**Risk:** λ too large re-tilts gradient mass back to the bulk; keep λ small and check heavy-class WP separately, not just the pooled mean.

### A8. Resampled batch composition ∝ w (sampler-based weighting; unconventional)
**Change:** Instead of weighting the loss, draw each batch's rows with probability ∝ |y|·ĝ and use the plain unweighted mean squared error on y_clip within the drawn batch.
**Why:** Mathematically a close cousin of the current estimator (both unbiased for the same conditional), but it stabilizes the denominator/effective batch size per step and, with multiplicative jitter on the sampling weight during training, doubles as augmentation on the weight-dependent gradient path. This is the same "attack variance, not the conditional" thesis as A1, through a different mechanism — running both together discriminates *where* your WMSE variance comes from (batch composition vs absolute scale).
**Test:** 1 fit (F0, 1 seed), only after A1's result lands so the pair is interpretive.
**Falsifier:** Delta ≤ 0 or no difference vs A1 (then composition wasn't the binding constraint).
**Risk:** Sampling w/o replacement per sequence epoch changes coverage per epoch and interacts with LR schedules; keep the epoch layout fixed and identical to control.

## Part B — Red team

**(1) Why NLL, Pearson losses, and distillation-loss variants failed.** Your explanation — objective consistent, variance-limited — is half right but your experiments didn't actually test it. NLL failed *at the optimizer*: clip factor 0.04 means the measured comparison is "WMSE vs a crippled LM run", not two consistent objectives under matched optimization (A3 is the corrected version of that test). Per-chunk Pearson failed because batch correlation is a noisy, batch-coupled estimator of pooled correlation — again an optimization-noise story, which is *your* thesis, yet you filed it as the objective being refuted. Distillation-loss variants at a=0.9 are structurally powerless: the target is 90% teacher, so student-vs-teacher-matching loss changes only touch the residual 10% of the target signal — expected deltas ~0.0001, exactly what you measured. The honest summary: **you have no negative evidence about loss design for the 100%-signal regime; you have negative evidence for three confounded or underpowered probes.** One genuine caveat to the Bayes note: consistency is about the population minimizer, but you optimize with estimated weights ĝ(x)·|y|, which injects multiplicative gradient noise — so variance-attacking loss changes (A1, A3, A6, A8) are exactly the class your note says can win.

**(2) The B300 Transformer overfit.** Read the numbers: valid WMSE 1.55 with WP 0.31–0.41 implies total weighted variance ~1.8, so the 1.54B model is *in the right regime, not absurd* — GRU students at WP 0.68 imply WMSE ~0.96 on the same scale. The failure is protocol, not architecture: LR 3e-4 for 30 epochs on a model 100× the effective-data-appropriate size, no early stopping, no kept early checkpoints (overwritten — that destroyed the only informative evidence in the run). Redesign before any re-fit: (a) 30–200M params; (b) dropout 0.1–0.3, weight decay 0.1; (c) LR 3e-5–1e-4 with cosine; (d) early stopping on *validation WP per epoch* (your actual metric), not WMSE — WP and WMSE rank checkpoints differently under the clip; (e) save every epoch checkpoint permanently; (f) augment at the sequence level: random chunk offsets/starts from 20k-row sequences so the net sees many alignments per sequence; (g) before any 30-epoch spend, run a 10-minute 100-step actual-device profile. Gate: a teacher earns distillation only at teacher WP ≥ ~0.55–0.60 on a clean fold; below that, distilling at a=0.9 drags the student and the student is already 0.68. Also reconsider a: with a weak teacher, a=0.5 or residual-only distillation is the informative arm.

**(3) Most likely wrong assumption.** The brief's framing "gains must come from variance" quietly assumes that *local CV endpoint deltas at the 3rd–4th decimal rank real improvements*. Public 0.668 vs local 0.682, a 22-rank spread of 0.0041, and seed noise 0.0007–0.0014 together suggest your measurement floor is at or above the effect sizes you've been chasing — for endpoints trained on different (fold, seed) cells, the selection noise across many tested variants is larger than 0.0014, so several of your recent "+0.0001–0.0004 gains" are likely sign-random. The runner-up suspect: ĝ (scored-propensity) is validated only on local data; if the hidden test's realized scoring mask deviates from the propensity model, both training weights *and* your internal class-weight accounting (the 43–48% figure) shift, and member ranking changes. Derive propensity uncertainty, and check endpoints against the realized valid mask vs ĝ-mask — a ranking flip there is a red flag worth more than any single loss screen.

**(4) Stop doing.**
- Stop distillation-loss variant screens at a=0.9 — underpowered by construction; only mean-side or representation-side changes can register.
- Stop 630M–1.54B teacher fits until checkpoints are preserved and the regularization/early-stop protocol exists; that program is currently burning B300 hours to produce overwritten evidence.
- Stop scoring "+0.0001–0.0004" as gains. Adopt: accept only deltas ≥ 2× seed-noise, replicated on ≥ 2 folds × ≥ 2 seeds; store everything else as inconclusive.
- Stop overwriting epoch checkpoints — permanently, as a written rule; it already cost you the teacher-scaling evidence.
- Stop corr-vs-WMSE re-litigations (both arms have terminally measured results) and stop Muon revisits (−0.005, mechanism-contradicted by your own stable-rank observation).
- Stop treating per-batch weight mass variance as a footnote: it is 10×, it is your most cited ungraded code artifact, and A1 tests it in 2 fits.

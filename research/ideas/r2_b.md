# Round 2-B: Why GRU saturates & missing information (Haiku ideas)

## Framing
GRU plateaued at 0.671–0.672 devB (~0.654 LB); 114 heuristic features gave zero linear lift; symreg finds ~0.11 corr residual terms in classes 0/1 (trade-flow driven). Class-3 branch AUC jumps 0.62→0.96 near moves; is_scored predictable but ~20% target variance remains unexplained. Why saturated? What's the GRU missing?

---

## 1. **Regime-mode bottleneck: Quiet vs. active (unconventional)**
**Concept + source**: Class-3 AUC far-from-move 0.62–0.70 vs. near-move 0.96–0.99 (tree A1 note; E-090). The GRU shares a single 128-d hidden state across ALL rows, but the prediction task has two distinct regimes: (a) quiet/stale (WP 0.36 after 170+ silent rows; 1/3 weight loss), (b) active (near price-change events). Each regime needs different attention to features. A single GRU pathway cannot factorize.

**Analogy to our data**: Like Kyle-Glosten toxicity: informed traders' flow looks different from uninformed noise. The GRU tries to average; the cost is saturation.

**Testable hypothesis**: If we inject a 1-hot "regime flag" (quiet=1 if last 50 rows silent, else 0) as a discrete input, and let a 2-head MoE (mixture-of-experts) route GRU output through regime-specific readouts, we gain ≥ +0.0015 devB. Falsified if gain < +0.0005.

**Cheapest test**: Modify scripts/train_rnn.py to add regime flag (Bash 5 min). Run 1 epoch debug fit. No new model code needed; use auxiliary head pattern (already in nn.py). Expected cost: 15 min GPU. Predicted outcome: +0.001–0.003 (low because GRU may already learn regimes implicitly via time-series memory).

**Falsifier**: Regime flag provides no lift, or gain <noise floor. Suggests GRU internal state already encodes quiet/active distinctions.

**Research tree**: C2 (diverse members), branch "regime-MoE heads for class-3".

---

## 2. **Trade-flow toxicity masking: Informed vs. noise blending (conceptual)**
**Concept + source**: Cartea et al. (Toxic Flow) predict per-trade toxicity (client can unwind ≥ 0). Symreg finds trade-volume EMA16 in small-tick residuals (brief E-019 line 11). Toxicity is *predictability of client loss*, analogous to *predictability of next move*. GRU trains on mixed flow (informed + noise); class-3 far-move residuals show GRU fails when flow is ambiguous (AUC 0.62).

**Analogy**: Our trade-flow features are like "client toxicity score"; they should gate confidence in direction prediction. High-toxicity flow = high entropy direction.

**Testable hypothesis**: If we compute a "trade-flow toxicity proxy" (sign(dv) x spread state x recent-volume regime), segment class-3 rows into high/low toxicity cohorts, and fit separate GRU readouts per cohort, class-3 AUC improves ≥ 0.04 on low-toxicity subset, stays ≤ 0.65 on high-toxicity (reveals GRU *is* picking up on information imbalance). Falsified if both cohorts stay ~0.62.

**Cheapest test**: Use scripts/lift_probe.py. Design `trade_toxicity = sign(dv) * I(spread > 2*median_spread) * log(recent_vol / base_vol)`. Probe it against GRU residuals by class, segmented. ~1 min. Falsifier is flat correlation across cohorts.

**Falsifier**: Trade toxicity has zero coherence with residuals by cohort; GRU residuals white-noise within each toxicity level. Suggests trade flow is already learned or irrelevant.

**Research tree**: B1 (heuristics don't lift linearly) refinement; B3 (i1 cross); branch "per-cohort residual analysis".

---

## 3. **Stale-row entropy trap: Ghost data corrupting state**
**Concept + source**: WP 0.36 after >170 quiet rows (1/3 of weight). Rows with zero trades, constant bid/ask, a5 (time since trade) >> median. The GRU's hidden state still updates on zero-information rows, accumulating numerical noise and corrupting memory. This is a type of "gradient noise" baked into the data.

**Analogy**: In NLP, padding tokens still update RNN state; they introduce spurious gradients. Stale rows are high-dimensional NaN-like signals.

**Testable hypothesis**: If we mask gradients on stale rows (a5 > 300 ticks AND dv=0 AND bid/ask=constant) during training, training loss smooths (lower variance), validation WP on stale rows improves ≥ 0.03, and overall devB improves ≥ +0.0008. Falsified if gradient masking hurts or stale-row WP stays ≤ 0.36.

**Cheapest test**: Modify scripts/train_rnn.py loss function. Add a mask that zeroes gradients for stale rows. 1 epoch debug fit. Expected cost: 5 min GPU. Predicted outcome: modest gain (0.0005–0.001) because downstream loss normalization may already handle it implicitly.

**Falsifier**: Masking worsens training (e.g., loss diverges or validation fails). Suggests GRU relies on stale-row structure for regularization (unlikely but possible).

**Research tree**: B4 (stale-row regime); C1 (GRU recipe refinement).

---

## 4. **Look-ahead leak in symreg closed-forms (validation idea)**
**Concept + source**: Symreg found residual terms corr ~0.11 in classes 0/1 (E-019). Symreg scans all rows to find patterns. But causal constraints are not enforced in the symbolic search. The terms might rely on future ticks (e.g., min(v9, 0) could encode a future imbalance sign). A causal term trained on past data should *generalize*, while a future-dependent term should *fail* on a later test split.

**Analogy**: From toxic-flow: a prediction at time t must use only [t0, t] data. Symreg terms trained on global data may violate causality.

**Testable hypothesis**: If we run symreg on train+valid (full history), extract top-N terms, then test them on a held-out "future" split (e.g., last 5% of sequences unseen during symreg), top terms lose > 50% correlation. If correlation is preserved, terms are robust; if it collapses, they're look-ahead artifacts. Falsified if correlation drops < 20%.

**Cheapest test**: Use src/wnn/symreg.py. Split data: train+val → symreg engine, extract top 5 terms, apply them to test-split residuals in scripts/decide.py. Compare corr on unseen splits. ~2 min. Falsifier is high correlation on future data (suggests true causality).

**Falsifier**: Top symreg terms maintain correlation on future data. Means they generalize; causality is likely true.

**Research tree**: A4 (symbolic explanations); D1 (validation methodology).

---

## 5. **Class-3 tick-path memorization overfitting: GRU learns surface patterns, misses invariants**
**Concept + source**: Class-3 has exact tick law (E-079): F = round(mean mid next 4 book events). This is fully deterministic given 4-step future ticks. But the GRU sees snapshots (current orderbook only). It might memorize *which tick sequences are common* without learning the underlying spread/direction mechanics. Memorization saturates at a low accuracy ceiling.

**Analogy**: Like learning to classify digits by their pixel patterns (works on train set) vs. learning the true digit generatively.

**Testable hypothesis**: If we compute "GRU fluency on tick-law" (fraction of rows where GRU correctly predicts the 3-way branch: F inside [bid,ask] / F > ask / F < bid; ignoring magnitude), and compare it to a "random baseline" (predict based on recent spread direction only), GRU fluency is > 0.65 but < 0.80. If GRU had learned the true mechanics, fluency should be > 0.90. Low fluency = memorization. Falsified if fluency > 0.85 (suggests GRU learned mechanics).

**Cheapest test**: Compute predictions on class-3 val subset. Decode the 3-way branch from GRU outputs (t0/t1 signs and magnitudes). Compute AUC on branch alone. Compare to baseline (constant recent-spread direction). ~1 min in scripts/decide.py. Falsifier is high AUC (>0.80) on branch task.

**Falsifier**: GRU branch AUC >= 0.78. Means GRU learned the predictable signal; saturation is not due to memorization but to actual information limits.

**Research tree**: A1 (target law); B5 (class-3 exact tick-path features).

---

## 6. **is_scored as a confounding proxy: Train-test shift in mask distribution (validation idea)**
**Concept + source**: is_scored predictable from features (AUC ~0.9; E-087). Test mask is ~2% softer than val (lambda=0.02 in devB metric). But we don't know if the CLASS or SEQUENCE mix of scored rows changes between train, val, test. If test has more class-0/1 rows (which have is_scored ~ 100%) vs. more class-3 rows (which have is_scored ~ 50%), the GRU's learned is_scored classifier becomes a confound: it predicts the direction indirectly by learning is_scored, not by learning true direction mechanics.

**Analogy**: Proxy confounding in causal inference. The GRU learns to predict is_scored > predict direction.

**Testable hypothesis**: If we train a separate logistic regression to predict is_scored from features on train, then use it to reweight validation rows by 1 / p(is_scored | x) (inverse-propensity weighting), the metric variance on val drops > 20% (less spurious variance), and reweighted val WP better predicts test WP (correlation of per-sequence WP > 0.75 vs. raw ~0.60). Falsified if reweighting worsens rank correlation.

**Cheapest test**: Fit logistic regression on train (is_scored ~ all features). Compute propensity weights on val. Reweight val sequences in scripts/decide.py. Compute correlation of per-seq WP_raw vs. WP_reweighted with test. ~5 min. Falsifier is correlation drops (suggests is_scored isn't a confound).

**Falsifier**: Propensity reweighting doesn't improve rank correlation. Means is_scored isn't a hidden confound; it's independent or GRU already marginalizes it.

**Research tree**: A3 (scoring mask process); A5 (train-vs-test shift).

---

## 7. **Time-delta opacity: GRU processes rows uniformly despite irregular spacing**
**Concept + source**: Rows arrive at irregular intervals (microseconds to seconds). The GRU's recurrence processes every row as "one timestep", losing information about elapsed time. In markets, a 1-ms gap is *very* different from a 1-s gap; order book state decays differently. The GRU might be forced to store time-gaps implicitly (harder to learn), or ignore them (losing signal).

**Analogy**: Like predicting speech without phoneme timing; the model learns average effects, not rhythm-dependent effects.

**Testable hypothesis**: If we add a "time-delta feature" (log10(1 + time_since_last_row_ms)) as an explicit input, a 1-epoch GRU train shows +0.0015 devB gain. If gain < +0.0005, GRU already encodes timing implicitly. Falsified if gain is +0.005 or more (suggests timing was a major gap).

**Cheapest test**: Add time-delta to feature set. Run 1-epoch debug fit in scripts/train_rnn.py. Cost: 5 min GPU. Predicted outcome: +0.0004–0.001 (modest, since GRU can infer timing from feature trajectories).

**Falsifier**: Time-delta hurts (e.g., validation WP drops). Suggests it's noise or correlation with other features.

**Research tree**: B1 (feature engineering); C1 (GRU architecture).

---

## 8. **Multi-horizon prediction as implicit capacity drain: t0 vs t1 asymmetry is model-agnostic** (unconventional)
**Concept + source**: Brief notes t0/t1 asymmetry (t1 easier: 0.672 vs. 0.658 globally). The GRU must predict two targets from one hidden state. If t0 and t1 require different "attention signatures" (different feature weights), the GRU's single 128-d state cannot hold both sharply, leading to compromise predictions and saturation.

**Analogy**: Multi-task learning bottleneck: shared representation struggles when tasks have antagonistic gradient signals.

**Testable hypothesis**: If we train a separate single-headed GRU for t0 alone (remove t1) and one for t1 alone, each achieves WP >= 0.660 (vs. joint ~0.658). If single-target models saturate at same point as joint model, bottleneck is NOT multi-task; it's information-limited. Falsified if single-task models reach ~0.670 each.

**Cheapest test**: Modify scripts/train_rnn.py to train 2 separate 1-epoch models (t0-only, t1-only). Cost: 10 min GPU. Falsifier is high single-target WP (>= 0.665 each), suggesting multi-task isn't the limiting factor.

**Falsifier**: Single-target models also saturate at ~0.658 WP. Means the bottleneck is not multi-task; it's raw information in the features themselves.

**Research tree**: C1 (GRU recipe); C3 (multi-head averaging).

---

## 9. **Symbolic regression as a leakage detector: Closing-the-loop hypothesis**
**Concept + source**: Symreg found terms like min(v9, 0) - tanh(trade-volume EMA16) with corr ~0.11. If these terms are truly causal (not future-dependent), adding them to GRU inputs should give linear lift. They didn't (E-088). This suggests: (a) terms are artifacts/noise, (b) terms are future-dependent (look-ahead), or (c) GRU already learned them implicitly. A direct causality test: regress residuals on symreg terms with 10-fold CV; if CV-error < test-error by > 10%, terms are fitting train-specific noise.

**Analogy**: Stability testing in feature selection (does the feature survive cross-validation?).

**Testable hypothesis**: Extract top 10 symreg terms. On train, fit OLS regression (residual ~ top-10-terms). Compute in-sample R^2 and OOF R^2 (5-fold CV). If |R^2_in - R^2_OOF| > 0.008, terms are overfit artifacts. If |diff| <= 0.002, terms are stable (likely real, but GRU doesn't use them). Falsified if OOF R^2 > 0.005 (terms predict reliably).

**Cheapest test**: Extract symreg terms from cache/heur_valid.npz or rerun src/wnn/symreg.py on train. Fit OLS in 5-fold CV loop. ~2 min. Falsifier is stable CV R^2 (suggests terms are real signals GRU ignores).

**Falsifier**: Symreg terms show high OOF R^2 > 0.004. Means they're robust; the mystery is why GRU doesn't learn them (capacity issue? architecture issue?).

**Research tree**: A4 (symbolic explanations); B1 (why heuristics don't lift).

---

## 10. **Adaptive mixture-of-sequences: Class-3 needs a different inductive bias**
**Concept + source**: Class-3 WP ~0.42–0.44 vs. 0.72–0.79 for classes 0–2 (a 10x difference in hardness). The GRU is agnostic to class; it learns one shared feature-to-hidden mapping. But class-3 direction is governed by precise tick mechanics (spread + recent-move sign), while classes 0–2 follow trend-like patterns. Classes need different *inductive biases*, not just different readout heads.

**Analogy**: A model trained on chess and poker together will saturate; they need different world models.

**Testable hypothesis**: If we train a 2-head GRU where Head-A processes class-3 rows (attention over last 50 rows for tick-path context) and Head-B processes classes 0–2 (attention over last 500 rows for trend context), with a routing gate (per-class), we gain ≥ +0.003 devB. Falsified if gain < +0.001.

**Cheapest test**: Sketch routing in nn.py. Add per-class affine transformation before final readout. Run 1-epoch fit. Cost: 15 min GPU. Predicted outcome: +0.002–0.004 (modest because auxiliary class heads already provide some relief).

**Falsifier**: Per-class routing provides no lift or worsens training (e.g., gradient issues).

**Research tree**: C2 (diverse members); B5 (class-3 specialized features).

---

## 11. **Closed-loop validation: Softened mask as a surrogate for test metric (methodology)**
**Concept + source**: Softened mask (lambda=0.02) tracks LB better than hard mask. But we still don't know if it ranks models in the same ORDER as the hidden test. A perfect surrogate should have per-sequence correlation with test > 0.95. If correlation < 0.75, validation is unreliable.

**Testable hypothesis**: If we estimate test-set metric distribution (via public LB and team submissions), then reweight val seqs to match estimated test distribution (e.g., upweight class-3, downweight class-0/1 if test has more large-tick instruments), the reweighted metric's per-sequence ranking of a few diverse models correlates > 0.80 with LB rankings. Falsified if correlation < 0.60.

**Cheapest test**: Fit a multinomial logit on class distribution in LB (vs. val). Reweight val rows. Compute metric on top-3 models with reweighted val. Check if their ranking matches LB. ~5 min. Falsifier is rank inversion (val says model A > B, LB says B > A).

**Falsifier**: Reweighted validation rank is unstable or inverts. Means the gap between val and test is not just a distribution shift; it's a fundamental metric property change.

**Research tree**: D1 (decision rule); D2 (mask-softening).

---

## 12. **Feature-target correlation collapse in class-3: Is the signal even there?** (reality-check)
**Concept + source**: Class-3 branch AUC 0.62 far from moves. That's barely above random (0.5). If the signal is ~arcsin(sqrt(AUC)) ~ 0.3 (correlation units), then any linear learner will struggle. But is the signal *actually there*, or is class-3 inherently random on silent rows?

**Testable hypothesis**: Compute point-biserial correlation between (class-3 direction label) and (best individual feature) on class-3 far-from-move rows. If max corr < 0.15, the task is information-theoretically hard (signal-to-noise ratio is low), and GRU saturation is expected and optimal. If max corr > 0.25, the GRU is missing exploitable signal. Falsified if median-feature corr > 0.30 (high signal available).

**Cheapest test**: For each of 112 features, compute point-biserial corr with t0 direction on class-3 far-from-move subset. Rank features. ~1 min in bash. Falsifier is top feature corr > 0.28 (signal is available; GRU is indeed leaving money on table).

**Falsifier**: All features have corr < 0.12 on class-3 far-from-move rows. Means the GRU is at or near the Bayes limit; no model will do much better. Saturation is *structural*, not algorithmic.

**Research tree**: A5 (shift across train/valid/test); B2 (class-3 specialized approaches).

---

## Summary (research tree integration)
All ideas test the same hypothesis in different ways: *Why is GRU saturated?* Answers:
- **Regime opacity** (idea 1): GRU can't factorize quiet vs. active.
- **Flow masking** (idea 2): Trade-flow entropy gates branch predictability.
- **Stale-row noise** (idea 3): Ghost data corrupts state.
- **Look-ahead corruption** (idea 4, 9): Symreg finds artifacts, not true signals.
- **Tick-memorization** (idea 5): GRU learns surface patterns, not mechanics.
- **Confounding masks** (idea 6): is_scored is a hidden confounder.
- **Time opacity** (idea 7): Timing is lost or implicit.
- **Task conflict** (idea 8): t0/t1 multi-task bottleneck.
- **Class-heterogeneity** (idea 10): Classes need different inductive biases.
- **Distribution shift** (idea 11): Validation metric is a poor test proxy.
- **Signal weakness** (idea 12): Class-3 far-move signal is weak by design.

Next: Run ideas 3, 4, 9, 12 (1–2 min each; cheapest probes). If any return strong signal (idea 12 finds corr > 0.25, idea 9 finds stable symreg, idea 4 finds look-ahead), escalate to modeling tests (ideas 1, 5, 10).

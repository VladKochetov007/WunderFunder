# Reframe 29: five mechanisms (Opus, fresh context)

Blend-law arithmetic used throughout: for k=3, dropping ρ̄ from 0.995 to 0.985 adds about 0.336·(2/3)·0.010 ≈ +0.0022, provided member strength holds. So diversity at constant strength is worth chasing. Diversity bought with weaker members (as TabM did) is not.

## 1. Gap ledger, plus data-scaling slope (diagnostic; family: measurement)
- **Mechanism:** split the remaining error into components that each call for a different next step: (a) teacher ensemble vs student (distillation gap); (b) best single member vs the oracle per-class blend (combination gap); (c) score vs training-set size (data gap).
- **Hypothesis:** the work is going to the wrong component. If (a) is ≥0.004, invest in student capacity up to 20 MB or in the distillation schedule. If (c) is steep, the final train+valid refit and any augmentation matter more than architecture.
- **Ceiling / zero-fit:** (a) and (b) come straight from existing OOF predictions: WP of the teacher-ensemble average vs the student, and WP with per-class least-squares blend weights fitted on fold A and scored on fold B. (c) comes from existing checkpoints trained on different data fractions, if any exist.
- **Transfer:** none. It decides where the next 20 fits go.
- **Cheapest test:** zero fits for (a) and (b). Two fits for (c): the same recipe at 50% and 75% of train groups, evaluated on a fixed fold.
- **Falsifier:** all three gaps are under 0.002. Then the ceiling is in the teacher signal itself, so go to #3 and #4.
- **Cost:** about 1 CPU-hour, plus 2 GPU fits.

## 2. Class- and regime-conditional fixed blend with per-class gain (family: post-hoc combination)
- **Mechanism:** replace the single coefficient vector with coefficients w[class, phase], where phase is a coarse bucket of sequence position or time since the last book event. Add a per-class output scale s[class]. Pooled WP is not scale-invariant across classes: class 3 (WP ≈0.4) is probably over-scaled relative to its signal, and that costs pooled correlation.
- **Hypothesis:** the members' relative strength differs by class (Transformer members vs the GRU student), and the shared scale is miscalibrated across regimes.
- **Ceiling / zero-fit:** fit w and s on OOF fold A, evaluate on folds B and C (about 12–40 parameters). Report the cross-fold delta and its spread.
- **Transfer:** a lookup table in the ZIP. Class has to be identified causally, from the warm-up rows or a known column, at no runtime cost.
- **Cheapest test:** zero fits.
- **Falsifier:** cross-fold delta under +0.001, or the sign flips across folds.
- **Cost:** minutes.

## 3. Event-clock state and output holding (distant field: event-triggered control and spiking neurons)
- **Correspondence:** in event-triggered control (Åström-style sampling) the estimator updates only when an event crosses a threshold. Phased LSTM (arXiv:1610.09513) gates updates by a time oscillator. Here the targets change only at book events, so the optimal prediction is piecewise-constant between events, apart from information arriving in between.
- **Mechanism:** a two-clock GRU. A fast per-row cell feeds a slow cell that updates only on event rows, through a learned soft gate driven by an event indicator. The output head reads mostly from the slow cell. Alternatively, a cheaper causal output filter: an exponential moving average of predictions that resets at each event.
- **Hypothesis:** the student puts out between-event jitter that is pure variance, and the targets cannot track it.
- **Ceiling / zero-fit:** take existing OOF predictions and apply a causal within-segment EMA, sweeping α on fold A and scoring on B. Also measure the share of prediction variance that falls inside event segments vs across them.
- **Transfer:** the filter costs about five lines in the serving loop. The architecture variant needs a refit and stays at the same size.
- **Cheapest test:** zero fits for the filter. Then 2+2 fits (control vs two-clock, two folds or seeds) only if the filter gains ≥0.001.
- **Falsifier:** the best α is about 1, meaning predictions already move only at events.
- **Cost:** 1 hour, then 4 fits.

## 4. View-diverse teachers via SpecAugment-style masking (distant field: speech; family: data views for diversity)
- **Correspondence:** SpecAugment (arXiv:1904.08779) masks frequency bands and time spans so that acoustic models stop depending on any single channel. Here the ~200 book columns play the role of bands, and runs of rows play the role of time spans.
- **Mechanism:** train teachers with random column-group masking (contiguous blocks in the original column order, not random single columns) and short time-span masks. Different teachers use different mask distributions. Distil the student from these teachers, and add one masked teacher directly as a blend member.
- **Hypothesis:** same-recipe teachers share failure modes (ρ ≈0.997). Masking spreads reliance across inputs and adds regularisation, so ρ can fall towards 0.985 without the loss in strength that shared weights caused, because these are separate networks. Teammate evidence of memorisation in large Transformers points to regularisation headroom.
- **Ceiling:** the blend law with measured member WP and ρ. From existing checkpoints, column-group occlusion sensitivity per teacher shows whether the teachers depend on the same few groups. High overlap means masking has room to help.
- **Transfer:** a better distilled student, or a direct third member at the same size.
- **Cheapest test:** 4 fits, masked vs unmasked GRU teacher × 2 folds. Measure single-model WP and ρ against the existing members.
- **Falsifier:** member WP drops more than 0.002, or ρ to the incumbents stays ≥0.994.
- **Cost:** 4 GPU fits.

## 5. Causal per-sequence normalisation from the warm-up window (family: distribution shift / instance adaptation)
- **Mechanism:** use the 99 unscored warm-up rows to estimate per-sequence feature location and scale, plus a sequence embedding, and condition the network on them. This is FiLM or reversible instance normalisation done causally, frozen after row 99 or updated causally.
- **Hypothesis:** much of the class gap comes from regime-dependent scaling. A network that sees globally normalised inputs spends capacity re-inferring the regime row by row.
- **Ceiling / zero-fit:** regress OOF residuals, or per-sequence WP, on warm-up summary statistics such as per-column volatility and sentinel rate, with a cross-fitted ridge. The out-of-fold R² of per-sequence WP or residual scale bounds the gain.
- **Transfer:** about 200 running statistics per sequence, negligible size and time. It resets with the sequence state.
- **Cheapest test:** zero-fit regression first. Then 2+2 fits (FiLM-conditioned vs control) if the residual R² is ≥0.02.
- **Falsifier:** the warm-up statistics do not predict residuals out of fold.
- **Cost:** 1 hour, then 4 fits.

**Order:** #1 and #2 first (zero-fit, decide everything else), then the #3 filter, then the #5 regression, then #4 fits.

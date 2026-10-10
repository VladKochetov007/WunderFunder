# Round 3 — Eight risky ideas (low probability, high payoff)

## Reframe first

Every lever we have pulled has returned ±0.002, and four teacher families agree at correlation 0.95–0.99. That agreement is the whole diagnosis: we are not under-optimizing a loose problem, we are hitting a **shared information ceiling on the causal features**, and the residual loss is concentrated where it hurts — class 3 (half the metric weight, corr 0.44, weight in runs older than 16 rows) and long runs (weighted sign accuracy decays 90–92% → 74% past 512 rows). The oracle experiment sharpens this: run-averaging the true boundaries gains +0.025 on class 3 but *loses* 0.008 pooled, so the prize is not smoothing and not the run mean — it is getting more of the within-run sign dynamics right from *past-only* features.

So the only ideas worth spending experimental oxygen on are ones that either **(a) raise the information ceiling** (a new observable, a new external relation between rows/sequences, a recovered state) or **(b) change the effective data economy of class 3**. Nine of the ten ideas below are in those two families; the tenth tests whether the plateau is an optimization artifact after all. Each is stated with the mechanism in three sentences, the cheapest 1-day falsifier with a numeric go/no-go, the expected effect, the rough P(success)/gain, and the failure mode I fear most.

---

## IDEA 1 (DATA) — "One long tape": the sequences are windows of a shared continuous process

**Mechanism.** The folds are *component-disjoint*, which hints sequences are grouped into a small number of continuous components rather than drawn i.i.d.; if so, sequence *i*'s head may continue (or literally overlap) sequence *j*'s tail from the same physical timeline. That has three consequences: the unscored warm-up rows 0–98 inherit a run state from rows *before* row 0, the scored region can be entered with a labeled run state instead of a cold one, and the same physical rows may appear in two sequences so labels leak across the split. Test it by aligning every sequence head against every sequence tail (z-scored per column, allowing block-clock phase shifts).

**Cheapest 1-day falsifier + numeric go/no-go.** For each sequence, correlate its first 200 feature rows against the last 200 rows of all other sequences, per block clock. **Go if ≥1% of sequences have a boundary match at correlation > 0.99 (or a consistent integer offset), i.e. a rate ≫ the 1/12000 ≈ 0.008% chance rate**, and a linear probe separates true tail→head pairs from random pairs at >65% accuracy. Anything near the chance rate means i.i.d. windows and the idea dies.

**Expected effect.** Bimodal: 0 if independent; **+0.01 to +0.03 pooled if contiguous or overlapping** (extended causal warm-up; possibly recovered labels on the 0–98 prefix). **P(success) ≈ 15%, gain conditional on success ≈ +0.01 to +0.03.**

**Failure mode I fear most.** The "component" structure is only class identity, the boundary correlation is pure noise, and the apparent grouping is an artifact of how folds were sliced. Secondary fear: even if contiguous, the 99 unscored warm-up rows already saturate the state, so carry-in only rescues the first ~100 scored rows and the gain evaporates.

---

## IDEA 2 (DATA) — Duplicate/replay mining and label transplant

**Mechanism.** If the corpus was built by replaying a fixed library of scenarios with per-replay randomization, some validation sequences have a near-twin in train whose label trajectory transfers almost exactly. Locality-sensitive hashing over the fully observable unscored prefix (rows 0–98) plus a partial-trajectory hash finds those twins; transplanting their labels (or using them as a strong prior) is near-oracle on the affected rows. This is a pure data-leak lottery with an unbounded ceiling and near-zero downside.

**Cheapest 1-day falsifier + numeric go/no-go.** MinHash/LSH over binarized prefixes and partial trajectories; for each of the 1,873 val sequences record the nearest train distance. **Go if ≥2% of val sequences have a twin with prefix relative-L2 < 1e-3 or trajectory Jaccard > 0.95**; then confirm the transplant actually raises WP on that subset above its size share.

**Expected effect.** Unknown, bimodal: **0 if unique, up to +0.01 to +0.04 pooled if a meaningful subset is duplicated.** **P(success) ≈ 10%, gain conditional on success ≈ +0.01 to +0.04.**

**Failure mode I fear most.** Twins exist in *features* but their labels were re-drawn with different randomization, so transplant silently poisons exactly the rows we weight most. Any twin, even a loose one, must be label-verified on train before it is trusted on validation.

---

## IDEA 3 (DATA) — Recover the low-rank latent behind the 112 columns and denoise noisy copies

**Mechanism.** 112 anonymous columns with clean multi-rate clocks may be a small number of latent factors pushed through per-column monotone nonlinearities plus per-column noise; if so, several columns are noisy copies of one factor and can be averaged, and the whole input can be replaced by the denoised latent. Studying the singular-value spectrum isolates the shared factors from the idiosyncratic sensor noise, raising the SNR without touching the model. If the columns remain predictive when projected onto a handful of components, the discarded dimensions were never signal.

**Cheapest 1-day falsifier + numeric go/no-go.** Per-sequence PCA on the 112 columns, report the cumulative variance curve and the participation ratio. **Go if the top 16 components explain ≥95% of variance (rank ≪ 112) AND a latent-only GRU retains sign accuracy within 2 points of the full-input GRU** — i.e. the extra dimensions are noise. Else the columns are near-independent sensors and there is nothing to denoise.

**Expected effect.** If a genuine latent exists, **+0.005 to +0.02**, mostly through long-run sign accuracy. **P(success) ≈ 12%, gain conditional on success ≈ +0.01.**

**Failure mode I fear most.** The true generative rank is high: each column is its own noisy driver, the top-16 projection throws away exactly the rare high-frequency events that carry the sign flips, and denoising buys nothing or actively hurts.

---

## IDEA 4 (TRAINING) — Servable weight-space merging of a seed population (mode connectivity)

**Mechanism.** Seed averaging gives +0.0015–0.002 per member but is not servable; a single *merged* net is. Align the hidden units of many independently trained width-256 GRUs by matching activations (permutation / optimal transport, à la Git Re-Basin), then merge weights into one width-256 net. If the independently trained solutions lie on a shared low-loss basin, the single merged net captures much of the ensemble gain at zero extra serving cost.

**Cheapest 1-day falsifier + numeric go/no-go.** Train four width-256 GRUs (different seeds), align pairwise, and measure the loss barrier along the linear interpolation on a held-out fold. **Go if the barrier is < 5% of the endpoint loss AND the merged net beats the best single seed by ≥ +0.006 WP on fold 1.** Otherwise abandon the family.

**Expected effect.** **+0.004 to +0.015 servable** (single merged net vs single seed). **P(success) ≈ 12%, gain conditional on success ≈ +0.01 to +0.015.**

**Failure mode I fear most.** At this depth recurrent nets do not share a linear connectivity basin; the permutation search is itself ambiguous in a GRU, and the "merge" silently collapses toward the best member (+0.002, under the bar) or below the mean.

---

## IDEA 5 (TRAINING) — Meta-learned state priming: a hypernetwork sets h0 from the unlabeled warm-up

**Mechanism.** The market class is decodable at 99.4% from the first 500 rows, so each sequence carries a per-sequence latent that a fixed initial hidden state cannot express. Train a hypernetwork that maps the first 99 (causal, unscored) feature rows to the GRU's h0, meta-learned with a leave-sequences-out episodic protocol so it generalizes across sequences rather than memorizing them. At serving it costs one extra forward pass and nothing per row.

**Cheapest 1-day falsifier + numeric go/no-go.** Freeze a trained GRU and train *only* the h0-hypernet on fold 1 for one day. **Go if val WP ≥ best fixed-h0 student + 0.008 on fold 1, with the gain concentrated in class 3.** If the gain is uniform across classes, it is just extra capacity noise and not a state-priming effect.

**Expected effect.** **+0.005 to +0.02, concentrated in class 3.** **P(success) ≈ 12%, gain conditional on success ≈ +0.01.**

**Failure mode I fear most.** The GRU's own 99 warm-up steps already absorb the class signal, so priming h0 is redundant; or the hypernet memorizes a per-sequence h0 key and the fold-1 gain does not transfer to the hidden test (a classic meta-overfit).

---

## IDEA 6 (TRAINING) — A learned optimizer meta-trained on the pooled weighted-Pearson metric

**Mechanism.** AdamW/Muon and schedules are exhausted (Muon +0.004), but those are hand-designed rules optimizing the *training* loss, not the *reported metric*. Meta-train a small optimizer network over many short GRU runs with the held-out pooled weighted-Pearson as the meta-objective, so the update rule can exploit the metric's sign/weight structure directly. Apply the learned rule to the real width-256 GRU.

**Cheapest 1-day falsifier + numeric go/no-go.** One day of meta-training on a cheap proxy (fewer sequences, shorter unrolls), then apply to the real run. **Go if final fold-1 WP ≥ Muon + 0.006.** Else drop.

**Expected effect.** **+0.004 to +0.015, uncertain.** This idea's *diagnostic* value is high: it directly tests whether the plateau is optimization or information. **P(success) ≈ 10%, gain conditional on success ≈ +0.01.**

**Failure mode I fear most.** Proxy→full transfer fails, or the rule finds a slightly different point on the same plateau (≈ +0.001). If a better optimizer does not clear the shared teacher wall, that is strong evidence the ceiling is informational and every training-side idea below should be de-prioritized.

---

## IDEA 7 (TRAINING) — Clock-phase self-supervision: predict each block's "rows since last update"

**Mechanism.** Every column is piecewise constant and updates on a fixed clock, so each block's phase (rows since last update, and the next-update indicator) is a fully observable, label-free target. Add auxiliary heads that predict, at each step, the elapsed time since each block last changed and the block's change indicator, forcing the recurrence to explicitly encode clock phase. Clock phase is the model's only internal proxy for run age, which is exactly where its error concentrates.

**Cheapest 1-day falsifier + numeric go/no-go.** Bolt the auxiliary heads onto the existing 256-GRU and train on fold 1. **Go if class-3 weighted sign accuracy rises ≥2 points (e.g. 80.7 → 82.7) without lowering pooled WP.** Else the model already encodes phase implicitly.

**Expected effect.** **+0.005 to +0.02** as a representation fix rather than a data fix. **P(success) ≈ 15%, gain conditional on success ≈ +0.01.**

**Failure mode I fear most.** The repeated rows *literally are* the clock signal, so the GRU infers phase for free and the auxiliary task adds nothing — in which case the long-run sign error is genuinely information-limited and this whole family is dead.

---

## IDEA 8 (TRAINING) — Programmable-regime generative simulator for class-3 long-run data

**Mechanism.** Class 3 is data-limited (half the data costs 0.009) yet holds half the weight and has the worst correlation, and hidden-test loss concentrates there. Fit a generative model of the feature process (per-block update distributions, change magnitudes, cross-block correlations) *conditioned on a programmed label trajectory*, then synthesize additional long-run class-3 sequences with their labels and add them to training. This multiplies the scarce resource (class-3 long-run rows) instead of touching model capacity.

**Cheapest 1-day falsifier + numeric go/no-go.** Fit the simulator on train features; check realism with a discriminator (**go if it cannot exceed 60% accuracy, i.e. near chance**) and train with 2× class-3 synthetic augmentation. **Go if class-3 WP ≥ +0.01 and pooled ≥ +0.008 on fold 1.**

**Expected effect.** Unknown; target **+0.01 to +0.02** if label-conditioned dynamics are simulable. **P(success) ≈ 8%, gain conditional on success ≈ +0.01 to +0.02.**

**Failure mode I fear most.** The simulator nails the *marginal* clock statistics but misses the hidden driver that determines the label, so synthetic class-3 rows teach a wrong mapping — consistent with the measured fact that synthetic labels already hurt. This is the most likely failure of the eight.

---

## The one I would stake the campaign on

**Idea 1 — the "one long tape" hypothesis**, and I would run the Idea 2 duplicate scan in the same afternoon because it is the same evidence viewed two ways.

Why this one: it is the only family that can **raise the information ceiling itself** rather than redistribute a fixed budget. Every training-side idea above lives under the shared 0.95–0.99 teacher wall; the only way to beat a wall of shared *information* is to add information, and cross-sequence structure (temporal continuation or literal duplication) is the one place fresh, causal, *supplied* information can enter the model. The test is a boundary-correlation probe that costs an afternoon, the payoff is bimodal and non-trivial (+0.01 to +0.04 pooled if it holds), and a null result is itself decisive — it tells us the corpus is genuinely i.i.d. and redirects effort to the ceiling probe and Ideas 5–8 with a clear conscience. I would gate the gamble on the running lookahead probe: if the future-aware ceiling is only marginally above 0.669, then no causal restructuring will reach 0.700 and the provenance/duplication bet is the *only* remaining route, which makes it the correct place to stake scare resources.

**Attack on the shared assumption.** All eight ideas — and the whole campaign — implicitly assume the 112 columns are the only observable and that rows are exchangeable samples of one process. Idea 1 and Idea 2 attack that assumption directly. If the boundary probe comes back at chance *and* the lookahead ceiling is low, then the honest conclusion is that 0.700 may be unreachable from causal features alone, and the strategy should shift from prediction to *leakage and dataset-forensics* rather than one more architecture sweep.

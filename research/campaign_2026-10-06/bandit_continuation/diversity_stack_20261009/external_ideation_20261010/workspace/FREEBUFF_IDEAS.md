# External Ideation: 8 mechanism-distinct ideas

All ideas respect the hard constraints: strictly causal inference, no external data, no future labels, served model fits 1 CPU core / 60 min for ~39M rows. The recurring anchor for "why" is the error decomposition: **direction, not magnitude, is essentially the whole loss** (true-sign substitution lifts WP 0.669 → 0.843), and sign accuracy is a **strongly non-stationary function of run age** (90.7% at age 0 falling to 74.0% beyond 512) with 19.9% of weighted sign errors in runs older than 128 rows.

---

## 1. Oracle-age conditioning, then a causal change-point state ("age-state input")

**Mechanism.** The label process is piecewise constant with mean run length ~9.5, but the served GRU never sees an estimate of *how old the current run is*; it must infer it implicitly from 32 features. Add explicit conditioning on (estimated, then true) age-of-run and change-phase: first as a warm-started input (is-change flag, one-hot age bucket, log(age) scalar), then replace the oracle with a trained causal change-point detector whose posterior change-probability and time-since-argmax-change feed the same inputs.

**Why it could help.** The measured age curve (90.7% → 74.0%) is exactly what a single-rate recurrence learns slowly and poorly; the model is at its *best* right after a label change, which suggests it has detected changes but has no calibrated representation of "nothing has changed for 400 rows". Explicitly stamping age lets the readout re-weight its sign prior per age bucket instead of averaging regimes that disagree.

**Cheapest experiment.** (2–3 h, one GPU) Train the current student with **oracle** age/change inputs. Note honestly: the oracle is *not* causal to the row being predicted (knowing the current run age requires the current label), so this is an upper-bound ablation, not a servable variant — if it moves WP beyond ~0.004 (2σ), the age signal is real and under-served, and the follow-up investment of 6–10 h goes to a small causal change-detector head distilled from oracle change points.

**Falsification.** If oracle-age inputs give ΔWP ≤ 0.002 on fold 1 (within 1σ of the fold noise), the implicit representation is already sufficient and the idea is dead; do not spend time on the causal detector.

**Expected gain.** Oracle version: +0.003–0.010 (sign accuracy concentrated in old runs, but old-run errors are hard). Causal version: unknown — detector errors at changes could eat the gain. Honest range: +0.000 to +0.008.

**Family.** Change-point statistics / regime modeling.

---

## 2. Causal forward-message sign decoding at inference (no retraining)

**Mechanism.** Keep the GRU's per-row outputs as *emissions*, not decisions, and run the decoder as a causal Bayesian filter over the piecewise-constant label path: a two-state-per-sign HMM with (i) a jump rate fitted from training run lengths (~1/9.5 per row) and (ii) emission probabilities from a temperature-calibrated mapping of the logits. Decode the posterior run-sign (forward messages, optionally a lag-k backward pass where a lag of k ≤ few hundred rows is causally acceptable if the harness permits scoring after row 100 — otherwise pure forward filtering, still O(1) per row).

**Why it could help.** The model's per-row softmax treats 19,600 statistically dependent decisions as independent and pays no price for flipping sign mid-run; structured decoding that *penalizes sign flips proportional to elapsed run length* directly attacks the measured age-dependence, especially the 19.9% of error mass in old runs. It is an inference-side fix, so it needs no GPU and cannot overfit.

**Cheapest experiment.** (2–3 h, CPU + one GPU hour only to re-calibrate logits on validation) Fit the transition/emission parameters on fold 1 training sequences, evaluate forward-filtered WP on fold 1 validation. No model retraining at all.

**Falsification.** If forward decoding at the fitted jump rate is within ±0.002 of argmax decoding on two folds, the emissions carry no exploitable temporal structure beyond what the GRU already impulse-decodes — falsified and never revisited.

**Expected gain.** +0.002–0.006 overall; plausibly concentrated (2–4×) in class 0–2 where runs are longer. Unknown for class 3 if its runs are short.

**Family.** Optimal filtering / structured sequential decoding (synthesis of signal processing + compression view of the label string).

---

## 3. Self-supervised OU/innovation features for the target channels ("hidden-drift filter features")

**Mechanism.** Treat (t0, t1) as noisy observations of a latent drift process and fit, per market class, a discrete-time Ornstein–Uhlenbeck / AR(1) state-space model to the label series (parameters by EM on training data only). Emit the *filter innovations* (one-step prediction residual, whitened), the filtered drift estimate, and the filter's constant-hazard "age since last innovation burst" as 4–8 extra causal features. A second self-supervised pass builds identical features from the predicted label channel at test time so no future information is used.

**Why it could help.** Innovation residuals are the optimal statistic for "has the hidden drift regime changed?", i.e. they are a feature-space version of the change-point signal the model currently has to rediscover from raw inputs. The age-curve measurement says exactly this statistic is missing; the filtered drift also provides a magnitude-free sign prior, aligned with the finding that magnitude carries almost nothing.

**Cheapest experiment.** (4–6 h, one GPU) Fit AR(1) per class (simple ELS estimator, no EM needed for AR), append the causal filtered/innovation features, retrain the standard student, evaluate on fold 1.

**Falsification.** If adding whitened innovations changes WP by ≤ 0.002 across two folds, the GRU's internal recurrence already computes the sufficient statistic; the features are redundant and the estimation-theory framing adds nothing.

**Expected gain.** +0.002–0.008. Class 3 could respond disproportionately if its label statistics are noisier (more innovation signal per row); if its AR fit is degenerate, the gain vanishes — unknown.

**Family.** Estimation theory / signal processing (Kalman-style).

---

## 4. Label-free test-time bias controller on predicted-run statistics

**Mechanism.** During serving, run a cheap feedback loop on the *predictions only* (no labels needed): an exponential tracker maintains the empirical run-length distribution and flip-rate of predicted signs; a slowly-integrating controller adjusts a single scalar output bias (sign pressure) and a softmax temperature so the predicted run statistics match those measured on validation folds for the local market class (class inferred causally by the existing class-3 detector). Cost is ~O(1) arithmetic per row, adding a few microseconds to the 74 µs budget.

**Why it could help.** Training plateaus are all in noise, but the *decision boundary offset* was never tuned against the run structure: a systematically skewed sign pressure shows up as predicted-run statistics deviating from the true geometric-ish run-length law measured on validation (mean ~9.5 rows). A slow controller corrects drift and label-prior shift between validation and hidden test without any gradient descent.

**Cheapest experiment.** (1–2 h, CPU) Simulate the controller offline on validation sequences with the packaged student: sweep controller gain, confirm ΔWP and stability (no oscillation within sequences).

**Falsification.** If the controller at any stable gain is within ±0.002 of a *fixed* global bias tuned on validation, the deployed model has no test-time drift worth adapting to — falsified.

**Expected gain.** +0.001–0.004 on validation by construction small; its real (but unmeasurable) value is hidden-test robustness, since hidden-test loss is concentrated in class 3. Unknown on the hidden set.

**Family.** Feedback control / causal test-time adaptation.

---

## 5. Sign-only information-bottleneck teacher pretraining

**Mechanism.** Pretrain a large teacher with an explicit information-bottleneck constraint: quantize the recurrent state to a discrete code (e.g. 6–8 bits via a straight-through quantizer or VQ) and train the code to predict **only the sign channel** of both targets while a bit-budget penalty forces maximal compression of everything else (including magnitude information). Then distill the *code* (not the raw teacher logits) into the standard-width student with an auxiliary code-matching loss added to the normal target loss.

**Why it could help.** The decomposition says magnitude is nearly worthless (true-magnitude substitution: 0.685 vs true-sign: 0.843), yet the current pipeline distills full teacher targets including magnitude, spending the narrow 256-wide student's capacity on a nearly uselessly entangled representation. All tried losses (Huber, correlation, mixes) operate at the *output* level and did nothing; a bottleneck operates at the *representation* level and changes which state dimensions survive distillation.

**Cheapest experiment.** (6–8 h, one GPU) Add VQ quantization + sign-only readout to the existing teacher training for ~half the usual epochs, then run the standard student distillation with a code-prediction auxiliary head; compare on fold 1.

**Falsification.** If the bottlenecked-teacher student is within ±0.002 of the current student (or the code prediction head fails to train above ~90% code accuracy), then a 256-wide GRU is not bit-starved and compression is not the binding constraint — falsified.

**Expected gain.** +0.002–0.006. Uncertain: the sign signal is 80% accurate even in the current model, so capacity may not be the limiter; class 3 data-limitedness is as likely the binding constraint.

**Family.** Information theory (information bottleneck / compression).

---

## 6. Mirror-decode of the anonymized event codes into explicit bid/ask flow features

**Mechanism.** The anonymizer presumably mapped real order-book fields to codes through some injective-but-scrambled table. Mine the code tables structurally: for each discrete event code, test for a mirror partner (a second code whose within-class conditional distribution of the next-step label flip is inverted, and whose feature-vector statistics are sign-flipped under a permutation recovered from covariance structure); use the recovered pairing to synthesize *signed* imbalance features (bid-side minus ask-side queue pressure, aggressor side, and book-direction conventions) as additional inputs.

**Why it could help.** Class 3 has the lowest correlation (0.44) and is data-limited (-0.009 for halving data); that is the signature of features that are *information-poor rather than sample-poor*. Recovering the physical bid/ask meaning costs no data and gives the tiny served model exactly the signed-flow summarization that a large model could otherwise learn only implicitly from 32 scrambled channels.

**Cheapest experiment.** (4–6 h, CPU + one GPU hour for training) The pairing search is pure CPU over code statistics; then retrain the standard student with ~8 signed-flow features appended and evaluate on fold 1.

**Falsification.** If no statistically robust mirror pairing exists (symmetry score not separable from permutation null), or if appending the recovered features gives ΔWP ≤ 0.002 on two folds, the codes carry no decodable side information in the used representation — falsified.

**Expected gain.** Signed flow is the strongest known predictor in real order-book data, so if the decode works: +0.005–0.015, plausibly the single largest upside here. If the anonymization is not side-recoverable: 0. Genuinely unknown until the symmetry test runs.

**Family.** Quant microstructure / domain-structural feature extraction (also partially a data-structure idea).

---

## 7. Age×class error atlas and age-stratified resampling data structure

**Mechanism.** Two-part, deliberately model-free. (a) *Measurement*: build a validation atlas — weighted sign accuracy, flip-rate, and error mass in class × age-decade × run-parity cells — so every future experiment is judged per cell, not only on pooled WP; the current single-number comparison at σ≈0.0022 hides where any real gain lives. (b) *Data structure*: reorganize training batches by attaching the true run age to each row and stratify sampling so long-age rows (which hold 19.9% of error mass but are the least represented in gradient terms) appear in balanced age buckets, replacing global loss weights with a fixed deterministic sampler.

**Why it could help.** Age-dependent accuracy is *measured*, but nothing in the training or evaluation pipeline is currently age-indexed. The model's sign prior is effectively trained and averaged as if run age did not matter; the stratified sampler is the minimal intervention that exposes gradients to the worst cells, and the atlas makes any improvement (or its absence) diagnosable in hours rather than across fold-runs.

**Cheapest experiment.** (2–3 h, mostly CPU; one GPU for the retrained run) Build the atlas (CPU only, existing checkpoints), then retrain the student once with the stratified sampler and compare per-cell and pooled.

**Falsification.** If the atlas shows error mass roughly proportional to row share in every class × age cell (no concentrated deficit), stratified resampling cannot help — that in itself is a decisive, cheaply obtained negative result. If the sampler version is within ±0.002 pooled (with per-cell evidence that old-age cells did *not* improve), falsified.

**Expected gain.** Uncertain; plausibly +0.001–0.004 pooled with outsized per-cell movement in old-run class 3 (the highest-value cell). The measurement half is guaranteed value at zero model risk.

**Family.** Measurement & data structure (evaluation protocol + sampling structure, not a model).

---

## 8. Surprise-gated plasticity in the recurrent update ("novelty-tagged memory")

**Mechanism.** Replace the GRU's learned input-gate with a *two-factor* plasticity rule borrowed from neuroscience: gate how much the hidden state's feature-specific components update by the normalized prediction error of a self-supervised input-prediction head (surprise creates "synaptic tags" that decays with eligibility traces over several rates). Concretely the hidden update x_t = gate (input omega; surprise_t) multiplies an internally generated novelty signal, distinct for distinct feature groups, so the state writes aggressively on surprising rows and holds steady through long unsuprising runs — the failure regime measured at ages 128+.

**Why it could help.** The measured degradation with age is the signature of the state *over*-integrating noise during long featureless runs and then needing many rows to re-commit after a change. A novelty gate is a mechanistically different fix than two-clock or chrono-init variants (tried, null): the gate is driven by an *externally self-generated error* on the 32 features, not by a learned gate on the same stream, so it can implement "hold steady until surprised" exactly.

**Cheapest experiment.** (3–4 h, one GPU) Implement the gate as a one-branch modification of the existing student (input- prediction head is 3 conv/linear layers) and train with the current recipe on fold 1; also measure whether feature-surprise correlates with the true change events (a free diagnostic).

**Falsification.** If surprise_t has low mutual information with true change rows (say, ROC-AUC < 0.6 against detected changes), the gate is noise and the mechanism cannot work; likewise if the modified student is within ±0.002 of seed-matched baseline across two folds.

**Expected gain.** +0.001–0.005, unknown; the mechanism is plausible but gated recurrences are exactly the family that several tried variants (gated, two-clock) failed to profit from, so treat as a coin flip biased slightly positive.

**Family.** Neuroscience / plasticity-based recurrence.

---

## Which two would you run first and why

**#2 (causal forward-message decoding)** and **#1-oracle (oracle-age input ablation)** together, in that order.

- **#2 costs essentially nothing** (a day of CPU, no training) and is a *pure inference-side* test of the single most surprising measurement: that the model is near its best at age 0 and worst in old runs. If emissions + a fitted jump rate buy as little as +0.002 pooled it already matches the best known cheap trick (seed averaging), and if it fails inside one day you have permanently eliminated an entire class of inference-side fixes. Nothing on the list has a better gain-per-hour than a decoder you can bolt onto the existing package.
- **#1-oracle is the highest-information *model-side* experiment**: it answers with one 3-hour training run whether the age-structure of the label process is under-served by the current recurrent state — the question on which five of the eight ideas (#1-causal, #3, #7, and indirectly #8) all hinge. A decisively positive oracle result routes all remaining GPU budget into causal age/change-point machinery (#1, #3, #8); a decisively negative one kills that whole branch in a single run and redirects effort to #6 (mirror decode), which has the largest *uncertain* upside — you would then run it as the third experiment since it is the only idea whose outcome is a big either/or (0 or +0.01+) rather than a small expected value.

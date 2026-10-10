The gap from 0.669 to 0.700 is a sign problem with a known ceiling. Replacing the student sign and keeping its magnitude reaches 0.843, while replacing the magnitude and keeping its sign reaches only 0.685. Weighted sign accuracy is 84.3% (80.7% in class 3). A rough conversion is about +0.011 WP per +1 point of weighted sign accuracy, so +0.034 WP needs on the order of +3 points of weighted sign accuracy. The long-run tail is large enough to matter and too small to do it alone: runs older than 128 rows hold 19.9% of weighted sign errors, so cleaning that entire tail is roughly a +0.03 WP ceiling, not a route to the sign oracle. Same-family capacity is used up (width, depth, schedules, distillation, teachers correlated 0.95–0.99).

## 1. Decreasing-hazard sign hysteresis

**Mechanism.** Treat each target as a telegraph signal. Keep a sign bit and the age of that bit. Flip only when the student logit disagrees by a margin that widens with age, using a hazard fit to the training run-length histogram (mean run 9.5 with mass past 512 is a decreasing hazard, not a constant leak). Emit the student magnitude with the held sign.

**Why it could help.** Sign accuracy falls from 90.7% at the change to 74% past age 512, so the usual failure is losing a sign that was right when the label moved, on rows that still carry a fifth of the weighted sign errors. A soft GRU update mixes every quiet row and can cross zero inside a constant label. Sticky odds that grow with dwell match a heavy-tailed run length. Magnitude is the wrong place to spend capacity: the magnitude oracle is only +0.016 WP.

**Cheapest experiment.** No retrain. On one fold’s existing predictions, grid about 20 margin schedules. Also record, within each true run, whether later mistakes reverse a sign that was correct at age 0. Under an hour on CPU.

**Falsify.** No schedule gains ≥0.004 WP, or sign accuracy past age 128 does not rise. The premise dies if late mistakes are not reversals of a correct age-0 sign.

**Expected gain.** About +0.005 to +0.02 WP if late errors are chatter around a correct entry sign. Near 0 if late rows carry real opposing information. Not enough by itself to clear 0.700.

**Family.** Point processes / hysteresis control.

## 2. Lattice recovery of the event alphabet

**Mechanism.** Split the 32 floats into discrete codes and continuous sizes using only the tables. Per column, measure cardinality, share of values on a low-denominator lattice, and exact low-degree relations with other columns (difference, sum, product, sign). From the discrete columns build an integer event id and any side or level bit they determine; from the continuous columns add a causal first difference. Train the current student on that augmented row.

**Why it could help.** Accuracy peaks on the row where the label changes (90.7%), so the change is legible in the current event. A GRU on undifferentiated floats can spend its state relearning a small code. Class 3 is where a rare code would hurt most, and a specialist already failed (−0.14), so the fix has to be a better observation for the shared trunk. The failed experiment was a block swap, which does not exhaust interleaved or arithmetic couplings.

**Cheapest experiment.** CPU, under an hour, on one fold: cardinalities, lattice residuals, pairwise exact relations, and mutual information of each recovered code with the label-change bit. Only if a change-aligned code appears, one student retrain on one fold.

**Falsify.** No column is a small exact alphabet and no recovered variable moves with label changes above the baseline MI of a shuffled column. A retrain then also fails if WP moves by <0.002.

**Expected gain.** Unknown until a code shows up. If a change-aligned code exists, about +0.01 to +0.03 WP. If the columns are genuinely continuous and independent, 0.

**Family.** Data structure / measurement.

## 3. Shallow program for the current event

**Mechanism.** Search for a short rule from `(x_t, x_t − x_{t−1})` alone to `sign(y_t)`: depth-3 oblique tree, or a beam of threshold rules on at most three columns. No recurrence. If a rule is accurate at changes and stable inside runs, serve it as the sign, hold it with the hysteresis of idea 1, and take magnitude from a linear map of the current row.

**Why it could help.** The sign oracle is worth +0.174 WP and the model is already best at the change. That is the signature of a shallow function of the current event, which a recurrent state can only approximate before it drifts. Unconstrained GRU width from 256 to 768 did not move this, which fits a missing feature interaction rather than a missing hidden unit.

**Cheapest experiment.** One fold, CPU or well under one GPU-hour. Report weighted sign accuracy overall, at age 0, and past age 128, next to the student (84.3% / 90.7% / 74%).

**Falsify.** The best depth-3 rule is near chance at age 0 even with first differences. The change is then not a shallow function of the current row.

**Expected gain.** Unknown, bimodal. If the rule is real and holds inside runs, this is the idea with a path through a large fraction of the +0.174 sign gap; a rule that only matches the student at age 0 and then dies is worth about 0.

**Family.** Program synthesis.

## 4. Exact state hold on non-events

**Mechanism.** A binary detector on the current row, annealed from a sigmoid to a hard threshold, decides whether the GRU state is updated. On a non-event the hidden state is copied, not passed through a forget gate. The readout may still rescale magnitude from the current row. A penalty keeps the event rate near the label change rate (about 1/9.5 per row, with a heavier tail).

**Why it could help.** Soft gates almost never sit at exactly 1 for hundreds of steps, so the state keeps integrating rows that do not change the label. That lines up with the monotone decay of sign accuracy and with 19.9% of weighted sign errors past age 128. Exact copy is a different primitive from a soft two-clock: the dwell is infinite until an event, which is the geometry of these labels. Class 3, at correlation 0.44, is where a drifted sign is most expensive, and it needs the shared trunk (the specialist collapsed).

**Cheapest experiment.** One fold, one student-width training run, a few hours. Compare the age curve, not only WP. Log the detector’s precision against true label changes.

**Falsify.** Sign accuracy past age 128 stays within 1 point of the student and WP moves by <0.004. Also falsified if the detector cannot beat a rate-matched random gate.

**Expected gain.** About +0.005 to +0.02 WP, overlapping idea 1. Larger only if the trigger uses a feature the logit has already thrown away.

**Family.** Hybrid systems / event-driven memory.

## 5. Completely monotone impact kernel

**Mechanism.** Build four to six causal EMAs of a one-dimensional signed event proxy, rates fixed on a log grid from 1 to 512 rows. Fit non-negative coefficients per target so the kernel is completely monotone: the effect of an event cannot change sign as it ages. Add those channels to the student, or first try the kernel alone as a linear predictor. Serving cost is a few multiply-adds per row.

**Why it could help.** A kernel that rings will flip the prediction inside a long constant label, which is one concrete way to get 90% accuracy at the change and 74% after age 512. Unconstrained TCNs can ring; a non-negative sum of exponentials cannot. The slow EMA holds a horizon a width-256 state does not have to learn, on a problem where extra GRU width bought +0.001.

**Cheapest experiment.** CPU, about an hour. Freeze a signed proxy (best single column, and the code from idea 2 if it exists). Non-negative least squares from the EMA bank to the two labels on one fold. Score WP and late-age sign flips against the same bank with unconstrained coefficients.

**Falsify.** The monotone kernel does not beat the unconstrained linear readout of the same EMAs on sign accuracy past age 64, and neither is within striking distance of the student (gap stays >0.01 WP, or the kernel adds <0.003 when used as extra features).

**Expected gain.** About +0.003 to +0.015 WP if a usable proxy exists. Unknown, and near 0, if every column is weakly signed.

**Family.** Signal processing / microstructure (transient impact).

## 6. Held discrete sign token

**Mechanism.** Quantize a one-dimensional projection of the state to a codebook of 2 or 8 tokens. Replace the token only when the new projection wins by a margin; otherwise reuse the previous token (straight-through estimator). The served direction is that token; a positive scale from the GRU supplies magnitude. The memory of direction is a few bits with an explicit dwell, which a float hidden state does not give.

**Why it could help.** Wider teachers and four teacher families do not change the distillation target (pairwise correlations 0.95–0.99), so more analog capacity is the wrong hypothesis class. The label is piecewise constant. A token that cannot drift by a small gradient step matches that geometry, and it attacks the same age curve as idea 1 inside the trained model rather than as a post-process the logit can fight.

**Cheapest experiment.** One fold, one student-width run with a 2-code and an 8-code bottleneck on the pre-logit. Log token age. A few hours.

**Falsify.** Tokens update almost every row (age ≈ 1), or token sign accuracy past age 128 is no higher than the baseline logit.

**Expected gain.** About +0.005 to +0.02 WP if the token locks the scarce object, the sign. About 0 if the codebook chatters at the event rate of the features rather than of the labels.

**Family.** Compression / discrete memory.

## 7. Shared jump, two amplitudes

**Mechanism.** Test whether `t0` and `t1` change on the same rows. If they do, serve one hazard and one regime bit, and two amplitudes that may depend on the current row. Each prediction is its amplitude times the shared sign. If the change indicators are independent, do not fit the joint model.

**Why it could help.** The metric pools both targets, so a missed shared jump is counted twice. A shared emission lets the cleaner target carry the noisier one, which is a structural constraint on when signs may move. Width, loss, and class reweighting do not impose that constraint; those knobs are already on a plateau. Class 3’s low correlation may be one target with a weak emission riding a jump the other target sees.

**Cheapest experiment.** CPU, minutes, on training labels: co-change rate against the independence baseline, and the fraction of weighted sign errors that hit both targets on the same row. A joint model is justified only if co-change is far above independence.

**Falsify.** Change correlation near 0, or a shared flip gate lowers one target’s weighted sign accuracy by more than it raises the other.

**Expected gain.** Likely 0 to +0.008 WP. Unknown until the co-change rate is measured. A small idea unless errors coincide.

**Family.** Multivariate point process.

## 8. Sequence-level class scale from feature moments

**Mechanism.** The four classes are defined from label statistics. Test whether causal feature moments over the warmup (event rate, per-column variance, imbalance of differences) predict the class. If they do, keep one trunk trained on all classes and let a frozen classifier choose among four scale-and-bias pairs. The trunk stays shared because a class-3-only model was far worse.

**Why it could help.** Class 3 carries about half the weight at correlation 0.44, and the hidden-test damage sits there. Halving the data costs class 3 only 0.009, so this is not a sample-size gap, and synthetic labels already failed. A last-layer correction is a different lever from a specialist: classes 0–2 still teach the representation, while class 3 gets its own output map. This helps only if class 3’s sign errors have a sequence-level convention (a flipped or rescaled market), which the row-level autopsy does not rule out.

**Cheapest experiment.** CPU logistic regression from the first 500 rows’ moments to class, one fold. Stop if balanced accuracy is near 25%. If it is clearly above chance (above about 50%), one short finetune of the four output pairs.

**Falsify.** Balanced accuracy below about 35%. Or the class is predictable and the scale-bias blend still moves WP by <0.003.

**Expected gain.** Unknown, and capped well below the sign oracle. About +0.003 to +0.015 WP if class 3 is a per-sequence sign or scale convention. About 0 if its errors are scattered inside the sequence.

**Family.** Shift measurement / output adaptation.

## Which two would you run first and why

Run **decreasing-hazard sign hysteresis** and **lattice recovery** first, in that order, both before another training job.

Hysteresis prices the main measurement directly: the age curve, the 19.9% tail, and the sign-versus-magnitude split. It needs no new model, finishes in under an hour, and the +0.004 falsifier is above fold noise (0.0022). If holding the age-0 sign lifts late accuracy, ideas 4 and 6 become implementations of a confirmed mechanism. If it does nothing, the late errors are new information and those memory constraints should be dropped.

Lattice recovery is the data-structure screen. Accuracy is highest on the change row, and every architecture result says more of the same GRU will not read the row better. An hour of cardinality and exact-relation checks either produces an event code worth a retrain or closes the alphabet hypothesis, including interleaved couplings that a block swap does not cover. The shallow-program search (idea 3) should be the immediate follow-up on the same machines if any column is discrete or any single difference lines up with changes; it is the only screen here with a plausible path through a large piece of the +0.174 sign gap.

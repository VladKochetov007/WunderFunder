The score gap lives in **class 3’s long runs**, and every architecture you have already agrees with the others (pairwise correlation 0.95–0.99). The next gain has to change the **prediction object** or the **information in the teacher**, not the cell type.

One arithmetic fact before the proposals: the fast clock covers 31% of rows and is 3.2× enriched for label changes, so it carries only about **59%** of changes. About **41%** of true jumps are off that clock. Any mechanism that treats “the 22-column block” as *the* change process is already incomplete. The 100%-update block is a continuous driver sitting on top of the event clocks, and it is the obvious source of slow sign drift.

Global error numbers are the wrong map. True sign plus your magnitude scores 0.843, and your sign plus true magnitude scores 0.685, **pooled over classes**. Class 3 is half the weight at correlation 0.44, with 73% of that weight in runs older than 16 rows, and sign accuracy there is 80.7% (74% after age 512). A 3–4 point sign gap does not explain 0.44 versus 0.74–0.81. Until the swap decomposition is repeated **inside class 3 × run age × target**, “sign is the bottleneck” is an easy-class result.

Unconditional sign-holding lowering WP is also narrower than it looks. It says late flips **as a pool** carry information. It does not say flips on quiet rows carry information.

---

## 1. Five reframings of the task

### R1. Reconstruct a marked point process on a clock calendar, then broadcast

The row series is the reconstruction of a jump process whose legal move times are the union of a few block-update calendars; between legal times the level is constant, and the scored series is that level held forward. This is easier because the mean run is 9.5 rows while class-3 weight sits in long runs, so most row-losses are the same label repeated against a moving 100%-clock input that shoves a GRU state off the level. The object to learn is a per-tick stay/jump decision plus a destination level, not a 20,000-step regression.

**One-day experiment.** On existing fold predictions, cross-fit a gate across folds: hold the output on rows where a chosen set of blocks did not update; on an update row, accept the model’s new output. Choose the block set on fold A by change coverage, score official WP on a hard fold (0.668) and on fold 1. Split the accepted flips into fast-clock, other event clocks, and the always-on block.

**Falsifier.** Gated replay loses WP on the hard fold, or class-3 true changes are spread across the always-on block so that “quiet” rows are not quiet. **Effect guess:** +0.005 to +0.02 if class-3 false flips concentrate off the enriched clocks; ~0 if the informative late flips already do.

### R2. Two weakly coupled jump processes, joined only by the pool

t0 and t1 co-move only 2.3× chance, so a shared recurrent state is a coupling the labels barely have. The task is two marked point processes, each allowed its own driving blocks, with a small shared term for the excess co-jumps; the metric is applied only after both reconstructed series exist. This is easier if one target’s long class-3 runs are being dragged by the other target’s jumps inside the same hidden state (the plateau across GRU/LSTM/TCN is what you expect when every net learns the same entangled compromise).

**One-day experiment.** Per target, per class: WP, weight share, and which block’s update row has the change enrichment. Then a linear probe: predict each target’s change event from each block’s delta, alone and cross-target. No new net.

**Falsifier.** The two targets show the same block-enrichment profile and similar age-decay curves, and a per-target probe does not beat the joint model’s change indicators on the weaker target. **Effect guess:** unknown, plausibly +0.005 to +0.015 on the pool if one target holds most of the class-3 damage; negligible if the weight and the errors split evenly and the blocks match.

### R3. Score and learn run-atoms, then paint them back onto rows

Inside a constant run the label is one number, so the correlation atom is the run, with weight proportional to |clip(y,−2,2)| times length. Row-wise weighted MSE treats a 512-row class-3 run as 512 draws and lets the prediction wander; a single late sign flip on that run is a large Pearson hit because the weight is exactly the large |y|. The easier problem is: at run start (and at each legal tick) rank and scale this run against other runs, emit one level, and broadcast it. Same official score after broadcasting.

**One-day experiment.** Collapse validation predictions to one atom per true run (mean output, and separately the output at run start). Broadcast each and rescore WP, overall and in class 3 × age. That isolates wander-within-run from wrong-level-at-birth. Fit nothing.

**Falsifier.** The run-start broadcast matches the row-wise WP within 0.002, including class 3 after age 512. Then wander is not the damage, and a run-level objective will not move the score. **Effect guess:** +0.01 to +0.03 on the hard fold if late sign decay is accumulated wander; under +0.005 if the level is already wrong at the first row of the run.

### R4. Invert a structural generator of the 112 columns, then read the level off the canonical state

The columns are a handful of vector processes plus sample-and-hold, not 112 streams. The task is to recover, per sequence, the nuisance (block signs, scales, and any exact affine or mirror relation) and the event masks, then to predict the label as a function of that small state. This is easier because a width-256 GRU currently spends capacity re-learning a per-sequence coordinate system from the prefix, which is what a 99.4% class decoder from 500 rows already proves is identifiable, and then the coordinate system drifts as the always-on block moves. An exact mirror or a per-sequence sign flip would also cap every architecture at the same place, which matches a 0.002 plateau and 0.95–0.99 agreement.

**One-day experiment.** On a few hundred training sequences, test exact and near-exact relations: duplicate blocks, constant scale, negation, permutation of coordinates within a block, and linear images stable inside a sequence but different across sequences. Compare relations fit on the first 500 rows with relations fit on all 20,000. Separately, lag histogram of (label-change row minus nearest update) per block, per class, per target.

**Falsifier.** No stable low-dimensional relation beyond the update masks you already know, and full-sequence refit does not change the prefix fit. Then there is nothing structural to invert. **Effect guess:** unknown; bimodal. Either ~0, or +0.02 to +0.05 if class-3 columns are a per-sequence mirror/scale of a simpler driver.

### R5. Carry an explicit posterior over the current level, and emit the metric mean of it

The population target m(H) = E[w y | H] / E[w | H] is the mean of a posterior over a sticky level, not a generic sequence regression. Sign accuracy 90–92% just after a change and 74% after age 512 is what a belief looks like when new feature deltas keep multiplying the likelihood and the level is stored only implicitly in a hidden state that the always-on block is free to overwrite. The easier problem is a belief state (mass on “same level”, mass on a new level drawn from a class-conditional destination law) updated mostly on clock ticks; the served number is the weighted mean, so true late flips survive as likelihood ratios. Same score, because that mean is the optimal predictor under your metric. This differs from R1: R1 restricts *when* the output may move; R5 represents *what* is being believed so a legal tick can flip the mean without erasing the level.

**One-day experiment.** Post-hoc filter on existing logits or predictions, cross-fit across folds: a two-atom belief (stay at the last committed level, or take the model’s new value), with the stay prior and the tick likelihood fit only on other folds, class-conditional if the audit says so. Serve the weighted mean. Compare with the raw model on age>512 class-3 sign accuracy and on WP.

**Falsifier.** The filter does not raise age>512 class-3 sign accuracy on a held-out fold, or it raises sign accuracy and lowers WP (the late flips you “fixed” were the informative ones, and the likelihood cannot tell them apart). **Effect guess:** +0.01 to +0.025 if the current outputs already contain a correctable belief; ~0 if the destination level itself is wrong, which is when a teacher rather than a post-hoc filter is required.

---

## 2. Five elephants

### E1. “Sign is the bottleneck” may be an easy-class average

Why it may be wrong: the jump from 0.669 to 0.843 with oracle sign is pooled. Class 3 has correlation 0.44 against 0.74–0.81 elsewhere and only a modest sign-accuracy gap (80.7% vs 84.3%). That pattern is what magnitude ranking, shrinkage, or a few ultra-high-weight runs look like, not a pure sign problem. Hidden-test loss concentrating in class 3 means the public 0.666 can move only if the class-3 component moves.

**Cheapest measurement:** on fold 1 and on the 0.668 fold, repeat the sign/magnitude swap inside class × target × run-age, and report each cell’s share of Σ|clip(y,−2,2)|.

### E2. The independent draw is the run (and the sequence), not the row

Why it may be wrong: labels are constant for a mean of 9.5 rows and much longer in the heavy class, so row-wise WMSE multiplies long class-3 runs into the gradient and calls them “more data.” “Half the data costs 0.009” and “class-3 specialists lose 0.14” can both be true if halving rows keeps most runs, while a specialist loses the shared geometry learned from easy classes. Synthetic labels hurting fits the same picture: the model needs real run structure, not more row-shaped noise.

**Cheapest measurement:** recompute the half-data curve by dropping whole runs, and by dropping whole sequences, stratified by class. If run-subsampling hurts much more than row-subsampling, the data-limited claim was counted in the wrong unit.

### E3. Column identity is assumed stable across sequences

Why it may be wrong: a 99.4% market-class read from the first 500 feature rows means the prefix already contains a sequence-level latent. The same fact is what you would see if signs, scales, or within-block layouts were redrawn per sequence and the class were a property of that draw. A shared GRU then re-estimates a coordinate system forever, and every architecture tops out together because they share the same unidentified nuisance. Teachers that distill each other’s predictions stay inside that nuisance.

**Cheapest measurement:** the relation-fit in R4, plus whether a class-3 linear map estimated on one set of sequences transfers to another set of sequences of the same class. Failure to transfer is the elephant showing itself.

### E4. Co-occurrence with the fast clock is being read as the causal calendar

Why it may be wrong: enrichment is not coverage (about 41% of jumps are off that block), and co-occurrence is not lag. If the label moves on the same row as a block, that block is a usable contemporaneous input under a causal serve. If the label moves first and the block prints it afterward, the model is fitting a consequence and will look good in-sample exactly on change rows (your 90–92% just after a change) while the long-run problem is untouched. t0 and t1 being pooled in one Pearson number has the same shape of mistake: two weakly coupled processes can hide one bad target behind one good one.

**Cheapest measurement:** for each target and class, the histogram of (change row − nearest update row) per block, including the always-on block, and the WP of each target alone.

### E5. Fold spread is being treated as evaluation noise, and the loss is assumed to match the weight cap

Why it may be wrong: the four folds already span 0.668 to 0.707, and the public score sits on the floor of that range. That is what a different class-3 weight mix, or a different run-length mix, looks like. Separately, metric weight caps at 2 while squared error does not, so if a small slice of rows has |y|≫2 they can own the gradient and not own the metric. Either mistake sends a teacher search into a 0.002 plateau on the wrong rows.

**Cheapest measurement:** class-3 weight share and run-length quantiles on each of the four folds, next to that fold’s WP; plus the fraction of WMSE coming from |y|>2 versus the fraction of metric weight coming from those rows. Influence of the top few percent of sequences on fold WP tells you whether 0.034 of public score is a handful of runs.

---

## 3. Three ways to raise the teacher ceiling

All three give the teacher an object the current student never sees, and distill that object only where it is still a function of the causal past. They are aimed at shared errors (the 0.95–0.99 agreement), which is the part ensembles cannot move.

### T1. Distill only the privileged residual that is still in the past

Fit the best causal teacher C you already have. Fit one privileged teacher L that may use future rows, future labels, and the true run segmentation. Form r = L − C. Regress r on causal features (and on C’s hidden state) **inside class 3, age>16, per target**. Distill into the student the sum C + r̂, or supervise r̂ as an auxiliary head, and drop any component of r orthogonal to the past.

A raw substitute of L’s future prediction trains the student on shocks that have not happened yet; that is the natural reason lookahead distillation sat in the noise. An ensemble of C’s stays in the span of the same errors. The regression is the test that a residual is actionable: if R² is about 0 on class-3 late rows, the missing information is genuinely future and no student of this filtration will reach 0.700. If R² is material, the student is under-extracting something the past already contains, which is the one failure mode a bigger GRU has been missing.

### T2. A cross-run memory teacher for rare levels

Index completed training runs by a prefix embedding and by market class. The teacher’s prediction for a query prefix is a kernel-weighted vote of **realized levels of other runs**, not another parametric net. Privilege, if you want it, is only in the index: build the keys with full-run feature statistics, then require the student to match the retrieved vote from a causal prefix embedding alone. At serve, either distill the vote into the GRU, or carry a small causal prototype table if the latency budget allows a lookup beside the existing GRU.

Ensembles average models that make the same mistakes on the same rare class-3 levels. Substituting y at t+h uses this sequence’s future. Retrieval uses other sequences’ finished outcomes, which are inside the training set and simply never stored. Class-3 specialists failing by 0.14 says a from-scratch net cannot learn the geometry from class 3 alone; it does not say other runs’ levels are useless as a memory.

**Falsifier inside the same day as the build:** kNN of completed runs, features restricted to the causal prefix, scored by WP on held-out sequences. If it does not beat C on class 3, stop; there is no neighbor structure to distill.

### T3. A structural canonicalization teacher, then a small student on canonical coordinates

The teacher sees the whole sequence and solves the nuisance E3 is about: block signs, scales, exact mirrors, and the lag from each block to each target. It writes a canonicalized causal feature stream (the canonicalization at row t may use only the prefix estimate, or the student is trained to match the full-sequence estimate using the prefix only). The level model is then fit in coordinates that mean the same thing in every sequence. Distill the canonical state and the block-responsibility labels (which block’s tick coincided with the true jump), not L’s raw forecast.

Responsibility labels are a low-entropy alignment the current y-regression never isolates; ensembles do not estimate them; a future-y substitute smears the alignment back into a real-valued target and reintroduces the unpredictable part of the next level. The student stays a causal GRU on the canonical stream, possibly still width 256, because the capacity is no longer spent identifying the sequence.

**What has to be true:** R4’s relation-fit finds something stable. If the nuisance is already identified by row 500, full-sequence privilege adds little and the win is the canonical training signal, which you can test by re-fitting one small model on prefix-only canonical features.

---

## 4. The experiment to run first

**Cross-fold clock-gated replay plus the class-3 swap table, on the predictions you already have.**

On one hard fold and on fold 1, tabulate weight share, sign accuracy, oracle-sign WP, and oracle-magnitude WP by class × target × run-age, and tabulate true change rate against model sign-flip rate on each block’s update rows and on the complement. In the same pass, hold outputs only where the train-fold-chosen blocks are silent, accept the live output on those ticks, and rescore official WP on the held-out folds.

That single CPU pass decides which story is even aimed at the public gap. If quiet-row flips are common and gated replay gains WP, R1/R5 become the training target and the old “holding hurts” result stays intact as a statement about unconditional holding. If class-3 oracle-magnitude is the large jump and oracle-sign is not, stop spending teacher runs on sign persistence. If both swaps are flat in class 3 and gated replay loses, the level itself is missing from this filtration and T1’s residual R² is the next day’s question, before any new architecture.

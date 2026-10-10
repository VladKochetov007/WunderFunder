# FREEBUFF_ROUND2.md — Round 2 external ideator answer

## 1. Five reframings of the TASK itself

### R1. From "predict the level" to "predict the change-hazard and the level, separately"
**Mechanism.** The label is a piecewise-constant jump process, so every row's prediction factorizes into P(change at row t | history) × E[level | change, history), integrated over run-age. The error decomposition (0.843 vs 0.685 says sign magnitude; sign accuracy 90–92% at change rows decaying to 74% in old runs) says the failure mode is *timing of the sign flip*, not the level scale. Factorized prediction lets the model express "I don't know when the current run ends" explicitly, instead of averaging it inside the level head, which is exactly what the population-optimal predictor m(H) = E[w·y|H]/E[w|H] requires when w is run-dependent.
**Cheapest 1-day experiment.** Add three auxiliary heads to the existing student: (i) per-row change-event head (BCE on the exact change row), (ii) run-age head (regression on rows-since-last-change, trained on the oracle-supplied run-age), (iii) keep the current level head but multiply its loss by the change-posterior. Change nothing else, evaluate all four folds. (~0.5 GPU-day, differential vs the current best student run.)
**Falsifies it.** If adding the oracle-supplied run-age input does not shift the sign-accuracy-vs-run-age curve upward in the >256-rows bucket (i.e. the student's 74% stays at 74%), the run-age signal is not learnable at width 256 and this reframing dies.
**Effect-size guess.** +0.005 to +0.015 on the pooled score (unknown; sign decay is the biggest identified error axis, but the information may simply not be in the features).

### R2. From "predict y_t" to "predict the residual to carry-forward" (predict the jump, not the level)
**Mechanism.** Reframe the target as R_t = y_t − y_last-change, i.e. the model predicts the deviation from the last observed label level only at run changes; otherwise the residual is exactly 0 by construction. The served prediction is then y_pred = y_last-change + E[R_t | H], where y_last-change is a trivially available causal input if past labels are accessible. This decouples "the value we're currently in" from "the jump we predict next", and the metric-optimal action at a row inside a run is dominated by whether the run has ended, not by a fresh regression on 112 columns.
**Cheapest 1-day experiment.** Retrain the same width-256 student with residual targets and a per-row replay flag; no other change, evaluate all four folds, compare to the matched direct-target run.
**Falsifies it.** Residual-target WP ≤ direct-target WP on every fold (within noise ±0.002).
**Effect-size guess.** +0.003 to +0.010 (unknown; hinges on whether y_last-change is in fact available at serving — see Elephant 4).

### R3. From "predict y" to "predict the sign-then-magnitude decomposition with a metric-aligned sign loss"
**Mechanism.** The free-parameters in the metric (w=|y|) mean the score is essentially driven by getting the joint density of the signed-magnitude right; but the error decomposition shows sign is the dominant lever: perfect sign gives 0.843 on the same data. Instead of a regression head for y, split prediction into (i) weighted binary sign classification with weight ≈ E[|y|·1{run-age}] per row (mirroring the metric's w exactly), and (ii) a magnitude regression on |y|; recombine with a calibrated rank of run-age and class as the sign decision. This is a *different supervision signal* from weighted-MSE, even though it aims at the same population-optimal predictor.
**Cheapest 1-day experiment.** Dual-head student: weighted-BCE sign head (per-row weight |clip(y,-2,2)|) + |y| regression head; combine with a calibrated per-(class, run-age-bucket) sign threshold fit on training data. Evaluate all four folds.
**Falsifies it.** Weighted sign accuracy in long runs (old >512) does not move off 74% under the dual head.
**Effect-size guess.** +0.004 to +0.010 (unknown; depends on whether the current model's sign errors are optimization failures of the MSE-intermediate rather than information limits).

### R4. From "one model over mixed classes" to "class-and-run-age-conditional prediction with per-bucket metric alignment"
**Mechanism.** The four market classes have wildly different correlations (0.44 for class 3 vs 0.74–0.81 for others) and different weight distributions (class 3's weight sits in old runs). Condition the aim: predict y in a per-(class, run-age-bucket) normalized parameterization, so the model's implicit loss-of-fit is reweighted toward the (class 3, old-run) cells that carry half the metric weight. This is a supervision normalization, not an architecture change and not class-specialists (which already fail at −0.14 because they drop shared features).
**Cheapest 1-day experiment.** Take the current best student checkpoint, retrain one run with (class, run-age-bucket)-normalized targets and the same loss; evaluate the four folds with the class decoder applied causally.
**Falsifies it.** Class-3 WP is unchanged within ±0.002 on folds 1–2 (the hardest folds) after reweighting; i.e. class 3's difficulty is an information limit that reweighting cannot recover.
**Effect-size guess.** +0.002 to +0.008 (unknown).

### R5. From "two independent level targets" to "one coupled 2-D jump process with a co-jump event frame"
**Mechanism.** t0 and t1 change together 2.3× more than chance, i.e. there is a shared latent event driving both targets. Frame the task as inferring that shared event distribution, with each target's own change as a marginal on top; the *co-jump flag* becomes a student-visible feature the level heads can condition on. If the co-jump is a real shared event, its occurrence is causally inferable from feature blocks (e.g. the 22-col fast-clock block at 31% coverage that already shows a 3.2x change-rate), and knowing "this row is likely a co-jump" lets both heads borrow a cleaner update step at the same time; whatever decay survives the co-jump rows is a per-target information limit, not a timing limit.
**Cheapest 1-day experiment.** Add a co-jump-event head (BCE on joint-change rows) plus a co-jump-hazard input to both target heads; evaluate all four folds.
**Falsifies it.** Conditional sign accuracy on high-co-jump-hazard rows equals ±0.5pp minus the sign accuracy on single-change rows → the co-jump frame has no extra causal information to transfer.
**Effect-size guess.** +0.001 to +0.004 (unknown, likely small).

---

## 2. Five "elephants in the room"

### E1. "No column is a small discrete alphabet", therefore y is not a deterministic function of the features.
The campaign has ruled out alphabet columns but not *algebraic* maps: piecewise-constant labels are extremely suspicious — they look like a snapshot of a continuous latent (e.g. a mid-price, a spread, an aggregation of a feature block) quantized at clock-alignment rows, and candidate columns include exactly the 22-col "fast-clock" block that co-times 3.2× the base change rate. Fit per-column, per-block, per-lag (lags 0..50) affine maps on training data and check whether any residual collapses to float-epsilon on change rows; if yes, the "student" becomes an exact decoder, not a GRU. **Cheapest measurement**: a few hours of sklearn linear fits on 112 cols × 51 lags × 6 blocks.

### E2. The metric bookkeeping matches the grader's bookkeeping at the boundary of |y| = 2.
w = |clip(y, −2, 2)| has a discontinuity at the clip boundary; if the grader clips at a slightly different value, or computes w on a rounded/off-line label, the *interior* weighting of the highest-information rows (|y| > 2 has presumably been excluded by our clip; those rows carry the biggest weight) changes non-trivially, and the model's population-optimal predictor m(H) was derived under a slightly different w. **Cheapest measurement**: recompute one validation fold's metric under clip ∈ {1.5, 2, 2.5, 3} from stored predictions; if the model *ranking* flips for any clip choice, the entire alignment of loss-to-metric is off and the "plateau" may be partly a bookkeeping root, not a modelling ceiling.

### E3. "Hidden-test damage concentrates in class 3" may be a *composition shift*, not a class-3 model weakness.
If the four validation folds differ in class composition and the hidden test has a different composition again, class 3 could be perfectly well modelled *within* each fold's class-3 share but worse *pooled*; the class-3 "data-limited" conclusion (−0.009 at half data) may partly be a pooling artifact. **Cheapest measurement**: decode class composition per fold with the 99.4% decoder, recompute each fold's WP decomposed as Σ_c share_c × WP_c; if the class compositions explain ≥80% of the fold-level spread (0.668/0.679/0.703/0.707), half the campaign's class-3 conclusions need to be re-drawn; ~1 hour.

### E4. The served student is *causal in features*, but nobody has verified what is available at serving about the label process itself.
The campaign treats labels as a training-time-only signal; but if y_{t−1} (or any recent label) is delivered to the served model at inference (as in most time-series competition serving setups, where y is revealed with a lag), then the entire reframing R2/R1 gains a nearly-free causal input that the current student never gets. **Cheapest measurement**: 5 minutes re-reading the serving spec; if the spec is ambiguous, a served dummy model that logs what arrives per row settles it in an hour.

### E5. "Sign accuracy decays to 74% in runs older than 512 rows" may be a *pooling artifact of runs*, not model laziness.
Long runs contain more rows per run AND more metric weight per run; within-run variation in sign accuracy may be flat across run-age if you condition correctly on the run's remaining information. If it is a weighting artifact, the "run-age is a hard regime" thread (and the run-age hypotheses in R1/R3) would be chasing a phantom; instead the ceiling lives in *which runs* get sampled, i.e. run-level composition — closer to Elephant 3's class-mix issue than to an information limit. **Cheapest measurement**: recompute error decomposition *per run* (1 error/run, weighted by run metric weight, stratified by class) vs *per row*; if per-run sign accuracy is near-constant across run-age, the decay is a weighting artifact.

---

## 3. Three ways to raise the TEACHER ceiling beyond hyper-parameters

### T1. Distill the *future-oracle teacher's run-age / change-hazard posterior*, not its prediction.
**Teacher gets**: non-causal lookahead — the exact future label change schedule y_{t+1..t+H} (or just the next change time) — from which it computes the per-row change-hazard posterior and the run-age-conditional sign flip probability.
**Turning it into a causal gain**: distill the *hazard posterior* into a small student head (not the level prediction, and not an output substitution); the student then keeps the hazard as a causal feature that modulates its own level output. Because the sign-error is a timing failure (E5's 90-92% → 74% curve), the hazard head compresses exactly the information that is causally extractable but not learnable at width 256 by a level-by-level loss.
**Different from 'substitute the teacher's prediction at t+h'** (we do not replace any output, only add a latent feature) and **different from ensembles** (no averaging of predictions; the teacher is a different *information regime*, not a different model of the same regime).

### T2. Cross-sequence retrieval teacher: more *data*, not more parameters.
**Teacher gets**: at training time, nearest-neighbour retrieval across the whole corpus (all ~12,000 training sequences plus validation) for the current prefix signature, producing an empirical posterior over the next run's direction and level; the teacher is "huge" only in the sense of search, not parameters.
**Turning it into a causal gain**: the student is trained against the retrieved posterior's two moments (E[y|prefix retrieval], posterior sign-probability) as supervision, i.e. the teacher provides a *label-space augmentation*, not a predictive output; at serving the student consumes its own causal features only (and its retrieval, if any, is skipped). This is label-space distillation — different from parameter distillation into a wider net, which already failed because the teacher's *output* was redundant with the student's.
**1-day falsifier**: if a per-(class, run-age-bucket) empirical sign-prior lookup built from unlimited retrieval does not beat the current student's own sign accuracy in the >256-run bucket, even the retrieval teacher has nothing to transfer, and this thread dies in one notebook cell.

### T3. Offline hindsight teacher: distill the *ex-post-optimal action*, not ex-ante level guesses.
**Teacher gets**: the full future label path, from which it computes the ex-post metric-optimal *decision* at each row — which of {hold, flip} minimises the counterfactual weighted loss against the realized label path (an oracle action, not an oracle level).
**Turning it into a causal gain**: train the student against the hindsight-optimal action distribution (per-row policy targets) instead of the level y; because hindsight supervision aligns directly with the metric's decision structure (hold vs flip, weighted by w), it sidesteps the "weighted-MSE ≈ metric" approximation that the current loss chain relies on. The teacher's contribution is a *changed supervision signal* — a policy, not a value, and not a prediction substitution.
**Differentiator vs 'substitute the teacher's prediction at t+h'** (no output substitution at serving; the student's policy consumes only its own causal features at inference) and vs **ensembles** (no averaging across models; the signal comes from the *label process*, which no model sees).

---

## 4. Final line

**Run the run-age + change-hazard differential experiment first (R1, extended by T1's oracle-lookahead teacher for the hazard head), because the error decomposition pins sign-timing as the single largest actionable error axis (0.843 vs 0.685), the mechanism is directly causally servable by the student's own features, the experiment is a ~1-day differential against the current best with a crisp falsifier (if adding oracle run-age doesn't move the >256-run sign curve off 74%, the thread dies), and it is the only proposal on this page that attacks a failure mode no hyper-parameter change has ever touched.**

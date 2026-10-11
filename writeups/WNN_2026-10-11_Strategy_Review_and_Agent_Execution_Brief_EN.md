# WNN / Connectome: Critical Strategy Review and Agent Execution Brief

**Prepared:** 11 October 2026  
**Language:** English  
**Purpose:** A self-contained, executable research brief for the next engineering/research agent. This document consolidates and expands the preceding Russian review, the follow-up ideas, and the supplied experiment/status ledgers.  
**Competition objective:** Improve the *official public leaderboard* score beyond approximately **0.672** without violating the runtime, packaging, causality, or evaluation constraints.  
**Best reported public score at this snapshot:** **0.6668**, from `4_z1c_z4_pair_q8vnni.zip`; gap to 0.6720: **0.0052**.  
**Important:** This is an analysis and an experiment specification, **not** evidence that the proposed combinations will work. All numerical local results below must retain their fold/lineage context.

---

## 0. Executive instruction to the agent

**Stop optimizing a single exposed fold as if it were the leaderboard. Stop substituting a high-performing local scratch model for a validated finished student. Do not run another broad HPO sweep.** The best near-term opportunity is to **finish and compare strong all-data student pipelines**, identify a *genuinely complementary* model under the actual serving budget, and then test one meaningful change in distillation data/teacher information rather than ten more variants of alpha and epochs.

In priority order:

1. Reconcile the latest state from receipts, not older narrative summaries. Preserve running work; do not duplicate it. Audit the exact current champion and all useful prediction/checkpoint artifacts.
2. Finish **H388/H390** strong all-data teacher/student production, verify teacher/cache provenance and native prediction parity, then produce fair fold analogues for quality selection. An all-data fit has **no honest held-out score**.
3. Complete **H391**: a width-384 student with the current strong three-teacher target, `alpha=0.9`, six-epoch student recipe, contrasted with the existing strong width-256 student and earlier `h384 alpha=0.7 / e2` student. Judge **single score and marginal contribution to the final blend**.
4. Test pre-registered small sets of **strong-new + strong-old student**, **narrow + wide student**, **two independently trained strong students**, and **student + strong teacher**. Select against matched clean-fold controls and *finish* a feasible native endpoint before any public claim.
5. Make **mixed synthetic-path online distillation** correct (teacher and student see the same modified *causal* history), finish the already admitted experiment, and compare with the **current** student rather than an obsolete weak baseline. Keep real **out-of-sample teacher targets** (cross-fitting on authentic sequences) separate from synthetic **out-of-distribution inputs**.
6. Explore **one small residual correction model on the frozen strong student's hidden state**, or **one training-only hidden-representation KD objective**, not both as sprawling HPO families. Confirm complementarity on unseen components.
7. Use the Q8VNNI serving implementation to maximize *validated score under the full test runtime and ZIP size*, not the number of models. Benchmark each actual ensemble with the official callback path.

**Do not claim** that +0.001 from one fold, +0.001 from another arm, and +0.002 from a selected blend add to +0.004 public. They are distinct, often correlated interventions tested under different controls.

---

## 1. Ground truth and the real selection problem

### 1.1 Public submissions (observed; not forecasts)

| Public candidate | Public score | Meaning |
|---|---:|---|
| `4_z1c_z4_pair_q8vnni.zip` | **0.6668** | Strongest reported public score at snapshot; two all-data students |
| `5_z1c_q8vnni_single.zip` | 0.6649 | One all-data student; pairing was valuable for this lineage |
| `7_dm_t3_a9_e6_fold1_q8vnni.zip` | 0.6652 | Fold-1-held-out student trained with strong T3 teachers, alpha 0.9, six epochs |
| `9_muonall_prop0_e4_s4_fold1_q8vnni.zip` | 0.6616 | Fold-1-held-out scratch Muon-all / prop0 / four epochs; large local-public drop |
| `8_dm_t3_a9_e6_muonall_prop0_e4_s4_pair_fold1_q8vnni.zip` | 0.6650 | Pair of the preceding student and scratch; did **not** improve on student public 0.6652 |
| `1_p2_student_distilled_from_scratch_h256_teachers.zip` | 0.6639 | Earlier fold-1-held-out student |
| `2_z5_seed9_student_muonparent_hft.zip` | 0.6664 | Earlier best all-data finished student |
| P1/P3 uploaded 10 Oct | **Timeout; no score** | No quality conclusion possible |

In the same lineage, the held-out student's local fold-1 performance improved from **0.66938** to **0.67103**, and its public score improved from **0.6639** to **0.6652**. The directions agree, but these two observations **do not establish a calibrated local-to-public conversion or a universal “student transfer class.”** Conversely, the selected scratch model scored **0.67195** on exposed fold 1 but only **0.6616** public. Its locally attractive blend scored **0.67329** on fold 1 and **0.6650** public. Selecting exposed-fold optima is hazardous.

The public results are not repeat measurements of an identical model; estimating “portal noise” from score dispersion across different submissions and declaring statistical significance would be invalid.

### 1.2 What constitutes a fair comparison

For each proposed model or blend, distinguish:

- **Exposed development fold:** useful for mechanistic screening, highly susceptible to adaptive selection.
- **Independent component-disjoint fold:** needed for material promotion of a direction.
- **All-data final:** may be packaged for public submission, but **cannot be assigned a held-out score using the same data it was fit on**.
- **Public score:** external evidence about the exact ZIP, not an explanation of why it scored that way.

Keep all ancestors out of the relevant evaluation component: parent, teacher, student, normalization, propensity estimator, any residual calibrator, and architecture/weight selector. A `fold-1-held-out` student whose teacher or parent touched that fold is not a truly clean fold-1 student.

Score exactly as the official implementation does: compute **separate weighted Pearson correlations for `t0` and `t1`**, clip predictions and true labels as prescribed, and average the two target scores. The existing `gmetric.py` performs this correctly. Do **not** “fix” the metric on the basis of an inaccurate prose description in an earlier report.

Use matched **paired deltas** and component resampling, recomputing the global score after selecting components, rather than taking the mean of component-level Pearson values. Report mean estimated improvement as the decision statistic and use CIs as uncertainty indicators, **not automatic rejection gates**.

### 1.3 The single most damaging methodological error

The chosen public scratch model was an exposed fold-1, **best-of-recipe/best-of-seed** endpoint. All-data public champion 0.6668 was a different, completed two-student pipeline. The local leaderboard ranked them together without matching data membership or finished training procedure. Recipe rank correlation between folds 0 and 1 is imperfect (reported Pearson ~0.82, Spearman ~0.76 for 29 recipes; residual score SD ~0.0015). In the scratch family, even the exact same recipe varied materially by epoch and seed.

**Immediate agent action:** before packaging a locally winning scratch member, reproduce its improvement against the same seed/control on another genuine clean fold. If unavailable, explicitly mark the expected public gain as **unknown**, not `fold1_delta − 0.0057` or any other fabricated calibration.

---

## 2. Current research evidence: what is already closed or has changed status

These findings correct older proposals. An agent must check live receipts before launching.

| Direction | Current archived observation | Consequence |
|---|---|---|
| H230 main + family-head output mixtures | Main head was best; all tested family mixtures lower | Do not repeat generic family-head blending |
| Head finishing on the newer student | Reported **neutral** in the newer run | Do not advertise an untried finishing fix for this exact student; may remain relevant for other lineages |
| Slow/fast linear probe | Apparent **R² 0.26** included simultaneous endpoint increment; excluding it gives **R² 0.006** | **Withdraw** that high-R² predictive-feature argument; only a separate conditional innovation mechanism remains open at low priority |
| Cached-row difference-channel / Pearson / run-level KD | Scores mostly +0.0000 to +0.0002 vs control; sum=0 hurt ~−0.0016 | Narrow KD-loss variants near current cache have low priority |
| Alpha / student epochs / born-again parent | `alpha .9 e6` ~0.67103; `.95 e6` ~0.67094; `1.0 e6` ~0.67068; `.9 e10` ~0.67094; born-again ~0.67054 | Current single-student output-matching chain is near a local plateau |
| Same-prefix small-LR scratch tails | H378 ~0.66891, H380 ~0.66885 vs 0.66897 same-seed control | More epochs alone, after the same prefix, did not improve this specific scratch family |
| Pure scratch Muon-all / prop0 e4 | Locally attractive; public 0.6616; fold/recipe generalization unresolved | Do not place this exact member in the public blend solely because local fold1 is high |
| Generic adaptive gates, ridge stackers, regime tables | Negative/neutral under tested setups | Equal fixed blend is default; only a fundamentally new complementarity-trained member justifies reopening |
| Weight INT4 and software prefetch | Existing attempted kernels negative for their tested performance objectives | No speculative Q4 project without a measured serving need and changed technical method |
| Windowed-feature LightGBM teacher | ~0.618 versus ~0.6985 student on its subset; blend roughly −0.0003 | Do not repeat the same snapshot/window GBM; hidden-state/residual GBM is a distinct untested question |
| Output holding/smoothing and earlier dual-clock GRU | Strong negative or neutral trials | Do not revive output hold as a long-memory fix |
| History volume28 E413 | +0.00096179, CI ~[−0.00008,+0.00206], single seed/fold | Interesting controlled weak positive; needs modern-recipe replication, not a claimed new best |
| OOD-only student | ~0.66730; 50/50 with strong student ~0.66993 on exposed fold1 | OOD-only endpoint not competitive; mixed consistent-view distillation not answered by it |

Additional **source-derived correction**: the complete future-input lookahead ladder (causal ~0.666; lead 2 ~0.680; lead 8 ~0.711; lead 32 ~0.780; lead 128 ~0.891; lead 512 ~0.960) shows that observing the future solves much of the task. **It does not estimate the best attainable strictly causal score**. No one should declare the present GRU “at the information ceiling” from this ladder.

---

## 3. Combination A — new strong student + old proven all-data student

### Hypothesis

The older Z-family and newer T3-distilled family may learn complementary errors because their teacher ancestry, propensity/row weights, initial parents and optimization trajectories differ. The relevant object is the **increment over an existing strong pair** on unseen components, not the novelty of a candidate and not raw prediction disagreement.

### Why this was underexplored

Most attention went to pairing a new student with the high-scoring fold-1 scratch model, which subsequently failed on public. The old all-data Z1c+Z4 pair, however, is the actual public champion at the snapshot. The right question is whether the new full-data student replaces or complements **one member of that proven pair**, not whether the selected scratch should be included.

### Required experiment

Construct truly held-out fold-analogue members of:

- `A`: old finished student (Z-like lineage, including its actual parent/head procedure);
- `B`: current strong T3 student (new alpha .9 / e6 recipe);
- `C`: independently initialized/seeded second current T3 student.

On identical component sets, score these **predefined endpoints**:

1. Best `A` single.
2. Equal `A + A'` old-student control pair (if a fold-clean analogue exists).
3. Equal `B + C` new-student pair.
4. Equal `A + B` old/new pair.
5. For each target separately, optionally one *pre-registered* conservative asymmetric weight (for example 0.6/0.4), evaluated on a different fold from the one used to nominate that weight.

Keep clipping **after** raw-output blending, following official semantics. Store target-specific WP and global score; no per-class oracle routing or label-derived gating at inference.

### Decision and failure interpretation

Promote if `A+B` improves the best fixed equal-pair control on a second component-disjoint panel at acceptable runtime, or if `B+C` produces a more robust improvement. A negative result closes only the exact member lineages, not the principle of older/newer complementarities.

**Do not** compare all-data `A` predictions on fitted valid rows against fold-held-out `B` as if both were OOF. For any all-data endpoint, rank using matched cleanfold **sibling fits**, then train final all-data weights.

---

## 4. Combination B — h384 strong student + h256 strong student (H391)

### Observed starting point

- Current strong `h256 alpha=.9 / e6` student: **0.67103** on exposed fold1.
- Existing **different-recipe** `h384 alpha=.7 / e2` student: **0.67041** fold1.
- Their frozen equal raw blend: **0.67203** fold1.
- A wide student with the strong target/alpha/epoch recipe is registered as **H391**, but at the latest archive snapshot remains **planned/not admitted**. Check current run state before doing anything.

### Mechanism

The wide student may retain subtle ensemble behavior that a width-256 student discards. The existing wide student is weaker by itself but contributes useful errors. This gives a *specific* reason to try a more thoroughly trained wide student. It does **not** show that width alone caused the blend gain, because the first wide and narrow experiments differ in parent, epochs, alpha and architecture.

### Exact arm and controls

**Candidate H391:** train one width-384 student, initialized from an independent strong width-384 parent, distilled from the exact three strong width-384 teacher sources (T3), alpha 0.9, the same six-epoch student schedule and loss formulation as the narrow champion. Fix the nominated checkpoint in advance or use the same checkpoint-selection policy on the same independent development fold as controls.

Controls:

- Unchanged h256 strong `alpha=.9/e6` student.
- Unchanged h384 `alpha=.7/e2` student.
- Equal blend `(h256 + h384 old)/2`.
- Equal blend `(h256 + h384 H391)/2`.
- Equal blend `(h256 strong + independent h256 strong)/2`, if available.

Preserve exact teacher source identities, input normalizers, component exclusions, run order and predictions/checkpoint hashes. Compare **blend deltas**, not only standalone WP. Also inspect `t0`, `t1` and the harder long-run class as diagnostics—but do not optimize exclusively for a class proxy.

### Serving boundary

Use actual Q8VNNI full-callback measurements. The ledger reports indicative local timings of **~113.14 µs/row h256+h384**, **~102.67 µs/row three h256**, **~158.98 µs/row two h384** for particular replay conditions. These are **not official portal times**; they need side-by-side benchmark against a reference package that has a known portal duration. Check complete-package size (<20 MB), target CPU ISA handling, deterministic reset, no fallbacks, and full-fold quantized score parity.

Do not assume a faster standalone student increases ensemble quality; if H391 becomes more correlated with h256, the old weaker wide member may remain a better partner.

---

## 5. Combination C — separate *real OOS targets* from *synthetic OOD inputs*

This is the most conceptually important change suggested by the user.

### 5.1 Real out-of-sample teacher-target experiment (H096 / E442 family)

**Hypothesis:** A student learns a more generalizable teacher function when the teacher supplies targets for **authentic sequences it was not trained on**, compared with conventional in-sample teacher targets.

The older E327 analysis compared teacher predictions on fitted and held-out sequences by class, found mixed signed gaps and inferred little teacher memorization. That diagnostic **did not train matched students on OOF versus in-sample teacher outputs**. E442 planned that scientific comparison, but a completed matched result has not been established. Therefore the mechanism is **not refuted**.

**Minimal clean test** (component-level groups, no overlapping ancestor membership):

1. Freeze outer evaluation components `E` and exclude them from every ancestor and calibration step.
2. Partition remaining fit components into `A` and `B`, matched as well as possible on coarse observable class/activity distributions.
3. Train teachers `T_A` and `T_B` on `A` and `B`, respectively, with identical architectures, budgets and teacher training protocols.
4. Produce teacher outputs on all `A+B` rows:
   - **In-sample control:** `A ← T_A`, `B ← T_B`.
   - **Real OOF target:** `A ← T_B`, `B ← T_A`.
5. Train two students with **identical parent weights, supervised rows, labels, distillation alpha, batch order, seeds, augmentations and optimizer/epoch budget**. Only the teacher-target assignment changes.
6. Evaluate both on outer `E` using exactly the official target-specific metric.
7. Optionally add a third condition using the existing strong teacher ensemble, but do not confound the primary A/B test with teacher training-set size.

**Critical caveat:** teachers trained on half the fit data may be worse; OOF targets are not automatically better. Also teacher in-sample prediction is not “data leakage” in ordinary supervised distillation. This experiment tests **generalization of the teacher function**, not the validity of standard KD.

**Decision:** compare student WP, teacher-to-student agreement and error correlation on outer `E`. The first is decisive. If OOF improves only imitation but not label WP, it is not a leaderboard improvement.

### 5.2 Mixed online distillation from transformed paths (H350 / v9)

**Snapshot:** original `w14_ood_{splice,spmix,swn,swn25}` runs initially died **before fitting** due to cache disk reservation. A compact shared-cache repair followed. Later `spmix` was actually reported fitting; check live terminal receipts before requeueing. The `OOD-only` result (~0.66730) is a specific negative endpoint, **not** a test of mixed real+OOD training.

**Objective:** keep real-data supervised/student-KD training primary and use a limited additional training-only consistency objective:

\[
L=(1-\beta)L_{\rm real}+\beta\,\mathbb{E}_{\tilde x}
  [v(\tilde x)\Vert S(\tilde x)-T(\tilde x)\Vert^2].
\]

Here `beta` is a separate synthetic-path fraction/weight; **it is not** the existing alpha mixing ground truth and teacher predictions. As a *proposed screening point*, one can start near beta 0.1–0.2 and a no-augmentation control. These are **not tuned, validated settings**.

**Invariants (must be tested in code before GPU fitting):**

- Teacher and student receive the **same transformed observed prefix**, with independent hidden states advanced through that same history.
- Changing `raw` inputs requires recomputing **all event flags, first differences and since-event counters** causally. Never pair altered raw paths with stale original engineered features.
- Never assume that original labels remain valid after splice/time warp/noise. Use new **online teacher outputs** as synthetic targets unless an exact label-preservation rule is demonstrated.
- Retain natural run-length and block-update/missingness structure where possible; IID row noise on held features can create fake events on nearly every row.
- Restrict transformations to the observed past/current data. Do not splice in future data relative to a prediction timestamp or carry in future teacher state.
- Check `.detach()`/gradient ownership of teacher, tensor shapes, reset boundaries and the target clipping/scaling path. If a teacher sees a different normalization history, “same raw values” is insufficient.
- Use a deterministic seed, fit-only distribution parameters, cache identity hashes and a fixed allowed sample budget. Do not claim OOD-only as a positive result of mixed KD.

**Recommended order:**

1. Identity-path test: online teacher equals cached teacher closely enough on unchanged real prefixes, with matching teacher selection and state semantics.
2. Natural contiguous subsegments + causal burn-in (authentic real paths, no synthetic label transfer).
3. Mild structured block-level perturbations that preserve update clocks and sentinel patterns.
4. Only then splices/time-warp, with explicit ablation versus weaker sham transformations.

**Primary comparison:** `strong current h256 student real-only` vs `same student real+augmented teacher KD`, with the same total real-row training and a matched compute/additional-steps control. Measure cleanfold score *and* fixed pair contribution. A gain relative only to obsolete `dsq_t256=0.66938` is not enough to promote the direction.

### 5.3 Strict label-free KD is a different hypothesis

An `alpha=1.0` setting does not automatically remove all label influence: the implementation may still use `|y|` row weights, auxiliary losses or label-dependent inclusion. Confirm on actual code. A strict label-free mode should yield identical loss/gradients under changed true labels when teacher outputs and input paths are fixed. That test establishes semantics, **not** expected WP improvement. Keep it as a single narrow contrast; prior `alpha=.95` / `1.0` results did not beat the current strong student.

---

## 6. Combination D — train a complementary correction, not another generic full network

### Core idea

A conventional second model minimizes its own weighted prediction error, and the team then hopes it is complementary. Instead, **fix the incumbent anchor** and explicitly train a cheap additional model for the missing function:

\[
  p_t = A(x_{\le t}) + \lambda\,g(h^A_t,z_t),
\]

where `A` is the frozen strong student, `h^A_t` its causal hidden state, `z_t` a few train-only-selected causal features (raw/event summaries are eligible), and `g` a very small output/residual network, optionally with compact recurrent state.

Potential supervision choices:

- **Label-residual mode:** `clip(y_t,2) − A(x≤t)`; tests whether the corrector can extract label signal not used by A.
- **Teacher-residual mode:** `T(x≤t) − A(x≤t)`; attempts to retain teacher information the student lost.

For label-residual mode, `A` must produce **OOF / excluded-component predictions on the residual-training rows**, or the corrector will learn a misleading in-sample residual distribution. Teacher-residual mode also requires an honest outer evaluation set for the whole ancestor lineage.

### Minimal implementation and controls

1. Freeze a cleanfold anchor `A`, export pre-head or post-GRU hidden activations **without leaking future rows**.
2. Zero-initialize `g`'s final output so initial `A + g` equals `A` numerically.
3. Train a tiny `g` (e.g. compact MLP from 16–64 projected features; exact size to profile first) with fixed `lambda` and limited head training steps.
4. Compare against a same-cost `g` trained from *only two predictions* and against an independently trained tiny GRU/MLP of similar serving cost. This distinguishes hidden-state information from extra parameters.
5. Compare incremental WP over `A` on true disjoint components. Record per-target improvement, clipping behavior, calibration drift, and whether `g` degenerates to an affine rescale.
6. Package only if both cleanfold gain and full callback cost are favorable. Optionally distill the composite prediction back into one model **after** validating that the composite is stronger.

This is **not** the same as the older post-hoc GBM/ridge/MLP stackers on model outputs. E276's negative residual booster addressed a tabular booster on predictions, not a purpose-trained compact network on strong hidden states. Existing LightGBM on windowed raw features was also weak. Do not open a large AutoGluon/GluonTS search before this controlled hidden-state test works.

### Avoid false complementarity

Minimizing pairwise prediction correlation or forcing an error-correlation penalty can make a model *worse and different*. The optimization and selection target is:

\[
\Delta=WP(A+\lambda g)-WP(A)
\]

on components not used to fit or select `g`. A weaker standalone corrector is acceptable **only if the joint endpoint is actually better**.

---

## 7. Alternative training-only mechanism: low-rank hidden-state KD (H373)

Output-only KD compares two numbers per row, while the teacher and student may use substantially different representations of market history. This is a possible reason why a strong teacher ensemble (~0.67254 on exposed fold1) is not fully reproduced by the best student (~0.67103). Merely increasing the weight on the teacher output difference-channel or altering Pearson KD did not solve this.

**Distinct experiment:** add a small auxiliary objective that makes the student predict a **fixed, low-rank projection of the teacher's causal hidden representation** on matching observed prefixes.

- Freeze each teacher and a *separate* projection for its hidden state (e.g. an 8–16-dimensional fitted/random orthogonal projection; this dimension is an unvalidated engineering default).
- Avoid averaging coordinate-aligned hidden units across separately trained teachers; their bases need not correspond.
- Use the student's projected hidden state or a training-only auxiliary head to regress the detached teacher features.
- Normalize auxiliary magnitude and confirm main-loss gradient scaling. Select a small fixed auxiliary coefficient against a zero-coefficient/equal-compute control.
- Do not store dense hidden trajectories for all 20,000-row sequences if online batch inference or low-dimensional caches suffice.
- Drop training-only heads at export; verify the public serving architecture is unchanged.

**Falsifier:** no improvement on the official label WP or no incremental contribution to a strong pair under matched controls. Higher teacher imitation alone is not evidence of predictive benefit.

This direction was present in the research registry as a proposal, not established as a completed scientific fit. It should receive **one controlled pilot**, not an unlimited parameter grid.

---

## 8. How to stack small improvements without double-counting them

Improvements must address **different failure modes** to be plausibly additive. Use the following factorization:

| Mechanism | Example | What it adds / changes |
|---|---|---|
| Better supervision | Strong T3 teachers, authentic OOF teacher targets | Quality and generalization of the target function |
| Better input coverage | Consistent-view natural/OOD path augmentation | Where student learns the function |
| Better student capacity | h384 student, representation KD | How much teacher information can be retained |
| Complementary error correction | Old+new fixed blend, trained residual branch | Errors left after the anchor predictor |
| More serving headroom | Q8VNNI full-callback optimization | Whether multiple useful components can be deployed |

Run each standalone against its exact baseline, then test its **incremental contribution after the previously accepted interventions**. For instance:

| | Real-path student training | Mixed, consistent-view additional paths |
|---|---|---|
| **h256 current strong recipe** | Existing control | Augmentation effect |
| **h384 current strong recipe** | Capacity/recipe effect | Interaction of capacity with input coverage |

Do not execute this as a four-way grid until its existing cells have been verified and the remaining jobs are affordable. The first two/three cells may already exist; reuse exact artifacts. This is a **small hypothesis-driven interaction test**, not broad HPO.

Past campaigns already found non-additivity: one propensity improvement disappeared under stronger EMA; short-recipe conv features did not transfer to the full EMA recipe; longer student tails and head variants flattened; naive T6 teacher ensemble underperformed T3. Do not simply sum their historical fold gains.

**Secondary cheap candidates:** a modern matched E413 volume-history replication and one predefined target-specific static blend. Both are lower priority than complete strong-student comparisons. For target-specific blending remember the official score averages two target correlations, but the extra fitted coefficients increase selection risk. Never use target-derived class/routing at serving.

---

## 9. Serving/runtime strategy — optimize the *quality frontier*

The competition's documented evaluation CPU is **AMD EPYC 9224** (verify current official documentation if specifications change). The archive notes Zen 4, 1 MB L2 per core, a 256-bit execution datapath, and relevant VNNI support. Final restrictions include **one CPU core**, a **60-minute test limit**, a **20 MB ZIP limit**, causal state reset, and the prescribed Python callback interface.

The archive reports Q8VNNI pair submissions near **25–27 minutes** public, while older i16 runs used ~33–39 minutes for singles and were more variable. These are *specific submitted artifacts*, not an exact law for other ensembles. In E525 the project's q8x was faster than its i16 path, and later a faster Q8VNNI engine was built. The optimization problem now is choosing the **highest-quality package that fits**, not merely speeding up a kernel for its own sake.

**Concrete candidate frontier:**

- Two h256 strong students.
- One h256 strong student plus one h384 strong student.
- Three h256 students, *only if third provides real incremental OOF gain*.
- One strong student plus h384 teacher.
- One strong student plus a tiny residual head/branch.

For each candidate, record: official cleanfold FP32 WP, exact native Q8 WP, per-target WP, score difference after clipping, ZIP size, total bytes resident, local full-callback time on ≥ multiple complete sequences and a conservative portal-time budget. Also check actual target-ISA dispatch and weight packing. A 10-sequence parity gate with correlations >0.99997 makes a gross numerical mismatch unlikely, but is **not** complete-fold score parity.

Int4/q4 or other weight compression becomes a priority **only when a specific validated useful blend otherwise fails time or package-size constraints**, and a new integer4 decode/kernel design addresses the previously measured loss. Fewer weight bits do not inherently mean faster single-core inference.

Don't assume timing scales linearly with member count or width; benchmark exact packed models in the full callback. Python overhead, cache working set, persistent state, CPU contention, ISA support and scorer characteristics can all change runtime.

---

## 10. Evaluation, selection, and scientific controls

### 10.1 No single-fold promotion

At minimum, nominate a mechanism using dev fold1, then check on a **second component-disjoint fold** with matching parent/teacher/normalization ancestry. If fold2/3 are heavily exposed through prior work, say so honestly; no repeated use turns an exposed panel into a “fresh” holdout.

When comparing algorithms rather than finished ZIPs, keep the same seeds, fit component counts, checkpoint selection and full recipe. When comparing real final pipelines, use **complete pipelines** (parent, teacher, student, finishing, packaging). A half-trained fold student must not be ranked as equivalent to a full-data finished pair.

### 10.2 Recommended bounded decision protocol

For an **existing-bank combination**:

1. Check all member prediction arrays correspond to the same scored rows, fold, source hashes and clipping semantics.
2. Score pre-registered equal blends first.
3. Report paired global WP delta and per-target WP.
4. Repeat on a second cleanfold with analogously trained members.
5. Test exact native end-to-end cost.

For a **new training mechanism**:

1. Write one-sentence causal hypothesis and what would refute it.
2. Confirm existing ledger does not already contain an identical completed run.
3. Implement unit tests for causality/reset/state/normalization/teacher target alignment.
4. Run **one matched seed and fold** against the unchanged strong recipe, not the weaker historical baseline.
5. Spend the second fold/seed only if the first gives informative evidence or a plausible needed control.
6. Compare it in the best current feasible blend, not only in isolation.
7. Only then prepare the all-data final, with no held-out score claim for the actual all-data weights.

Do not introduce arbitrary hard thresholds like “halve fold1 delta,” “ignore all gains below 0.003,” or “reject if CI crosses zero.” Previous review retracted these rules as unsupported. Select by mean; uncertainty determines replication budget and how forcefully you describe a finding.

### 10.3 A reporting template for each experiment

```text
EXPERIMENT ID:
MECHANISM / ONE-LINE HYPOTHESIS:
EXACT COMPARISON / WHAT CHANGES:
SOURCE REVISION / INPUT HASHES / CACHE + CHECKPOINT HASHES:
DATA COMPONENT MEMBERSHIP / EXCLUSIONS / ANCESTRY:
TRAINING RECIPE / SEEDS / SAMPLER / RUN COUNT:
STATUS (prepared | queued | fit-started | fit-finished | exported | graded | packaged | public):
OFFICIAL WP control / candidate / paired delta:
WP_t0 / WP_t1:
PAIRED COMPONENT CI + N COMPONENTS:
MARGINAL GAIN IN PRE-REGISTERED BLEND:
NATIVE QUANTIZED WP PARITY / CALLBACK TIME / ZIP SIZE:
INTERPRETATION, CONFOUNDS, AND DECISION:
NEXT ACTION OR CLOSURE REASON:
```

---

## 11. Resource, reproducibility, and execution instructions

The research archive documents repeated cases where a purported experiment failed *before scientific fitting* (Vast OOD disk-budget failure, Kaggle input-footer collision, missing pipeline on host) or completed fitting but lacked export/grade. **Treat infrastructure failure as execution evidence, not a negative scientific result.**

Before running anything, inspect the **live source of truth**:

- `research/RESEARCH_NOTES_INDEX_20261010.md`
- `research/hypotheses.jsonl` and `research/experiments.jsonl`
- `research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md`
- `research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/IDEA_STATUS_AND_EXPLOITATION_PLAN_20261011_R1.md`
- `prediction_bank/` and especially the corresponding `BANK_FOLD*.jsonl` and latest grade files
- `research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/`
- `research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/review_public_gap_20261011/REVIEW_PUBLIC_GAP.md`
- `research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/distillation_deep_dive_20261011/`
- `native/INT8_VNNI_REPORT.md`, `SCORER_CPU_SPEC_20261010.md`, and exact package build receipts
- `.runtime/`, job status, latest alerts (historically `scripts/alerts_unread.sh`) and provider billing/remaining quota.

Some paths above are **repository-relative pointers recorded in the supplied archive**; verify their current existence before scripting. This document does not have live access to a future research host or its job scheduler.

**Specific historical resource snapshot:** Vast B credit was reported at ~$4.70 and auto-topup OFF around 12:56 UTC 11 Oct; local queue was empty at one later report; H388 all-data h384 teachers had reached fit/export terminal; H390 cache/final student and mixed OOD were undergoing delivery/fit; Kaggle H382 had had multiple pre-fit package errors. These are time-stamped statuses, **not current instructions to rent, kill, restart or spend**. Read new terminal receipts and user-authorized budget before any paid dispatch.

Preserve source, models, OOF predictions, and teacher caches by hash. Do not overwrite prediction bank files when cloning jobs. Atomically write exports. Deduplicate jobs by (data split, source, config, seed, parent, teacher cache). Keep deterministic same-seed replications distinct from *independent* seeds. Maintain safety margin for RAM, cache staging, local workload and paid-credit exhaustion. Commit meaningful source/evidence updates with file references and score tables. A monitor is only useful if someone **reads and reacts** to its alerts.

Do not blindly try “everything possible”: computation should maximize **completed hypothesis comparisons that change decisions**, not GPU utilization percent.

---

## 12. Ranked work packages and acceptance criteria

| Rank | Work package | Available evidence | Deliverable / decision | Main risk |
|---|---|---|---|---|
| **P0** | H390 all-data strong student pair + cleanfold analogues | Old all-data pair public 0.6668; strong fold1 new student 0.67103 | Two or fewer robust finished Q8 finalists, backed by valid lineage | All-data artifacts may lack fair CV analogues |
| **P0** | Fixed old/new student blends | Public old pair works; new student gain partially transferred | Predeclared cleanfold pair table and full-callback package | Apparent diversity may vanish on unseen components |
| **P1** | H391 strong h384 student + h256 pair | Existing mismatched wide/narrow blend 0.67203 fold1 | Single and incremental pair comparison; native feasibility | Better wide model may become too similar |
| **P1** | Finish mixed consistent-view OOD KD (not OOD-only) | Earlier mixed jobs blocked; later spmix fitting | Real-only versus real+OOD on same strong recipe | Invalid synthetic event clocks or off-distribution teacher bias |
| **P2** | Authentic component-cross-fitted teacher outputs | E327 not a student crossfit test; E442 remained uncompleted | Paired student comparison with fixed A/B teacher sizes | Half-data teachers weaker; high compute |
| **P2** | Tiny hidden-state residual member | GBM on predictions negative; hidden-state correction not established | OOF incremental global WP and measured CPU cost | Residual overfit / in-sample anchor residual mismatch |
| **P2** | H373 low-rank hidden representation KD | Proposed; no completed matched fit established | One training-only KD pilot | Teacher hidden-coordinate mismatch; no label gain |
| **P3** | E413 volume-history modern replication | +0.00096 imprecise single fold | Matched history/current control on second panel | Gain disappears under stronger student |
| **P3** | Strict label-free KD semantics and one fit | Alpha1.0 close to but below leader; label-based weights may remain | Semantic unit test + one matched student run | Objective no longer aligns with official scored distribution |

**Promotion standard:** a candidate should be better than the **strongest feasible incumbent ensemble** under a predefined cleanfold comparison, and must pass exact native callback, reset and size constraints. Exposed fold1-only “0.673” is not permission to forecast a public 0.673. A submission ZIP is ready only after the entire pipeline passes; public uploading remains a user-controlled action.

---

## 13. Instructions for the next research agent (copy directly as the mission)

> Your mission is to improve the Wunder Connectome public weighted-Pearson score beyond the current 0.6668 (target >0.672) without violating causal inputs, official metric, 1-vCPU/60-min runtime, or ZIP constraints. Treat the preceding document as the experiment design; **confirm all live experiment statuses against the repository before starting work**. Prioritize completed, genuinely component-disjoint comparisons of end-to-end pipelines.
>
> **First**, inventory H388/H390/H391, the mixed OOD jobs, current strong-student fold analogues and exact existing ZIP receipts. Recover finished fits/exports/grades before launching anything; keep running unrelated branches intact. Report statuses as actual `fit_started`, `fit_done`, `exported`, `graded`, not optimistic `queued` summaries.
>
> **Second**, build a small fixed strong-member blend comparison: old Z-like student plus new strong T3 student; two independent strong T3 students; h256 strong plus existing/stronger h384 student; strong student plus strong h384 teacher. Use genuinely excluded components for each member's entire ancestry and compute the official target-separated weighted Pearson, paired delta, and quantized callback runtime.
>
> **Third**, if H391 is still unrun and preflight passes, execute exactly one h384 strong-target/alpha.9/e6 arm with an equal-work h256 comparison, preserving the existing h384 alpha.7/e2 benchmark. Report marginal ensemble gain, not just its solo score.
>
> **Fourth**, finish the correct mixed OOD online distillation against the *current* strong student. Require identical transformed causal prefixes for teacher and student and regenerated event features. Maintain a separate design for **real component-OOS teacher targets** using two equally sized teachers; do not conflate it with artificial OOD paths or claim OOD failure from startup crashes.
>
> **Fifth**, implement only one inexpensive residual-on-hidden-state or low-rank hidden-KD pilot if core deliveries are secure. Use zero-initialized correction, exact OOF anchor predictions for residual-label training and a parameter-matched control. Select by incremental global WP on unseen components.
>
> **Throughout**, avoid broad HPO, repeated alpha/epoch sweeps, fixed label-run output holding, generic stacking over old weak models, and speculative q4. Check current watchdog, remaining Vast credit and Kaggle quota before any new dispatch; preserve unrelated experiments and evidence. Git-commit each meaningful change and emit a machine-readable result receipt plus concise Markdown report. Keep user approval for public uploads and any new paid compute not already authorized by the latest live budget.
>
> **Final output:** a ranked list of at most a few genuinely competitive ZIPs, their exact model/teacher/parent lineage, cleanfold comparison against the 0.6668-class incumbent, official FP32/Q8 difference, actual full-callback runtime/size, reason for expecting (or not expecting) a public improvement, and explicit unresolved uncertainty. Do not invent a numeric public forecast from one exposed fold.

---

## 14. Research sources and provenance

The following materials were provided in the conversation and are the principal basis of the empirical claims. Any design suggestion not directly measured here should be treated as a **proposed hypothesis**.

### Attached snapshot sources

1. `Pasted markdown(20261011-132018).md` — review of the three 11 Oct public submissions, quantization parity, local/public mismatches, retractions of overstated transfer rules, latest KD and same-state-tail grades.
2. `Pasted markdown (2)(7).md` — latest controller/current-continuation snapshot; all-data work, constraints and handoff.
3. `Pasted markdown (3)(1).md` — idea-ledger classification, latest T3/width/diversity evidence, OOD delivery statuses, H391 proposal and runtime observations.
4. `Pasted text (4)(2).txt` — archived hypotheses, including H350, H353–H373, H387–H391 and canonical `REV-*` clarifications.
5. `Pasted text (5).txt` — archived experiments, including E327/E413/E442 and recent H350/H378/H380/H388/H382/H390 receipts.
6. `LOOP_LOG_20261010.md`, `RESEARCH_NOTES_INDEX_20261010.md`, and the accompanying `wnn_research_notes_20261010.tar.gz` — previous-day full research context.
7. Prior English consolidated critique: `WNN_Unified_Review_EN.md`; earlier solution description `Pasted markdown(20261010-151809).md`.

### Specific reproduction pointers

- `review_public_gap_20261011/REVIEW_PUBLIC_GAP.md` (especially corrections/addenda 11–14).
- `ops_continuation_20261011/H384_STUDENT_GRADE_FREEZE_R1.json`, `H384_STUDENT_SCORES_R1.txt`.
- `ops_continuation_20261011/QUALITY_DIVERSITY_PAIRED_R1.jsonl`.
- `ops_continuation_20261011/LOCAL_TERMINAL_AND_SEED_SUMMARY_R5.json` and `H378_LOCAL_TERMINAL_R1.json`.
- `ops_continuation_20261011/MONITOR_TERMINALS_R6.json`, relevant H390/H388 manifests and cache preflight receipts.
- `distillation_deep_dive_20261011/CEILING_AND_PLAN.md`, `CACHE_ONLY_DIAGNOSTICS.md`, `LITERATURE.md`.
- `gbm_teacher_20261011/RESULTS.md`, `blend_probe_20261011/RESULTS.md`, `eda_residuals_20261011/RESULTS.md`.
- `native/INT8_VNNI_REPORT.md`, `SCORER_CPU_SPEC_20261010.md` and the actual Q8 submission build receipts.
- Historical E327 in-sample/held-out teacher diagnosis; E413 volume-history matched test; E442 proposed genuine crossfit test; E276 negative post-hoc booster.

### Outside conceptual research (not empirical proof for this dataset)

- Beyer et al., *Knowledge Distillation: A Good Teacher Is Patient and Consistent*, CVPR 2022, https://arxiv.org/abs/2106.05237 — consistency of teacher/student input views and training schedules; originally evaluated in image distillation.
- Stanton et al., *Does Knowledge Distillation Really Work?*, NeurIPS 2021, https://arxiv.org/abs/2106.05945 — teacher/student imitation accuracy and downstream generalization need not track each other.

The first two papers motivate focused experiments; **they do not establish gains on anonymized order-book sequences**. The quantitative values here originate from the supplied research archive or user-provided public results, not from those papers.

---

## 15. Bottom line

The core opportunity is not another recipe sweep. It is to **turn already strong teacher/student work into finished, genuinely comparable ensembles**, while testing exactly one new source of information about the teacher's function. The strongest combinations to prioritize are **new student + robust old student**, **strong narrow + strong wide student**, and **real+consistent-view teacher distillation**. For larger gains, the distinct hypotheses are **authentic out-of-sample teacher targets**, **hidden-representation transfer**, and a **small model trained to correct the champion's residual**, not a generic additional predictor.

The path to 0.672 public is uncertain. The scientific workflow should make the next result interpretable regardless of whether it improves. A reproducible +0.0005 on two truly excluded component folds and a deployable endpoint is more useful than +0.003 from another selected model on fold 1 that collapses in public.

# All-data teacher-history wave ADH-1

Status: recipe preparation. This document admits no fit, no rental, no submission, and no use of the Vast machine that is already staging the fold-0 pair.

Objective mode: competition. Utility is information about the next served student target. Public score is an external observation that this wave does not forecast.

Verification:

| Surface | Tier | What it can support |
| --- | --- | --- |
| Row coverage, coefficient sums, reset parity, determinism, clip-shell counts, residual orthogonality | A | Artifact validity |
| `global_wp_clip2_v1` on supplied train 10606 + valid 1873, or on the clean fold 10535 + 1405 | C | Exposed proxy only. These rows are already fitting rows, teacher rows, or previously inspected rows |
| Public leaderboard | outside this wave | No promoted claim and no numeric forecast |

A gain on an exposed proxy stays a proxy. It is not added to the public best 0.6660. The teammate report 0.6674 stays unreplicated.

## Contract

Question. Does the current heterogeneous student target lack all-data teacher history, lack a unit scale under the official ±2 clip, lack a component orthogonal to that scale, or some combination of those?

Deliverable of a later admitted run, not of this card: frozen full-row prediction artifacts, a target-assembly manifest, reset evidence, and at most the predeclared student screens below.

Invariants.

- Own independently trained weights only. The existing all-data GRU256×2 epoch-3 parent is eligible because it is already our checkpoint.
- Causal features only. Sequence state resets at each sequence boundary. No future inputs at export or at student inference.
- Official evaluation clips predictions and labels at ±2, then applies the fixed scored mask and the original clipped-label magnitude weights. Training-target clipping stays an explicit switch. This wave saves unclipped teacher outputs.
- All-data fitting has no held-out claim.
- Family-target pairs stay beside the primary blend. They are not folded into the 0.8125 mass.
- The fold-0 Vast job (TCN128×12 versus GRU256×2, raw scratch, 4 epochs, AMP, 4096) is a different experiment. Its live metrics do not set coefficients here, and its checkpoints are fold histories if they later succeed.

Baseline target C0. Unnormalized primary blend, coefficient total 13/16 = 0.8125:

| Slot | Each seed | Pair mass |
| --- | ---: | ---: |
| Own strong pair | 0.15 | 0.30 |
| Own full-finetuned pair | 0.10 | 0.20 |
| Own raw-WMSE pair | 0.125 | 0.25 |
| Own GRU128 raw-baseline pair | 0.03125 | 0.0625 |
| Total |  | 0.8125 |

Those eight teachers were fit on the clean fold (10535 train + 1405 valid). Their saved predictions already cover all 10606 + 1873 sequences. Coverage is not an all-data training history. Four raw family-target pairs accompany this blend. Their numeric coefficients are a parser audit; the student family-loss weight 0.1 is a loss weight, not a license to invent per-pair blend coefficients.

Current student recipe, held fixed across every student screen in this wave: two epochs, checkpoint used at epoch 1, batch 16, chunk 2000, raw chunk WMSE, learning rate 0.0001, AdamW weight decay 0.01, OneCycle, propensity power 2 with floor 0.03, valid-mask weight 0.03, structure auxiliary 0.3, next auxiliary 0.3, family loss 0.1, distill coefficient 0.7. Initializer: the same own all-data GRU256×2 epoch-3 parent for every screen. The two in-flight students are two seeds of one initializer, so seed agreement does not replicate a mechanism. Seeds stay secondary in this wave.

Known timings, local only, not 3090 timings: all-data parent 4 epochs 3635 seconds; student 2 epochs about 2000 seconds. Any other cost below is a qualitative statement or a labeled projection.

## Mechanisms

Three different objects. Two of them are models. The third is a scale operator on predictions that already exist.

1. History (replication-gap model). Gated recurrent state of the existing all-data GRU256×2 at the fixed epoch-3 checkpoint, exported for every supplied row. Same architecture family as the strong pair. The membership of the fit is the thing that changes: all supplied sequences, versus the clean-fold strong pair.
2. Scale (no new model). Multiply the frozen C0 blend by 16/13 so the same relative weights sum to 1. This is the clip/calibration contrast. Coefficient total 0.8125 changes which predictions land inside ±2.
3. Orthogonal residual (new combination, second model). A from-scratch all-data TCN128×12 raw teacher, then the component of its predictions that a single scalar multiple of the all-data GRU cannot match. Dilated causal convolution has a finite receptive field and a buffer reset. The GRU has a carried hidden state and a hidden-state reset. Those are different mechanisms. A weaker standalone TCN score does not retire this arm.

Deferred same-family capacity check, not part of the first wave: an all-data GRU128 raw teacher. The blend already contains a fold-trained GRU128 pair at mass 0.0625, and Card S already isolates global scale. A new GRU128 fit waits unless the TCN lane is blocked by quota.

## Shared artifact law

Every teacher file is a full-row export before any student starts.

- Rows: all 10606 train sequences and all 1873 valid sequences. Missing keys are a hard failure. No zero-fill.
- Channels required for this wave: two primary targets, unclipped. Family channels stay on the existing accompanying pairs until a teacher head is shown to emit them.
- Alignment keys and the on-disk schema (column names, parquet versus npz, sequence id versus row order) are a parser audit of the current export path. Position-only joins are refused if the existing schema has sequence identifiers.
- Manifest fields: checkpoint content hash, data hash, fit membership (`all-supplied` or `clean-fold-10535-1405`), epoch index, code revision, export command, dtype, row counts, per-target coefficient, parent head width.
- Canonical dtype is float32 predictions computed from the checkpoint in full precision. An AMP training run does not make fp16 the stored target. Record the source dtype if the checkpoint is fp16.
- Determinism: a second export of the same checkpoint matches within a predeclared absolute tolerance of 1e-5 per value.
- Reset test, both families: export sequence A, sequence B, and the concatenation A‖B with the implementation's sequence boundary removed. After a correct reset, predictions on B equal standalone B. A carry into B fails the artifact. GRU evidence is a zero initial hidden state at each sequence start. TCN evidence is an empty causal buffer at each sequence start, causal padding only, no bidirectional kernel. Kernel size, dilation schedule, and receptive field are a parser audit; the reset test does not need those numbers first, and a failed reset blocks use of that file.
- Clip diagnostic, stored beside the target and not baked into it: for each primary channel, count rows with absolute value in `(2, 32/13]`. `32/13 = 2 / (13/16)` is the open-closed shell where a unit-scale prediction is clipped by the official metric and the 0.8125-scaled prediction is not, when the two differ only by the factor 16/13. Also store the count of rows with absolute value above 2 under C0, under the unit-scale target, and under each candidate.
- Student heads: the current endpoint has 23 ordinary head rows extended to 31 with 8 fresh family rows. This wave changes only the two primary channels of the distill target. Structure, next, and family channels keep their current sources. Head-width of the epoch-3 parent is a parser audit. If that head emits more than two channels, record the widths and still publish the two primary channels. Do not silently map parent outputs onto the eight family rows.

Exposure budget. Primary scored student screens, if later admitted: at most four, one seed each, same initializer and same student hyperparameters. The in-flight student counts as the C0 observation only when its frozen config hash matches C0. Extra seeds, extra epochs, and rebuilding the other seven teachers are outside this budget. Selecting a winner on the exposed proxy spends that proxy. It does not create confirmation.

## Card S — scale control

Purpose. Measure how much of the present target is the coefficient total 0.8125 interacting with official clipping.

Ancestry. Same eight fold-trained teachers as C0. No new checkpoint. Fit membership remains clean-fold training with full-row prediction coverage.

Definition. Let `B` be the current unnormalized blend. In float64,

`B_unit = B * (16/13)`

then store float32. Multiply the vector. Do not re-round the eight coefficients separately and add them again. Relative weights stay exactly the C0 weights. Sum of the scale factors is 1. Family pairs are unchanged and stay outside this product.

Candidate versus control. Control is C0 (`B`). Candidate is `B_unit`. The student recipe is unchanged, including distill coefficient 0.7. Whether 0.7 mixes two loss terms or blends the teacher into the label is a parser audit. Either reading, Card S changes only the teacher tensor.

Cheap evidence, before any student. The clip-shell count defined above, plus per-target Pearson of `clip(B, ±2)` versus `clip(B_unit, ±2)` on all covered rows. A large shell count means scale reaches the official clip. A tiny shell and a Pearson of essentially 1 means the 0.8125 factor is idle at the clip.

Falsifier for "scale is the active difference." The shell is empty on both targets and clipped Pearson is 1 within 1e-6. Card S then has no student value. The factor remains recorded so later blends do not drop it by habit.

Failure mode. Raising the blend from 13/16 to 1 moves more rows onto the clip boundary. Clipped Pearson can change because saturated predictions lost rank variation, while the underlying teachers are identical. A proxy move in that direction is a clip artifact. Report the shell count next to any proxy so the clip artifact stays visible. Unit scale can also hurt raw chunk WMSE while helping or hurting weighted Pearson; those two numbers answer different questions.

Cost. CPU assembly inside the existing prediction cache. No GPU fit.

## Card R — replication-gap teacher

Purpose. Put one real all-data recurrent history into the blend without retraining eight teachers.

Ancestry. Checkpoint: own all-data strong GRU256×2 parent at fixed epoch 3. Independently trained. Prediction export is the missing piece. This checkpoint is also the student initializer. Declare that dual use. If a checkpoint at epoch 4 from the measured 3635-second run is already on disk, export it as a side artifact for a fit-length diagnostic. Do not train a new run to create epoch 4.

Definition. `A` is the two-channel unclipped export of that epoch-3 checkpoint on all 10606 + 1873 sequences.

`B_hist` keeps every C0 coefficient except the strong pair. Strong-pair mass 0.30 is removed from the two fold-trained strong seeds and placed on `A` as a single component of 0.30. One checkpoint is one component. Copying it into a fake second seed is refused.

| Component | C0 | B_hist |
| --- | ---: | ---: |
| Fold strong seed 0 | 0.15 | 0 |
| Fold strong seed 1 | 0.15 | 0 |
| All-data GRU256×2 epoch 3 | 0 | 0.30 |
| Fold full-finetuned pair | 0.20 | 0.20 |
| Fold raw-WMSE pair | 0.25 | 0.25 |
| Fold GRU128 raw pair | 0.0625 | 0.0625 |
| Primary sum | 0.8125 | 0.8125 |

Family pairs stay on their current fold histories.

Cheap evidence, before any student.

- Reset, coverage, determinism, and head-width audit on `A`.
- Per target, Pearson of `A` versus the mean of the two fold strong seeds, both unclipped and after ±2 clip, on all covered rows.
- Relative residual RMSE of that strong-pair mean after a single scalar projection onto `A`, fit in float64 on all covered rows: `c = (A · S) / (A · A)`, residual `(S - c A)`, RMSE divided by the standard deviation of `S`.
- The same two statistics restricted to the label-magnitude shell `|clip(y, ±2)| >= 1` and, separately, the label-free shell `|A| >= 1`. The label shell is a metric-weight diagnostic. It is not an inference feature.
- Clip-shell counts for `B` versus `B_hist`.

Allocation rule, predeclared, and not a mechanism refutation.

- Skip the Card R student when both targets have clipped Pearson at least 0.995 and relative residual RMSE at most 0.02 on all covered rows. Record the student effect as `not_tested`. Spend the student slot on Card S if Card S's clip shell is nonempty. Queue Card N.
- Otherwise the Card R student is the replication-gap screen.

A skipped student does not mean all-data history is useless. It means this 0.30 swap is almost a rescaling of the fold strong pair, so the informative next spend is scale (Card S) or a different mechanism (Card N).

Falsifier for "the epoch-3 all-data export changes the served target." The skip rule above fires, and the clip-shell count between `B` and `B_hist` is 0. Later student disagreement would then come from optimization noise.

Failure mode. The student is initialized from the same checkpoint that now occupies mass 0.30 of its teacher. Distill loss can fall because the student is matching itself, including on rows where the parent memorized all-data labels. Judge the screen by official weighted Pearson on the predeclared exposed scored mask, and store two diagnostic distances: student versus `A`, and student versus `B_hist`. A drop in the first with a flat official metric is self-distillation. In-sample all-data targets can also raise the proxy by copying teacher errors on the same rows the student fits. That result still has no held-out claim.

Second-step mass, only if the first Card R student is admitted and its exposed proxy differs from C0 by a predeclared reporting margin. The margin is descriptive because this proxy has no fresh sampling unit: absolute official-metric gap of 0.002 or larger. The follow-up replaces the fold raw-WMSE pair mass 0.25 as well, moving that mass onto `A` (new coefficient 0.55 on `A`, raw-WMSE pair at 0). That second step is still one student and still inside an extended budget the controller must name. It is not an eight-teacher rebuild. A flat first screen falsifies the 0.30 swap at this student budget only.

Cost. One inference export. Projection: well below the measured 3635-second, 4-epoch local train, because no optimizer runs. Confirm VRAM on the first batch inside the 5 decimal GB local envelope. On export OOM, reduce the inference batch. Leave the weights unchanged. Student cost, if later admitted, projects near the measured 2000 seconds only on a comparable local device. That projection is not a 3090 measurement.

## Card N — scale-orthogonal TCN residual

Purpose. Test a new combination: all-data convolutional history added only in the part the all-data GRU does not already span, held against an equal-mass scale control so diversity is not confused with calibration.

Ancestry. New own training. Architecture TCN128×12, raw objective, from scratch, all supplied sequences 10606 + 1873, own initial weights. No GRU weight transfer, no teammate weights, no reuse of the fold-0 Vast checkpoint as an all-data teacher. That fold-0 run, if it completes, remains a fold-membership architecture screen.

Family coefficients. Let `T` be the two-channel unclipped all-data TCN export. Estimate one scalar per target on two frozen masks:

- `M_all`: every covered row
- `M_valid`: the supplied valid sequences only (1873)

`c = (A · T) / (A · A)` in float64 on that mask. Publish both scalars. If `|c_all - c_valid| / max(|c_all|, 1e-8) > 0.05` on either target, the projection is unstable. Stop Card N before a student and record `insufficient_evidence` for the residual mechanism. Do not retune the mask after seeing a student score.

When the scalars agree, freeze `c = c_all` and define the residual `R = T - c A`. On `M_all`, `A · R` is zero up to float64 roundoff. That dot product is an admission test for the artifact.

Two targets at the same extra mass `λ = 0.15` (one current strong-seed coefficient, frozen):

- Diversity candidate: `B_div = B_hist + λ R`
- Scale companion: `B_cal = B_hist + λ (c A)`

`B_cal` adds pure anchor scale. `B_div` adds the orthogonal piece. The C0-relative primary mass of `B_hist` stays 0.8125. `R` is not a probability weight. `λ` is not chosen from a score.

Disagreement shell, label-free. Let `δ` be the 75th percentile of `|A - T|` on `M_all`, computed per target and then frozen. Define

`B_shell = B_hist + λ R * 1[|A - T| >= δ]`

`B_shell` is the unconventional arm. `B_div` is its uniform-residual parent. The shell uses teacher disagreement only, so the training target is still a precomputed function of teacher outputs. It does not read the label, and it does not run the teachers at student inference.

Family pairs remain the accompanying fold pairs at family-loss weight 0.1.

Cheap evidence, before any student.

- Card R export `A` must already have passed reset and coverage. Card N is undefined without that anchor.
- TCN reset test as specified above.
- Per-target standard deviation of `R` divided by standard deviation of `T`.
- Fraction of residual variance sitting in the disagreement shell.
- Clip-shell counts for `B_hist`, `B_div`, `B_cal`, and `B_shell`.
- Standalone official metric of `T` and of `A` on the exposed proxy, stored as description. A weaker standalone `T` does not block the residual. A numerically empty residual does. Empty means relative residual standard deviation below 0.02 on both targets.

Student rule, only after those artifacts pass, and only as a later admission. Two paired screens, one seed each, same student recipe and same parent initializer: `B_div` versus `B_cal`. Add `B_shell` only when the shell holds at least half of the residual variance and the clip-shell count of `B_shell` differs from `B_div`. Otherwise `B_shell` stays an untested target file, not a third silent score.

Falsifiers.

- Projection instability across `M_all` and `M_valid` falsifies this scalar construction for the current exports.
- Relative residual scale below 0.02 falsifies "TCN adds a direction the GRU misses" at the level of a global scalar. A nonlinear calibration leftover can remain; see the failure mode. Do not launch the student to hunt it inside this wave.
- A student gap that appears equally on `B_div` and `B_cal` supports a scale/calibration reading. It does not support a diversity reading.
- A student gap on `B_div` with a flat `B_cal`, concentrated in the disagreement shell under the official magnitude weights, supports conditional value of the TCN residual at this budget. It still does not support a public-score claim.

Failure mode. A single scalar misses nonlinear calibration: tail squashing, per-sequence gain, and the ±2 compression already measured in Card S. Leftover `R` then looks like mechanism diversity and is still scale. The two-mask check catches mask-dependent linear scale and misses nonlinear scale that is stable across those masks. The honest bound is: Card N identifies linear-scale-orthogonal disagreement, not every kind of diversity.

Second failure mode of the shell. If `|A - T|` is mostly optimization noise, `B_shell` adds variance on a quarter of rows and the official magnitude weights can amplify it. That is a reason to keep `B_cal` paired with `B_div`, and a reason to leave `B_shell` unscoreed when the residual is diffuse. A label-gated shell (`|clip(y, ±2)| >= 1` as a training-target switch) was considered and is not the candidate. The student would have to learn a target that jumps with the unseen label. Teacher disagreement is the gate that stays a function of inputs the teachers already observed.

Third failure mode. All-data TCN predictions on rows included in its own fit are in-sample. The residual can be a memorized correction. The exposed proxy can reward that correction. Public transfer stays unknown.

Cost. One new all-data TCN training run, then one export, then at most two student screens. Epoch budget matches the fold-0 raw screen: 4 epochs. Wall time is unknown. Do not convert 3635 seconds or 2000 seconds into a TCN or a 3090 quote. Device batch must be profiled. The number 4096 belongs to the fold-0 AMP pair already staging; it is a candidate token or batch setting only after a parser audit and a memory probe. It is not automatically legal on a 5 decimal GB GPU or on an all-data row count. Prefer a free Kaggle GPU lane when the controller later admits the fit. Leave the staging Vast job alone. Paid credit stays inside the existing $20 ceiling with no top-up; this card does not reserve a new rental.

## Unconventional transferable piece

Name. Linear residualization of a new teacher against a frozen all-data anchor, then an optional disagreement shell.

Correspondence. The scalar `c` is the same object as a one-covariate residual regression: remove from `T` the part linearly explained by `A`, and distill the remainder. Any later teacher (another recurrent depth, a longer TCN, a raw-WMSE rerun) can reuse `A`, the mask test, `λ = 0.15`, and the shell quantile without a new blend search. The anchor file is the reusable artifact. Rebuilding eight teachers is not required to try the next family.

What would make the modification travel badly. If every new teacher is nearly a scalar times `A`, the residual files are empty and the modification correctly refuses to launch students. If residuals are large and unstable across `M_all` versus `M_valid`, the modification correctly stops. Those outcomes are successful uses of the instrument.

## Discriminating order

The smallest useful wave never waits for eight rebuilt teachers.

1. Synthetic admission test, CPU, no data fit. Build a tiny two-channel fixture with a known blend sum 13/16, a known 16/13 multiple, a known orthogonal residual, a known clip shell, and a deliberate cross-sequence carry. The assembler and the reset checker must accept the first four and reject the carry.
2. Card S from the existing blend cache, and Card R export of the epoch-3 all-data parent. Diagnostics in the cheap-evidence lists. No student yet.
3. Apply the Card R skip rule and the Card S empty-shell rule. Admit at most one student from `{C0 if the in-flight hash mismatches, Card S, Card R}` under a later execution decision. Matching in-flight C0 is reused rather than rerun.
4. Card N training and export only after step 2 has a valid `A`, and only on an accelerator that is free without touching the staging fold-0 pair.
5. Card N students only after the projection is stable and the residual is nonempty, as `B_div` versus `B_cal`.

Scale-up after that evidence, still without an eight-teacher precondition.

- Proxy gap on Card R at the 0.002 descriptive margin: one follow-up student that moves the raw-WMSE pair mass onto `A`.
- Diversity reading on Card N: keep the TCN residual in the teacher set at `λ = 0.15` and consider a longer or wider own TCN as a large teacher whose compression path is the same student recipe. Serving size stays a student problem.
- Scale reading on Card N (`B_div` and `B_cal` move together) or empty residual: stop TCN residual students. A later teacher must change the receptive field, the objective, or the fit membership in a way this residual can see. Another seed of GRU256 does not satisfy that.
- Empty Card R and empty Card S: all-data export of this parent is not a new target. Spend the next new fit on Card N, because a weak or unused TCN can still be orthogonal.
- Family-channel extension is a separate later card. It starts when some all-data teacher head actually emits family rows. This wave does not block on it.

## Parser-audit list

Facts above that are recipe values, not confirmed flag spellings: GRU depth 2 and width 256; epoch index 3; student epochs 2 with the checkpoint taken at epoch 1; batch 16; chunk 2000; raw chunk WMSE; learning rate 0.0001; AdamW weight decay 0.01; OneCycle; propensity power 2 and floor 0.03; valid-mask weight 0.03; structure auxiliary 0.3; next auxiliary 0.3; family loss 0.1; distill coefficient 0.7; TCN width 128 and depth 12; raw scratch; 4 epochs; AMP; the token 4096 on the fold-0 pair.

Unknown until the existing parser and export schema are read. Do not invent these flags in a launcher.

- Argument names, config filenames, and which binary exports predictions.
- Meaning of fixed epoch: restore epoch 3 from a longer run, or a run whose length is 3. Identity of the measured 4-epoch directory versus the epoch-3 parent file.
- Whether 4096 is batch size, chunk length, or a token cap, and whether AMP stores fp16 weights.
- Parent head width: 2, 23, 31, or another width. Map from head row to the two official targets.
- Numeric coefficients and column names of the four accompanying raw family pairs.
- Whether distill 0.7 weights a loss term or mixes teacher and label before one loss.
- Whether the current target builder clips at ±2, standardizes channels, or already divides by 0.8125.
- OneCycle `pct_start`, `div_factor`, and `final_div`.
- Sequence id, step, split, and mask column names.
- TCN kernel, dilation schedule, normalization, and causal padding implementation.

Runnable only after that audit, with values taken from this card and flag names taken from the parser: export the hashed epoch-3 GRU on full supplied membership; assemble `B_unit` and `B_hist` as pure functions; train a student only under a later admission by pointing the existing distill-target input at one assembled file and leaving every other student setting at the values listed here; train TCN128×12 all-data raw for 4 epochs only under a later admission, with AMP enabled and with batch or chunk chosen by a memory probe rather than by copying 4096 blindly.

## What this wave refuses to claim

No expected public score. No claim that the in-flight students have improved on 0.6660. No claim that all-data student training creates all-data teacher histories. No claim that a weak standalone TCN or GRU128 is useless as a conditional target. No claim that a flat 0.30 swap falsifies heterogeneous all-data histories in general.

## Controller priority

Do the no-fit pair first: assemble Card S from the current blend, and export Card R from the existing all-data GRU256×2 epoch-3 checkpoint. Run the synthetic admission test and the cheap prediction-space diagnostics before any new training. Admit at most one two-epoch student from that evidence, reusing a hash-matched in-flight C0 artifact. Queue Card N as the genuinely new combination on the next free Kaggle GPU, or on another free own accelerator that is not the Vast instance currently staging fold 0. Keep `λ = 0.15`, the two-mask projection test, and the `B_div` versus `B_cal` pair frozen. Leave the other seven teachers on their fold histories until a Card R or Card N screen shows a target-space gap at the budgets written here.

Evidence that changes this order.

- Card R skip rule fires and the Card S clip shell is empty: do not spend a student on either card. Move the next admitted fit to the all-data TCN residual.
- Card S shell is large and Card R histories are near-duplicates: the first student is Card S. Card N still follows, because scale being real does not answer diversity.
- Card R histories disagree inside the high-magnitude shell: the first student is Card R at mass 0.30. A descriptive official-metric gap of at least 0.002 on the exposed proxy is the gate for the single follow-up that also moves the 0.25 raw-WMSE mass onto `A`.
- Projection scalars disagree across the two masks, or the TCN residual scale is below 0.02: do not train Card N students. Keep the TCN checkpoint as a negative artifact of this residual construction.
- `B_div` and `B_cal` move the exposed proxy together: treat the TCN as added scale. Stop residual students. A later model has to change receptive field, objective, or fit membership beyond a scalar.
- `B_div` moves and `B_cal` stays flat: keep the orthogonal residual at `λ = 0.15` and scale the TCN as a large teacher through the same student compressor. Add `B_shell` only under the variance rule already stated.
- The epoch-3 export fails coverage, reset, determinism, or head-width identity: repair the export. Do not substitute the fold-0 job's checkpoints.
- Export or TCN setup OOMs or loses quota: that is execution evidence. It is not a negative result about recurrent histories or about convolutional residuals. Reduce the inference or training batch and keep the recipe identity.
- A hash-matched C0 student fails or its target cache is corrupt: repair that pipeline before a new student. Card S and Card R diagnostics still proceed, because they use teacher files.

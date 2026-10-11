I'll read the three distillation notes and the campaign's current evidence so the six methods sit on this dataset's actual constraints.Distillation can recover part of the blend, and the current +0.0008 is a weak test of that, not the ceiling. A realistic target for one causal GRU-256 is about **+0.001 to +0.002** over continuation (~0.6686), toward the diverse-pair blend (~**+0.003**, roughly 0.672–0.673). Beating the blend needs a target no causal teacher output contains. Seed noise is ~0.001, so anything under that needs three seeds before it counts.

The blend gain is the small slice of error that is **not** shared: members share 96–99.9% of the residual, 94% of residual sum of squares is a per-run level, and class 3 holds 69% of that residual at correlation 0.44. Row-wise MSE to the mean of two similar teachers, mixed 30% with the raw stepped labels, for 3 epochs at 1e-4, spends its gradient on the shared part the warm start already has. Wider teachers already failed this way (h384 distill 0.66836 vs continuation 0.66861). Scratch runs that keep the 1e-3 one-cycle past epoch 3 memorize (0.667 → 0.649). Every method below uses a long low-LR tail instead of a longer high-LR cycle: AdamW from the same parent, 2 epochs at 1e-4, then 8–12 epochs cosine from 3e-5 to 1e-5. Label weight stays ≤0.1 during the tail so the stepped labels do not paint the ensemble correction back out.

## 1. Event-run level distillation

**Mechanism.** Labels are piecewise constant (mean run ~9.5 rows) and change at book events. The metric pools rows, so a wrong level on a long run dominates weighted Pearson. Distill the ensemble’s level on each event-run, and match shape only inside the run.

**Loss.** Segment runs with the existing event flags. Let `L̂` be the ensemble mean over the run. Loss = `0.8 · WMSE(run_mean(student), L̂) + 0.2 · WMSE(student − run_mean(student), teacher − L̂)`, label weight 0.1, zero extra weight on duplicated intra-run rows. Same long low-LR tail. Aux head: predict the run level from the GRU state at the event row and hold it until the next event.

**Signal.** Ensemble outputs, collapsed to a per-run level, plus within-run residual.

**Data.** Train split only. Runs from train sequences.

**Falsifier.** On cached fold-1 predictions, replace each model’s within-run mean with the ensemble’s run mean and rescore. If that oracle is under +0.001, there is no level headroom. If it is ~+0.002–0.003, this is the whole prize and a real fit should land +0.001–0.002.

**Pointer.** Kim & Rush, Sequence-Level Knowledge Distillation (EMNLP 2016): supervise the sequence statistic, not every token.

## 2. Orthogonal residual of diverse teachers

**Mechanism.** Same-recipe seeds buy ~+0.001; an h256+h384 pair buys ~+0.003. The useful target is `ensemble − parent`, a few percent of the residual. MSE to the mean ignores that and the raw h384 target already transferred nothing.

**Loss.** `WMSE(student, y)` with weight 0.3, plus `β · a_r · MSE(student − parent, ensemble − parent)`, `β` from 0.2 to 1.0 over the low-LR tail. Row weight `a_r` = variance across the diverse pair, times the metric weight, times 0.25 on class 3 (there disagreement is mostly the 0.44-correlation noise, not a second opinion). Do not distill h384 outputs directly; distill `h256_teacher + (pair_mean − h256_teacher)`.

**Signal.** Residual-of-teachers, weighted by multi-teacher disagreement.

**Data.** Train-split caches of one h256 and one h384 teacher. Same parent as continuation.

**Falsifier.** Weighted correlation of `(pair − parent)` with `(y − parent)` on fold 1. If that is ~0, the +0.003 is cancellation that inputs do not determine, and no student can learn it. If it is clearly positive, expect +0.001–0.0015 over continuation.

**Pointer.** Mirzadeh et al., Improved Knowledge Distillation via Teacher Assistant (AAAI 2020), for the h384→h256 gap. Furlanello et al., Born-Again Neural Networks (ICML 2018), for a student trained on another model’s outputs from the same family.

## 3. Low-rank slow state

**Mechanism.** Training detaches state every 1000–2000 rows, so the sticky per-run offset is exactly what backprop through time does not carry. Full 256-d hidden MSE copies teacher noise. Match only the few hidden directions that actually store the run level.

**Loss.** From a teacher hidden dump, take the top 4–8 principal directions of `h_teacher` that predict the run-level label residual beyond the teacher’s own output. Loss = MSE of student hidden projected on those directions vs the teacher projection, weight 0.5, plus method 1’s run-level output loss at weight 0.5. Freeze the encoder for the first epoch. Then the same 8–12 epoch low-LR tail on the GRU. Output KD only, no labels, for the last 4 epochs.

**Signal.** Hidden state, low-rank, the slow/run component only.

**Data.** Train split. One extra forward pass to store teacher states at event rows (not every row).

**Falsifier.** If those directions do not predict run offset better than the teacher output, skip the fit. Expected gain is smaller than method 1, about +0.0005–0.001, and only if the level lives in state the output head is too weak to read.

**Pointer.** Romero et al., FitNets (ICLR 2015); Sun et al., Patient Knowledge Distillation (2019) for intermediate-layer matching. Use the low-rank cut so it is not full hint regression.

## 4. Pseudo-labels on warped and held streams

**Mechanism.** In-sample teacher outputs sit on the same rows the student already fits, so they barely move it (+0.0008). The ensemble’s edge has to be learned off those rows. Labels are constant between events and the book is multi-rate (columns update on 12–100% of rows), so stretching time or inserting event-holds changes the path without inventing a new label.

**Loss.** Two stages. Stage A: 4 epochs, lr 1e-4→3e-5, loss = WMSE to the ensemble on augmented streams only, no labels. Augmentations: causal time-stretch/compress of 0.8–1.25 inside a sequence, and random insertion of 1–4 held rows after an event (inputs frozen, teacher re-run). Stage B: 6 epochs at 1e-5 to 3e-6 on real train rows, `0.2 · label + 0.8 · ensemble`. Early-stop stage B on a train-internal slice, not on fold 1.

**Signal.** Ensemble outputs on augmented paths. Separately, ensemble pseudo-labels on valid-input streams whose labels are not in the fit (other folds’ inputs are already in training; use a held-out component slice of train, or valid inputs only for a later confirmation, never fold-1 labels).

**Data.** Augmented train paths first. Valid-input-only pseudo-labels only after the train-augmentation falsifier passes, and not for the fold you score.

**Falsifier.** If teachers’ outputs barely change under stretch/hold (shift ≪ seed noise), augmentation has nothing to teach. If they move and the student matches them on warps but fold-1 does not move by ≥0.001, the blend is not a missing-coverage problem. Expected gain +0.0005–0.0015.

**Pointer.** The sibling-challenge note in `DISCORD_TRICKS_DEEP_DIVE_20261011.md` (median-variance rescale and random stretch, claimed useful there, untested here). Same mechanism as Xie et al., Self-training with Noisy Student (CVPR 2020): unlabeled or synthetic inputs, teacher targets.

## 5. Consensus future price, with class-3 quantiles

**Mechanism.** `t0 = F − ask`, `t1 = bid − F`. The sum is the spread (almost observable); the difference is `2(F − mid)` and is under-dispersed. Capacity is being spent shrinking the forecast of `F`. Class 3 is half the metric weight and most of the error; matching it with MSE fits noise.

**Loss.** Pin `s0 + s1` to the teacher sum (or the book spread when that identity holds) with MSE weight 0.3. On the difference, maximize weighted correlation with the ensemble difference, weight 0.4, so the student is penalized for under-dispersion. On class 3, use a pinball loss at quantiles 0.3 and 0.7 of the ensemble prediction and take the midpoint as the served value; outside class 3 use WMSE to the ensemble, weight 0.3. Label weight 0.1. Schedule: 2 epochs at 1e-4 on the difference head only, then 10 epochs from 3e-5 to 1e-5 on the full net.

**Signal.** Decomposed outputs (sum vs difference), quantile targets on class 3, ensemble variance as the pinball band.

**Data.** Train split, same ensemble cache. No new forward passes.

**Falsifier.** Split the pair’s +0.003 into sum-channel and difference-channel on cached predictions. Train only the channel that actually holds the gain. If class-3 pinball targets are just a shrunk copy of the mean, drop the quantile term. Expected gain +0.001 if the missing mass is the `F` forecast.

**Pointer.** Park et al., Relational Knowledge Distillation (CVPR 2019), for matching a relation between outputs rather than the outputs. Kendall & Gal, What Uncertainties Do We Need (NeurIPS 2018), for a variance-aware treatment of the noisy class.

## 6. Privileged short lookahead (the unconventional one)

**Mechanism.** The same GRU with future rows scores 0.680 at 2 rows ahead and 0.711 at 8, against 0.666 causal. Labels also lead the features by about one row, and they are a smooth function of a long future path. A 2-row lookahead teacher is a denoised version of the stepped label plus a little real future. Distill only the part of that privileged target that still correlates with the past. That is learning using privileged information, not a softer copy of the causal ensemble. This is the only method here whose target can sit above the causal blend, and it can also fail completely if the ladder gain is pure future leakage.

**Loss.** Fit a lookahead-2 teacher on train (lookahead-8 only as a second arm). On a train-internal cut, regress the lookahead run level on the causal ensemble’s run level and the causal features; keep the projection `P(lookahead | past)`. Student target = that projection. Loss = method 1’s run-level WMSE against `P`, label weight 0. Rows where the internal-cut R² of the projection is below 0.1 get weight 0 (the unpredictable future is dropped). Schedule: 12 epochs, lr 3e-5→1e-5, no 1e-4 phase, so the student does not chase the leaked component. Serve the causal student only.

**Signal.** Lookahead teacher outputs, gated by causal predictability. Not the raw lookahead number.

**Data.** Train split, where a lookahead teacher is a legal training target. Do not build the target from fold-1 labels. Score the causal student on fold 1 as usual.

**Falsifier.** Before any student fit: R² of lookahead-2 run levels given the causal ensemble, on a train-internal cut. If R² equals the causal model’s own fit, the extra ladder points are not distillable; stop. If R² lifts the reconstructed target by ≥0.002 WP on that cut, run the student and expect +0.001–0.003 over continuation, with a real chance of matching the pair blend. A gain past the blend is possible only on this arm, and only up to the predictable fraction of the 2-row gap (+0.014 at the oracle, most of which will not transfer).

**Pointer.** Lopez-Paz et al., Unifying Distillation and Privileged Information (ICLR 2016); Vapnik & Vashist, Learning Using Privileged Information (2009).

## What to run first

The two cache-only checks decide the budget. Method 1’s oracle run-mean swap, and method 2’s correlation of `(pair − parent)` with `(y − parent)`. If the level oracle is large, fit method 1. If the residual correlates, fit method 2. Fit method 6’s R² check in parallel; it is the only arm with a path above the blend. Methods 3–5 wait on those three numbers.

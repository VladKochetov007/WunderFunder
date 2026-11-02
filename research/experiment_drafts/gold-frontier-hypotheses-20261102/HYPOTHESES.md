# Gold frontier: five specific research proposals

These are unimplemented hypotheses, not measured improvements or runnable jobs.
They complement the [portfolio in PR55](https://github.com/VladKochetov007/WunderFunder/pull/55),
[temporal KD PR56](https://github.com/VladKochetov007/WunderFunder/pull/56),
and [disagreement KD PR57](https://github.com/VladKochetov007/WunderFunder/pull/57).

## Common experimental contract

The current exposed development reference is TL3 0.692377425 and TL3 + 10% LSTM
0.692502063 on the previously used full validation set ([PR53 evidence](https://github.com/VladKochetov007/WunderFunder/pull/53#issuecomment-6545905799)).
This tiny delta is not independent confirmation and does not forecast public score.
The full-data PR54 models fitted both supplied train and valid and cannot be used
as clean parents or teachers for a retrospective comparison on those labels.

For proposals A–D, initial allocation is 2 control + 2 treatment fits each, **16
primary new fits in total**, before missing teacher/parent fits and caches; each
proposal can be admitted independently. Proposed cells are F0/seed1001 and
F2/seed1002, conditional on verified ownership, actual component membership, clean
ancestry and matching data. The folds and seeds are confounded; these two cells do
not establish significance or independently estimate both sources of variance.
Where exact clean parents or source-qualified teacher caches do not exist, make
and account for them separately before admission, never substitute full-fit PR54
or a parent that has seen the evaluation component. Proposal E is a fixed-prediction
analysis with **zero new model fits only when qualifying out-of-fold predictions
already exist**; otherwise teacher/fold refits and exports are extra. No existing
control is assumed reusable until its full source, recipe, seed and ancestry match.
A teacher-only advantage is not a student advantage; separately budget student
transfer (minimum 2+2 paired fits), teacher export and serving measurements.

Each implementation PR must freeze exact input/model/optimizer, effective batch,
updates, LR and schedule, objective weights, loss masks, fold identities, clean
parent and cache digests, checkpoint selection by inner-fit data, and training
precision before the first fit. For these new GRU scientific fits use strict FP32,
TF32/autocast off; the previously approved BF16/BF16x9 Transformer amendment applies
only when explicitly selected as an identical matched recipe. No training-time
quantization. Evaluation uses full 20k-row sequences in order with reset on `seq_ix`;
training windows such as C must stay within a sequence and preserve prefix order.
All inputs and state are prefix-causal; true past targets must not become inference
inputs. Future targets may enter the training loss only as permitted labels.
Any normalization or thresholds are fitted on fitting components only. Verify
prefix invariance, warmup and sequence resets on the receiving host. Audit
component/duplicate separation and all ancestor exposure. Anonymous component IDs
alone do not establish chronology or representative future regimes.

Use the official `global_wp_clip2_v1` mask, clipping, per-target weighted Pearson
and prescribed pooled aggregation, not an average of per-sequence correlations.
Report all candidate/control deltas per cell, t0 and t1, counts, age/target-scale
slices, and failed attempts. A **+0.0005 WP** paired improvement is a provisional
resource-allocation threshold, not statistical significance, equivalence, or a
promise of leaderboard impact. Small or mixed effects warrant targeted follow-up;
a confidence interval crossing zero is not a kill rule. A component bootstrap of
fixed predictions is conditional on the fits and does not estimate training
variance or correct adaptive selection. Old folds and public feedback are exposed;
freeze the full selection process before any truly unexposed or prospective check.
Run the exact exported candidate on a Linux Python 3.11, one-vCPU, 16 GB,
60-minute, offline receiver; verify full-sequence parity, 4000-row boundaries,
resets, quality and ZIP <=20 MB. These are unmeasured constraints, not an
acceptance claim. Record source/config/data/split/ancestor hashes, progress, LR,
preclip gradients, resource/time, all predictions/grades and separate evidence
and weights archives. No measurements are asserted here.

## A. Multiscale residual objective on causal model outputs

**Lineage:** student objective. **Question:** Does emphasizing both slow drift
and faster changes in target prediction improve pooled WP without adding serving
cost? [FreDF](https://arxiv.org/abs/2407.12415) shows frequency roles can vary
across forecasting tasks; the following filtered-residual loss is our proposed
adaptation, not the FreDF method or its measured result. The team also noted a
frequency-loss analogy in its [earlier mechanism audit](../../campaign_2026-10-05/review_20261005/MECHANISM_AUDIT.md); that multi-horizon discussion did not test this recipe.

**Pair:** matched existing supervised/KD baseline versus the same loss plus a
fixed coefficient times squared errors of past-only finite impulse filters applied
to predictions and permitted labels, at two fixed scales. Fix filter coefficients,
window/mask normalization, weights and loss coefficient on inner-fit development
before any outer result. Do not filter across sequence or invalid-mask boundaries;
exclude windows without full eligible labels. All model inputs remain causal;
label windows serve loss only. Keep main loss and effective updates identical and
log loss/gradient scales; a change in total gradient scale is a plausible alternative,
so a gradient-scale-matched control is a separate two-fit follow-up. Inference model
is unchanged. Hypothesis predicts paired pooled-WP improvement and predeclared
low/high-frequency residual reductions; mixed or worse WP and noise amplification
would constrain these exact filters. Distinct from #56, which matches teacher
prediction differences rather than filtered ground-truth residuals.

**Budget:** 2+2 student fits, plus any qualified teacher prerequisites. Receiver
measurements are still necessary even though no inference layer is added.

## B. Target-wise gradient conflict intervention

**Lineage:** student optimizer. Existing [R6 diagnostic](../../campaign_2026-10-19/idea_screens_r6/RESULTS.md)
reported 4/16 negative-cosine windows and pooled cosine +0.333540 on one narrow
probe; it does not imply persistent or harmful conflict. The team already proposed
PCGrad, so this is a targeted continuation, not a new discovery. [PCGrad](https://arxiv.org/abs/2001.06782)
projects conflicting task gradients in multi-task settings; transfer to our two
targets remains untested.

**Pair:** fixed student and loss versus identical student with PCGrad applied only
to shared-trunk t0/t1 task gradients when their dot product is negative. Preserve
head-specific gradients; explicitly define treatment of regularizers, KD and empty
masks before implementation. Aggregate identical global gradients before projection
in DDP; fix projection order, precision, normalization and clipping policy. Compare
same data, initialization and successful optimizer updates; report overhead, per-target
metrics, layerwise conflict frequency and update norms. A gain in one target alone
may be offset in the other; lower conflicts do not by themselves prove improved WP.
If gradients rarely conflict or both target and pooled scores fail to improve,
this particular intervention is unsupported, not every multitask method.

**Budget:** 2+2 student fits, with a 1-B300-per-fit starting assumption requiring
real profiling; cached gradients or fixed-window diagnostics are not trained fits.

## C. Prefix burn-in for truncated recurrent training

**Lineage:** student/training time. If training examples reset state at arbitrary
chunk boundaries but deployment carries state across the whole sequence, a
state-distribution mismatch may harm later predictions. This premise must first
be checked against the actual chosen trainer; if it already carries the exact same
state under the matched recipe, this variant is ineligible and should not be run.

**Pair:** same eligible loss rows and optimizer updates in both arms. Control
resets hidden state at each sampled train window. Treatment starts at the preceding
prefix from the *same* sequence, advances state without loss or backward through a
fixed burn-in, detaches it, and then scores the same target window. Prefix length,
start sampling and checkpoint rule are frozen on inner fit. Report extra forward
rows, elapsed time, sequences processed and steps separately; do not call this an
equal-compute test. For an equal-wall-time follow-up, prespecify a separate panel.
Full-sequence, state-carrying inference is identical for both. The prediction is
more accurate late-sequence WP or lower chunk-boundary discontinuity; no effect,
lost quality at startup or disproportionate train cost limits this version.
Related to [adaptive TBPTT PR22](https://github.com/VladKochetov007/WunderFunder/pull/22),
but the manipulated variable is prefix-state initialization at matched scored rows,
not the adaptive backpropagation horizon. Avoid stale states after optimizer steps.

**Budget:** 2+2 student fits; no teacher fitting only for a genuinely label-only
recipe, otherwise add exact qualified teacher/cache cost. Profile the burn-in cost.

## D. Parallel causal convolution teacher and recurrent student

**Lineage:** teacher, then connection/student. Causal dilated temporal convolutions
can process training positions in parallel; [Bai et al.](https://arxiv.org/abs/1803.01271)
compared temporal convolutional and recurrent sequence models in other tasks.
That paper does not establish GPU speed or quality on this dataset.

**Pair:** same source-clean data and inputs, GRU teacher control versus one
left-padded dilated TCN teacher treatment with frozen receptive field and comparable
parameter count. Match seeds, loss rows, updates, data and endpoint rule; measure
actual GPU-hours, memory, receptive field and full-sequence causal parity rather than
asserting speedup. All temporal padding must be strictly causal and windows must
not cross sequence boundaries. Fixed context limits are a plausible disadvantage.
If treatment improves teacher scores, assess transfer with a separate matched GRU
student + label/KD panel using qualified teacher exports; stronger teachers alone
do not establish a better submission. The deployable path is a streaming GRU, so a
TCN inference implementation is not assumed; evaluate exact deployed student and
blend on CPU. Distinct from #55's Transformer multiscale memory and Mamba/DeltaNet
hybrids, and from #48's Transformer width/context screen.

**Budget:** 2+2 teachers, plus at least 2+2 students and cache construction for
transfer; any clean parent fits are extra. No implementation or kernel timing exists.

## E. Nested OOF shrinkage of fixed blend weights

**Lineage:** ensemble selection; a focused continuation of prior OOF/mixture work,
including PR55 card 4, not a claim of novelty. Stronger standalone models need not improve the
actual TL3 mixture when their residuals are correlated; separately choosing weights
on exposed valid can overfit selection. Compare the frozen TL3 weights against one
predeclared nonnegative, sum-to-one shrinkage rule toward those weights, fitted
using only source-qualified OOF fitting-component predictions and labels. Freeze
candidate members, lambda, objective, target handling, selection procedure and
cost before outer evaluation. For each outer held-out component, every underlying
prediction and every ancestor must exclude that component; nested split or an
independent inner OOF panel must exclude the outer units from weight selection.
Missing cross-fitted predictions are a blocker, not an excuse to reuse in-sample
or PR54 full-data models. Do not choose lambda or members on the outer result.

**Budget:** zero new fits only if a fully qualified OOF matrix already exists;
otherwise count each teacher-by-fold fit and all exports. Record the exact matrix,
provenance, and evaluation/selection history. Compare pooled WP for both targets,
component-paired contributions, all member removals only as diagnostics, and the
exact full-blend CPU latency; fixed-prediction bootstrap conditions on those models.
A gain smaller than selection uncertainty, a loss on outer components, or latency
beyond the 60-minute bound prevents promotion of this recipe. This is a specific
nested evaluation of shrinkage, not proof that another blend search has failed or
that a held-out/relabelled exposed fold is fresh.

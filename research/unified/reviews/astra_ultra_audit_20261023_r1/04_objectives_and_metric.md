---
kind: review
status: complete
created_utc: 2026-10-23
updated_utc: 2026-10-23
hypothesis_ids: [H-ASTRA23-04-A, H-ASTRA23-04-B, H-ASTRA23-04-C, H-ASTRA23-04-D]
experiment_ids: []
evidence_paths:
  - src/wnn/gmetric.py
  - src/wnn/nn.py
  - src/wnn/training/losses.py
  - src/wnn/training/targets.py
  - src/wnn/training/updates.py
  - research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/RESULTS_F0_R1.md
  - research/unified/sources/NLL_GRADIENT_WEIGHTING_20261021_R1.md
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
  - research/unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md
---

# Objectives, metric geometry, and the missing effective-weight audit

## Question and scope

The strongest opportunity in this lens is to identify **which observations and parameter directions the current training recipe actually emphasizes**, before adding another nominally metric-aligned loss. This is a source-based review, verification tier C. It proposes no immediate execution: the October 23 stop remains binding. I read source and existing result documents, including the historical NLL implementation through `git show`; I ran no imports, tests, profiles, fits, array scans, or exports. No sibling report informed this assessment.

There is no evidence here of a new best submission. The relevant measured comparisons are the matched exposed-F0 WMSE/NLL continuation and the two exposed i1 fold/seed cells. Their absolute scores belong to different protocols. Public 0.66832 remains the historical project reference, not a score inferred from these experiments.

## Five findings that change the research question

### 1. Weighted MSE already has the right population target under explicit assumptions

The [official evaluator](../../../../src/wnn/gmetric.py) computes two separate correlations and averages them. For one target, write `Z=clip(Y,-2,2)`, `P=clip(f(X),-2,2)`, and `w=M|Z|`, where M is the fixed scored/prediction mask. It pools rows before centering; it does not average sequence correlations. The evaluator uses centered float64 moments to avoid cancellation.

Under the probability measure Q proportional to w, the score is `Corr_Q(Z,P)`. For causal information X, the best unrestricted squared-error predictor is

`m(X) = E[M|Z|Z | X] / E[M|Z| | X]`.

This also maximizes correlation, up to positive affine transformations, when its variance is nonzero. To see the connection, `Cov_Q(Z,f(X))=Cov_Q(m(X),f(X))`; Cauchy–Schwarz bounds the latter, with equality for an affine function of m. Since Z lies in [-2,2], m itself lies there, so an admissible maximizer need not incur prediction clipping. Zero-denominator regions carry no metric mass. This is a population argument, not a claim that finite GRU optimization attains the optimum.

The derivation is already reflected in [H-RR17-20](../../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md#h-rr17-20--cross-fitted-scalar-calibration-under-the-weighted-metric); it should not be repackaged as a discovery. Its consequence deserves more weight: “WMSE does not optimize Pearson” is insufficient motivation for another correlation loss. The meaningful departures are restricted capacity, finite data, sampling, normalization, teacher targets, and optimization.

The [implemented WMSE](../../../../src/wnn/nn.py) uses clipped targets and original clipped-label magnitude. Raw-target fitting instead learns a different conditional mean; Huber also changes the population estimand. Both can still help through finite-sample regularization or representation learning, but neither is automatically more faithful. Preserve [explicit target-treatment arms](../../../../specs/adr/0004-independent-training-and-target-variants.md), and distinguish a raw auxiliary target from changing the deployed mean objective.

### 2. The effective training measure is much more than `abs(clip(y))`

In [main_row_weights](../../../../src/wnn/training/targets.py), propensity weighting can replace the initial mask choice; class weights, actual-mask redistribution, event age, and episode redundancy can modify it further. These are optional code paths, not assertions that all were enabled in every run. The [NLL recipe](../../../campaign_2026-10-20/submission_delivery_r1/nll_adaptation/recipe.json) specifically uses propensity power 2, epsilon .03, `valid_true_mask=.03`, chunk normalization, alpha .9, and two auxiliaries at .3.

For its main objective, the coefficient on a residual gradient therefore includes row weighting, clipped-label magnitude, the inverse chunk weight sum, and—under NLL—inverse predicted variance. The global clipping factor acts subsequently on the combined gradient. AdamW then transforms the history of these gradients. These factors must not be collapsed into a single “metric weight.”

The WMSE denominator sums across rows **and targets**. Official WP gives each target half the final score after its own normalization. Shared finite capacity therefore need not be allocated equally by WMSE. Separately, averaging gradients from chunk-normalized objectives differs from a globally normalized weighted risk when chunk weight mass varies. Neither observation proves an implementation bug or predicts that equalizing targets/chunks will help: current normalization may stabilize training.

An additional execution trap is concrete. In current [main_loss](../../../../src/wnn/training/losses.py), cached teacher targets enter a hard-coded WMSE branch before `objective()`. Merely changing `--loss` does not change that main loss, even with alpha zero. The running-normalizer branch is also below the teacher branch. Any intervention needs proof of its executed branch; a flag-only comparison can silently test nothing.

### 3. The NLL result is a useful tie with a major unresolved mechanism

The [completed matched result](../../../campaign_2026-10-20/submission_delivery_r1/nll_adaptation/RESULTS_F0_R1.md) gives NLL−WMSE = **−0.0000436109**, with conditional component interval **[−0.0006230104,+0.0004552434]**. Their fixed oldZ4 mixtures differ by only **−0.0000038521**. This neither establishes an advantage nor establishes equivalence. There was one seed on exposed F0, with shared historical mixed-precision ancestry and strict-FP32 continuation.

NLL clipped every update; WMSE clipped roughly seven percent. Epoch-zero mean clipping factors were .0418 and .9866. The readout dominated aggregate squared gradients, but the diagnostics do not separate mean, variance, and auxiliary rows. In [optimizer_step](../../../../src/wnn/training/updates.py), clipping precedes AdamW, scheduler, and averaging. A .04 clip factor is therefore **not** evidence that the recurrent parameters learned at four percent of the control rate. Constant common gradient scaling can approximately cancel in Adam normalization; time-varying scaling, epsilon, shared loss directions, and weight decay complicate that cancellation.

Historical producer `68ff682977b822f38351aac0f621146f5f13b440`, inspected as text, implements weighted `exp(-s)*(mean-target)^2+s`, with s clamped to [-4,4]. The current [mean_variance module](../../../../src/wnn/training/mean_variance.py) only restores dormant checkpoint-layout rows. Those are different source states; current root source alone cannot reconstruct the historical NLL execution.

The target was .1 clipped label plus .9 cached teacher signal; the cache itself stores .8125 times a three-teacher mean. In the idealized case of a deterministic teacher given the same X and no further target clipping, the label-noise contribution to mixture variance is scaled by .1². Thus small learned residual variance can be an expected consequence of teacher-heavy supervision. It need not indicate discovery of low market noise or a pathological head. Different teacher/student capacity and context make approximation error another contributor.

Retain heteroscedastic learning, but modify its claim to **adaptive training emphasis for this target mixture**. Defer a broad variance-head sweep. The [existing gradient-weighting note](../../sources/NLL_GRADIENT_WEIGHTING_20261021_R1.md) already correctly separates clipping diagnostics from parameter-step suppression.

### 4. Complementarity survives scalar shrinkage, but nonlinear calibration remains open

The [complete i1 grade](../../../campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json) finds fixed 90%B/10%C improvements of **+.00035953** and **+.00027145** despite C losing alone in both cells. The [F0 geometry analysis](../../../campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json) reports +.00036506 without prediction clipping versus +.00035953 officially. Pure positive scalar shrinkage cannot improve unclipped Pearson; clipping-only explanations are also inadequate for this measured gain.

However, positive covariance with the target residual after an **affine** projection on B does not establish new information beyond B. A nonlinear function of B can have that property. C might partially recalibrate B's curved conditional response rather than recover missing temporal information. That could still be a valuable competition improvement, and might admit a much cheaper serving implementation.

The [history progression](../../ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md) identifies a different live alternative: cold C activates random input columns that B masks, changing the initial function. Moreover, the A mixture also helps F2. Retain the function-preserving warm comparison, and add scalar sufficiency as a diagnostic question; do not declare “history proven” or “just calibration” from the current evidence.

### 5. Sequence moments should be judged through global covariance, not pretty geometry

The [sequence-moment investigation](../../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) establishes descriptive arcs and heterogeneous tiny-data outcomes. It also explains why an arc can arise algebraically from occupancy, without identifying a physical regime. Longer tiny-pool training reverses some earlier moment advantages; dictionary mixtures can help while concatenated features lose. These facts argue against a universal normalization story and against closing the family.

Weighted pooled covariance decomposes into weighted within-sequence covariance plus covariance of sequence weighted means. Input-derived causal moments might improve either term. Between-sequence gains are legitimate under this metric, but potentially sensitive to sequence composition. De-meaning each sequence during training or evaluation would erase a component the official score counts. Conversely, treating whole-sequence target moments as serving inputs would be invalid.

Retain raw levels plus causal moment state, and test a function-preserving readout before another large concatenation. Report within/between components as diagnostics alongside the unchanged official score, not as a replacement metric or a retrospectively chosen selection rule.

## Questions whose answers would change an allocation

1. For the exact strong-parent recipe, how different are distributions of **scored metric mass, actual training mass, and normalized main-gradient mass** across targets, position bins, and dependency components?
2. Are teacher-mixture residual variance and clamp occupancy primarily associated with label magnitude, teacher disagreement, target type, or recurrent approximation error? Existing aggregate head norms cannot answer this.
3. Does NLL change recurrent **AdamW update direction and size**, after separating weight decay, or mainly change a preclip diagnostic that the optimizer largely absorbs?
4. Does the i1 mixture correction retain useful covariance after a low-complexity, component-held-out nonlinear calibration of B, in each cell?
5. Do moment/history gains live mostly in within-sequence dynamics or between-sequence location/scale, and is that decomposition stable under predeclared component-composition sensitivities?
6. Which proposed objective/normalizer switches actually reach the teacher-cache branch, and which are currently bypassed? Is every future recipe bound to the historical implementation it claims to extend?
7. Does a raw-target or robust-loss proposal seek better official mean prediction, useful representation regularization, or ensemble diversity? Which fixed endpoint would make that proposed value observable?

## Proposed hypothesis cards

### H-ASTRA23-04-A — Audit and isolate chunk-mass normalization

**Family/change:** effective sampling measure. Replace only the main-loss chunk denominator with a fixed fitting-only reference mass; preserve target weights, target mixture, auxiliaries, clipping, and sampler. This requires an explicit teacher-branch implementation, because `loss_norm=running` is bypassed there.

**Assumptions/mechanism:** chunk mass varies enough that equalized update windows distort useful training emphasis. A fixed denominator may restore mass-proportional emphasis. It can also increase gradient variance and worsen clipping.

**Predicted observable:** main-gradient contributions become closer to the declared weighted fitting risk; a useful effect must survive official grading of the fixed endpoint, rather than merely lowering the altered loss.

**Cheapest test:** after resume, inspect a predeclared tiny set of fitting windows and their existing weight components without updating parameters. Compare denominator ratios and output-gradient directions; no validation-based choice of reference mass.

**Matched controls/falsifier:** four new continuations, two per arm across two clean fold/seed cells, zero new teachers. Same reference-loss scale at initialization, update budget, checkpoint, RNG, and ancestry. If mass variation is negligible, or normalization changes gradients but adds no useful endpoint contribution at the screen's resolution, defer this recipe. Do not simultaneously equalize the two targets.

**Cost/reuse/exposure/serving:** reuse exact parents, teacher caches, and evaluator. Historical F0 continuations cost 4,249–4,391 seconds each, suggesting roughly 4.8 hours for four comparable fits before exports; this is planning arithmetic, not current throughput. All folds remain exposed. No serving overhead. Related: [current loss source](../../../../src/wnn/training/losses.py), [H-055 rejection audit](../../ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md). **Status: not_tested.**

### H-ASTRA23-04-B — Separate variance learning from its mean-gradient weighting

**Family/change:** objective/optimizer interaction. Compare WMSE, detached-factor β-NLL with β=1, and β=.5. Keep the historical original-weight denominator, clamp, teacher mixture, auxiliaries, initialization, and optimizer fixed. Do not introduce a variance-initialization or clipping-policy sweep.

**Assumptions/mechanism:** variance-weighted mean updates, or shared variance gradients, change capacity allocation. At β=1 the direct mean-output derivative equals WMSE, but variance gradients still enter shared parameters and global clipping. Therefore β=1 is informative, not an exact WMSE control.

**Predicted observable:** WMSE versus β=1 exposes the joint variance-branch effect; β=.5 versus β=1 changes variance weighting. The latter also changes variance gradients, so it does not uniquely isolate mean weighting. Measure head-row and trunk update diagnostics rather than telling a causal story from clip rates alone.

**Cheapest test:** a bounded frozen-state gradient/optimizer-step replay, with copied state discarded and no fitted checkpoint. Use a saved state with learned, nonconstant variance; at zero logvariance all β factors initially equal one. Include a detached-trunk, separately optimized variance-only diagnostic: under controlled RNG it should leave the mean trajectory unchanged. This is a future engineering probe, not an admitted job.

**Matched controls/falsifier:** six new scientific fits total, two per arm across two cells, zero teachers. Existing F0 β=0 results remain context, not new replication. If predicted update differences are absent, the optimization explanation weakens; if β variants improve diagnostics but not the fixed official endpoint, they do not earn scale. Wide score intervals alone do not falsify the family.

**Cost/reuse/exposure/serving:** about 7.2 fit-hours by the historical per-fit times, plus export/grading; actual admissible hardware must be profiled later. Reuse the pinned historical source and ancestry, not the dormant current head. Exposed folds; historical FP16 parent ancestry remains. Discard variance outputs at serving if only the mean is used. Related: [H-NLL-GRADIENT-WEIGHTING](../../sources/NLL_GRADIENT_WEIGHTING_20261021_R1.md). **Status: not_tested.**

### H-ASTRA23-04-C — Test whether history adds information beyond scalar recalibration

**Family/change:** sufficiency and endpoint calibration. Fit a fixed, low-complexity scalar map from B to the weighted label mean, and a separate map from B to C. The latter decomposes C into a part predictable from B and a residual; it is a diagnostic, not a new temporal model.

**Assumptions/mechanism:** some mixture gain comes from nonlinear score calibration. If C supplies distinct information, its residual should still help after accounting for that map. Calibration fitting error can obscure both conclusions.

**Predicted observable:** compare identity B, calibrated B, original fixed B/C mixture, and its residual-correction diagnostic on components not used to fit maps. Freeze map complexity and the residual coefficient; never tune them on grade components.

**Cheapest test/matched controls:** four two-output calibration jobs total: two map types in each existing F0/F2 cell, using a predeclared component split within each cell. This means eight scalar maps, zero new GRU/teacher fits. Identity and the existing fixed mixture are reused controls. Train/calibration/grade memberships must be explicit; previous exposure persists despite the new within-cell split.

**Falsifier:** stable residual contribution after calibration weakens a calibration-only account. Conversely, no residual contribution plus comparable calibrated-B quality weakens the claim that a second history model is necessary at this fidelity. Neither establishes absence of all temporal information.

**Cost/reuse/serving:** stream saved predictions with a predeclared small knot count; propose a 15-minute CPU ceiling after safe admission, actual cost unknown. No recurrent exports needed if artifacts are complete. A retained scalar map has negligible state and small arithmetic cost; preserving the original mixture still requires both models. Related: [H-RR17-20](../../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md#h-rr17-20--cross-fitted-scalar-calibration-under-the-weighted-metric), [i1 progression](../../ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md). **Status: not_tested.**

### H-ASTRA23-04-D — Causal moment correction with an unchanged initial predictor

**Family/change:** causal location/scale adaptation. Add a small zero-output residual readout to a frozen strong parent. Compare base prediction/current-input controls with the same-width readout also receiving a fixed set of causal prefix moments; zero disabled slots after shared fitting-only normalization.

**Assumptions/mechanism:** moment state describes systematic location/scale errors that a cheap readout can correct. Raw levels remain available. A frozen backbone may also make this screen too weak to detect representational benefits.

**Predicted observable:** official fixed-endpoint improvement with an interpretable within/between-sequence covariance change. Require gains on held-out components and report both target deltas, not only an improved moment reconstruction loss.

**Cheapest test/controls:** four new head fits total, two per arm across two clean cells; zero recurrent or teacher fits. Same parent caches, parameter count, sampling, initial predictions, objective, and fixed checkpoint. Explicitly count cache construction and full endpoint grading; a cached-head fit is not automatically cheap.

**Falsifier:** no conditional gain beyond the same-capacity control, or gains confined to composition-sensitive between-sequence effects, lowers priority for this readout. It does not reject richer moments inside a trainable teacher.

**Cost/reuse/exposure/serving:** reuse complete parent caches if their columns support the frozen feature definition; otherwise defer until total cache/fit/grade cost is measured. Propose a one-hour inclusive screen ceiling, not a throughput claim. Folds are exposed; moments reset per sequence and use current/past rows only. Serving adds bounded counters and a small head. Related: [moment results](../../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md), [occupancy proposals](../../ideas/DISCRETE_OCCUPANCY_HEURISTICS_20261017_R1.md). **Status: not_tested.**

## Primary literature and limits

I inspected Seitzer et al., ICLR 2022, [sections 2–4 of the full paper](https://arxiv.org/html/2203.09168v2), not only its abstract. Its analysis distinguishes noise-related inverse-variance weighting from neglect of hard-to-fit regions; its method modifies gradients with a stopped variance factor. That motivates card B but does not establish a failure in this GRU. The [pinned author implementation](https://github.com/martius-lab/beta-nll/blob/669c9f251eb41464c1ec8b43751c7c742dd75cf9/README.md#quick-implementation-of-beta-nll) confirms detachment. Our clipped, magnitude-weighted, teacher-heavy target and shared auxiliary losses differ from the paper's ordinary regression setting. No paper experiment or code was reproduced.

## Serious counterargument and best next decision

This report may overvalue explanations relative to competition utility. WMSE has a useful population optimum but finite-capacity training can benefit from deliberately biased weighting. The NLL tie could mean there is little available gain, while architecture or data coverage matters more. Calibration maps can overfit exposed components; a residual head can become another incomplete delivery. Better bookkeeping has no competition value unless it redirects a finite allocation.

Accordingly, my best next test is **proposed and unexecuted**: one bounded effective-weight and optimizer-step fingerprint on the exact historical WMSE/NLL recipe, after explicit resume. Freeze a small fitting-only window sample, record target/mask/propensity/chunk masses, variance-clamp occupancy, mean/variance/auxiliary gradient contributions, and the resulting trunk/mean-head AdamW step excluding weight decay. Compare fixed WMSE, β=0, β=1, and β=.5 at the same copied learned-variance state; discard all changed states. First establish whether actual optimizer moments were retained. If absent, report gradients and an explicitly cold-moment counterfactual, without claiming reconstruction of historical updates. No quality claims, new checkpoints, or validation-based parameter choices follow.

Give this engineering probe a 30-minute inclusive ceiling on suitable authorized compute, with zero scientific fits and no automatic successor. If it reveals a specific material interaction, choose **one** ensuing screen, rather than all four cards. If it does not, retain the NLL tie, keep the pooled-memory rejection at its exact scope, and spend the next scientific allocation elsewhere. The current review itself is complete without that execution.

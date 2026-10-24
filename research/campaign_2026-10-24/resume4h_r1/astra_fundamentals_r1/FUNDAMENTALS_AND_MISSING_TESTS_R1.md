---
kind: review
status: source_review_complete_probes_proposed
created_utc: 2026-10-24
hypothesis_ids: [H-ASTRA24-FUND-01, H-ASTRA24-FUND-02, H-ASTRA24-FUND-03, H-ASTRA24-FUND-04, H-ASTRA24-FUND-05, H-ASTRA24-FUND-06, H-ASTRA24-FUND-07]
experiment_ids: []
evidence_paths: [INITIAL_REFRAME_R1.md, DIRECTION_PROBE_RECIPE_R1.md, HYPOTHESIS_CARDS_R1.json, SOURCE_RECEIPT_R1.json]
---

# What could be misleading us, and which missing tests matter

The strongest concern is that several experiments do not isolate the question named in their title. The online-head comparison mixes added readout capacity, evolving input summaries, relationship estimation and correction scale. A field-age comparison also changes the grouping of observation events. A low-data innovation screen changes both data diversity and optimizer exposure. These are useful experiments, but their outcomes cannot tell us which of those ingredients deserves the next allocation.

There is no evidence here of a single hidden trick that explains the gap to the aspiration above 0.700. There is evidence that a few inexpensive identification tests can prevent us from repeatedly spending on the wrong explanation. My first choice is a fixed-prediction direction decomposition, followed—only if its result makes the question worth buying—by the absent raw/event-only readout control. This preserves ambitious representation and teacher research without treating more architectures as a substitute for an answer.

## Scope and evidential status

This is the one explicitly requested Astra/ultra source/literature reframe. It ran no fits, inference, scientific payload scans, native execution, provider jobs or public queries. It did not alter shared sources, recipes, ledgers or navigation. The initial proposal was saved before detailed feasibility reports, the PR summary and the old idea register were opened. Mandatory current-state/frontier reading exposed four topic labels from the earlier synthesis; that limitation is recorded in [the initial note](INITIAL_REFRAME_R1.md). The source review is an independent formulation, not an independent replication.

The objective is expected performance of a valid causal deterministic Challenge2 endpoint. The public reference 0.66832 is historical context, not an anchor to which local deltas can be added. Existing folds are exposed. Fixed-prediction grades are measured conditional empirical evidence; their transfer to new data remains uncertain. Primary papers support possible mechanisms in other settings, not Challenge2 gains. Proposed tests below are not admitted runs.

## The three most consequential assumptions

### 1. “If a named feature head helps—or fails—we have tested that feature mechanism.”

**Measured/source facts.** All six saved F0 cells, five unique predictions, lose against the frozen parent. Parent WP is 0.6811541494; the best raw head is 0.6806097840. All five fixed additional-0.1 corrections gain about 0.000129–0.000136 in that cell; A gains 0.0001313682, almost the same as C/lag1's 0.0001362175. Conditional intervals cross zero. [Raw grade](../feasibility_scout/RESULTS_R2.md), [shrink grade](../feasibility_scout/correction_shrink_r1/RESULTS_R3.md).

The code already has a zero-initialized residual, `parent + 0.1 * head(features)`. Its hidden layer receives 211 raw/event features plus 208 relationship channels. A is not an untouched or no-history baseline: it includes matured i0 movements, continuing decayed means and their errors, with slopes forced to zero. The parent is added to the output; its prediction is not a hidden-layer input. A-versus-parent therefore asks whether this trained correction helps, not whether online means help. The absent equal-width zero-stat head is the simple missing control. [Control interpretation](../feasibility_scout/CONTROL_INTERPRETATION_R1.md), [fitter](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/online_relationship_fit.py), [head](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/online_relationship.py).

**New counterevidence received after the initial proposal.** F1 A's raw correction loses 0.0015674131 against its parent. Its fixed additional-0.1 correction changes WP by −0.0000046962, with conditional interval [−0.000112863,+0.000109508]. Parent/shrunken WPs are 0.6699228637/0.6699181675. Thus the positive F0 common-shrink result is not a dependable observed gain across these two cells. This is a completed grade whose independent scalar result review was pending when root sent it; I checked its JSON metadata/scalars, not arrays. The two cells change fold and fitting seed together. [F1 grade](../feasibility_scout/f1_mean_shrink_r1/F1_MEAN_SHRINK_GRADE_R2.json).

**What is still possible.** Useful new information can be present but assigned too much weight. A generic readout can repair parent error without useful relationship slopes. Fit-side parent predictions can be too optimistic, because they are in-sample while evaluated predictions are out-of-fit. A correction that does not condition on the parent output or hidden state may average across different parent confidence states. None of these explanations is established by shrinkage, and none requires calling the existing implementation buggy.

**What changes the decision.** Separate a direction's score value from its magnitude, remove parent-calibration and shared-A directions descriptively, then fit Z versus A if shared value remains worth investigating. Do not jump from these results to “online adaptation does not work,” nor to “shrink all heads and release.” The F1 counterexample makes a generic release claim particularly weak.

### 2. “More precise clocks recover the true time of the problem.”

**Measured/source facts.** Ninety-six sequences yield 65 distinct full-sample event bitstrings among 112 fields. Seven-group ages differ from field ages on 14.3923% of scored field–row pairs; nine-family ages differ on 11.5332%. Boundary censoring is rare in the sampled field–row denominator. These statements concern exact float32 changes and row counts. They do not identify physical time, exchange events, missing observations, useful feature identity or future equality. [Clock findings](../feasibility_scout/clock_eda/FINDINGS_R1.md).

The canonical parent already receives 99 event features, including changes, event flags, recencies and held event changes. Consequently “give the model events instead of raw rows” is not a new treatment here. The raw task also explicitly says numerical feature index is not book-depth order. Same-index correspondence between two anonymous instruments is an assumption to test, not a semantic guarantee. [Event source](../../../../src/wnn/features.py), [task specification](../../../../docs/data_overview.md).

**The alternative formulation.** Treat observed value and observed update process as separate inputs. An unchanged value could reflect unchanged economics, a held observation, quantization, saturation or a representation invariant. Event order and event counts may matter even when age does not; common activity may explain an apparent field-age effect. Individual clock superiority must be judged against a specified group/activity control, not against a vague “coarse time” concept.

Primary work on irregular sampling explicitly models observed values together with their indices, while GRU-D uses observation masks and intervals. These are useful decomposition principles. Their missingness is observed; our equality mask is a proxy for value change. Replacing one with the other is an unverified analogy. [Li and Marlin, ICML 2020](https://proceedings.mlr.press/v119/li20k.html), [Che et al., GRU-D](https://arxiv.org/abs/1606.01865).

**Strongest counterevidence.** Existing event features are already strong enough to be in the parent. Teammate age screens report small negative field-versus-group deltas at both 10% and 25% fitting data. That reduces the value of repeating exactly “more clocks.” It does not disprove an update-pattern/value interaction or a different observation representation. The raw private grades remain unavailable locally. Neither low average censoring nor nonzero clock disagreement establishes predictive importance.

**What changes the decision.** Audit numeric equality before assigning physical freshness: do event differences track stable atoms, float resolution, saturation or consistent group transitions? Preserve full-row clocks and source-qualified sampled membership. If the representation is mostly a common activity counter, buy a cheaper common-state control. If field identity survives these controls, test that specific interaction. Do not force cross-sequence continuity or use test-set value frequencies: serving resets at each sequence and receives only the current/past rows.

### 3. “A cheap stand-alone score ranks mechanisms for the final endpoint.”

**Measured/reported facts.** The innovation shortscreen reports paired deltas +0.00085285, +0.00064007 and +0.00134467 at 10% data, versus −0.00101477, +0.00045897 and +0.00046894 at 25%. The second mean is near zero because one cell reverses substantially while two remain positive. Calling this universal disappearance with more data is too strong. The cells and nested subsets are reused, so these are neither six independent replications nor a causal estimate of sample size. [Source-checked recipe and comments](../kaggle_age_r3/FEASIBLE_NEXT_SCIENCE_R1.md).

The actual recipe keeps whole 20,000-row sequences, six epochs and a OneCycle schedule. Moving from 10% to 25% therefore increases distinct components, examples and optimizer updates, and changes the schedule in update units. Truncated context is not the reported fidelity difference. Repeating the positive 10% pair cannot answer why 25% behaves differently. The reported 6.18-hour faithful four-fit serial B300 projection already exceeds a two-hour follow-up window; infrastructure infeasibility is not evidence against innovation.

In other domains, low-fidelity rankings can depend on optimization phase; the MORL paper demonstrates this for vision learning-rate schedules. It justifies inspecting the proxy, not attributing our data-fraction reversal to the same cause. [Lee et al., WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Lee_Improving_Multi-Fidelity_Optimization_With_a_Recurring_Learning_Rate_for_Hyperparameter_WACV_2023_paper.pdf).

**A second ranking error.** A weaker stand-alone predictor can improve a mixture if its disagreement aligns with the deployed mixture's remaining error. Conversely, a stronger model can be redundant. The local i1 history blend and the current replacement packages make the complete endpoint the relevant comparison. The TL3/H390 diagnostics use fitted or unknown ancestry; their fixed native replay can establish operational behavior and fitted-data diagnostics, not clean CV gain. [PR synthesis](../ALL_PR_RESEARCH_SUMMARY_R1.md), [quota packet](../quota_submission_r1/README.md).

**What changes the decision.** Use `(reference endpoint, fixed intervention, training exposure, serving cost)` as the research unit. A feature name, family or isolated score is insufficient. For fidelity, record update counts, unique groups, sequence coverage and LR progress before discussing scale transfer. For ensembles, test the frozen replacement/addition against the full comparator under the actual clipping order. Do not select a coefficient from the public leaderboard or add local increments to a public score.

## A tempting diagnosis that needs narrowing: “WMSE is the wrong objective”

For one clipped target Y, let w=|Y| on the fixed selected population and define Q by `dQ = w dP / E_P[w]`. Weighted Pearson is ordinary correlation under Q. The maximally correlated predictor among functions of the available causal information X has the direction

`m(X) = E_Q[Y|X] = E_P[wY|X] / E_P[w|X]`,

where the denominator is positive. This follows from conditional expectation and Cauchy–Schwarz: only the projection of centered Y onto functions of X contributes covariance. Weighted squared error has exactly this conditional mean as its minimizer. Because clipped Y lies in [−2,2], m also lies there. Thus prediction clipping does not invalidate the existence of a shared population optimum. Zero-weight states do not affect either criterion.

This does **not** assert finite-sample or optimization equivalence. The current `primary_pair_wmse` correctly clips labels and uses their original clipped magnitudes, but divides each batch's weighted error sum by that batch's weight sum. The fitter consumes ordered 8,192-row batches spanning complete sequences. The expected average of such ratios need not equal the pooled ratio. Fit row weights, mask policy, two targets sharing parameters and denominator, finite capacity, weight decay, clipping and in-sample residuals all change the practical question. [Loss source](../../../../src/wnn/conditional_readout.py), [fitter](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/online_relationship_fit.py).

This derivation and the batch-mass issue were independently reached in this pass but already appear in reviewer 04's [earlier report](../../../unified/reviews/astra_ultra_audit_20261023_r1/04_objectives_and_metric.md). They are unresolved work to reuse, not new discoveries. A cheap batch-mass audit can establish whether the suspected measure difference is material. A new correlation loss or distributional head should earn its added complexity against that specific problem.

## Questions we are not yet answering cleanly

1. Does A beat the same trained 419-wide head with its 208 relationship slots zeroed? This is missing even though it is simpler than C-versus-B.
2. Does a correction retain score-relevant direction after accounting for parent shift/scale through clipping, and after removing A's common direction? A tiny endpoint delta alone cannot answer.
3. Are F0 and F1 differences about correction direction, scale, parent composition or initial sequence behavior? F1's positive early-bin diagnostic is post hoc and does not justify an early-only serving gate.
4. Which information is unavailable to a correction head: useful past input state, the parent's confidence, or the mapping from an anonymous field to its counterpart? Adding a larger generic network bundles these questions.
5. How much of the effective objective is defined by per-sequence row weights and batch denominators rather than the advertised label weight? Inspect the actual producer contract before blaming the loss family.
6. At 10% versus 25%, how much of the contrast remains at equal optimizer exposure? At equal exposure, any remaining result is still a data-diversity intervention; it is not a complete full-budget forecast.
7. Is a purportedly useful component still useful against today's full served endpoint, at native precision and clipping order, with its actual ancestry? A successful Python cache comparison against a different parent cannot answer.

## One decisive cheap test, with zero new fits

The [direction-probe mini-recipe](DIRECTION_PROBE_RECIPE_R1.md) is ready for a separate implementation/grader owner. It freezes the five F0 corrections, the already-chosen alpha=0.1, and optionally the source-checked F1 A. It streams fixed arrays and accumulates per-component moments, requiring no Torch, model inference or new calibration fit. The cap is one CPU, 400 MB, no swap and 900 CPU seconds, additionally bounded by the active deadline. Actual admission and runtime verification remain root-owned.

For each endpoint it calculates the exact right derivative of official correlation through clipping at alpha=0 and 0.1, existing fixed endpoint changes, and clipping-entry/exit mass. It decomposes the direction into parent-calibration span, common-A span and remaining direction. It handles exact ±2 boundaries and numerical rank explicitly. This is descriptive geometry; evaluation-derived projection coefficients cannot become a serving calibrator.

The decision is concrete. If A's direction survives calibration but B/C add little beyond A, spend on Z-versus-A rather than more slope variants. If B/C retain a useful distinct direction, a matched relationship follow-up remains live. If the shared direction is mainly calibration, test a small honest calibration comparison before expensive feature fits. If direction changes by cell, prioritize transfer/conditioning rather than extrapolating the F0 shrink result. Zero-crossing intervals cannot establish equivalence; report costs, estimates and all intervals without a universal significance gate.

## One prospective four-fit screen

**Question:** Does the actual mean-only online state add value beyond the same trained raw/event correction under the existing residual-fitting regime?

Use two existing clean parent/cache cells, F0 and F1, with newly declared fitting seeds 2201 and 2202 respectively. Before execution the actor must verify these seeds have not already defined the same runs and pin actual source-qualified membership, parent/caches and recursive ancestry. If an identity is unavailable, leave that cell unadmitted rather than substitute a different source silently.

At each cell fit two arms: Z has all 208 appended relationship slots zeroed; A uses the existing mean-only state with lag1. Both have the identical 419→64→2 architecture, paired initialization, 211 base features, normalization, source order, three epochs, fixed epoch2, AdamW settings, clipping and strict FP32. Z still computes whatever shared preprocessing is needed for identical base inputs; its disabled inputs are explicitly zero. Parameter count is equal, although the feature intervention intentionally changes active input directions. It is a feature-access control, not a guarantee of equal effective capacity.

This is **four total new small-head fits: two Z and two A; zero new parent, GRU, teacher or student fits**. Reuse the two exact parents and their caches. Do not count historical A predictions as independent replications or substitute them for a newly seeded arm. Primary outcome is `WP(parent + 0.1*(A-parent)) - WP(parent + 0.1*(Z-parent))`, using the fixed additional shrink chosen from the present exposed analysis. Report both arms versus parent, raw corrections as secondary outcomes, both targets, every cell and their equal-cell mean. The new fits are prospectively specified; the evaluation remains exposed. Two fold/seed cells cannot disentangle fold and seed effects.

A positive A−Z direction in both cells would make mean-state access worth a targeted follow-up; a small inconsistent difference with substantial cost favors a cheaper generic correction. Neither outcome establishes a universal mechanism. The screen deliberately keeps fit-side in-sample residuals unchanged to isolate feature access within that regime. If the no-fit probe indicates residual distribution is the dominant unresolved issue, prefer the existing honest-residual descendant instead and document the allocation change.

**Cost and admission:** existing readout caching/thermal failures show that “four small heads” is not a laptop-time estimate. Reuse verified disk caches, measure one bounded batch/sequence cost and charge all setup/export/grade time before admitting a complete panel. No wall-hour promise is supported. Do not draw on the production reserve or silently extend today's deadline. Root owns the draft/allocation; a named actor must own fits through exported predictions and an independent grader must own the frozen comparison. This report hands off a proposed recipe, not an ownerless implementation task.

## Seeded cross-domain reading and what actually transfers

The seed `20261024` ranked SHA256(seed + newline + path) over twelve existing non-README Markdown/text files under `research/unified/sources`; the first two were the Rogii first-place writeup and the Santander talk translation. There was no reroll. Full pool/procedure are in [the selection receipt](RANDOM_SOURCE_SELECTION_R1.json).

From [Rogii](../../../unified/sources/competition-writeups/rogii_top1.txt), the selected brief passage is: “PF features no longer improved my best single model and I kept them in my final ensemble for diversity.” This supports asking about conditional ensemble value after the base strengthens. It does not establish our component's value. Its geological alignment, particle smoothing and whole-input geometry cannot be copied into a causal row callback; its scores are external author reports, not independently reproduced here.

From [the Santander translation](../../../unified/sources/competition-writeups/santander_transcript_english_translation.md), the selected passage is: “The first thing that caught my attention was the number of repeated values in each row.” The transferable operator is to interrogate anonymous numerical encoding before assuming continuous economic units. The talk's global test-count construction, target-class shuffling and cross-row reconstruction are not a valid recipe for this causal task. Its “magic” is not evidence of a corresponding leak here. The bounded successor is a fitting-only/causal atom-and-resolution audit, with any predictive representation subsequently tested on source-qualified groups.

## Novelty, counterarguments and allocation

The old [forty-card register](../../../unified/reviews/astra_ultra_audit_20261023_r1/IDEA_REGISTER.md) already covers dual clocks, atoms, correspondence, batch mass, honest residuals, ordinary diversity and context probes. Those ideas have not become new because a fresh agent rediscovered them. The seven accompanying cards explicitly identify these ancestors. The incremental deliverables are the recovered-head tangent decomposition, the Z/A screen made concrete, and the fidelity/interpretation update using current F0/F1 and PR26/27 evidence.

The strongest counterargument is that none of this may unlock a material gain: the existing parent can already summarize the useful events, all cheap corrections may be too redundant, and teacher/data scale may remain the better investment. That is why the recommended first test is bounded and has an explicit outcome that redirects spending. Conversely, a tiny or uncertain effect does not justify discarding an entire representation family. Current resource failures measure delivery difficulty, and fitted package diagnostics measure technical behavior at their stated scope; neither settles the scientific objective.

The useful change in research unit is a **frozen intervention against a named complete endpoint, with a control that rules out its cheapest alternative explanation**. The near-term deliverable should be that comparison and its reusable predictions, not another architecture proposal or a public-score promise.

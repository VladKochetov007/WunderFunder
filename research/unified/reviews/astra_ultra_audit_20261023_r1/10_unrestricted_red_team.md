---
kind: review
status: complete
created_utc: 2026-10-23T17:14:28Z
updated_utc: 2026-10-23T17:14:28Z
hypothesis_ids: [H-ASTRA23-10-A, H-ASTRA23-10-B, H-ASTRA23-10-C, H-ASTRA23-10-D]
experiment_ids: []
evidence_paths:
  - research/unified/reviews/astra_ultra_audit_20261023_r1/CONTRACT.md
  - research/experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json
  - research/unified/ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md
  - research/campaign_2026-10-04/E421_BRANCH_REPORT.md
  - research/campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/README.md
  - src/wnn/gmetric.py
---

# Reviewer 10: ask what correction is missing before naming the architecture

## Scope and principal judgment

The strongest overlooked question is: **what inexpensive, causal change to an existing prediction would actually improve the complete endpoint?** The answer may call for more capacity, better memory, or a different target treatment. Those are possible consequences, not established diagnoses. A project can accumulate increasingly elaborate models while repeatedly learning the same conditional signal.

My preferred next discriminator is a tiny, fixed **response-speed test**: compare the parent with a bounded causal lead correction, its opposite smoothing correction, and a gain-only control. This could cheaply overturn my hypothesis as well as a vague “more temporal modeling” story. It is proposed and unexecuted. No experiment, import, array scan, model replay, export, worktree, provider job, or publication occurred in this review. Scientific work remains stopped under the [October 23 receipt](../../../experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json).

This is tier C judgment over historical evidence. All existing folds are exposed. The user-reported TL3 public 0.66832 is a historical endpoint observation, not a freshly checked leaderboard value. Nothing here estimates a public score or places a ceiling on the >0.700 objective. Novelty is unnecessary if an ordinary ensemble is the better purchase.

## Four findings that change the questions

**1. Temporal dependence is not synonymous with a need for smoothing or persistent regimes.** The [fixed output-filter experiment](../../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/README.md) measured 0.6812265 for the raw parent, 0.6755207 for fixed-Q/R smoothing, and 0.6647620 for prefix-estimated smoothing. Both filters lost across all four descriptive partitions. Separately, the [early-to-late expert selector](../../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/README.md) scored 0.693871 versus TL3's 0.697669 on the later window; expert-choice agreement did not exceed its marginal-rate baselines. These are different protocols and cannot be pooled into one statistic.

My inference is narrower: the evidence gives little support to these forms of slow output adaptation. An alternative worth attempting to falsify is that part of the useful signal is already present but attenuated or delayed. The opposite alternative is equally serious: smoothing simply discards information, while the parent is already optimally timed. Then sharpening will also lose. A positive lead-filter result would demonstrate an exploitable output defect, not establish physical latency, target maturity, or a causal market mechanism.

**2. The best recent memory observation is also evidence for ordinary diversity.** The [complete i1 screen](../../ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md#complete-panel-update--2026-10-22) reports fixed 90%B/10%C gains of +0.00035953 and +0.00027145, although history model C loses standalone in both cells. Its F0 gain survives removal of prediction clipping, weakening a clipping-only explanation. However, 90%B/10%A, where A removes i1, gains +0.00054446 on F2 and loses only 0.00002308 on F0. Changing information access can produce useful errors without validating the intended memory explanation.

The recorded function-preserving warm-history comparison is sensible and should be retained. A separate allocation question is missing from that mechanism test: would the same training and serving budget buy more from another ordinary seed? The answer need not disprove history to change spending. “Complementary” should describe a measured correction against a declared endpoint, not merely low prediction correlation.

**3. Mathematical metric alignment is not an unused source of free quality.** Inspection of [gmetric.py](../../../../src/wnn/gmetric.py) confirms clipped-label-magnitude weighting and clipping of both targets and predictions. For the selected-row population, write clipped target as Y and weight as w=|Y|. Under the tilted distribution, the conditional mean is

`m(X) = E[wY | X] / E[w | X]`.

Weighted squared error estimates this quantity; an exact distributional model is another route to it. With finite nonzero variances, this conditional mean supplies the maximum-correlation direction, up to admissible positive affine transformations. Clipping, finite capacity, estimation, and the actual fitting population still matter. Thus a distribution head can improve optimization or inductive bias, but its readout formula alone does not create additional information.

This is not a new discovery: [H-111's design](../../../campaign_2026-10-05/joint_target_model/README.md) already states the same limitation. [E421](../../../campaign_2026-10-04/E421_BRANCH_REPORT.md) measured unchanged 0.682090, joint CE 0.630341, and factorized CE 0.593983 on its sparse frozen-state screen. Its newly fitted WMSE control was only 0.492579. Beating that degraded control would have been misleading. The [pooled-objective audit](../../ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md) likewise preserves E291's adverse pooled-memory comparison. I initially considered distributional regression; this evidence changes my disposition to defer unless a specifically different optimization test is justified.

**4. Model-family stories currently exceed what the comparisons identify.** The [B300 review](../B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md) reports a partial wide/shallow advantage of 0.00077732 across twelve matched pairs. Width and depth both change. It is a useful recipe prior, not an isolated scale law. The same record finds the tested geometry bundles negative and no scored field-age or innovation pairs. Retain width as a pragmatic candidate, modify any capacity diagnosis, deprioritize repeating those exact geometry bundles, and keep age/innovation unresolved. Pair pooling and TensorSketch remain plausible proposals, but their semantic or dimensional elegance is not evidence that they contain the missing correction.

## Distant transfers and the actual source draw

**Control engineering: compensate a response rather than replace the estimator.** MIT's [Feedback Systems recitation 17](https://ocw.mit.edu/courses/6-302-feedback-systems-spring-2007/fb1e56673aeec2190270f6a45c7b6765_6_302rec17.pdf), inspected on pages 3–4, describes lead versus lag compensation and explicitly limits lead gain because of noise. The [Caltech course explanation](https://murray.cds.caltech.edu/Why_do_we_need_pole_in_the_lead_compensator%3F), inspected in full, explains why unrestricted differentiation amplifies high-frequency error. The transferable structure is a causal stream with a possibly inadequate response rate and a bounded correction operator. This task is prediction, not closed-loop control; phase margin and plant stability results do not establish predictive benefit. Card A uses a discrete finite difference, not a claimed reproduction of the continuous-time controller.

**Geosteering: local fit residuals describe where a geometric approximation is unreliable.** The authorized text pool was exactly these four sorted relative names under `research/unified/sources/competition-writeups/`: `discord_tricks.txt`, `rogii_top1.txt`, `santander_transcript_english_translation.md`, `semenov.txt`. I used seed **2026102310**, unsigned LCG32 `s=(1664525*s+1013904223) mod 2^32`, with modulo indexing. First draw 446324557 gives index 1: Rogii. The file has 528 lines; second draw 1724086344 gives start `1 + draw mod (528-16+1) = 49`. The actual selected window is lines 49–64. Its passage includes **“Quadratic fitting coefficients / Quadratic fitting residual RMSE”**, amid slope and typewell/horizontal interaction features. [Original text](../../sources/competition-writeups/rogii_top1.txt).

The narrow transfer is to distinguish a smooth correction from irregular correction noise using local residual structure, rather than treating all disagreement as one scalar. Card B implements an even simpler causal slow/fast decomposition first. A learned roughness gate would be a later descendant, not silently added to this screen. The original task uses geological structure and visible-region information; its coordinates, particle smoother, labels, RMSE, and reported competition success do not transfer. The bounded text draw avoids PDF extraction entirely. Context lines 40–84 were inspected after the draw; the quoted passage itself lies in the selected window.

Directed paper search also inspected [Imani and White, 2018](https://proceedings.mlr.press/v80/imani18a/imani18a.pdf), abstract and Section 3, including the local gradient bound. Its relevant claim is that soft histogram targets may improve regression optimization; it does not prove a superior Bayes estimand or a Challenge2 gain. [Bellemare et al., 2017](https://proceedings.mlr.press/v70/bellemare17a.html), abstract only, motivates distributional value learning in reinforcement learning; Bellman dynamics are not part of this task. These papers inform the deferred card D, with E421 as direct counterevidence. Friedman’s original gradient-boosting paper was sought, but the author PDF failed and the publisher returned an access barrier; no detailed claim here depends on an unread paper. Several older Caltech chapter URLs also failed; the accessible teaching sources above supply the inspected control material.

## Hypothesis cards

### H-ASTRA23-10-A — bounded causal response-speed correction

**Family/status:** temporal output response; `not_tested`. Related opposite intervention: [fixed Kalman output filtering](../../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/README.md). This changes the prediction operator, not model memory capacity.

**Change and assumption:** from existing prediction p, define `z_t=clip(p_t,-2,2)` and `d_t=z_t-z_(t-1)`, resetting `d_0=0`. Freeze five endpoints: z; `z+0.125d`; `z-0.125d`; `1.125z`; and previous z. Apply unchanged official clipping/scoring afterward. The small coefficient is a declared exploratory choice, not tuned or physically calibrated. The first difference has bounded discrete-frequency gain; its maximum amplification here is 1.25 before final clipping.

**Mechanism/prediction:** if predictions underreact to informative recent changes, positive correction should beat both the parent and gain-only control. Opposite smoothing and one-row delay diagnose direction, without claiming that either is a matched architecture control.

**Cheapest test/controls:** one streamed pass over already exported B predictions for F0/1001 and F2/1002 from the completed i1 panel, scoring all five fixed transforms on the full unchanged mask. Primary contrast is positive correction versus parent; gain-only distinguishes some saturation/scale effects. No coefficient search. Report both target deltas, fixed component uncertainty and sequence-position contributions.

**Falsifier:** consistently nonpositive primary deltas, gains explained entirely by gain-only, or unstable correction direction across cells defeats this specific underreaction remedy. Any future-row access or reset error invalidates it. A successful small-model result still requires a separate strong-parent replay before endpoint recommendation.

**Cost/reuse/ancestry/serving:** zero fits and teacher fits; reuse two exact prediction caches, masks and membership manifests. Budget the future cache verification, two full-fold streams and grade explicitly; elapsed time is unmeasured, so no “minutes” promise. O(1) extra state per target and a few arithmetic operations per callback; native parity/runtime remain future work. Both cells are exposed, and fold/seed effects remain confounded.

### H-ASTRA23-10-B — locate the useful part of history disagreement

**Family/status:** conditional component value and temporal decomposition; `not_tested`. Child/refinement of [i1 history](../../ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md) and H-203's output-residual direction, not a new history mechanism.

**Change:** on unmodified raw B/C predictions set `d=C-B`; form causal `s_t=0.5*s_(t-1)+0.5*d_t`, initialized with `s_0=d_0`, and `f=d-s`. Freeze B, `B+0.1d`, `B+0.1s`, `B+0.1f`. Retain the original 90/10 blend exactly as the known reference.

**Assumption/mechanism/observable:** useful complementarity may reside predominantly in persistent offsets or recent changes. One isolated component may retain most of the full correction's benefit while the other degrades it. This does not identify the underlying information source; the decomposition changes scale as well as frequency content.

**Cheapest test/controls:** zero-fit streaming evaluation in the two original cells with the parent and original mixture as controls. Fix one half-life/operator only, with no best-frequency search. Inspect additive covariance/variance diagnostics as explanations alongside full official endpoints; do not replace pooled WP with average bucket correlations.

**Falsifier:** neither component preserves useful endpoint contribution, inconsistent signs, or dependence on saturation alone weakens this decomposition. Do not infer that all history is useless. A negative result costs no additional recurrent fits.

**Cost/reuse/ancestry/serving:** two paired cache streams and fixed grades, elapsed cost unmeasured; zero new teachers or fitted models. Reuse B/C source identities and full sequence order. All exposure persists. Serving still requires both original models and C's unsupported native history path; this diagnostic supplies no immediate cheap submission. If a correction later proves compressible, that is a separately costed distillation claim.

### H-ASTRA23-10-C — make ordinary seed diversity the spending comparator

**Family/status:** ensemble allocation; `not_tested`. Related to the existing i1 mixture and H-305's earlier seed-versus-snapshot comparison; no novel ensemble theorem is claimed.

**Change:** reproduce the exact cold B recipe with two new predeclared independent seeds in each original fold. Evaluate each fixed `0.9B+0.1B_new` against the existing `0.9B+0.1C`. Average the two new-seed deltas within each fold; never choose the better seed. New seeds must differ from each other and the original cell seed.

**Assumption/mechanism/observable:** optimization diversity may buy comparable target-aligned correction without the history representation. If ordinary members match history's contribution at lower serving cost, they are attractive regardless of whether history carries some real information.

**Cheapest meaningful screen/controls:** **four new scientific fits total**, two ordinary-seed models per fold, zero teachers. Reuse the two exact historical B anchors and two C candidates only after recipe/data/source identity is established. Different source or sampling semantics would invalidate that reuse and require redesign rather than quietly changing the control. This is an allocation screen; it does not cleanly isolate history from every source of stochasticity.

**Falsifier:** the predeclared average ordinary-seed contribution loses to history in both folds at comparable inclusive cost; retain history's priority. Mixed results remain uncertain and may justify neither expansion. Four models do not establish significance or public transfer.

**Cost/reuse/ancestry/serving:** four full cold fits plus exports and grades, not four tiny heads. Derive admission from the original B runtime under the future authorized hardware; no current budget is spent. Reuse exact masks and memberships, with the same exposed folds. An ordinary H512 member avoids history-feature implementation but still adds recurrent inference cost. Measure complete ensemble runtime before claiming a serving advantage.

### H-ASTRA23-10-D — soft distribution targets only as a specific optimization retest

**Family/status:** distributional optimization, `not_tested`, **defer**. Explicit descendant of H-111/H-147/E537-F and E421, not a rediscovered untouched family.

**Change:** only if future diagnostics justify it, compare a parent-anchored scalar residual, hard-bin auxiliary CE, and soft-histogram auxiliary loss with identical continuous serving heads, shared capacity and update budget. Soft/hard arms use the same fixed bins and auxiliary weight; only target smoothing differs. Parent prediction is unchanged at initialization. This tests an optimization hypothesis, not a new metric readout theorem.

**Assumption/mechanism/observable:** supervised soft targets might make useful updates easier than hard quantization; the predicted observable is improvement over both the anchored scalar and hard-bin arms, while beating the untouched parent. Lower auxiliary loss alone is insufficient.

**Cheapest test/controls/falsifier:** six head fits, three arms × two ancestry-clean exposed cells, zero new backbone/teacher fits if exact cached states exist. First require evidence that the proposed comparison differs from prior executed recipes; otherwise spend zero. Failure to exceed the competent scalar arm and untouched parent closes this configuration. A tiny frozen-state negative does not establish an end-to-end impossibility.

**Cost/reuse/ancestry/serving:** six head fits plus state streaming/export/grade; state generation may dominate and is not free. If caches are absent, reassess the whole cost before admission. Fit bins/scales only on fitting groups; keep all producer ancestry explicit. Auxiliary bins disappear at serving, leaving a scalar correction, but native parity is unmeasured. Existing adverse evidence makes this lower priority than A–C.

## Seven questions whose answers change a decision

1. Does a fixed causal lead correction improve a complete endpoint beyond a gain-only change? A yes directs effort toward response dynamics; a no removes this cheap explanation.
2. Is the history mixture's useful correction slow, fast, or inseparable? That decides whether the next representation should preserve levels, transitions, or both.
3. Does another ordinary seed buy at least as much fixed-mixture gain per full training-and-serving cost? A yes can change allocation without disproving history.
4. Do warm history gains survive the already planned function-preserving initialization? A failure weakens information-retention explanations while leaving cold optimization diversity open.
5. Which target and score-weight mass actually account for a proposed improvement? Broad average accuracy can hide damage to the heavily weighted observations that determine WP.
6. Does a new head beat the untouched parent, or only a newly trained control that has forgotten it? E421 makes this a concrete risk, not a hypothetical warning.
7. What would distinguish a genuinely different distributional experiment from H-111/E421/E537-F? Without that answer, another literature citation should not reopen spending.

## Counterargument, limits, and one next decision

The strongest objection to my recommendation is that it could overfit a compelling pattern in the negative results. Smoothing usually changes timing and amplitude; its failure does not imply exploitable lag. Current models may already predict the appropriate conditional expectation, in which case finite differences amplify noise and the only useful next step is better representation learning or scale. Ordinary seed ensembles also demand repeated serving cost and can conceal the genuinely useful information that history adds. The original warm-history test has more direct project support than my lead-correction idea.

That objection argues for a **single bounded A test**, not a new response-model campaign. After an explicit resume and cost admission, freeze the two cached B endpoints and the five operators above, then grade once. No raw-input pass, new fit, coefficient sweep or artifact packaging is necessary for that scientific question. A failed test redirects attention to the existing warm-history comparison and simple seed alternative. A positive test earns a strong-parent replay; it does not earn a public-score forecast.

I read the assigned contract, common project navigation and selected historical records, not sibling reports, assignments or synthesis. The common source index incidentally displayed the title/description of the newly added audit source register; I did not open or use that register. Some broad text outputs were truncated; numerical claims above rely on the subsequently inspected focused result notes. Source search was selective, not an exhaustive duplicate audit. Exact payloads, current provider access and runtime feasibility remain unverified. These limitations are why every card remains `not_tested` and all execution remains stopped.

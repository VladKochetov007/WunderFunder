---
kind: review
status: complete
created_utc: 2026-10-23T17:05:30Z
updated_utc: 2026-10-23T17:05:30Z
hypothesis_ids: [H-ASTRA23-08-A, H-ASTRA23-08-B, H-ASTRA23-08-C, H-ASTRA23-08-D]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-17/research_reset_r1/eda/FINDINGS.md
  - research/campaign_2026-10-17/research_reset_r1/eda/EVENT_LAG_CONTROLS.md
  - research/campaign_2026-10-13/strategy_execution_r1/DOMAIN_PANELS_R1.md
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
  - research/unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md
  - research/unified/validation/VALIDATION_PROTOCOLS.md
  - src/wnn/gmetric.py
---

# Transfer depends on what is observed, selected, and preserved

## Question, scope, and recommendation

Which promising mechanisms exploit transferable information, and which might fit the supplied observation and selection process? The most useful next step is a **proposed, unexecuted stress grade of the completed i1/history endpoints using already frozen component domains**. It would address a missing comparison before another representation screen. It cannot identify the hidden deployment distribution.

This independent review follows the [audit contract](CONTRACT.md). It used text/source reads and primary literature abstracts only; no data or prediction arrays were opened, and no fit, import, test, profile, export, provider job, or service action ran. The October 23 stop remains effective. Findings below are tier C interpretation of existing evidence. All local folds are exposed; source independence does not repair that exposure. Historical TL3 public 0.66832 is not a ceiling or a forecast. No sibling reports or synthesis were inspected.

## Five findings

### 1. Missing representation and missing information are different problems

The [data contract](../../../../docs/data_overview.md) supplies two instrument streams, anonymous future-movement targets, and row position. It does not disclose target horizon or maturity, physical elapsed time, or instrument identity. A model may infer useful proxies, but we cannot assume that one row means a fixed duration, that a price-like code is invertible into price, or that a previous target becomes available after one row. Component indices also do not establish book depth.

The [lagged-label proxy](../../../campaign_2026-10-18/online_features_r1/causal_label_proxy_r1/README.md) achieved 0.521679 WP on 32 exposed evaluation components. This measures predictability of the previous row's hidden label from current inputs; it does not create observed feedback. A deterministic proxy can denoise or summarize available history, but it cannot introduce information absent from that history. The same distinction applies to causal moments, field ages, world-model latents, and larger teachers: their value can be better learning, memory, or regularization without new information.

For one target, write $H_t$ for legal current/past observations, $S_t$ for the official selection indicator, and $Y_t$ for the clipped target. Under the distribution weighted by $S_t|Y_t|$, a correlation-optimal predictor is the following conditional mean, or a positive affine transform that respects the output bounds:

\[
m(H_t)=\frac{\mathbb E[S_t|Y_t|Y_t\mid H_t]}{\mathbb E[S_t|Y_t|\mid H_t]}.
\]

This is a population characterization, not a fitted estimator or numerical bound. It follows by conditional expectation under the weighted law; $m$ itself lies inside the clipping interval wherever its denominator is positive. It shows why residual error, input-reconstruction accuracy, and ordinary conditional means answer different questions. We do not know this conditional law. Nearby current inputs with different labels do not establish irreducible noise because their histories, latent states, and selection can differ. Current scores therefore do not justify an attainable-score ceiling.

**Disposition:** retain larger teachers and state representations, but require claims to identify the information made easier to use. Deprioritize explanations that treat pseudo-labels as realized labels or input novelty as target novelty.

### 2. Scoring selection is a second transport problem

The [validation protocol](../../validation/VALIDATION_PROTOCOLS.md) reports 3,528,273 scored validation rows, 9.4188% of all validation rows; train has no official scoring mask. The callback cannot observe that mask. The [canonical scorer](../../../../src/wnn/gmetric.py) additionally weights by clipped target magnitude. Matching the distribution of all rows is therefore insufficient to establish matching of the evaluated problem.

The [October 17 census](../../../campaign_2026-10-17/research_reset_r1/eda/FINDINGS.md) makes the distinction tangible: retrospective class 3 holds approximately 47.69% of combined selected magnitude weight, and 43.690% of its weight occurs at age at least 64 relative to the chosen i0 conjunction. Those percentages are not additive score contributions. The classes use labels; they cannot be a deployed router. A model can look good on typical rows while missing high-weight quiet intervals.

A scoring-propensity network is a useful approximation, not knowledge of why the organizer selected rows. If selection depends only on observed history and its mechanism is stable, propensity weighting can help align training. If selection depends on unobserved future outcomes, or the mechanism changes between sources, a model of $P(S\mid H)$ alone cannot identify $P(Y\mid H,S=1)$. Strong mask classification would not settle that issue. [Sugiyama et al.](https://jmlr.org/papers/v8/sugiyama07a.html) explicitly define covariate shift using an unchanged conditional output law; that assumption has not been established here.

The earlier [first-principles assessment](../../ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md) already proposes propensity powers zero/one/two and rejects treating five retrospectively correct composite-validation signs as a calibrated veto. This is an unfinished diagnostic direction, not a new discovery by this review.

**Disposition:** modify mask-aware training investigations to report concentration and transport sensitivity. Keep the actual official mask unchanged. All-row and synthetic-mask results remain secondary diagnostics.

### 3. Leakage units are not necessarily deployment domains

[Campaign split code](../../../../src/wnn/splits.py) validates component exclusion against a feature-derived relationship graph. That is valuable protection against known related-sequence overlap. It does not prove that different components represent different instruments, periods, encoding regimes, or independent market conditions. Conversely, a component-wide feature distribution can differ without representing temporal drift.

The October 17 all-feature EDA sampled 256 sequences equally across retrospective classes. Its class-conditional i1-volume histogram discrepancies reach 0.2349 nats, but composition can explain them. The [lag controls](../../../campaign_2026-10-17/research_reset_r1/eda/EVENT_LAG_CONTROLS.md) use only 64 sequences inherited from those strata. Their opposite alignment effects are reasons to investigate conditional usefulness, not evidence that i1 is unstable or causal.

An important existing asset is the [frozen domain specification](../../../campaign_2026-10-13/strategy_execution_r1/DOMAIN_PANELS_R1.md): component activity, i1 change frequency, and feature-support size, with 25%/75% component-frequency mixtures and cutoffs fit on eligible valid-source fitting groups. It preserves original row weights and recomputes pooled WP. These are artificial sensitivities, not estimates of future prevalence. Quiet/transition, source-similarity, and cross-fitted propensity panels remain explicitly pending in that record.

Public/local reversals are consistent with several explanations: domain mixture, selection, ancestry, adaptive choice, or package differences. They do not uniquely identify any one. The [protocol note](../../validation/VALIDATION_PROTOCOLS.md) records a locally positive low-LR continuation accompanying public .6680 versus .6682 previously; its lesson is to diagnose transfer, not discard local evaluation.

**Disposition:** retain component CV and add structural stresses. Do not rename a new partition a fresh holdout or claim that synthetic worst-case composition approximates the private set.

### 4. Feature geometry may encode observation conventions as well as dynamics

The [moment investigation](../../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) shows that arcs can arise algebraically from mixtures with a frequent atom. Near-arc examples have more than two values; a second parabola root can be virtual rather than a support bound. This weakens a literal two-state market interpretation while leaving occupancy and state estimation useful.

Its first pilot's mean moment gain was +0.00088332, but exposed-fold means opposed: −0.00788477 and +0.01306417. Longer-budget R3 made moments/scale worse on average, while a fixed control/dictionary mixture gained +0.007337 on the same tiny panel. That is a reason to separate atom identity, occupancy, and useful complementary error; it does not establish a stable latent regime or reject the family.

Per-field age has a related semantic trap. No observed change since sequence start is **left-censored duration**: the last true change could predate the visible window. Resetting a counter is correct state hygiene, but it does not turn the reset into an observed market event. Likewise, unchanged values are not established missing observations. [GRU-D](https://arxiv.org/abs/1606.01865) motivates timing representations for actual missingness; its clinical mechanism does not establish that anonymous repeated values here mean stale measurements.

**Disposition:** retain raw values alongside relative/occupancy features, explicitly represent whether a change has been seen, and avoid imposing universal normalization, exchangeability, or a hard event clock. The census's 1,253 exceptions, including 92 scored valid exceptions, already falsify the selected i0 conjunction as a universal target-change gate.

### 5. Complementarity is a stronger lead than an unverified semantic story

The [completed i1 grade](../../../campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json), interpreted with its [progression note](../../ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md), supports three distinct statements. Removing i1 costs about 0.01590/0.01111 in the two cells. History C loses alone, but the fixed 90%B/10%C mixture gains +0.00035953/+0.00027145. A fixed mixture with i1-removed A also helps F2 but not F0. Thus useful i1, complementary history, and evidence specifically for temporal memory are different claims.

These were six cold fits on 10% of fitting components, with one seed per fold. C's active random input columns changed its initial function as well as information access. The warm function-preserving comparison is consequently sensible, but incomplete delivery is no negative mechanism result. The same applies to the unfinished online-relationship panel and unstarted field-age scientific screen.

[D'Amour et al.](https://jmlr.org/papers/v23/20-1335.html) show that similar in-distribution performance can coexist with different deployment behavior. The useful correspondence here is to inspect conditional prediction behavior, not infer that every gain must be a shortcut.

**Disposition:** preserve fixed mixtures and reversible raw bypasses. Deprioritize wider architectural explanations until the existing small contribution has a stress profile; do not impose universal positive-domain or nonzero-interval gates.

## Four hypothesis cards

### H-ASTRA23-08-A — Composition sensitivity of the history increment

- **Family/change:** evaluation transport; apply the three existing frozen component panels to B, fixed 90%B/10%C, and the already specified 90%B/10%A comparator. This extends S30 and the existing i1-history hypothesis rather than creating another memory architecture.
- **Assumptions/mechanism:** activity or support composition may explain a fragile correction. The two cells' artifacts and panel memberships must align exactly.
- **Predicted observable:** a transferable local lead retains useful covariance contribution under both prescribed mixtures; concentration in one domain or sign reversals identify where it needs modification. Survival is robustness evidence only.
- **Cheapest test/controls:** zero new fits; stream fixed prediction pairs through one official mask per source-qualified component, then recompute pooled WP at natural, 25%, and 75% high-domain frequency. Use all three already frozen panels, no cutpoint or coefficient search. Report both targets, each cell, and paired component uncertainty.
- **Falsifier:** the claim of broad composition stability fails if the positive ordinary delta reverses systematically under the declared mixtures. That does not falsify conditional usefulness; an interval crossing zero alone is insufficient.
- **Cost/reuse:** reuse the six existing cold fits; zero new model or teacher fits. Planning ceiling: one future bounded CPU grading job of at most one worker-hour, contingent on a separate small profile and streamed storage. Existing panel/evaluator code is reusable, but current compatibility has not been executed here.
- **Ancestry/exposure/serving/status:** preserve exact original fitting memberships and two exposed fold/seed cells; no additional serving cost. `not_tested`.

### H-ASTRA23-08-B — Separate target learning from selection concentration

- **Family/change:** selection transport; revisit the existing [S05/S31 power ladder](../../ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md) with propensity powers 0, 1, and 2 at matched continuation budget.
- **Assumptions/mechanism:** excessive concentration on an estimated selection mechanism may sacrifice useful information or amplify source-specific selection. Alternative explanation: weighting accurately focuses a stable official task.
- **Predicted observable:** reduced power improves ordinary official score or its domain stability while reducing effective component/row concentration. An all-row improvement alone is not success.
- **Cheapest test/controls:** first inspect already saved, ancestry-qualified propensity outputs and fixed residual moments after authorization; do not claim mask predictability diagnoses missing-at-random selection. A scientific screen is six continuations: two cells per power, no new teacher fits. Reuse exactly matched clean parent and propensity artifacts; if eligible propensity artifacts are absent, defer and redesign the complete cost rather than silently add fits.
- **Falsifier:** lower powers consistently lose official and fixed-stress endpoint comparisons, or the expected concentration difference is absent. This narrows this weighting recipe, not every shift remedy.
- **Cost/reuse:** six full continuations plus exports/grades; potentially substantially more expensive than small heads. Runtime is unmeasured and no laptop allocation is requested. Exact existing controls may reduce new fits only after identity checks.
- **Ancestry/exposure/serving/status:** propensity training and parent ancestry must exclude each evaluated component; all folds remain exposed. No extra inference network if weights affect training only. `not_tested`.

### H-ASTRA23-08-C — Separate atom identity from occupancy

- **Family/change:** representation transport; a decomposition child of [discrete occupancy](../../ideas/DISCRETE_OCCUPANCY_HEURISTICS_20261017_R1.md). Compare equal-width residual heads using zero extra channels, fitting-defined atom identity, or causal occupancy/rank summaries; retain the identical raw parent path.
- **Assumptions/mechanism:** relative occupancy may preserve useful state when exact numeric atoms change across components. Exact atoms may instead carry stable semantics and deserve retention.
- **Predicted observable:** the occupancy arm has a more stable fixed-endpoint increment across feature-support domains; identity may win where fitting support is shared. Define support diagnostics before outcome inspection.
- **Cheapest test/controls:** six small-head fits, two per arm across two cells; zero new GRU/teacher fits. Freeze initial parent parity, common normalization membership, head capacity, updates, and any fixed blend. A shuffled feature-to-field mapping is a later falsifier, not an uncounted extra arm.
- **Falsifier:** occupancy fails to improve the raw control, or identity remains superior across all declared support domains. A tiny-head failure does not rule out recurrent integration.
- **Cost/reuse:** reuse verified frozen-parent caches; causal counting and any missing caches require profiling. Six-head delivery can take hours, as recent operational history warns; no speed estimate or currently runnable claim is made.
- **Ancestry/exposure/serving/status:** dictionaries and scaling fit only on fitting components; validation remains exposed. Bounded counters/dictionaries reset per sequence; native integration and runtime are additional work. `not_tested`.

### H-ASTRA23-08-D — Treat unseen prehistory as censored age

- **Family/change:** observation-process representation; a child of [per-field age](../../ideas/PER_FIELD_AGE_20261022_R1.md). Add whether each field has changed at least once in the observed prefix, retaining current age and raw inputs.
- **Assumptions/mechanism:** a counter before its first observed reset has different meaning from an equally long interval after an observed change. Before saturation, the flag can be reconstructed from exact age and row position if both are available; it is then a learning aid, not new information. The [current age implementation](../../../../src/wnn/research_features.py) caps ages at 10,000, where the flag can preserve a distinction discarded by that representation. The parent might already encode it.
- **Predicted observable:** a useful increment concentrates in fields/positions with censored histories without degrading the full endpoint. Effects tied only to row position suggest a clock shortcut.
- **Cheapest test/controls:** first measure the relevant scored weight after authorization; negligible coverage argues against spending on this recipe. If informative, six residual-head fits: padded age baseline, correct field-specific flags, and fixed deranged flag-to-field mapping, each on two cells. All flags use past/current inputs only; identical width, zero-output initialization, parent cache, and updates. The deranged control preserves the contemporaneous flag population while disrupting field correspondence.
- **Falsifier:** correct flags do not outperform both controls, or a gain vanishes outside the early-position slice. A weak early-only gain may still justify a narrowly scoped component; it does not establish physical duration.
- **Cost/reuse:** zero new parent/teacher fits; six heads plus feature production and complete grades, all unprofiled. Reuse the age-screen implementation and caches where identities match; do not restart its stopped panel.
- **Ancestry/exposure/serving/status:** fitting-only normalization, clean parent ancestry, exposed folds. At most 112 additional Boolean states plus readout work, subject to measured callback cost. `not_tested`.

## Seven decision-changing questions

1. Does the completed history mixture survive the existing component-frequency stresses, and does its gain remain after separating within-domain covariance from between-domain offsets?
2. Which independently documented target properties establish maturity or horizon? Without them, which online proposals remain valid using observable-input outcomes only?
3. Are propensity errors concentrated by feature support, sequence position, or source, and do lower weighting powers change official endpoint ranking?
4. Do apparent i1 distribution differences persist under comparable component composition, rather than equal label-class sampling?
5. Are exact feature atoms stable across components, or does useful occupancy survive when a fitting dictionary reports unknown values?
6. How much scored magnitude weight occurs before a field's first observed change, and can an existing recurrent state already distinguish those intervals?
7. Which local/public disagreement has a completely matched package, inference path, mask protocol, and known ancestry, sufficient to isolate transfer rather than implementation or selection?

## Counterarguments, limits, and the next decision

The strongest objection is that these stresses may delay a real improvement to study imaginary shifts. Whole-sequence activity/support summaries can mix economic information with nuisance; deliberately making a model invariant could delete predictive signal. The public set may closely match ordinary validation composition. I therefore recommend a bounded diagnostic, not robust optimization, universal invariance, a worst-domain veto, or a requirement to explain every gain causally.

A second objection is scale: cold 10%-component history results may say little about a strong parent. True. The proposed zero-fit grade prioritizes the next comparison; it cannot replace the already motivated function-preserving warm test. Conversely, a new teacher could improve the conditional predictor substantially without resolving hidden target semantics. Nothing here supplies an information ceiling or a reason to abandon scale.

**Best next test, proposed and unexecuted:** card A, with one frozen three-panel report and no new coefficients, domains, or fits. After explicit research resume, root would own admission and artifact binding, a grader would own paired computations, and an independent reviewer would check interpretation. If alignment/profile fails, stop that job and preserve the reason; do not rebuild huge caches automatically. If composition sensitivity is substantial, prefer a targeted descendant of the history/occupancy line. If the increment is stable, proceed toward the matched strong-parent comparison with more confidence in allocation, not a public-score forecast.

Literature inspection was deliberately limited to the linked primary abstracts: Sugiyama et al. for the conditional-law assumption behind covariate shift; D'Amour et al. for underspecification; Che et al. for missingness/time representations. None establishes gains in this competition. No random-literature sample was conducted in this bounded review. Existing reports and selected original source/grade text were inspected; reported historical numbers were not recomputed, masks were not revalidated rowwise, and hidden deployment data remain unavailable. Root owns indexing and ledger registration of this report.

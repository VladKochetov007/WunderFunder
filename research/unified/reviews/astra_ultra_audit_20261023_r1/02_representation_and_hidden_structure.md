---
kind: review
status: complete
created_utc: 2026-10-23
updated_utc: 2026-10-23
hypothesis_ids: [H-ASTRA23-02-A, H-ASTRA23-02-B, H-ASTRA23-02-C, H-ASTRA23-02-D]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-17/research_reset_r1/eda/FINDINGS.md
  - research/campaign_2026-10-17/research_reset_r1/eda/EVENT_LAG_CONTROLS.md
  - research/unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md
  - research/unified/ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md
  - research/campaign_2026-10-15/vast_a_cpu_probe_r1/PER_FIELD_CLOCK_INPUT_SCREEN_20261015_R1.md
  - research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json
  - research/experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json
---

# Representation review: distinguish the observed encoding from the latent process

The largest opportunity is to test which aspects of the encoding carry transferable information: numerical level, identity of a repeated value, order of changes, or elapsed rows. Current evidence supports useful i1 inputs and heterogeneous temporal structure. It does not identify a universal event clock, prove that same-index i0/i1 channels form meaningful pairs, or establish that anonymous components are exchangeable. Several proposed architectures quietly require those stronger assumptions.

This is a tier-C source and literature review under the [audit contract](CONTRACT.md). I read small text artifacts and selected source, browsed primary papers, and wrote this report. I ran no fits, imports, tests, profiles, data/array scans, exports, services or provider jobs. All proposals below are **not_tested**. The [October23 stop](../../../experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json) remains binding. Existing local folds are exposed; no new score or public forecast is inferred.

## Four findings that change the framing

### 1. i1 usefulness does not establish a cross-instrument correspondence or a lead–lag mechanism

**Measured:** the completed cold screen loses 0.01590144 on F0/1001 and 0.01110844 on F2/1002 when standard i1 inputs are removed. Explicit history C loses against B alone in both cells, but the fixed 90%B/10%C mixture gains 0.00035953 and 0.00027145. The conditional component intervals are respectively [−0.00001187, +0.00070893] and [+0.00008223, +0.00043398]. The no-i1 A mixture also improves F2, by 0.00054446. These are two exposed fold/seed cells, not independent confirmation of memory. [Progression and primary grade links](../../ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md), [independent grade review](../I1_SIX_FIT_HISTORY_GRADE_REVIEW_20261022_R1.md).

**Source fact:** C activates 156 randomly weighted input columns that B masks. Identical stored weights therefore did not preserve the initial function. The subsequent zero-extension warm design addresses this confound, but the stopped work has not delivered a completed warm quality comparison.

**Inference:** i1 may supply contemporaneous state, sequence context, event intensity, nonlinear interactions, or history. Removing the entire block cannot separate these. The [online proposal](../../ideas/ONLINE_RELATIONSHIPS_20261022_R1.md) explicitly treats its 52 same-index correspondences as assumptions. The data documentation establishes two instruments and persistent column identities; it does not establish equal transformations or a physical correspondence between their same-numbered components. A successful regression from one code difference to another would still be a predictive relationship between encodings.

**Decision:** retain i1 and the function-preserving history question. Modify descriptions that equate the small blend gain with “discovered temporal information.” Defer a universal cross-impact narrative until a correspondence-specific comparison exists. The [seven-of-twelve online terminal receipt](../../../experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json) is an incomplete delivery outcome, not negative evidence about online learning.

### 2. There are at least three different clocks, and none is a verified target clock

The clocks are row position, observed feature-change count, and rows since a selected field last changed. They need not agree. The [official local documentation](../../../../docs/data_overview.md) supplies ordered rows and resets, but no physical elapsed-time column or target definition. Consequently “stale,” “event,” and “duration” should name observables rather than imply missing observations or market-time units.

**Measured:** the [249.6-million-row census](../../../campaign_2026-10-17/research_reset_r1/eda/FINDINGS.md) found 1,253 eligible target changes outside the selected i0 conjunction, including 92 scored valid exceptions. One documented exception changes the two targets by approximately −1.6893 and +2.1720 while both selected codes remain unchanged. The all-feature sample found no identical adjacent 112-vectors; an extra feature changes on every transition. Thus selected-price persistence is compatible with changes elsewhere and cannot justify freezing the predictor. Conversely, all-row change in extras makes an indiscriminate “any feature changed” clock nearly useless.

The [64-sequence lag control](../../../campaign_2026-10-17/research_reset_r1/eda/EVENT_LAG_CONTROLS.md) found persistence but heterogeneous i1/i0 association. At validation lag1, observed-minus-shifted association varies from positive to negative across retrospective strata. Its 1,000-row circular shift is an offline phase-association diagnostic, not a causal serving feature or a significance test. Positive event persistence alone does not identify target-relevant history.

**Boundary issue:** the [current age recipe](../../../experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json) starts ages at zero. Before a field's first observed change, age is time since observation began, only a lower bound on its true dwell duration. After a change it is an observed duration. A seen-change flag distinguishes these states. However, the [eight-sequence clock receipt](../../../campaign_2026-10-15/vast_a_cpu_probe_r1/PER_FIELD_CLOCK_INPUT_SCREEN_20261015_R1.md) found only 0.368% of field-rows with age equal to sequence position. That argues against promoting censoring alone to the main explanation without checking its concentration on relevant rows.

**Decision:** retain field ages as an unresolved representation. Modify them conceptually to include observation confidence; compare against position and activity explanations. Deprioritize hard skipping/freezing as a first quality intervention. Sequence boundaries remain mandatory state resets even if empirical dependency components are used for fitting splits and uncertainty; those components never authorize cross-sequence serving state.

### 3. Repeated numerical atoms suggest a mixed representation, not a recovered physical law

**Measured:** the [moment investigation](../../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) found near-arc sequences with 8 and 23 distinct values; one selected example placed 92.285% of rows on its two most frequent values. It also explains why a point mass mixed with a broader distribution can produce the same quadratic mean–variance curve as a two-point distribution. An apparent arc endpoint can therefore be virtual. Selected illustrations do not establish prevalence or physical meaning.

**Inference:** a channel can contain both an atom identity and an excursion coordinate. A single affine normalization may make a frequently held code numerically convenient while failing to distinguish “at a common atom” from “near that number.” But exact equality could also reflect quantization of an ordinary continuous quantity. Both models remain possible. The useful representation preserves raw magnitude while exposing atom membership and a continuous residual, rather than forcing every sequence onto one curve.

This overlaps [discrete occupancy](../../ideas/DISCRETE_OCCUPANCY_HEURISTICS_20261017_R1.md); it is a narrower child, not a fresh family. The small geometry pilots were heterogeneous, and longer-budget broad feature bundles lost. Those negatives lower the priority of another large concatenation. They do not distinguish atom identity from scaling, occupancy, dwell and altered optimization, which were often changed together.

**Decision:** retain a small pointwise encoding contrast, using fitting-only vocabularies and raw bypass. Defer transition graphs, topology and large moment banks until they beat that simpler explanation. Do not call equality an economically meaningful category or numeric distance a physical price distance.

### 4. Anonymity does not imply symmetry, and a bad negative control can manufacture a mechanism story

The [documentation](../../../../docs/data_overview.md) says component indices are stable identifiers rather than depth order. This rejects “neighboring index means neighboring level.” It does **not** imply that swapping component identities leaves the target distribution unchanged. The target concerns i0, so swapping i0 and i1 is also not a justified symmetry. Price-like and volume-like same-index pairing across streams needs its own evidence.

The [pair-set proposal](../../ideas/FRESH_REFRAME_20261023_R1.md) sensibly retains a raw bypass and proposes a deranged-pair comparison. That comparison is meaningful only insofar as a restricted shared interaction encoder makes pairing consequential. A fixed permutation of a complete age vector before an unrestricted dense layer is merely a permutation of the layer's learnable columns. It preserves the function class and cannot test whether “correct ages” matter. Likewise, globally relabeling all dictionary categories is a naming change, not a semantics-destroying control. These are algebraic observations, not measured outcomes.

The current age F−G comparison is useful operationally, but repeating seven group ages across 112 slots changes rank and optimization geometry relative to 112 field ages, as its [selection note](../../../experiment_drafts/b300-fast-local-kaggle-20261023/SELECTION_R1.md) acknowledges. Equal width does not make it a pure clock-resolution experiment.

**Decision:** retain identity-preserving shared encoders as regularization hypotheses. Defer hard permutation invariance, cross-side pooling and arbitrary channel sorting. Require every proposed null control to state exactly what information or inductive bias it removes.

## Seven questions whose answers would change allocation

1. **Where does i1's value survive?** Does a current-row nonlinear i1 representation improve the matched frozen endpoint without extra temporal features? A positive answer raises encoding/interactions above longer memory.
2. **Is field age useful conditional on row position and aggregate activity?** If not, a small nuisance-state summary may replace 112 clocks and their serving cost.
3. **Are long ages mostly recurrent atoms or ordinary quantized excursions?** Different behavior at known atoms versus other levels would favor a mixed categorical/numerical representation over generic decay.
4. **How much official weight lies before the first observed change of a field?** The fitting fixture's 0.368% unweighted field-row figure is insufficient. A negligible weighted footprint lowers censor-aware modeling priority; concentration in early failures raises it.
5. **Does same-index i0/i1 pairing beat a fixed within-family derangement inside a restricted shared encoder?** Failure would weaken the current 52-pair prior while preserving i1's overall value.
6. **Does a row-lag effect remain stable when sequences have very different event rates?** If not, a change-count clock may be more transferable; if yes, imposing event-time invariance may erase useful duration information.
7. **Can an apparent feature gain survive an initial-function-preserving comparison and the complete endpoint?** If only altered initialization or a weak cold model benefits, prioritize ordinary ensemble diversity or capacity rather than attributing the gain to hidden structure.

## Hypothesis cards

Shared future protocol: unchanged official `global_wp_clip2_v1`, strict FP32, full required-row predictions, all state reset at sequence boundaries, and no target/mask-dependent features. Use source-qualified F0/1001 and F1/1002 with their respective clean frozen parents; compare within cell because ancestry differs. Existing parent predictions may be reused only after identity and coverage checks. Their fit-side predictions are in-sample, a known residual-learning limitation. No existing adapted head is assumed to be an exact control. All folds are exposed; component intervals condition on fitted models. Costs below are planning estimates, not measured admissions or additions to the stopped allocation.

### H-ASTRA23-02-A — Clocks beyond position and activity

- **Family/change:** observation-process representation; compare three residual heads: N uses standard211 plus row position and seven fixed group change-rate summaries; F adds per-field ages; FC adds seen-change flags to F. Reserve identical slots, zero disabled channels, preserve the parent with zero output initialization. Freeze one rate half-life and clock cap before outcomes.
- **Assumptions/mechanism:** per-field dwell contains useful state beyond generic activity; distinguishing censored duration helps where it matters. This is an explicit child of [per-field age](../../ideas/PER_FIELD_AGE_20261022_R1.md), not a GRU-D missingness claim.
- **Predicted observable:** F−N is positive in complete endpoints if clock resolution matters; FC−F concentrates in pre-first-change rows if censoring matters. A benefit only from N identifies a cheaper explanation.
- **Cheapest test/controls:** first the zero-fit diagnostic below; subsequently six heads, two per arm, zero GRU/teacher fits, two reused parents. Match head capacity, order, objective and checkpoint; acknowledge activation/rank differences rather than claiming perfect mechanism isolation.
- **Falsifier:** no incremental endpoint contribution from F/FC on both cells, or only a training-fit residual gain. A small censored mass specifically deprioritizes FC.
- **Future cost/reuse:** six small heads plus streaming features, full exports and grade. Treat the historical 7,200-second allowance as a ceiling to re-profile, not a credible runtime estimate. Reuse verified parent caches and age implementation; no new funding assumed.
- **Ancestry/serving/status:** shared protocol; 112 ages plus seen flags and seven rate states; native callback and runtime unverified. **not_tested**.

### H-ASTRA23-02-B — Change-count history with a duration escape path

- **Family/change:** temporal representation; retain raw inputs and add two EMA innovation summaries updated on a predeclared sparse price-family change event, retaining row-age and observed event count. Compare against the same two summaries updated every row.
- **Assumptions/mechanism:** row density partly reflects the observation process; approximately equal numbers of relevant changes may align history better across sequences. Time-warp tolerance is only a hypothesis. The duration bypass protects information discarded by event-time compression.
- **Predicted observable:** event-clock benefit is larger in sequences where row/event-rate ratios differ; a duration bypass prevents losses in long quiet intervals.
- **Cheapest test/controls:** input-only stability summaries on a bounded fitting panel after resume, then four fits, two per clock, zero teachers. Use identical zero-initialized extension, rates declared in their respective units, exposure and fixed checkpoint. Do not skip state updates or outputs.
- **Falsifier:** selected events fire nearly every row, summaries collapse to the row-clock basis, or the matched complete endpoint loses. Stable event association without target value is insufficient.
- **Future cost/reuse:** four head or continuation fits, never silently interchangeable; prefer heads for the screen. Feature state is linear in selected channels; full delivery cost is unmeasured. Reuse canonical event/EMA components. Related [H-RR17-03/08/09](../../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md).
- **Ancestry/serving/status:** shared protocol; deterministic counters/EMAs, no physical timestamps or target maturity. **not_tested**.

### H-ASTRA23-02-C — Separate frequent atoms from continuous excursions

- **Family/change:** pointwise encoding; retain raw channels and compare an atom-membership plus bounded continuous-residual encoding against an equal-width piecewise-linear numeric encoding on a small predeclared channel set. Exclude occupancy and temporal state from this first contrast.
- **Assumptions/mechanism:** some repeated values deserve discrete treatment, beyond nonlinear numerical resolution. Fit all atoms, bins and scales on fitting components only; include unknown-value behavior. No assertion that atoms identify instruments or regimes.
- **Predicted observable:** the atom encoding adds endpoint value beyond the numeric control, especially at atoms and nearby excursions; otherwise numerical curvature is the simpler explanation.
- **Cheapest test/controls:** audit existing support summaries first; a predictive contrast uses four heads, two per arm, zero teacher/GRU fits and two reused parents. Freeze channels before labels and preserve identical parent outputs initially. A raw parent is a secondary reference, not a stage-matched adapted control.
- **Falsifier:** atoms fail to recur outside their fitting components, unknown categories dominate, or the numeric control matches/beats the atom endpoint. This does not reject the broader dictionary family.
- **Future cost/reuse:** fitting-only vocabulary construction plus four small heads and grade; no extra recurrent inference. Reuse [moment/dictionary manifests](../../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) only where memberships are compatible. Runtime unmeasured; cap vocabulary at three atoms per selected channel.
- **Ancestry/serving/status:** shared protocol; small fixed lookup and residual transform with raw bypass; no randomization of ties during serving. **not_tested**.

### H-ASTRA23-02-D — Test correspondence before enforcing shared structure

- **Family/change:** relational inductive bias; small shared two-input encoder with pooling and raw bypass, comparing same-index i0/i1 pairs against one fixed within-side/feature-family derangement. A third arm receives the same marginal encoders without pair interaction.
- **Assumptions/mechanism:** repeated interactions transfer across documented feature families, and proposed pairing provides information beyond marginal summaries. This sharpens [H-395](../../ideas/FRESH_REFRAME_20261023_R1.md); it does not assume exchangeability of the complete predictor.
- **Predicted observable:** same-index interactions beat both deranged interactions and marginal sharing. If both paired arms beat marginal sharing equally, generic interaction capacity is supported, not correspondence.
- **Cheapest test/controls:** six heads, two per arm, no new teachers/GRUs; fixed shared map dimensions, raw bypass, zero residual output and checkpoint. A source/algebra audit must first confirm that the derangement changes the restricted interaction family rather than merely relabeling unrestricted coordinates.
- **Falsifier:** the same-index arm has no endpoint advantage, or the raw bypass makes all three effectively indistinguishable. Do not infer lack of i1 value.
- **Future cost/reuse:** six small heads and repeated pair-map evaluation over 20k rows; measured full-path admission required. Reuse cached raw features, parents and group-map manifests. More engineering than card C; lower priority until clock/encoding alternatives are resolved.
- **Ancestry/serving/status:** shared protocol; bounded shared map, no cross-sequence adaptation, unknown native overhead. **not_tested**.

## Primary literature and transfer limits

- [Che et al., GRU-D](https://arxiv.org/abs/1606.01865): inspected primary abstract; the PMC full-text route returned a browser challenge. It combines masks and time intervals for missing-data prediction. Transfer only the separation of value from observation pattern. All supplied channels here are observed; unchanged values do not establish missingness or justify imputation.
- [Tallec and Ollivier, time transformations](https://arxiv.org/abs/1804.11188): inspected primary abstract; HTML did not expose the body. It links learnable recurrent gates with quasi-invariance under time transformations and motivates chrono initialization. This supports asking which clock is useful, not imposing invariance when waiting itself predicts the target.
- [Gorishniy et al., numerical embeddings](https://arxiv.org/html/2203.05556v4): inspected Sections3.1–3.2 and the abstract. Piecewise-linear encoding changes numerical representation; their feature embeddings retain feature-specific parameters. Transfer the numeric-control construction. Their tabular results neither validate atom semantics nor establish a sequential forecasting gain.
- [Zaheer et al., Deep Sets](https://papers.nips.cc/paper/6931-deep-sets.pdf): inspected Section2, including the countable-domain theorem and stated fixed-size qualification for the uncountable case. Sum-pooling is appropriate for functions with the specified symmetry. The theorem does not demonstrate symmetry in these columns.

No new random library draw was performed. The existing seeded source passages were provenance context, not independent evidence for these cards. No literature model, dataset or code was imported.

## Counterargument and the best proposed test

The strongest objection is that the existing GRUs already encode much of this history, while every extra representation changes optimization and adds delivery cost. A small frozen residual head may miss an interaction that requires changing the recurrent state; conversely, in-sample parent residuals may make an apparent head effect unrepresentative. The 0.368% boundary figure and negative broader geometry runs make a large semantic reconstruction project especially hard to justify. We need not recover hidden physical meaning to win.

**Best next test, proposed and unexecuted:** a bounded, zero-fit clock-versus-state diagnostic before reopening a head panel. Freeze a source-qualified sample of eight complete dependency components from each of F0 and F1 using membership alone, and use existing clean-parent predictions. On available small raw caches, stream only predeclared fields and maintain row position, existing group activity, exact age, seen-change status and fitting-defined atom membership. Report official magnitude-weight mass and parent residual moments in fixed cross-tabs of position/activity/atom status versus age; report censored cells separately. Keep each target and cell separate. Target changes may be diagnostic outcomes but never inputs or a maturity clock. Do not search fields, bins, lag thresholds or coefficients from these results.

This is diagnostic prioritization, not a calibrated predictor or evidence of a score gain. It can reveal that age associations are largely position, activity or atom occupancy, or that a material unexplained residual pattern survives. In the latter case, card A earns a complete matched screen. A negative tiny-panel result lowers priority without closing history mechanisms.

Propose a **300 CPU-second hard bound, ≤1GiB RAM, no GPU and zero new fits**, subject to explicit resume and a real source/cache feasibility check under the operational CPU cap. This is an admission ceiling, not measured cost. If the needed bounded cache is absent, do not open large Parquet metadata or build caches under this diagnostic label. A controller owns any later execution; an independent grader owns the resulting evidence. Until then, the completed deliverable is this report alone.

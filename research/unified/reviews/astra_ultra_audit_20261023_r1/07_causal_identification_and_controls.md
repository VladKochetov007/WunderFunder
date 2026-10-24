---
kind: review
status: complete
created_utc: 2026-10-23
updated_utc: 2026-10-23
hypothesis_ids: [H-ASTRA23-07-A, H-ASTRA23-07-B, H-ASTRA23-07-C, H-ASTRA23-07-D]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json
  - research/experiment_drafts/online-relationships-20261022/recipe.json
  - research/experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json
  - research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json
  - research/experiment_drafts/dynamics-aux-20261021/recipe.json
  - research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_PROFILE_RESULT_R2.json
---

# What the current contrasts identify, and the controls worth buying

## Scope and main judgment

The strongest opportunity is to turn broad mechanism labels into narrower, useful comparisons. History may provide a complementary error direction without exposing a GRU memory deficit. Online estimation may help by becoming adequately initialized without tracking nonstationarity. Field age may summarize a distribution of inactivity without requiring field identity. Future-input supervision may improve retention or regularization without learning an anticipatory world model. Each alternative suggests a potentially cheaper model, rather than merely weakening a claim.

This is a tier-C source review under the [audit contract](CONTRACT.md). I read source, recipes, existing text receipts and primary-paper abstracts; I did not load arrays or checkpoints, run imports/tests/profiles, fit models, or query evaluation outcomes. The October 23 stop remains binding. Every experiment below is **proposed and unexecuted**. Existing exposed folds remain exposed, and component intervals condition on the fitted models. Root owns indexing and ledger registration.

## Five findings that change interpretation

### 1. The i1 result identifies a useful treatment package, not missing memory

The [terminal grade](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json) records six cold H512×1 fits, two fold/seed cells, fixed CP2 and 10% fitting-component subsets. History C loses to standard-input B by −0.00353711 and −0.00390342, but fixed 90%B/10%C improves by +0.00035953 and +0.00027145. Removing i1 also produces a useful correction on F2: 90%B/10%A gains +0.00054446 there, versus −0.00002308 on F0. These are measured conditional results, not independent public-transfer evidence.

The [geometry receipt](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json) weakens clipping-only and scalar-shrinkage explanations on F0. It does not identify where the complementary direction came from. Identical initial tensor hashes coexist with different initial functions because C activates random projection columns masked in B; the [existing source review](../I1_HISTORY_FIXED_MIX_GEOMETRY_REVIEW_20261022_R1.md) correctly recognizes this.

Retain the function-preserving warm-history proposal. Its zero appended projection columns remove an immediate perturbation to the inherited predictor. A remaining alternative is that any accessible nonlinear input expansion would improve optimization or diversify errors. A current-input-only feature expansion is therefore a more informative next control than another all-zero suffix, **if** the warm pair first justifies further spending. A matched independent standard-input member is also the relevant eventual ensemble-cost comparator. Neither extra control is required before reporting the exact existing fixed-mixture result.

### 2. The online warmup control is mostly an unspent prior

The [online recipe](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/online-relationships-20261022/recipe.json) and [actual kernel](../../../../src/wnn/online_relationship_numba.py) initialize means/covariance to zero and input variance to one. B freezes all relationship statistics after row 98; updates begin at row 16. It therefore receives 83 updates, only 82/67 with observed backward regressors for lag 1/16. The decay is `lambda = 2^(-1/512)`.

By direct algebra, the inherited unit-variance contribution at freezing is `lambda^83 ≈ 0.894`; mean estimation has accumulated only about 10.6% total observation weight. This is not a measured effective sample size: overlapping changes, heterogeneous variance and placeholder regressors matter. Nevertheless, B is structurally a heavily prior-dominated estimate. C can beat it under stationary relationships simply by acquiring more information and removing initialization bias. B versus C also changes ongoing intercept estimation, not only slopes.

The scientific question worth preserving is whether **recent** relationships improve target corrections over equally mature **older** relationships. That requires matching estimator age and weighting mass. A later-freeze arm alone narrows startup bias but still changes sample count. This distinction reveals a neighboring mechanism: reliability-aware shrinkage toward fitting-population relationships, with adaptation strength driven by observation mass or variance support. Such a mechanism could be useful even if nonstationarity is weak.

The [terminal receipt](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json) contains seven saved heads and no quality grade. None of these alternatives has been empirically selected. The [mean-only lag identity receipt](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/online-relationships-20261022/MEAN_ONLY_LAG_IDENTITY_R1.json) is also useful: duplicated A fits were not independent evidence, and future designs can reuse an exact control after verification.

### 3. Zero output preserves predictions, not learning geometry

The [age head](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/age_readout_model.py) has random hidden weights and a zero output layer. At initialization it returns the parent exactly. For `p + 0.1 V tanh(Wx+b)`, however, the derivative with respect to V already depends on the feature treatment. The first output-layer update sees different hidden activations, while hidden-layer data gradients initially vanish because V is zero. Thus initial-function parity is valuable, but it does not establish matched initial learning dynamics.

Likewise, [age A/G/F](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json) match 323 input slots while appending respectively zero, seven repeated group clocks, or 112 field clocks. Their effective ranks and column correlations differ. For a group of size m, repeated group age contributes through the sum of m hidden weights: equal nominal width is not equal regularization or update geometry.

There is a sharper, source-derived observation: under the recipe's common reset, cap and exact-change semantics, a group's age equals the **minimum field age within that group**. F versus G therefore contrasts the entire age vector with repeated minima, not simply finer clocks versus equally rich coarse clocks. An age-distribution control retains information that G discards while removing field identity. This can reveal a compact set-summary mechanism if full field identity proves unnecessary. Unchanged values still do not establish missingness or elapsed physical time.

The [geometry history](../../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) reinforces this caution. R1/R2 changed input width and initial draws; within-stage zero controls were matched, across-stage controls were not. Longer R3 training reversed some short-budget advantages, while dictionary mixtures retained conditional value. Keep the representation family; deprioritize explanations that assign these movements uniquely to feature semantics or model capacity.

### 4. Dynamics weight zero isolates the package, not anticipation

The [dynamics recipe](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/dynamics-aux-20261021/recipe.json) is a sensible first comparison: both arms allocate the same auxiliary head, share initialization and target-statistics work, and differ in Huber-loss weight 0 versus .05. A positive result would establish utility of that added training objective in this recipe. It would not distinguish future-dynamics information from extra representation supervision, altered gradient direction, or changed clipping behavior.

A past-window dynamics target is a stronger next alternative than random labels. It asks the same state to retain comparable statistics already available causally. If past and future auxiliaries help similarly, a memory-shaping explanation becomes competitive with anticipation; if future wins, horizon-specific forecasting earns attention. Random-label auxiliaries are poor universal placebos because they change predictability and gradient noise. Matching target variance does not match gradient magnitude.

There is no quality evidence yet: the [profile receipt](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_PROFILE_RESULT_R2.json) deferred the four-fit panel at a conservative 12,118.9-second projection against 10,800 seconds. This is execution evidence only.

### 5. Shared ancestry can protect a comparison while limiting its scope

The [warm-parent catalog](../STRONG_PARENT_F0_F1_WARM_INIT_CATALOG_20261022_R1.md) describes different F0/F1 parents, teacher mixtures, child seeds and historical precision. Matching treatment/control **within** a cell protects the incremental contrast; averaging those deltas does not isolate fold variation or replicate one common training lineage. An alpha-zero continuation from a distilled initializer also retains historical teacher influence: it tests removal of current distillation, not absence of teacher knowledge.

The online and age residual heads learn against **in-sample parent predictions**. This is not evaluation leakage when the whole ancestry excludes the evaluated components, but it can suppress exactly the residual errors the head must correct at deployment. Before abandoning a cheap readout family after a weak result, inspect that mismatch. A future calibration-head experiment on a reserved subset of parent-held-out components, evaluated on disjoint remaining components, needs zero parent fits; it sacrifices evaluation coverage and remains exposed exploration. It would test the alternative training route, not cleanly isolate in-sample bias because calibration membership also changes. A fully matched cross-fitting study would cost additional parent fits and should not be hidden in a supposedly cheap head screen.

## Use shams that answer a question

A fixed permutation of appended features is invertible by a dense first layer; it does not remove their information. Permuting those features and compensating the corresponding weight columns is an exact functional symmetry in real arithmetic. This is a valuable implementation control, not a null-information treatment. Similarly, changing hidden-unit order with corresponding adjacent weights should preserve the model. Floating-point tolerance and optimizer-state transformations matter when continuing training.

Conversely, row shuffling or circular shifts can destroy autocorrelation, alter support, or introduce future inputs. A worse shuffled model need not prove temporal causation. Specify the property retained and the property removed. Within-group age sorting below retains every row's multiset, norm and group minimum while removing field assignment; it does not retain each field's temporal trajectory. Past-only delays preserve causality but alter freshness. These limitations are tolerable when they match the decision. No single sham separates every alternative explanation.

## Four hypothesis cards

### H-ASTRA23-07-A — History versus an equally accessible current-input expansion

**Family/change:** temporal representation versus optimization/capacity. After a useful warm-history result, compare zero suffix, the existing 156 history coordinates, and 156 deterministic current-i1 coordinates, for example three fixed transforms per standardized current field. Freeze transforms and fitting-only scales before grading. Use the same strong parent, suffix width, zero appended projection weights and continuation schedule.

**Assumptions/mechanism:** history may improve accessible temporal information beyond what an equally sized nonlinear current representation supplies. The current-only branch introduces no observations unavailable to the base input, but can still change approximation and optimization.

**Predicted observable:** history beats both the zero and current-only controls in fixed endpoint contribution across the two cells. Similar current-only benefit redirects effort toward a better input encoder or cheaper diversity.

**Cheapest test/matched controls:** six continuations, two per arm; first reuse the already planned warm pair if it is completed under exactly compatible settings. No teacher fits. Check initial prediction parity and shared sampler/statistics before later fitting. Do not reuse a superficially similar control with different normalization or order.

**Falsifier/counterargument:** history fails to exceed the current-only expansion. This weakens a history-specific claim, not all temporal representations; an arbitrary nonlinear basis may be an unfairly weak or strong competitor. Additional independent-seed baseline ensembles remain a separate cost comparison.

**Cost/artifacts/ancestry/serving:** at most six new GRU fits, up to four if exact completed controls qualify; no reliable current wall estimate. Reuse frozen parents, folds and teacher caches, retaining their differing ancestry and exposed status. History still needs native state support; current transforms are cheaper to implement but have unmeasured runtime. **Status: not_tested.** Related: [history progression](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/unified/ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md), old [H-RR17-05/-06](../../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md).

### H-ASTRA23-07-B — Recent information versus equally mature old information

**Family/change:** online system identification. Construct two causal finite-window relationship estimators using the same N matured pairs, decay weights, prior, standardization and lag. One retains a predeclared early window; one rolls over the most recent N pairs. Before N pairs exist, both use identical behavior. Choose one lag in advance for this diagnostic; do not silently replace the historical two-lag panel.

**Assumptions/mechanism:** recency, rather than removal of startup bias, supplies useful target information. N=2,048 is an illustrative future recipe choice, not a tuned optimum. A rolling finite window can use maintained sufficient sums and a ring buffer; recomputing N updates at every row would be unnecessarily expensive.

**Predicted observable:** recent-window corrections beat old-window corrections after common burn-in, including the official full-sequence endpoint; matured input-forecast loss is secondary. If only both beat the 83-update state, estimator maturity is the simpler explanation.

**Cheapest test/controls:** first a future stationary synthetic probe with known fixed slope, then four head fits: recent/old × two clean-parent cells. Keep common raw inputs, matured outcomes, head and fitting objective. Existing A endpoints can be contextual references only if ancestry, precision and coverage match exactly.

**Falsifier/counterargument:** recent and old windows tie or reverse on target contribution despite better recent input prediction. Stationary streams can produce finite-sample differences; even a repeated gain supports usefulness of recency, not a unique economic regime explanation.

**Cost/artifacts/ancestry/serving:** zero GRU/teacher fits; four heads, feature generation and exports. Reuse complete parent caches; wall cost requires a new complete-path estimate because the old twelve-head job timed out. A roughly megabyte-scale ring buffer for 52 paired coordinates at N=2,048 is plausible; exact native cost is unmeasured. Preserve in-sample residual and exposed-fold limitations. **Status: not_tested.** Related: [online relationship proposal](../../ideas/ONLINE_RELATIONSHIPS_20261022_R1.md), [H-RR17-05/-24](../../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md).

### H-ASTRA23-07-C — Age identity versus age distribution

**Family/change:** asynchronous representation. Compare field ages with the same ages sorted within each of the seven documented groups; retain the raw211 input path. A third zero-age arm anchors endpoint value. Sorting is current-row causal and retains the 112-coordinate multiset, its norm, every group's minimum and all order statistics.

**Assumptions/mechanism:** the useful variable may be how many fields are stale, rather than which fields are stale. This is a concrete neighboring mechanism, not an assertion that fields are exchangeable.

**Predicted observable:** sorted-age and field-age heads both improve over zero, with little consistent field advantage. That outcome motivates a few group quantiles/threshold counts instead of 112 identified clocks. A repeatable field advantage supports identity-dependent interactions.

**Cheapest test/controls:** six small heads, two per arm, same zero-output initialization and fitting objective. This is a prospective alternative design, not a rewrite of frozen A/G/F. If the historical six-head screen later completes, reuse an exact F/A pair only after establishing identical execution semantics; add two sorted-age heads.

**Falsifier/counterargument:** sorted ages lose while identified ages help, or neither helps. Sorting changes temporal coordinate continuity and learning difficulty, so the contrast cannot prove that all gains derive from identity. A fixed feature permutation would be a weaker sham because the dense head can invert it.

**Cost/artifacts/ancestry/serving:** six new heads initially, or two with four exact reused controls; zero parent/teacher fits. Reuse verified caches. The frozen historical 7,200-second allowance is not evidence this new design fits it. Sorting has per-row cost; compact quantiles would require their own follow-up and native measurement. Parent ancestry and exposed folds remain unchanged. **Status: not_tested.** Related: [per-field age](../../ideas/PER_FIELD_AGE_20261022_R1.md), old [H-RR17-03/-22](../../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md).

### H-ASTRA23-07-D — Anticipation versus retention supervision

**Family/change:** auxiliary representation learning. Compare no auxiliary, frozen future-input dynamics targets, and equivalent past-window mean-change/log-energy targets, using the same 52-output head and horizons 16/256. Match valid target-origin support by excluding origins unavailable to either construction; preserve full official evaluation coverage.

**Assumptions/mechanism:** future targets may shape an anticipatory state beyond what reconstructing recent dynamics supplies. Alternatively, both may teach useful memory or regularize the main task.

**Predicted observable:** future beats past and zero in target/complete-endpoint quality. Similar past/future gains favor retention supervision and motivate cheaper past-state compression. Auxiliary-loss improvement alone is insufficient.

**Cheapest test/controls:** six cold fits, two per arm; zero teachers. Fitting-only standardization, same output width and same update budget. Log main/auxiliary gradient contributions and clipping frequency as explanations, not as promotion criteria. The unchanged old zero control is not automatically reusable if the common validity support or objective implementation changes.

**Falsifier/counterargument:** future fails against past or zero. Past is usually easier to reconstruct, so this does not equalize task difficulty or rule out longer-horizon forecasting. A random-target sham would add a different confound rather than solve that one.

**Cost/artifacts/ancestry/serving:** reuse frozen membership and target-generation components, with six genuinely new fits. The old four-fit 3.37-hour conservative projection implies a larger panel needs more than the old three-hour envelope under comparable conditions; no new allocation is assumed. All folds remain exposed. Strip training-only heads; actual native compatibility still requires work. **Status: not_tested.** Related: [dynamics packet](../../../experiment_drafts/dynamics-aux-20261021/README.md), old [H-RR17-08/-18/-25](../../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md).

## Seven questions whose answers change a decision

1. Does the repeated i1 mixture benefit exceed the contribution of a comparably costly second standard-input member? If not, diversity may be the cheaper route.
2. Does warm history improve after initial-function and normalization matching? If not, defer this warm recipe while preserving the cold complementary endpoint evidence.
3. What fraction of any online gain survives equal estimator age and weighting mass? This determines adaptation versus initialization work.
4. Does age identity add value beyond its within-group distribution? This determines whether 112 clocks need individual model connections.
5. Do residual corrections fail mainly where in-sample parent residuals differ from parent-held-out residuals? This decides whether head-training membership deserves a bounded change.
6. Does a future auxiliary beat an analogous past auxiliary, and does either improve the actual endpoint? This distinguishes anticipation investment from representation supervision.
7. Which results survive within-cell comparisons when F0/F1 parents, teachers and fit seeds differ? This decides whether to generalize a recipe or retain a lineage-specific option.

## Recommendation, counterarguments and best next test

Retain history complementarity, age, online relationships and dynamics as distinct possibilities. Modify the **claims** now; amend experiments only when the extra answer changes a spending choice. Deprioritize interpreting feature counts as matched capacity, alpha-zero descendants as teacher-free, or online versus row-98 freeze as evidence of nonstationarity.

The strongest counterargument is practical: this is a competition, and a useful endpoint need not establish its mechanism. Adding controls can consume the resource budget and leave every panel unfinished. The cards are alternatives and conditional follow-ups, not a cumulative checklist. In particular, an ungraded partial online panel should not automatically trigger four new head fits, and the dynamics control expansion has low immediate priority under the observed cost limit.

**Best next discriminating test, proposed and unexecuted:** after explicit resume and disposition of existing saved work, run H-ASTRA23-07-B's maturity-matched recent/old comparison if online target utility warrants follow-up. Its initial stationary synthetic probe is a cheap falsifier of the existing mechanistic interpretation: the old C-versus-B advantage can occur without any changing relationship. The subsequent four-head screen asks whether recency adds target value beyond that artifact. It can redirect an expensive fast-memory program toward a much smaller reliability-aware estimator. No work follows automatically from this report.

## Primary literature inspected and transfer limits

- [Chen, Goodfellow and Shlens, Net2Net](https://arxiv.org/abs/1511.05641), abstract inspected: function-preserving transformations transfer a predictor when changing architecture. This motivates initialization control; it does not guarantee identical subsequent training.
- [Jacot, Gabriel and Hongler, Neural Tangent Kernel](https://arxiv.org/abs/1806.07572), primary abstract/search record inspected: parameter-gradient geometry governs function evolution. The finite-head derivative above follows directly from project source; no infinite-width prediction is claimed for these GRUs or AdamW fits.
- [Che et al., GRU-D](https://arxiv.org/abs/1606.01865), abstract inspected: missingness masks and observation intervals can be predictive in clinical time series. Our exact-value age is neither a verified missingness mask nor a physical observation interval.
- [Sun et al., Test-Time Training](https://arxiv.org/abs/1909.13231), abstract inspected: unlabeled test inputs can support self-supervised adaptation. Its image evidence does not identify stationary versus changing relationships here or establish target alignment for the input surrogate.

The repository's [25-idea portfolio](../../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md) supplies the old-family links above. Its historical CP30 protocol is preserved history, superseded prospectively by the current 4–6-total-fit policy. No literature result or reviewer agreement is counted as a new project measurement.

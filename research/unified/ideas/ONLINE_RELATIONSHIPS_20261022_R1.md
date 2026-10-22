---
kind: experiment
status: running
created_utc: 2026-10-22
updated_utc: 2026-10-22
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-ONLINE-RELATIONSHIPS-20261022-R1]
---

# Online input relationships with a task-trained residual readout

## Resumed implementation profile

After the user resumes, exact serial compiled features pass the [new cost profile](../../experiment_drafts/online-relationships-20261022/FEATURE_PROFILE_COMPILED_RESULT_R1.json): one pass projects 523.40 seconds and three-epoch recomputation 1,286.24 seconds, using eight complete fitting-valid sequences per fold. Its [execution receipt](../../experiment_drafts/online-relationships-20261022/FEATURE_PROFILE_COMPILED_EXECUTION_R1.json) records 9.53 wall seconds and zero swaps. These optimistic feature-only estimates reopen the full cost profile, with zero scientific fits admitted. The [independent source review](../reviews/ONLINE_RELATIONSHIP_NUMBA_FP32_REVIEW_20261022_R1.md) and [fitter test receipt](../reviews/ONLINE_RELATIONSHIP_READOUT_FIT_RECEIPT_20261022_R1.md) establish implementation checks, not target gains. Parent-cache creation and complete delivery cost remain unmeasured. The historical Python deferral below is preserved.

## Question and progression

Can continually learning input relationships provide useful information for target prediction beyond running means or initial calibration? Online learning is a hypothesis, not an established explanation for the performance plateau. The completed [history progression](I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md) found fixed-blend complementarity (+0.00036/+0.00027), while history alone lost. The strong-parent continuation remains unchanged and running. Saved checkpoint averaging added only about +0.000011 to its fixed endpoint. Earlier proxy routing/filtering were negative or weak and innovation features mixed; none tests this exact task-trained readout. The future-input auxiliary panel remains untested for quality after its cost gate failed.

The user accepted the online screen, then explicitly requested **both backward lags** when asked whether the i1 difference meant one or sixteen rows. This dated amendment expands the original six fits to **12 readout fits**, six per lag, while preserving the **2.5-hour inclusive local-worker ceiling**, the 12-hour production reserve, and the overall 30-hour allocation. No fitting is admitted until a sustained profile supports the complete amended panel. A missed cost gate defers it intact.

## Mechanism and frozen comparison

The [machine-readable recipe](../../experiment_drafts/online-relationships-20261022/recipe.json) defines A (mean only), B (update through zero-based row98), and C (continual), separately for backward lags1/16. Each fits F0/seed1001 and F1/seed1002 on the same deterministic 10% fitting-component subsets with complete sequences. The existing clean fold parents remain frozen; there are zero new GRU/teacher fits and two reused unchanged parent endpoints.

At row t, mature the pair from t−16, measure its error using pre-update coefficients, update permitted statistics, and forecast using the current backward i1 difference. The 52 index correspondences are a tested representation assumption, not known economic relationships. All state depends only on observed inputs, never targets, scoring masks or request flags. Initialize means/covariance at zero, input variance at one; use the specified EW covariance update, half-life512 and ridge0.1. Standardize input differences on fitting membership only, floor standard deviation at0.001, clip differences and added state to ±8.

Unavailable backward differences are zero placeholders. Updates begin at row16: B receives83 matured i0 outcomes through98, with82 genuinely observed backward regressors for lag1 and67 for lag16. Overlapping pairs and differing startup availability limit interpretation; B versus C does not establish nonstationarity.

Every arm retains419 slots: standard211, matured i0 changes52, forecast/error/slope156. A64-unit tanh head has a zero output layer and serves parent +0.1×residual. Fixed three epochs/CP2, AdamW0.001, weight decay0.01, clipping1, batches8192, canonical weighted WMSE, strict FP32. Fit-side parent predictions are **in-sample**, not OOF features; their residual distribution may differ on held-out components.

## Evaluation and decision

Primary contrasts are C−A, C−B and every corrected endpoint minus its unchanged fold parent, separately by lag/cell. Lag16−lag1 within each arm is secondary. Use complete source-qualified folds, unchanged global_wp_clip2_v1, both targets, four fixed position bins, and paired component uncertainty. All folds are exposed; two cells confound fold/seed, and intervals condition on fitted models. A better input forecast without a useful target/endpoint gain earns no larger run. Negative results narrow this readout implementation, not online learning generally.

Verify prefix causality, maturity timing, chunk equivalence, A–B–A reset, constant inputs, finite state, checkpoint restoration, future-input/evaluator-label invariance, zero-head parent parity, and exact prediction-cache coverage including NPY headers. Actual native endpoint quality/runtime are required before release; preserve PredictionModel.predict and leave online behavior off by default.

## Ownership, budget and sources

Root controls admission; online_actor owns causal features/model, online_delivery owns streaming delivery, and online_grader independently owns comparisons/review. All native roles use Luna/max with isolated worktrees. One GPU worker, two aggregate CPU cores,5GB aggregate task RAM/VRAM, no swap and physical scratch remain binding. The [existing dynamics profile](../../campaign_2026-10-21/next_cycle_r1/DYNAMICS_PROFILE_RESULT_R2.json) charged550.672seconds; reserving9000seconds leaves1249.328seconds of that10800second allocation. Earlier attempts remain separately preserved, and admission must reconcile all charges.

[Test-Time Training](https://arxiv.org/abs/1909.13231) motivates learning from unlabeled test inputs; its image results do not establish task alignment here. The [source note](../sources/ONLINE_RELATIONSHIP_SOURCES_20261022_R1.md) records primary literature and a reproducible random library passage. Separate [per-field age](PER_FIELD_AGE_20261022_R1.md) and [burstiness](BURSTINESS_20261022_R1.md) cards receive no fits. Percentile encoding stays within the existing quantile lineage.

Implementation, verification, profile, fit, native serving and grade are distinct stages. Follow the [packet](../../experiment_drafts/online-relationships-20261022/README.md) and [campaign frontier](../../campaign_2026-10-21/next_cycle_r1/FRONTIER.md) for actual state; this proposal asserts no measured online gain.

## Measured implementation cost and terminal disposition

The [R2 receipt](../../experiment_drafts/online-relationships-20261022/EXECUTION_RESULT_R2.json) defers the complete12-fit panel: fastest-sample one-pass feature-generation projection15,818.88seconds(4.394h) exceeds9,000seconds before all other delivery costs. Three-epoch recomputation projects38,860.94seconds. Actual cost probe92.16seconds,386,316KiBpeakRSS, zeroswap; earlier contained supportOOM9.183seconds preserved. Samples are8complete fitting-valid sequences per fold; trainreader overhead is unmeasured. This is an optimistic empirical estimate for this implementation, not a mathematical lower bound.

The causal state/head have23passing focused tests and an independent scalaroracle. No readout fits, scientific scores, completefit/export runner, artifactgrader, or nativeendpoint were produced. A faster implementation could earn a new profile; no new allocation is inferred. The scientific hypothesis remains **not_tested**, while this implementation is deferred for opportunity cost. The warm-history panel continues under its separate frozen admission.

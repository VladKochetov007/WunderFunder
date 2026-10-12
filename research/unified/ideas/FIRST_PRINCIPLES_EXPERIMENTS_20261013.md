---
kind: idea
status: proposed
created_utc: 2026-10-12T23:38:27Z
updated_utc: 2026-10-12T23:38:27Z
hypothesis_ids: [H-STRATEGY-COMPACT-20261013, H-STRATEGY-AVERAGING-20261013, H-STRATEGY-CROSSFIT-20261013, H-317, H-096]
experiment_ids: [E-083, E-154, E-164, E-181, E-264, E-230, E-302]
evidence_paths:
  - research/unified/reviews/FIRST_PRINCIPLES_REVIEW_PLAN_20261012.md
  - hpo/VALIDATION_PROTOCOL.md
  - research/campaign_2026-10-13/strategy_execution_r1/COVERAGE.json
---

# Experiments from the first-principles review

## Question or claim

Which additions to the active strategy would buy more reliable prediction per unit of training and serving cost?

The user requested merging and considering `origin/claude/stoic-euler-70wff3`. Main merge `20c904da` preserves its source commit `4737e515` and its static [review plan](../reviews/FIRST_PRINCIPLES_REVIEW_PLAN_20261012.md). That branch contains the plan/verdict, not the promised full review or ranked idea-card document. This assessment supplies concrete experiments without attributing new measurements to the imported review.

The original [strategy campaign](../../campaign_2026-10-13/strategy_execution_r1/README.md) remains active. The new review does not cancel the user's request to implement and deeply test its ideas, change the top-1 / public >0.700 objective, or authorize paid capacity. Current local GPU and Kaggle allowance authorization applies to useful new comparisons; each runnable recipe still needs clean ancestry, a measured resource bound and an execution owner.

## Evidence and interpretation

| Review claim | Source check and implication |
| --- | --- |
| A single layer costs less without sacrificing score | E-083 compared **H128**, with GRU128×1 at 0.6675 and GRU128×2 seeds at 0.6669/0.6672 on the old development recipe. These are recorded historical scores, not a matched modern H256/H390 or runtime experiment. Good reason to test compact models; insufficient evidence that current capacity is redundant. |
| Restore EMA because it gained 0.0037/0.0033 | E-154/E-164 concern mean-teacher consistency with EMA and a four-epoch versus two-epoch comparison. Duration, consistency and averaging are not isolated. Test EMA-only first at a common update clock; separate consistency and SWA arms. |
| Reject all effects below 0.0064 | The [power analysis](../../../hpo/VALIDATION_PROTOCOL.md) derives that number from a particular scratch-model bank and assumed variance decomposition. Its four-fold/two-seed table gives 0.0034; it is not a universal promotion threshold. Report paired effects, training variation, exposure and cost. Follow [ADR0009](../../../specs/adr/0009-uncertainty-is-not-a-kill-switch.md) when allocating uncertain improvements. |
| Four-fold validation resets evaluation | F0–F3 are all exposed. More folds and distinct pipeline seeds improve robustness evidence but do not restore an untouched holdout. F1 has 457 evaluation sequences; the old protocol's 468 is stale. Use actual split manifests. |
| Composite validation matched all public signs | The cited result is five selected historical comparisons, using a separate train holdout with a propensity-estimated mask. Treat it as a useful extra domain diagnostic, not a calibrated veto oracle or an official train-source score. Parents, teachers and propensity estimation must also exclude those held-out components. |
| Variance, rather than capacity, is the limiting factor | Seed averaging has useful evidence, but flat old H128 depth comparisons do not establish a capacity ceiling for different teachers, representations or students. Allocate a compact-model arm alongside the existing scale/compression ladder. |
| Encoders and class weighting are universally negative | E-181/E-264 constrain specific encoder/EMA/partner recipes; E-104 constrains a particular class-mass treatment. Preserve those negatives without ruling out materially different interactions. |
| A recovered target law is an oracle | E-079's selected subset does not establish a universal future-price law. E-230 records substantial failure mass, and E-302 rejects the proposed mean-next-N winner. Do not build augmentation or a target estimator on the unverified universal claim. |

These are source checks against append-only historical records, not recomputed model grades. The imported November 15 calendar and final dataset size have not been independently verified here; neither is used as an admission deadline. A hypothetical doubling of row count approximately doubles a fixed callback's cost, but does not establish the actual private workload.

## Mechanism and method

All new fits use supplied permitted data, own weights, source-qualified component splits and the unchanged official metric. First screens use F0/F3 and paired seed 3, then seed 4 when informative; complete finalists follow the existing four-fold/two-repeat plan. Report mean seed performance of fixed endpoints, per-fold paired deltas and component-resampling uncertainty separately. Final private selection retains the accepted maximum 0.001 loss in ordinary CV relative to the strongest supported endpoint.

### 1. Compact ensembles at measured serving cost — S36

**Hypothesis H-STRATEGY-COMPACT-20261013.** Lower-cost recurrent members may permit additional independent seeds or complementary families within the same CPU budget.

Start with H256×1 and H192×2 against H256×2. Match fitting membership, optimizer, target treatment and attempted update budget. Architecture changes cannot reuse an incompatible parent while claiming a pure capacity contrast: use own scratch parents under a common recipe, or declare independently trained architecture-specific parents as part of the full pipeline. Start with two seeds per selected architecture, not a large width sweep.

The cheap discriminator is parameter/operation accounting and actual recurrent callback timing on full reset sequences. It supplies a cost estimate, not a quality result. After fits, compare each fixed equal-weight ensemble and its fixed contribution to the reference endpoint, with actual Q8 quality and runtime. Numerical effects remain unknown. Defer an exact architecture if its measured serving savings cannot compensate for its standalone and conditional losses; a weak member can still be complementary.

**Interaction:** compact capacity × independent fitting diversity, parents S27/S29. Do not assume twice as many members is twice as much independent information, particularly with shared teachers.

### 2. Isolate weight averaging — S37

**Hypothesis H-STRATEGY-AVERAGING-20261013.** Averaging an existing training trajectory may reduce seed variation at unchanged inference cost.

Use a common six-epoch student schedule and fixed CP5 endpoint. Compare raw weights against EMA-only with decay 0.999 updated **after successful optimizer steps**, initializing from the same weights. Record attempted and completed updates, skip behavior and EMA clock. The actor must check existing trainer semantics before using any historical flag. No consistency loss in this first contrast.

A separate SWA candidate may average the predetermined epoch-3/4/5 weights. It needs all selected checkpoints from the same trajectory and explicit buffer semantics; it is not equivalent to averaging predictions. Existing eligible checkpoints can support a cheap fixed-prediction screen before a new fit. Mean-teacher consistency is a later separate arm against the matched-duration EMA-only control.

Measure paired official scores and candidate seed variation, plus fixed-blend contribution. Averaging that oversmooths useful late updates or hurts complementary predictions falsifies that recipe's practical value. No extra served network is required; score the actual averaged checkpoint after export.

### 3. Mirror symmetry as a falsifiable representation — S38 / H-317

First reconstruct the permitted feature schema and test candidate bid/ask permutation/sign maps on fitting data. Anonymous coordinates and an approximate target law do not establish a symmetry. Require an involution, valid transformed values, correct warmup/reset behavior and a justified target transformation before training augmentation.

If inputs represent reflection about a price axis and targets are distances `F−ask` and `bid−F`, the corresponding target map would swap the two outputs; a generic sign flip is not interchangeable. This is conditional algebra, not proof that the supplied anonymous encoding implements that map.

Use deterministic complete fitting sequences for the diagnostic. Compare original and transformed feature distributions and an existing model's equivariance residual. A large model residual alone neither disproves the data symmetry nor establishes that augmentation helps. If the mapping fails, preserve the counterexample and defer the claimed symmetry; do not train on implausible examples. If it passes, compare a paired augmentation arm with unchanged evaluation and ordinary inference. Two-pass test-time averaging is a separate cost-bearing arm.

### 4. Causal cross-fitted teachers before privileged variants — S39

**Hypothesis H-STRATEGY-CROSSFIT-20261013**, extending H-096 and S20's horizon-zero control. In-sample teacher errors may transfer differently from component-excluded predictions.

Inside each outer fitting set, split complete source-qualified components into halves A/B. Fit own matched causal teachers T_A and T_B with equal population and schedule. Cross-fitted targets use T_B on A and T_A on B. The matched in-sample control uses T_A on A and T_B on B. Thus both caches use the same two teachers, the same student rows, the same target amplitude, and the same teacher training-set sizes; a full-data teacher is a separate scaling comparison. Repeat partition/teacher seeds when this screen justifies it.

Teacher normalization, propensity models, initializers and all ancestors must respect their respective fitting memberships; a full-outer-fit warm parent would invalidate an inner out-of-sample claim. Scratch teachers are the straightforward valid starting point. Export caches once, fit paired identical students, and judge the students on the outer excluded groups. A better teacher score alone does not support the hypothesis. Preserve cross-fit machinery for the later horizon-2/8 privileged teacher experiment rather than duplicating it.

### 5. Reconcile mask concentration and domains — extend S05/S31

Add propensity power **0** to the existing power-1/current-power-2 ladder. Keep all other weight factors fixed and define power zero explicitly as disabling propensity weighting; do not let `0**0` or an accidental floor policy choose the semantics. Record effective row/episode weight concentration. A reduced-concentration representation interaction already exists as S10.

The train-holdout composite is a later full-ancestry experiment, not a cheap rescore of parents fitted on all train components. First use the existing frozen feature-domain panels. A proxy calibrated against the 28-recipe bank is an exposed retrospective diagnostic; any reusable selection rule needs a fixed subsequent recipe cohort and cannot claim pristine validation.

### 6. Runtime robustness — extend S29/S35

Prefer a fixed compact deterministic endpoint and measured callback headroom. The imported proposal to switch ensemble after N sequences makes predictions depend on previous sequences. The [supplied data rules](../../../docs/data_overview.md) require independent sequences, state reset and current-or-earlier rows of the same sequence. That proposed global counter is not admitted.

A within-sequence gate can instead use a fixed step schedule or allowed causal features and reset every sequence; it must process every row and handle any expensive member's hidden state correctly. Merely skipping a recurrent member and later restoring its stale state changes the endpoint. Evaluate its entire trajectory, actual package and runtime before adopting it. This is lower priority than the fixed compact ensemble.

## Decision and next step

Keep the current amplitude, innovation and gradient-credit cohort. Add S36–S39 to coverage; these are proposals with bounded preparation, not claims of four admitted fits. Preparation priority is compact cost accounting and the symmetry diagnostic, then checkpoint-availability audit for averaging and clean inner-component manifests for cross-fitting. Root owns this allocation until handing an exact recipe to an available delivery actor.

The review's concerns about setup failures and documentation replacing measurements are valid operational prompts. Reuse the existing scorer, split/provenance tools and training/export interfaces. Successful checkpoint delivery and matched grades are the milestones. No new controller framework, blanket scientific stop list, public-score regressor or extra external ideation round is introduced by this assessment.

## Source records

- [Imported static review](../reviews/FIRST_PRINCIPLES_REVIEW_PLAN_20261012.md) and [original strategy](../reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md).
- [Hypothesis ledger](../../hypotheses.jsonl): H-096, H-317 and the three named extension hypotheses above.
- [Experiment ledger](../../experiments.jsonl): E-083, E-154, E-164, E-181, E-264, E-104, E-079, E-230 and E-302; recorded historical evidence only.
- [Historical validation protocol](../../../hpo/VALIDATION_PROTOCOL.md), with the exposure and count corrections above.
- [Active coverage and status](../../campaign_2026-10-13/strategy_execution_r1/COVERAGE.json) and [accepted campaign plan](../../campaign_2026-10-13/strategy_execution_r1/PLAN.md).

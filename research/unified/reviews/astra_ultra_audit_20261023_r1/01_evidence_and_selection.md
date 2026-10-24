---
kind: review
status: complete
created_utc: 2026-10-23
updated_utc: 2026-10-23
hypothesis_ids:
  - H-ASTRA23-01-A
  - H-ASTRA23-01-B
  - H-ASTRA23-01-C
  - H-ASTRA23-01-D
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_PRECISION_PROBE_R2.json
  - research/campaign_2026-10-21/next_cycle_r1/TL3_STUDENT_EXPOSURE_RECOVERY_R1.json
  - research/campaign_2026-10-22/quota_retained_pair_r1/FIXED_SAMPLE_GRADE_R2.json
  - research/campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json
  - research/experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json
  - research/experiment_drafts/online-relationships-20261022/MEAN_ONLY_LAG_IDENTITY_R1.json
  - research/unified/reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md
---

# What the completed evidence establishes, and what we may be mistaking for progress

## Question and scope

The main missing comparison is often **whether a proposed mechanism beats an equally costly alternative way of improving the same endpoint**. A history model improving a one-model baseline through averaging is useful evidence, but it does not yet show that history beats another ordinary seed. A cold model losing to a production student does not isolate architecture. A complete implementation with no quality grade is still an unanswered scientific question.

This is an independent tier-C source review under the [review contract](CONTRACT.md). I inspected primary JSON receipts for the completed i1 panel, native replacement diagnostics, two-parent diagnostic/public comparison, and stopped online panel, plus selected metric source and interpretation notes. I did not load arrays, rerun graders, inspect weights, or execute experiments. Every test below is **proposed and unexecuted**. The [October 23 stop](../../../experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json) remains binding.

## Four findings that change the interpretation

### 1. Several apparently competing scores answer different questions

The [i1 terminal grade](../../../campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json) is a useful matched six-fit experiment: three arms, two fold/seed cells, fixed CP2 after three epochs, and fixed 90/10 mixtures. Its cold models use a 10% fitting-component screen. The [same-host native grade](../../../campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json) instead compares saved complete production/replacement endpoints on exposed F0. The [public receipt](../../../campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json) measures submitted artifacts on a separate benchmark, with filename/user-report rather than independently verified portal payload hashes.

| Completed comparison | Measured result | What it establishes |
| --- | --- | --- |
| i1 F0/1001: history C versus standard B | −0.00353711 standalone; fixed 90B/10C gains +0.00035953 | A small complementary direction exists in these fitted predictions despite worse standalone quality. |
| i1 F2/1002: same comparison | −0.00390342 standalone; fixed mixture gains +0.00027145 | The mixture sign repeats in a second exposed fold/seed cell. |
| Native F0: H512 CP5 replacement versus current-host TL3 | 0.69445462 versus 0.70181753; delta −0.00736292 | This fixed replacement endpoint is worse on these data. |
| Public: two retained parents versus original TL3 | 0.66771 versus 0.66832; delta −0.00061 | Removing the student in this submitted package did not improve the reported public score. |

The native gap is real, but the [exposure recovery](../../../campaign_2026-10-21/next_cycle_r1/TL3_STUDENT_EXPOSURE_RECOVERY_R1.json) links the original student's exact exported payload to a final train+valid distillation recipe. H512 excludes the 468 F0 evaluation groups. Teacher targets, initializer, objective, and other training details also differ. Exact final-student row-group execution and recursive ancestry remain incomplete. Thus “H512 cannot generalize as well as the mature student” and “depth is essential” are not identified by this comparison. Conversely, ancestry asymmetry cannot dismiss the separate negative public observation.

Two tempting excuses have already been weakened. Changing the native runtime moves TL3's full-fold score by only −0.00000059. On the eight-component precision probe, replacing only H512's int8 contribution with its FP32 counterpart gains +0.00004035 while leaving −0.00328280 versus TL3. That sample cannot establish the full-fold precision effect, but it makes precision a poor leading explanation for its measured gap. See the [primary precision receipt](../../../campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_PRECISION_PROBE_R2.json).

**Disposition:** retain H512 as an architecture with evidence in its own matched studies; deprioritize another unchanged TL3 member replacement. Require a common-ancestry endpoint comparison before explaining the replacement gap through capacity. Do not treat 0.7018 on production-exposed F0 as a local stepping stone to public 0.700.

### 2. History has a real exploratory blend signal, but its essential competitor is missing

The primary i1 receipt gives 468 groups/141 components on F0 and 443/144 on F2. Its 2,000 component resamples give mixture intervals [−0.00001187,+0.00070893] and [+0.00008223,+0.00043398]. They condition on fitted predictions. There is one fit per arm per cell; changing fold and seed together prevents estimating their separate contributions. The positive target/bin breakdowns are correlated views of those same fits.

The larger and cleaner result is that **existing i1 inputs matter under this recipe**: masking them costs −0.01590144 and −0.01110844. That is considerably better established than the proposition that additional temporal history is useful. It says neither that the useful information is economically understood nor that this feature dependence transfers unchanged to stronger models.

History's mixture advantage also needs the right rival. The same grade shows 90B/10A, where A removes i1, gaining +0.00054446 on F2—more than history's +0.00027145—while being approximately flat on F0. A weaker information set can therefore supply beneficial errors without supporting the proposed temporal explanation. There is no matched **90B/10B-prime** endpoint from another ordinary standard-input seed. Until that exists, paying for an extra history model has not beaten paying for ordinary model averaging.

This is not a claim that the effect is “just clipping.” The [primary F0 geometry receipt](../../../campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json) records positive weighted residual covariance of the clipped mixture correction after projecting the target on B: approximately 0.0007130 and 0.0003388 for the two targets. Clipping fractions are small; the [progression note](../../ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md) records that clipping slightly reduces the mixture improvement. Together with the [metric source](../../../../src/wnn/gmetric.py), this rules out pure scalar shrinkage or clipping alone as an explanation for that F0 point gain. It does not identify what produced the corrective direction.

The [independent six-fit review](../I1_SIX_FIT_HISTORY_GRADE_REVIEW_20261022_R1.md) identifies another limitation: equal initial parameter hashes do not imply equal initial functions when C activates random history projection columns that B masks. The planned zero-extension warm treatment addresses that problem. Its incomplete execution means the question remains open; it does not erase the cold result.

**Disposition:** retain the fixed-mixture observation, modify the next comparison to include an ordinary-seed competitor, and stop describing complementarity as evidence uniquely favoring memory. The missing comparison is practical as well as causal: extra native history state and a second model must earn their cost.

### 3. The public disagreement is evidence about selection reliability, not a universal failure of local validation

The [two-parent sample grade](../../../campaign_2026-10-22/quota_retained_pair_r1/FIXED_SAMPLE_GRADE_R2.json) shows +0.00257354 over TL3 on eight adaptively exposed components/10,988 scored rows. Both targets improve. The same receipt explicitly has no full-fold result, uncertainty estimate, or independent replication. The reported public comparison reverses sign, at −0.00061.

The correct update is narrower than “CV does not work”: this selected, ancestry-qualified eight-component diagnostic was not a reliable direction-of-transfer signal for this package. Nor does one reversal identify why. Possibilities include selection, population composition, fitted-data exposure, training/serving differences, and ordinary uncertainty. No public subgroup labels exist here to discriminate them.

This suggests asking **how concentrated a gain is**, before optimizing another scalar. Official WP pools weighted covariance and variance; it is not an average of sequence or position-bin correlations. A gain can depend on which components supply target mass and on between-group means. The source implementation preserves additive statistics and centered moments, so a future fixed-prediction influence analysis is possible without another fit. Its purpose would be to expose fragile rankings, not manufacture a fresh holdout or tune weights to public feedback.

The public pair remains a useful speed/quality observation: 1,711 seconds versus TL3's 2,352 seconds, a 641-second difference in these runs. It is not the best recorded public endpoint, and the timing difference is not guaranteed capacity for another model. Still, a faster parent pair plus a genuinely complementary small correction is a more concrete endpoint hypothesis than chasing a high standalone score whose contribution is unknown.

**Disposition:** retain local evidence as the selection basis, strengthen complete-endpoint and concentration checks, and keep public feedback as separate transfer evidence. No coefficient search against the public leaderboard follows from this diagnosis.

### 4. Some directions are contradicted; many others have never received a quality test

The [B300 review](../B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md) reports 12 matched width/depth pairs, with deep−wide −0.00077732 on equal-fold means and wide winning 9/12. Its partial geometry bundle is also negative; age and innovations have zero scored pairs. This is meaningful directional evidence for the completed recipes, despite an unfinished historical CP30. It is not evidence against all depth, geometry, age, innovations, or large Transformers. I could not inspect the remote primary B300 stop/grade packet in this checkout, so those figures remain source-qualified secondary reporting here.

Likewise, the [warm remainder receipt](../../../campaign_2026-10-21/next_cycle_r1/WARM_REMAINDER_COST_DEFERRAL_R1.json) preserves one completed control and defers three fits for cost. The [online terminal receipt](../../../experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json) preserves seven exported heads: all six F0 arms and one F1 control. One additional head was interrupted without a durable checkpoint; four never started. Neither supplies a complete paired quality result. Delivery failure is not negative science.

There is nevertheless recoverable information in complete within-cell contrasts. The current frozen contract deliberately closes grading until the entire panel exists. That must remain the historical record. A **new, explicitly scoped descriptive analysis** of all six completed F0 heads could answer a narrower question after authorization, without pretending the twelve-head panel completed or resuming fits. Completion occurred in serial order, not because F0 scores were favorable; quality was not graded. This is preferable to leaving all scientific value hostage to a further allocation, provided the narrower selection and exposure are explicit.

The [mean-only identity receipt](../../../experiment_drafts/online-relationships-20261022/MEAN_ONLY_LAG_IDENTITY_R1.json) also reports identical producer prediction digests for F0's two A controls. That is a potential future control reuse, not two independent replications; payload verification remains absent. Future panels should buy distinct information, not duplicate it.

## Seven decision-changing questions

1. Does fixed 90B/10C beat fixed 90B/10B-prime at the same additional-fit and serving cost? If not, allocate to ordinary ensembling or compression instead of history engineering.
2. Does history still contribute when its introduction preserves the initial function, and does that contribution survive a strong parent? The stopped warm panel does not answer this.
3. Which purported clean reference has complete source-qualified fitting, teacher, initializer, and propensity ancestry? If none matches TL3, distinguish the deployable incumbent from the research comparator instead of using one scalar for both.
4. Would the i1 mixture rank change after removing one connected component, or under modest predefined component reweighting? Concentration would lower the value of another small aggregate gain.
5. What do all six saved F0 online heads say relative to their fixed parent, mean-only control, and frozen-after-warmup control? A null result there could save completion costs; a useful contribution could justify a bounded second cell.
6. Are tiny gains from short cold training durable, or do they disappear as the control converges? The meaningful object is a treatment advantage along a matched learning budget, not the best selected checkpoint.
7. Can the strong i1 dependence be regularized into one standard-input model rather than purchasing a second temporal model? This changes both the modeling direction and the serving burden.

## Proposed hypothesis cards

### H-ASTRA23-01-A — History versus an ordinary second seed

- **Family/change:** conditional ensemble utility. Compare fixed 90% existing B-anchor + 10% new B-prime against the same anchor + 10% new C-prime in F0 and F2.
- **Assumptions/mechanism:** history is worth specializing only if it adds a correction beyond ordinary fitting variation. Use zero appended projection weights in both new arms to match their initial functions; this is an explicitly amended descendant, not a replay of old C.
- **Predicted observable:** the history-containing endpoint beats the ordinary-seed endpoint within both cells, with target contributions reported separately.
- **Cheapest test and controls:** four new fits, two per arm, with paired fresh seeds, identical membership, update/checkpoint rule and normalization. Reuse the two exact frozen B anchors; do not refit or select anchors and do not search blend coefficients. Zero teacher fits.
- **Falsifier:** history's endpoint is no better than the ordinary-seed endpoint in the tested cells, especially if both beat the single anchor similarly. This weakens history-specific spending, not all history representations.
- **Estimated future cost:** four small-screen fits plus exports and one fixed grade; order of the prior six-fit screen's cost, with fewer fits. Wall time under the latest safe resources is unmeasured and requires a new bounded estimate; no local feasibility is asserted.
- **Reused artifacts/ancestry/exposure:** exact B predictions, split manifests and canonical i1 inputs; all evaluation folds remain exposed, and two cells still confound fold and seed. No production TL3 ancestry is imported.
- **Serving:** an ordinary B-prime avoids additional history-state logic; compare complete runtime before choosing either two-model endpoint.
- **Related/status:** [history progression](../../ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md); **not_tested**.

### H-ASTRA23-01-B — Structured i1 masking as a cheaper source of robustness

- **Family/change:** representation regularization. Train the standard-input model with the existing A-style i1 channel mask applied with one fixed probability, proposed 0.1, once per fitting sequence; evaluate with all standard inputs present.
- **Assumptions/mechanism:** i1 is useful but may encourage brittle reliance on one signal family. The weak A model's F2 complementarity suggests a competing representation can sometimes correct B. A single model might learn some of that robustness through training augmentation.
- **Predicted observable:** masking improves the fixed standard-input endpoint, or its fixed contribution to the same anchor, without requiring history at inference.
- **Cheapest test and controls:** four new fits, ordinary B versus masked-training B across two source-qualified fold/seed cells; identical initialization, sample order, updates and CP2. Use a separate mask RNG so random masking does not alter sampler order. Zero teachers; no probability grid.
- **Falsifier:** no endpoint benefit in either cell, or improvement confined to a selected slice while overall quality declines. The existing large A−B losses are serious counterevidence: suppressing i1 may simply underfit the most useful features.
- **Estimated future cost/reuse:** four existing-size fits, exports and grade; negligible added parameter count but actual wall cost unmeasured. Reuse canonical A masking semantics, feature construction and fixed evaluator.
- **Ancestry/exposure/serving:** same clean fitting memberships, exposed evaluation, fresh paired fitting seeds; ordinary 211-input inference with no stochastic masking at serving. Native conversion/parity must still be verified before release.
- **Related/status:** [i1 six-fit review](../I1_SIX_FIT_HISTORY_GRADE_REVIEW_20261022_R1.md); a new descendant of the ablation result, **not_tested**.

### H-ASTRA23-01-C — Recover a bounded answer from completed online heads

- **Family/change:** causal adaptation and evidence recovery. Grade the complete F0 subpanel under a separately registered exploratory analysis; retain the original full-panel failure unchanged.
- **Assumptions/mechanism:** continually updated input relationships may add target-relevant information beyond running means or initial calibration. The six saved F0 heads contain the required within-cell contrasts, subject to actual payload validation.
- **Predicted observable:** C−A and C−B are positive, and C improves the unchanged parent, separately for lags 1 and 16. Better input forecasts alone do not count.
- **Cheapest test and controls:** zero new fits; verify every F0 prediction digest, coverage, source/checkpoint identity and parent, then publish all originally specified F0 contrasts with component intervals. Include both lag results; do not choose the better lag and label it the panel result.
- **Falsifier:** neither lag improves the parent or distinguishes C from the controls at a practically useful scale. An imprecise result remains inconclusive; it is not a universal rejection of online learning.
- **Estimated future cost/reuse:** one streamed validation/grade of saved caches and seven endpoints, with a proposed 600-second ceiling borrowed only as a planning reference from the old grading allowance. I/O feasibility is unmeasured; no allocation is transferred here. Zero GRU, teacher or head fits.
- **Ancestry/exposure/serving:** one exposed F0/1001 cell; fitting parent predictions are in-sample, which can make fitted residuals unrepresentative. No second-cell, native-runtime or production claim. Any positive result needs separate completion/confirmation authorization.
- **Related/status:** [online hypothesis and frozen contrasts](../../ideas/ONLINE_RELATIONSHIPS_20261022_R1.md); existing mechanism, narrower proposed analysis, **not_tested**.

### H-ASTRA23-01-D — Test whether the apparent gain is concentrated

- **Family/change:** transport robustness diagnostic. Freeze a component-influence and population-weight sensitivity report for existing fixed i1 endpoints before inspecting those outputs.
- **Assumptions/mechanism:** a small pooled gain may rely on a few components or between-component offsets, making it sensitive to population composition. The public pair reversal motivates this possibility but does not establish it.
- **Predicted observable:** unstable sign under leave-one-component-out evaluation or a small prespecified set of bounded component weights, despite a positive original aggregate delta.
- **Cheapest test and controls:** zero fits; use the existing B, A, C and fixed mixtures on full F0/F2. Recompute official pooled moments for each fixed perturbation, retaining original label weights, clipping and masks. Report within/between-component moment decomposition as explanation only. Never average per-component Pearson scores as a replacement metric.
- **Falsifier:** the history gain remains broadly distributed and positive under the frozen perturbations in both cells. That weakens this concentration account, without proving public transfer.
- **Estimated future cost/reuse:** one bounded streamed pass over saved predictions/targets plus small moment calculations; actual seconds unknown. A proposed one-hour ceiling is a cap for future admission, not an estimate or permission.
- **Ancestry/exposure/serving:** unchanged fixed models and exposed folds; perturbations are sensitivity analyses, not new independent test sets. No serving changes and no blend optimization.
- **Related/status:** [fixed-mixture geometry](../I1_HISTORY_FIXED_MIX_GEOMETRY_REVIEW_20261022_R1.md), [public receipt](../../../campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json); **not_tested**.

## Counterargument, best next decision, and limits

The strongest counterargument is opportunity cost: another comparator or sensitivity analysis can consume the budget that should produce a better model. Competition utility does not require proving a temporal mechanism. A small, repeated fixed blend gain can justify follow-up even when its interval crosses zero, and a sufficiently cheap useful correction should not wait for a complete causal theory. These recommendations should therefore change a concrete allocation, not become permanent prerequisites.

My best next test, **only after explicit authorization**, is H-ASTRA23-01-C: validate and grade the already-complete F0 online contrasts under a new, openly exploratory analysis. It buys an unanswered mechanism result without another fit, retains every control and lag, and could change whether completing that lineage is worthwhile. If new scientific fitting is later preferred, H-ASTRA23-01-A is the most important missing comparator for the history narrative.

Coverage limits are material. I inspected the cited JSON fields and source, not their numerical payloads or execution environments. Historical hash bindings are reported evidence, not hashes recomputed in this review. B300 results are secondhand from the linked review because its remote primary packet was absent locally. Public observations are user-reported with incomplete portal payload binding. No literature-based performance claim is needed for these proposals; their basis is the repository's own completed contrasts and explicit unresolved alternatives. All new cards are proposals, and none upgrades generalization, native serving, or public-score claims.

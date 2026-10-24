---
kind: review
status: complete
created_utc: 2026-10-23
updated_utc: 2026-10-23
hypothesis_ids:
  - H-ASTRA23-06-A
  - H-ASTRA23-06-B
  - H-ASTRA23-06-C
  - H-ASTRA23-06-D
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/TL3_STUDENT_EXPOSURE_RECOVERY_R1.json
  - research/campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json
  - research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_CP3_CP5_PAIRED_GRADE_REVIEW_R1.json
  - research/experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json
  - research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json
---

# Review 06: measure the contribution that survives into the endpoint

## Question and scope

Which components add useful information to the eventual deterministic predictor, and which apparent gains arise from comparison, inheritance, or fitting-distribution differences?

My recommendation is to keep ensemble complementarity and distillation open, but change the immediate question from “which model is better?” to “what correction survives a matched training and serving comparison?” The strongest missing control is an ordinary second model against the history model's mixture contribution. A separate, broadly useful risk is training residual heads against in-sample parent errors and deploying them against out-of-sample errors.

This is a tier-C independent source review under the [audit contract](CONTRACT.md). I read text, JSON receipts and metric source; I did not inspect arrays, execute models, run tests, alter frozen recipes or consult sibling reviews. The [October 23 stop][stop] remains binding. All tests below are proposed and unexecuted; none adds allocation or resumes work. Existing folds are exposed, and no public forecast is made.

## Five findings that change the research questions

### 1. The history result supports a complementary component, but has not priced ordinary seed diversity

The [terminal i1 grade][i1] is unusually informative because standalone and mixture rankings disagree under a frozen coefficient:

| Exposed cell | History C minus standard B | Fixed 90%B/10%C minus B | Conditional component 95% interval for mixture gain |
|---|---:|---:|---|
| F0 / seed1001 | −0.00353711 | +0.00035953 | [−0.00001187, +0.00070893] |
| F2 / seed1002 | −0.00390342 | +0.00027145 | [+0.00008223, +0.00043398] |

These six fits use a small cold-training recipe. The gain appears in both targets, but two fold/seed cells cannot separate fold effects from seed effects. The [F0 geometry receipt][geometry] shows that the gain persists before clipping; it is not simply clipping-induced shrinkage. Conversely, removing i1 entirely produces a +0.00054446 fixed-mixture gain on F2 and −0.00002308 on F0. Complementarity therefore does not uniquely identify a temporal-memory mechanism.

**Inference:** history deserves a conditional endpoint test, not rejection because C is weaker. However, a second ordinary B model could supply equally useful diversity at lower implementation cost. The existing comparison has no equal-cost B-plus-another-B competitor. The [function-preserving warm follow-up][history] addresses activation of random input columns; it does not itself answer this economic comparison. Both questions are useful, and they should not be conflated.

The score geometry makes this precise. For one target, let `p` be the base prediction, `d` a proposed correction, and use the official label-magnitude weights. Away from clipping boundaries, the derivative of weighted correlation for `p + a*d` at zero is

`Cov_w(y,d)/(sd_w(y)*sd_w(p)) − Corr_w(y,p)*Cov_w(p,d)/Var_w(p)`.

With output clipping, the local correction is additionally masked where the base is saturated; finite mixtures still require direct official scoring. This is an algebraic diagnostic, not a new measurement or optimization objective. Low error correlation and high standalone score are each insufficient. The relevant quantity is useful correction after accounting for what the base already predicts. The actual [metric implementation][metric] fixes clipping, centering, weights and target aggregation.

### 2. TL3 ablations reveal endpoint value, while exposure blocks an architecture verdict

The [same-host native receipt][native] reproduces the H512 replacement loss: CP5 minus original-host TL3 is −0.00736292; changing the host reference itself moves the score by only −0.000000588. This is persuasive evidence about the tested endpoint. It is not persuasive evidence that H512 is intrinsically inferior.

The [recovered student lineage][exposure] links the replaced TL3 payload to a final train-plus-valid distillation script, alpha 0.7 and an inherited `fbc1k_s5@ep3` parent. H512 excludes the 468 F0 evaluation groups. Exact recursive teacher/initializer and retained tied-lead membership receipts remain incomplete. The comparison changes exposure, initialization, objective and family, besides architecture.

The [two-parent public receipt][pair] adds a useful contradiction: removing the student saved 641 seconds relative to reported TL3, but scored 0.66771 versus 0.66832. Its earlier eight-component local advantage did not transfer. Portal binding is filename/user-report based, without an independently verified uploaded hash. This supports retaining the original TL3 as the current reference; it does not identify why the student helped publicly or justify coefficient tuning.

**Decision:** retain a quality/runtime frontier containing the faster pair and the stronger reported TL3. Deprioritize more interpretation of the same H512 replacement as an architecture contest. A matched clean family comparison or a direct marginal-contribution comparison is more valuable than another precision explanation already narrowed by the [precision probe][precision].

### 3. Distillation has a measured retention benefit; new-information transfer remains unresolved

The [quality-bottleneck note][bottleneck] reports only +0.00021415 between a mature T3 student and its exact initializer. Read alone, that can make KD look exhausted. The later [matched alpha receipt][alpha] changes the interpretation. Removing the current teacher target gives alpha-zero minus alpha-0.9 deltas of −0.00394686, −0.00874373 and −0.01431282 at CP1, CP3 and the primary CP5 endpoint. This is one warm-parent seed, with earlier distillation inherited by both arms. Historical fitting used AMP with slightly different skipped-update counts; it is not a new strict-FP32 experiment.

Thus “little improvement over the parent” does not imply “little value from KD.” The target may protect a strong inherited function while label-only continuation degrades it. That retention interpretation is a hypothesis, but the alpha contrast itself is measured. The [fit/evaluation diagnostic][alpha_review] also reports alpha-zero improving on fitting-valid groups while losing on excluded F0; target RMS differs between memberships. Overfitting, distribution shift and target regularization remain competing explanations.

Nor does a stronger teacher ensemble guarantee a useful served addition. The [equal-trio primary grade][trio] scores the three-teacher mean at 0.68090199, above the distinct-seed score mean of 0.67901819. Yet its fixed half-mixture with the older student gains only +0.00002562, with conditional interval [−0.00138485,+0.00139836]. The student already carries related information; old-student FP16 versus new-teacher FP32 is another recorded limit.

**Decision:** retain the supported warm KD target. Modify the [teacher-to-student integration proposal][integration]: negative TL3/R3 output mixtures do not establish that moving the same signal into a teacher is the best route. Transfer through training is a candidate mechanism needing a matched student experiment. Conversely, teacher scaling may remain worthwhile for teacher quality under the user's later teacher-first policy; student transfer is not a universal permission gate.

### 4. Parent exclusion at evaluation does not make fit-side residuals representative

The [online recipe explanation][online] explicitly states that fit-side parent predictions are in-sample. The [age recipe][age] reuses those frozen parents/caches and trains a zero-initialized, 64-unit residual head, serving `parent + 0.1*head`. Its control and feature arms can fairly compare feature variants on that pipeline. However, they learn how to correct a parent on observations the parent fitted, then are evaluated where it did not fit.

This is not automatically leakage into evaluation: parent exclusion may be correct. It is a distribution mismatch in the object being learned. An in-sample parent can leave attenuated or qualitatively different residuals; a head can learn a correction that disappears or reverses on excluded components. Conversely, a negative head result can wrongly discourage a useful feature if its learnable out-of-sample residual never appears during head fitting.

F0 and F1 also use different parent families/ancestry and head seeds. A repeated positive delta would support portability across those two particular settings, but an average cannot isolate the parent, fold or seed axis. All ancestors, including teachers, initializers and propensity models, must exclude the claimed evaluation components. Standard cross-fitting that excludes only a row from the top-level parent is insufficient.

**Decision:** keep the age/online mechanisms unresolved. Add a small explicit test of residual training membership before interpreting this pipeline as a reliable feature-ranking oracle. Do not rename newly partitioned exposed components “fresh.”

### 5. The deliverable cost is dominated by more than parameter count

The [online terminal receipt][online_terminal] records seven of twelve heads saved and no complete-panel grade, with 8,185.64 known seconds including prior work, service and collection. This is delivery evidence, not negative online-learning evidence. The [mean-only identity receipt][identity] shows F0 lag1/lag16 controls reporting the same prediction digest; those two actual fits do not provide independent evidence. Future panels can reuse an exactly verified control, while preserving the historical panel unchanged.

The age [native-route note][route] requires the exact parent, duplicate event-feature computation in the proposed wrapper, reset-local age state and an MLP callback. Existing recurrent packagers do not directly package this head. FP32 cached parent outputs can also differ from native parent outputs: a correction fitted around one base must be judged around the base actually served. Original TL3 took 1,587.6 seconds for the recorded full-F0 host replay; this is a host-specific reference, not an age-wrapper runtime prediction.

**Decision:** count cache transport/reconstruction, fitting, exports, grade, conversion and replay before calling a head cheap. A saved corrected-prediction cache is an intermediate artifact. Keep a named owner and explicit cost for the native parent-plus-head endpoint.

## Questions whose answers would change allocation

1. Does a fixed 90/10 history mixture beat a fixed 90/10 mixture with an ordinary second B seed at equal fitting cost? If not, simplify the endpoint before adding a history serving path.
2. Which proposed complement remains useful after accounting for the fixed incumbent, separately for both targets and broad component strata? This determines whether to improve, compress or omit that component.
3. Does the current KD target improve on a frozen initializer mainly by retaining its behavior, or does a genuinely different teacher add a reproducible correction? The answer selects anchoring versus new-teacher work.
4. Do fit-side and excluded-component residuals differ in useful correction direction, not only RMSE or variance? A positive answer changes residual-head training membership before more feature screens.
5. Can a proposed clean endpoint's entire teacher/initializer/propensity ancestry exclude its evaluation components? If unavailable, should that study remain a fitted diagnostic or fund new parents explicitly?
6. Does a cached-parent gain survive replacement by the exact native parent and callback implementation at a fixed coefficient? If not, fix that interface before another scientific feature arm.
7. How much of a teacher's useful correction is retained by a matched student, and at what complete fitting/serving cost compared with keeping the teacher or a second ordinary model? A lower teacher score is not an automatic veto; a higher teacher score is not a transfer result.

## Proposed hypothesis cards

### H-ASTRA23-06-A — Is history better than ordinary seed diversity?

**Family/change:** ensemble composition; add the missing equal-cost standard-model competitor to the i1 lineage. **Assumptions/mechanism:** the current history mixture may owe its benefit to generic decorrelation, whereas history-specific signal should beat a fresh standard companion. **Predicted observable:** compare `0.9*B_old + 0.1*C_new` with `0.9*B_old + 0.1*B_new`, using the unchanged official metric.

**Cheapest discriminating test/control:** two existing fold cells; train B_new/C_new with paired new seeds and the original matched cold recipe. Freeze both 90/10 endpoints; no coefficient search. Reuse the two old B anchors exactly, plus the six historical fits as context. **Falsifier:** history repeatedly offers no material endpoint advantage over ordinary companions at equal cost; this deprioritizes extra history machinery, not temporal features universally.

**Future cost:** B_new two fits, C_new two fits: **four new GRU fits, zero teachers/students**. The completed six-fit i1 accounting is 15,296.58 seconds; proportional four-fit planning is roughly 10,200 seconds before a fresh hardware/cost correction, with separate complete-endpoint serving work. This is an estimate, not current admission. **Ancestry/exposure:** preserve each cell's source exclusions and old-anchor lineage; two cells still confound fold and seed. **Serving:** the standard companion already has a known family route; active history requires additional callback implementation. **Related:** [i1 progression][history]. **Status:** `not_tested`.

### H-ASTRA23-06-B — Does honest residual fitting change what the head can learn?

**Family/change:** residual estimation; compare heads trained on parent-in-sample residuals with heads trained on components excluded by the parent and all ancestors. **Assumptions/mechanism:** reusable parents have enough excluded components to split into calibration and evaluation portions; in-sample errors hide deployment-relevant correction structure. **Predicted observable:** the honest-residual head improves its corrected endpoint relative to the in-sample head on the remaining excluded components.

**Cheapest test/control:** use the age A head with all age slots zero, avoiding simultaneous feature selection. For each F0/F1 parent, freeze an input-only component split of its excluded set into head-calibration H and evaluation E. Compare a head fitted on H with one fitted on an equal-sized, valid-source parent-fitting subset; match row budget, updates, seed, normalization and fixed coefficient 0.1. Grade only E. Reuse both frozen parents and available caches; fit no new parent. The sample-membership difference is an unavoidable remaining confound, so match source/input strata and describe this as a pipeline screen, not causal proof of overfitting.

**Falsifier:** honest-residual fitting gives no useful correction advantage in both cells and residual diagnostics fail to show the predicted mismatch. **Future cost:** two heads per membership arm, **four new head fits total; zero GRU/teacher/student fits**. Propose a 7,200-second inclusive screen ceiling, including new H/E cache reads and grading, subject to a successful complete-cost profile after resumption. Existing twelve-head delivery proves no automatic feasibility. **Ancestry/exposure:** H becomes fitting data and is never scored as holdout; E remains previously exposed development data. **Serving:** same parent/head architecture in both arms; native verification is a separate conditional cost. **Related:** [online][online], [age][age]. **Status:** `not_tested`.

### H-ASTRA23-06-C — Separate inherited-function retention from new teacher information

**Family/change:** distillation target; compare frozen-parent anchoring with a fixed complementary teacher correction. **Assumptions/mechanism:** current KD protects inherited quality, while an additional teacher may supply information beyond that protection. **Predicted observable:** a student trained toward the predeclared mixed teacher improves over an equally anchored control, and retains part of the teacher correction on excluded components.

**Cheapest test/control:** use two cells with already available, source-qualified parent and candidate-teacher caches. Control target is the frozen parent; treatment replaces a fixed 10% of that target with one candidate family. Match label coefficient, target scale/clipping, warm initialization, schedule and fixed checkpoint. Include the untrained parent as a zero-fit reference; do not sweep teacher weights. **Falsifier:** the teacher correction has useful held-out endpoint value, but the matched student does not retain it at the declared budget. That narrows this transfer route, not teacher quality.

**Future cost:** two anchor students plus two correction students: **four new student fits; zero new teachers**, conditional on verified reusable teacher caches. If caches are missing, stop preparation and re-cost teacher fits explicitly. A recent warm fit plus accepted CPU export cost about 9,038 + 1,937 seconds; four similarly expensive deliveries would exceed 12 hours before grading. This route is therefore deferred behind cheaper discriminators, not advertised as a four-fit bargain. **Ancestry/exposure:** shared teachers are not independent replications; verify recursive exclusion. **Serving:** ordinary student path, no inference teacher required; exact native endpoint still must preserve the gain. **Related:** [quality bottleneck][bottleneck], [integration][integration]. **Status:** `not_tested`.

### H-ASTRA23-06-D — Is the correction portable to the served parent?

**Family/change:** numerical endpoint interaction; substitute exact native parent outputs beneath a frozen residual head without refitting it. **Assumptions/mechanism:** small base discrepancies can be comparable to the proposed incremental gain. **Predicted observable:** the sign and useful size of parent-plus-head minus parent persist after native substitution.

**Cheapest test/control:** after an independently accepted complete head panel exists, compare four frozen outputs on identical rows: cached parent, cached parent plus head, native parent, native parent plus the same head. Freeze component membership and coefficient first. **Falsifier:** the improvement vanishes or reverses under native substitution, or callback state/parity fails. **Future cost:** **zero fits in every arm; zero teachers/students**; reuse the accepted head and parent artifacts. Budget two complete callback passes plus streaming grade, using measured parent timing and a bounded wrapper profile. Twice the recorded TL3 reference is about 3,175 seconds, but the actual parent/head route must be measured rather than assumed equal. **Ancestry/exposure:** same exposed components, no independence claim. **Serving:** directly tests the complete deterministic endpoint. **Related:** [native route][route], [H512 recovery][native]. **Status:** `not_tested`.

## Primary literature and transfer limits

- [Brown, Wyatt and Tiňo, Managing Diversity in Regression Ensembles](https://jmlr.csail.mit.edu/papers/v6/brown05a.html): inspected the journal abstract. It ties squared-error ensemble utility to accuracy and covariance jointly. This motivates the ordinary-companion control; its quadratic-loss conclusions do not directly optimize clipped weighted Pearson.
- [Cho and Hariharan, On the Efficacy of Knowledge Distillation](https://arxiv.org/abs/1910.01348): inspected the abstract. More accurate/larger classification teachers need not teach a given student better. This supports measuring transfer, not a claim that our teachers are too large or should be stopped early.
- [Dao et al., Knowledge Distillation as Semiparametric Inference](https://arxiv.org/html/2104.09732v1): inspected the abstract, problem definition, overfitting discussion and Section 4.2 cross-fitting construction. It isolates teacher data reuse as a potential student-learning problem. Our proposed residual-head screen borrows the exclusion principle; its classification, independent-sample and loss assumptions do not establish a regression or competition guarantee here.

## Best next decision, counterargument and limits

**Best proposed and unexecuted test: H-ASTRA23-06-B.** After an explicit resume, freeze the two-parent, four-head membership comparison before selecting another age/online feature. It addresses a common failure mode across multiple inexpensive-head ideas, reuses existing parents, and can change whether those screens are trusted. It neither rewrites nor partially grades the stopped online panel. First verify ancestor exclusion and complete delivery cost; inability to assemble honest calibration/E memberships is a concrete reason to defer this test.

The strongest counterargument is that splitting parent-excluded components sacrifices evaluation precision and changes the head's training distribution. A small honest-head loss could reflect fewer independent examples or source shift rather than absence of residual signal. Matching source and input strata reduces but cannot remove that ambiguity. If this design cannot be assembled without broad new engineering, H-ASTRA23-06-A is the clearer direct competition experiment, although more expensive and still tied to a cold proxy.

Distillation may also help despite negative direct blending, and cheap generic diversity may fail against a mature ensemble even when it helps a small B anchor. These are reasons to test complete endpoints, not reasons to declare either family closed. No current evidence justifies a public >0.700 forecast. My strongest supported conclusions concern conditional endpoint utility, the substantial matched warm-KD retention contrast, and the need to separate honest residual learnability from in-sample correction fitting.

## Source records

[stop]: ../../../experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json
[i1]: ../../../campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
[geometry]: ../../../campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json
[history]: ../../ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md
[metric]: ../../../../src/wnn/gmetric.py
[native]: ../../../campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json
[exposure]: ../../../campaign_2026-10-21/next_cycle_r1/TL3_STUDENT_EXPOSURE_RECOVERY_R1.json
[pair]: ../../../campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json
[precision]: ../../../campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_PRECISION_PROBE_R2.json
[bottleneck]: ../../ideas/QUALITY_BOTTLENECK_AND_STUDENT_TRADEOFFS_20261015_R1.md
[alpha]: ../../../campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_CP3_CP5_PAIRED_GRADE_REVIEW_R1.json
[alpha_review]: ../ALPHA0_WARMPARENT_CP3_CP5_GRADE_REVIEW_20261015_R1.md
[trio]: ../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json
[integration]: ../../ideas/TEACHER_STUDENT_SUBMISSION_COMBINATION_20261018_R1.md
[online]: ../../ideas/ONLINE_RELATIONSHIPS_20261022_R1.md
[age]: ../../../experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json
[online_terminal]: ../../../experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json
[identity]: ../../../experiment_drafts/online-relationships-20261022/MEAN_ONLY_LAG_IDENTITY_R1.json
[route]: ../../../experiment_drafts/b300-fast-local-kaggle-20261023/NATIVE_ROUTE_R1.md

- [October 17 model catalog](../../models/CURRENT_MODEL_CATALOG.md): source of fixed TL3 coefficients, family descriptions and package distinction; later receipts above supersede its dated status.
- [October 23 cycle accounting](../../../campaign_2026-10-21/next_cycle_r1/CYCLE_ACCOUNTING_POSTBOOT_R1.json): measured i1 and warm fit/export cost anchors used for planning, with historical allocation limits preserved.
- [Current frontier](../../../campaign_2026-10-21/next_cycle_r1/FRONTIER.md): stopped execution state, incomplete warm/online panels and protected production allocation. This report grants no spending authority.

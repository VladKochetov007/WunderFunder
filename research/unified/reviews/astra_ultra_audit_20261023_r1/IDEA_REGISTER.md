---
kind: review
status: complete
created_utc: 2026-10-23T17:17:50Z
updated_utc: 2026-10-23T17:28:29Z
hypothesis_ids: []
experiment_ids: [E-ASTRA-ULTRA-AUDIT-20261023-R1]
evidence_paths:
  - research/unified/reviews/astra_ultra_audit_20261023_r1/MANIFEST.json
---

# Register of all 40 independent hypothesis cards

This is a navigation and disposition layer over the ten preserved reports. Full
mechanisms, predictions, controls, falsifiers, ancestry, exposure and serving
limits remain in each linked card. No proposal is admitted or executed by this
review. Do not add the fit counts: cards overlap and many are alternative designs.
The normal prospective screen is 4–6 **total** new fits, with exact reuse stated.
All current folds remain exposed; null numerical priors are intentional.

Thirty-nine cards describe unexecuted proposed comparisons. Card 09-D mistakenly
asks to repeat an already completed comparison. Card 08-D retains an untested
idea but needs a corrected control. These dispositions, rather than the original
cards' blanket `not_tested` labels, govern further use. See [reconciliation](RECONCILIATION.md).

## Reviewer 01

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-01-A — History versus an ordinary second seed](01_evidence_and_selection.md#h-astra23-01-a--history-versus-an-ordinary-second-seed) | history attribution | 4 recurrent fits; anchors conditional | Retain; choose warm or cold recipe explicitly |
| [H-ASTRA23-01-B — Structured i1 masking as a cheaper source of robustness](01_evidence_and_selection.md#h-astra23-01-b--structured-i1-masking-as-a-cheaper-source-of-robustness) | robustness | 4 recurrent fits | Exploratory reserve; distinct from masking at evaluation |
| [H-ASTRA23-01-C — Recover a bounded answer from completed online heads](01_evidence_and_selection.md#h-astra23-01-c--recover-a-bounded-answer-from-completed-online-heads) | online delivery | 0 new fits; one declared F0 analysis | Priority recovery candidate; frozen full panel remains incomplete |
| [H-ASTRA23-01-D — Test whether the apparent gain is concentrated](01_evidence_and_selection.md#h-astra23-01-d--test-whether-the-apparent-gain-is-concentrated) | transfer robustness | 0 fits; fixed component diagnostics | Combine with 08-A; no tuned stress weights |

## Reviewer 02

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-02-A — Clocks beyond position and activity](02_representation_and_hidden_structure.md#h-astra23-02-a--clocks-beyond-position-and-activity) | age and clocks | 6 heads | Retain after position/activity controls; ancestor of censoring follow-up |
| [H-ASTRA23-02-B — Change-count history with a duration escape path](02_representation_and_hidden_structure.md#h-astra23-02-b--change-count-history-with-a-duration-escape-path) | event time | 4 heads | Retain dual clock; preserve duration and row recurrence |
| [H-ASTRA23-02-C — Separate frequent atoms from continuous excursions](02_representation_and_hidden_structure.md#h-astra23-02-c--separate-frequent-atoms-from-continuous-excursions) | numeric representation | 4 heads | Exploratory reserve; distinguish atom identity from nonlinear basis |
| [H-ASTRA23-02-D — Test correspondence before enforcing shared structure](02_representation_and_hidden_structure.md#h-astra23-02-d--test-correspondence-before-enforcing-shared-structure) | correspondence prior | 6 heads | Retain restricted architecture; not an unrestricted permutation null |

## Reviewer 03

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-03-A — measure useful recurrent context before lengthening it](03_memory_and_time.md#h-astra23-03-a--measure-useful-recurrent-context-before-lengthening-it) | memory access | 0 fits; recurrent replays | Diagnostic; reset-induced distribution shift limits interpretation |
| [H-ASTRA23-03-B — separate readout access from missing history](03_memory_and_time.md#h-astra23-03-b--separate-readout-access-from-missing-history) | state readout | 6 heads plus possible state export | Retain if cache cost is justified; hidden state is not already free |
| [H-ASTRA23-03-C — a second clock without state deletion](03_memory_and_time.md#h-astra23-03-c--a-second-clock-without-state-deletion) | event time | 6 heads | Related to 02-B; alternative design, not an additional default panel |
| [H-ASTRA23-03-D — complete a clean temporal-credit comparison](03_memory_and_time.md#h-astra23-03-d--complete-a-clean-temporal-credit-comparison) | gradient credit | 4 recurrent continuations | Retain existing S14 descendant; strict-FP32 costs unmeasured |

## Reviewer 04

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-04-A — Audit and isolate chunk-mass normalization](04_objectives_and_metric.md#h-astra23-04-a--audit-and-isolate-chunk-mass-normalization) | effective objective | 4 continuations plus branch audit | Retain only with verified executed teacher-loss path |
| [H-ASTRA23-04-B — Separate variance learning from its mean-gradient weighting](04_objectives_and_metric.md#h-astra23-04-b--separate-variance-learning-from-its-mean-gradient-weighting) | variance learning | 6 continuations | Diagnostic first; separate mean/variance gradients and Adam steps |
| [H-ASTRA23-04-C — Test whether history adds information beyond scalar recalibration](04_objectives_and_metric.md#h-astra23-04-c--test-whether-history-adds-information-beyond-scalar-recalibration) | scalar sufficiency | 4 two-output calibration jobs / 8 scalar maps | Retain if fixed split and capacity; calibration is fitting |
| [H-ASTRA23-04-D — Causal moment correction with an unchanged initial predictor](04_objectives_and_metric.md#h-astra23-04-d--causal-moment-correction-with-an-unchanged-initial-predictor) | causal moments | 4 heads | Exploratory reserve; preserve between-sequence information |

## Reviewer 05

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-05-A — Recover the actual learning curves](05_teacher_scaling_and_capacity.md#h-astra23-05-a--recover-the-actual-learning-curves) | learning curves | 0 new fits; up to 16 forward evaluations | Inventory first; evaluate only decision-changing missing points |
| [H-ASTRA23-05-B — Learn longer credit before buying more state](05_teacher_scaling_and_capacity.md#h-astra23-05-b--learn-longer-credit-before-buying-more-state) | gradient credit | 4 recurrent continuations | Merge allocation with 03-D, preserve original card |
| [H-ASTRA23-05-C — Compare capacity with buying more training](05_teacher_scaling_and_capacity.md#h-astra23-05-c--compare-capacity-with-buying-more-training) | capacity versus exposure | 6 teacher fits | Separate future scale allocation; compact anchors do not cap ambition |
| [H-ASTRA23-05-D — Diversity versus repetitions on a fixed architecture](05_teacher_scaling_and_capacity.md#h-astra23-05-d--diversity-versus-repetitions-on-a-fixed-architecture) | data diversity | 4 fresh-backbone fits | Retain only with ancestor-restricted data memberships |

## Reviewer 06

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-06-A — Is history better than ordinary seed diversity?](06_ensembles_and_distillation.md#h-astra23-06-a--is-history-better-than-ordinary-seed-diversity) | history attribution | 4 recurrent fits | Related to 01-A/09-B/10-C; compare exact cold allocations |
| [H-ASTRA23-06-B — Does honest residual fitting change what the head can learn?](06_ensembles_and_distillation.md#h-astra23-06-b--does-honest-residual-fitting-change-what-the-head-can-learn) | residual estimation | 4 heads | High-value question; parent-excluded fit data must stay disjoint from evaluation |
| [H-ASTRA23-06-C — Separate inherited-function retention from new teacher information](06_ensembles_and_distillation.md#h-astra23-06-c--separate-inherited-function-retention-from-new-teacher-information) | KD preservation | 4 student fits; zero teacher fits if caches match | Retain; current matched KD evidence argues against removing it casually |
| [H-ASTRA23-06-D — Is the correction portable to the served parent?](06_ensembles_and_distillation.md#h-astra23-06-d--is-the-correction-portable-to-the-served-parent) | serving transfer | 0 new fits; four frozen endpoint outputs | Required before claiming head transfer to a different parent path |

## Reviewer 07

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-07-A — History versus an equally accessible current-input expansion](07_causal_identification_and_controls.md#h-astra23-07-a--history-versus-an-equally-accessible-current-input-expansion) | history attribution | 6 recurrent fits; exact reuse conditional | Current-input nonlinear control distinguishes history from accessible basis |
| [H-ASTRA23-07-B — Recent information versus equally mature old information](07_causal_identification_and_controls.md#h-astra23-07-b--recent-information-versus-equally-mature-old-information) | online adaptation | 4 heads | Retain after utility evidence; equal estimator maturity before drift claim |
| [H-ASTRA23-07-C — Age identity versus age distribution](07_causal_identification_and_controls.md#h-astra23-07-c--age-identity-versus-age-distribution) | age identity | 6 heads; possibly 2 new with four exact controls | Retain distribution-versus-identity contrast; learning geometry still differs |
| [H-ASTRA23-07-D — Anticipation versus retention supervision](07_causal_identification_and_controls.md#h-astra23-07-d--anticipation-versus-retention-supervision) | auxiliary supervision | 6 backbone fits | Defer behind cheaper tests; equal valid support is essential |

## Reviewer 08

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-08-A — Composition sensitivity of the history increment](08_transfer_and_information_limits.md#h-astra23-08-a--composition-sensitivity-of-the-history-increment) | transfer robustness | 0 fits; existing three fixed stress panels | Priority cached diagnostic; synthetic shift is not a private-set forecast |
| [H-ASTRA23-08-B — Separate target learning from selection concentration](08_transfer_and_information_limits.md#h-astra23-08-b--separate-target-learning-from-selection-concentration) | selection weighting | 6 continuations | Diagnostic first; propensity cannot repair unknown outcome selection automatically |
| [H-ASTRA23-08-C — Separate atom identity from occupancy](08_transfer_and_information_limits.md#h-astra23-08-c--separate-atom-identity-from-occupancy) | numeric representation | 6 heads | Related to 02-C; test identity versus occupancy and unknown atoms |
| [H-ASTRA23-08-D — Treat unseen prehistory as censored age](08_transfer_and_information_limits.md#h-astra23-08-d--treat-unseen-prehistory-as-censored-age) | age censoring | 6 heads as proposed | REVISE CONTROL: fixed dense-head derangement preserves information |

## Reviewer 09

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-09-A — Recover the continual-versus-warmup relationship contrast](09_research_value_and_delivery.md#h-astra23-09-a--recover-the-continual-versus-warmup-relationship-contrast) | online delivery | 0 fits now; conditional later 4 missing unique F1 heads | Merge with 01-C; reuse requires exact source/membership proof |
| [H-ASTRA23-09-B — History must earn its cost against ordinary model diversity](09_research_value_and_delivery.md#h-astra23-09-b--history-must-earn-its-cost-against-ordinary-model-diversity) | history attribution | 4 recurrent fits | Related to 01-A/06-A/10-C; no winner-picking among seeds |
| [H-ASTRA23-09-C — Complete evaluation sequences can be the export unit](09_research_value_and_delivery.md#h-astra23-09-c--complete-evaluation-sequences-can-be-the-export-unit) | delivery cost | 0 scientific fits; engineering parity/export work | Retain whole evaluation sequences; no truncated-prefix shortcut |
| [H-ASTRA23-09-D — Resolve one saved trajectory mean before inventing another](09_research_value_and_delivery.md#h-astra23-09-d--resolve-one-saved-trajectory-mean-before-inventing-another) | trajectory averaging | 0 additional work to answer the stated question | ALREADY GRADED: +0.0000109615 fixed-endpoint increment; exclude duplicate |

## Reviewer 10

| Card and question | Family | Proposed complete-screen fitting cost | Root disposition |
|---|---|---|---|
| [H-ASTRA23-10-A — bounded causal response-speed correction](10_unrestricted_red_team.md#h-astra23-10-a--bounded-causal-response-speed-correction) | temporal response | 0 fits; five fixed causal transforms in two cells | Bounded exploratory reserve; failed smoothing does not prove underreaction |
| [H-ASTRA23-10-B — locate the useful part of history disagreement](10_unrestricted_red_team.md#h-astra23-10-b--locate-the-useful-part-of-history-disagreement) | correction decomposition | 0 fits; four fixed transforms in two cells | Related diagnostic; no frequency or coefficient search |
| [H-ASTRA23-10-C — make ordinary seed diversity the spending comparator](10_unrestricted_red_team.md#h-astra23-10-c--make-ordinary-seed-diversity-the-spending-comparator) | ordinary diversity | 4 recurrent fits; two new ordinary seeds per fold | Distinct allocation design with exact old B/C reuse; do not count as matched warm history |
| [H-ASTRA23-10-D — soft distribution targets only as a specific optimization retest](10_unrestricted_red_team.md#h-astra23-10-d--soft-distribution-targets-only-as-a-specific-optimization-retest) | distributional optimization | 6 heads only after a distinct rationale | Defer; H-111/E421 already provide relevant adverse evidence |

## How to allocate these ideas

Use the [synthesis](SYNTHESIS.md) for the proposed order and the [question register](QUESTION_REGISTER.md)
for all 70 original questions. History attribution appears in six cards because
reviewers worked independently; six mentions are not six experiments or six
empirical replications. Likewise the two gradient-credit cards share S14, and
the two online-recovery cards share the same saved outputs.

The [hypothesis ledger](../../../hypotheses.jsonl) records each card with its
source and root disposition. The [experiment ledger](../../../experiments.jsonl)
records this review as a zero-fit documentary activity, not a forty-arm admission.


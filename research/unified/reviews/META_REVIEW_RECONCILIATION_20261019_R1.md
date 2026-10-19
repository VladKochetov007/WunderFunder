---
kind: review
status: complete
created_utc: 2026-10-18T22:46:00Z
hypothesis_ids: [H-FROZEN-PARENT-ANCHOR-20261015, H-RR17-01]
experiment_ids: [E-KD-ALPHA0-MATCHED-TRAJECTORY-20261015-R1, E-STUDENT-FRONTIER-H512L1-F0S3-20261015]
evidence_paths:
  - ../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md
  - EXTERNAL_STUDENT_TRANSFER_META_REVIEW_20261019_R1.md
---

# Integrating the student/transfer meta-review

## Decision

Keep the accepted cold i1 A/B/C comparison first. Add frozen-parent anchoring as
a conditional, separately budgeted **matched FP32 pair**. Use the newer F3
wide/deep study before allocating duplicate architecture fits. If i1 history
helps, test its transfer through a fixed student rather than assuming a larger
teacher or wider student automatically preserves the gain.

The [imported review](EXTERNAL_STUDENT_TRANSFER_META_REVIEW_20261019_R1.md) supplies
useful competing explanations. Its recommendation to run one missing control is
not executable unchanged under this campaign's precision and memory constraints.

## Independent source audit

Reviewer: native `gpt-6-luna`, max effort, role `review_direction_audit`.
Scope: read-only source/ledger/receipt review; no fits, payload reads or regrading.
Root requested a second pass on precision and the newer research-team branch.
Main source was `9defbcbc0f2a6f1b35941dc657496993b48889c6`; the newer branch
was inspected at `758db928d20f61e9189839c284d092eed2bb415d`.

| Review proposition | Audited evidence | Consequence |
|---|---|---|
| T3 may preserve the parent rather than add useful information | Six-epoch CP5 T3 0.681155 vs label-only 0.666843; parent 0.680941. The label-only-minus-T3 conditional interval is about [−0.031167, −0.001122]. | Supports testing anchoring; does not establish its mechanism or transfer. |
| Only one new frozen-parent fit is needed | A compatible historical two-epoch C/CP1 score is 0.681227, but it was trained in FP16. The S arm only completed a profile. | Cannot reuse C as the strict-FP32 control; need a fresh matched C/S pair. |
| Frozen-parent S is ready for our laptop | Its 50-update profile used FP16 and peaked at 5.85 GB worker RAM. | Exceeds our 5 GB cap. Reuse code, then change/profile the memory path for both paired arms; no launch-and-see. |
| Wide H512×1 already has a strong gain | One-epoch exposed-F0 score 0.673238 vs H256×2 0.669778; Δ+0.003460, conditional interval [+0.002003,+0.004904]. | Short-screen evidence only. No additive prediction for the best endpoint. |
| No longer width/depth study exists | Main was stale: the newer F3 wide arm has a structurally verified six-epoch terminal checkpoint. Its deep counterpart/paired grade was not complete in the inspected receipts. | Collect the existing study before repeating a full width/depth pair. |
| H256→H384 gained on four folds by the quoted amounts | The reviewer could not trace all four quoted numbers to a specific indexed result. | Keep those exact numbers unverified; do not use them to justify allocation. |

The six-epoch alpha-zero endpoint is not a matched control for a two-epoch S/C
comparison. Parent/T3 targets can also differ in scale, calibration and inherited
teacher ancestry. Even a valid S≈C outcome supports a recipe-level conclusion,
not “the teacher contains no new information.”

## Exact prior receipts

- [Two-epoch C/CP1 paired grade](../../campaign_2026-10-15/student_quality_tradeoff_r1/WARM_MECHANISM_F0_S3_C_R3_CP1_PAIRED_GRADE_R1.json)
  and [C resource profile](../../campaign_2026-10-15/gru_parent_growth_r1/WARM_MECHANISM_PROFILE_C_RESULT_R1.json):
  historical external-T3 comparator, FP16 training. FP32 prediction export does
  not make training FP32.
- [Frozen-parent S profile](../../campaign_2026-10-15/student_quality_tradeoff_r1/frozen_parent_anchor_r1/S_PROFILE_TERMINAL_RESULT_R1.json):
  50 updates, no completed scientific fit/checkpoint/grade, 5.85 GB worker RAM.
- [Anchoring proposal](../ideas/FROZEN_PARENT_ANCHOR_CONTROL_20261015_R1.md)
  and [October 15 research-loop reset](../methods/RESEARCH_LOOP_RESET_20261015_R1.md):
  source semantics, interpretation and existing implementation route.
- [F3 wide terminal verification at the inspected research-team revision](https://github.com/VladKochetov007/WunderFunder/blob/758db928d20f61e9189839c284d092eed2bb415d/research/campaign_2026-10-18/priority25_execution_r4/receipts/RANK01_WIDE_F3_S1001_TERMINAL_VERIFICATION.json)
  and [root acceptance](https://github.com/VladKochetov007/WunderFunder/blob/758db928d20f61e9189839c284d092eed2bb415d/research/campaign_2026-10-18/priority25_execution_r4/receipts/F3_WIDE_TERMINAL_ROOT_ACCEPTANCE_20261018T2220Z.json):
  `rr18_r01_wide_f3_s1001_e6`, 89,280 updates, zero skips, six-epoch terminal
  checkpoint. Structural verification is not a quality score. The latest archived
  deep-arm observation was running at sequence index 3 on October 18 at 22:15 UTC;
  recheck its owner/live state before collection. Do not relaunch its work.

## Bounded additions and combinations

### 1. Anchoring versus external teacher information

Compare S (label + frozen own-parent target) against C (label + fixed T3 target)
from the exact same parent. Both new arms use strict FP32 and identical data,
sampler, auxiliary losses, optimizer, update count and fixed checkpoint rule.
Audit target clipping order and fitting-only calibration. Preserve the existing
direct-parent implementation; no cloned trainer or general distillation engine.

Before admission, name the exact parent/T3/cache identities and clean evaluation
components, pass the memory profile, and reserve the **whole pair plus exports**.
Use the targeted 3h allocation only if measured cost fits; otherwise leave the
pair queued for available Kaggle or a later authorized budget. The historical
1.9h estimate belongs to a different cold H512 recipe and cannot price this pair.

Interpret C>S as evidence that this external target is useful beyond the chosen
anchor at this horizon. S≈C favors cheaper anchoring in this recipe while leaving
teacher information and transfer limitations unresolved. If uncertainty dominates,
another matched seed may be more informative than another teacher architecture.

### 2. Representation → teacher → fixed student

If the i1-history teacher has a repeatable incremental or fixed-mixture benefit,
compare baseline-teacher versus history-teacher caches in **identical cold
students** with clean ancestry. This tests whether explicit state relieves the
student's information bottleneck. Student improvement is measured directly; it
cannot be inferred from teacher quality or prediction disagreement.

If both representation and anchoring earn follow-up, a later interaction design
can cross baseline/history parent with own-parent/T3 targets. Register it as a
separate interaction with both parent hypotheses. Do not add the two effects or
spend on the four cells before either parent comparison is informative.

### 3. Capacity and gradient horizon

First inspect and grade the existing F3 study. A matched H256×2 control for the
new three-epoch/10% B arm is optional only if it answers a distinct unresolved
question within remaining budget. Parameter parity and callback-cost parity are
different comparisons; neither follows from the width/layer labels alone.

Keep detach 500 vs 2000 at fixed update 2000/full forward state as a bounded
deferred arm. Inference resets do not establish a gradient-horizon benefit. The
label-only macro-update path still needs implementation before a cold-fit claim.

Kalman remains a separate exploratory family with explicit reopening conditions;
weak existing results do not justify making it this cycle's primary allocation.

## Limits

All folds are exposed. Quoted scores are historical inspected receipts, not
newly recomputed results. The current cycle has no completed fit, public forecast
or new candidate. The [accepted spec](../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md)
and [frontier](../../campaign_2026-10-20/plateau_review_r1/FRONTIER.md) govern the next action.

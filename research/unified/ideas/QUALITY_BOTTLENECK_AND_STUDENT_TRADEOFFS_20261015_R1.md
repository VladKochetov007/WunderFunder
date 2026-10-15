---
kind: idea
status: proposed
created_utc: 2026-10-15T11:55:00Z
updated_utc: 2026-10-15T11:55:00Z
hypothesis_ids: [H-KD-CONTINUATION-VALUE-20261015, H-STUDENT-LADDER-20261014]
experiment_ids: []
evidence_paths: [research/campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md]
---

# What limits student quality, and how to distinguish the causes

## Question or claim

The user asks whether teacher quality or distillation limits quality, and directs continued student tradeoff experiments while CPU-host measurements are unavailable. Current evidence does not isolate either cause. It does establish a useful architecture effect under the short, fixed recipe, and shows that the strongest historical students inherited much of their quality from their starting checkpoints.

This is a prospective allocation and retrospective evidence synthesis, not a new measured quality result. Root owns implementation, admission, predictions and grading. Native subordinate roles remain Luna/max: independent evidence review, matched student recipe delivery, and clean F3 teacher export delivery.

## Evidence and interpretation

All scores below use official `global_wp_clip2_v1` on repeatedly exposed F0. Different starting weights, training exposure, auxiliary objectives and targets prevent treating the rows as one causal comparison.

| Endpoint | WP | What it establishes |
|---|---:|---|
| Fresh H256×2, one epoch, B8 FP32 | 0.669777642 | Current architecture screen control; not a mature training baseline. |
| Fresh H512×1, same screen recipe | 0.673237761 | Matched architecture gain +0.00346012; conditional component interval [0.00200307, 0.00490448], one fold/seed. |
| Clean H384 teacher trio, equal mean | 0.680901989 | Quality of this particular homogeneous teacher cohort; not a teacher-family ceiling. |
| Current T3 warm student, six epochs | 0.681155360 | Mature continuation endpoint. Its exact starting checkpoint scores 0.680941207, giving a raw +0.000214154, without a paired interval. |
| Older eight-source warm student | 0.683141813 | Different initializer/targets/family loss. Its own parent scores 0.682823341; continuation gain +0.000318472 has component interval [-0.00132008, 0.00179901]. |

The current T3 seed-4 parent/student point scores likewise move from about 0.681029 to 0.681258. These are two student/initializer seeds sharing a teacher cohort and exposed F0, not independent teacher replications. No evaluated per-epoch quality curve establishes when either student converges.

Native Q8 changes the fresh H256×2 score by only +0.00000513, and H256×1 by +0.00001541. These particular serving conversions do not explain the large gap between fresh short fits and mature warm students. This does not certify every architecture or W4 export.

The current equal-prediction teacher trio is close to the mature student's score. There is little observed score separation on this exposed F0 comparison, but this does not establish that teacher limitations caused the result: the student inherits a strong parent, and there is no matched label-only continuation. The older student's stronger parent accounts for most of the point-score difference from the current mature student. Attribution to its heterogeneous targets requires a controlled target comparison.

## Mechanism and method

The first discriminator is a matched continuation pair from `c20261003_combo_s3_f0@ep3`: label-only alpha=0 versus current homogeneous T3 alpha=0.9. Hold initial weights, fitting membership, normalization, propensity weighting, auxiliary losses, optimizer, sampling/order, precision, batch/chunk sizes and schedule constant. Use six epochs per arm and a fixed final checkpoint. Preserve checkpoints at two/four/six epochs and evaluate those declared positions to obtain a quality trajectory; do not select the best epoch afterward. The existing alpha=0.9 endpoint is contextual evidence until exact source and numerical recipe equivalence justify reuse.

This comparison estimates the incremental value of teacher targets for a warm model. It does not answer fresh-student compressibility. A subsequent cold-start comparison should use the leading two shapes, equal exposure and target treatments, with distinct seeds and broader folds. If label-only matches or exceeds KD, test a lower teacher weight or a genuinely different cohort rather than assuming additional KD epochs solve the problem. If KD helps substantially but the larger student still improves, investigate student capacity and learning dynamics. If a stronger teacher improves its own held-out score but not the matched student's result, investigate target accessibility, shared errors and capacity mismatch.

Keep the original eight-shape screen intact: H256×1, H192×2, H256×2, H512×1, H384×2, H640×1, H512×2 and H448×3. H384×2 is fitting under its immutable original recipe. Remaining shapes need root admission. Teacher versions stay fixed within that comparison. The screen ranks finite-budget performance; finalists also need longer fits and runtime measurements. Do not discard a slow-learning architecture solely from a short screen.

In parallel, export the already-fitted clean F3 H384 teacher checkpoint rather than refit it. This unblocks grading and source-qualified fold coverage; it does not yet constitute a stronger teacher cohort. Later teacher work should compare heterogeneous family targets and larger recurrent teachers by both their own quality and their effect on matched students.

## Theory and useful falsifiers

For squared-error distillation with fixed row weights, fitting a blended target combines label and teacher objectives up to a parameter-independent term. This makes alpha a bias/variance tradeoff, not a guarantee of extra information. Here the deployed training target is `0.1*clip(y,2)+0.9*clip(T,2)`, with stored `T=float32(0.8125*mean(three teachers))`; weighting, clipping and finite optimization remain material. Audit the actual consumer rather than interpreting cache metadata as the objective.

Teacher prediction quality is not a mathematical ceiling on a student: [Born Again Neural Networks](https://proceedings.mlr.press/v80/furlanello18a.html) reports students outperforming teachers. Conversely, [Cho and Hariharan](https://arxiv.org/abs/1910.01348) find that better/larger classification teachers need not produce better students. [Beyer et al.](https://arxiv.org/abs/2106.05237) identify training duration and consistent teacher/student inputs as important in image distillation. These are external precedents, not evidence that the same effects explain our recurrent regression task. Their transferable test is to measure student benefit under matched training, not infer it from teacher score or parameter count.

The prior target-amplitude and trajectory-average experiments produced only small uncertain effects. Preserve those negatives; repeating a scale-only change is a lower priority than separating inherited quality, label-only improvement and teacher contribution.

## Decision and resource allocation

Continue the healthy H384 fit and its profile/export/grade chain. Prepare the matched continuation packet for review and the next useful GPU admission. No new fit is admitted by this note. Preserve remaining architecture candidates and broader-fold/seed confirmation. Defer CPU-host searches, initializer native-build repair and additional TPU numerical engineering while these quality deliverables are ready work.

At 2026-10-15T11:50:33Z, typed Kaggle quota is GPU 108000 seconds total / 81372.611 used / 0 reserved (7.3965 hours remaining), TPU 72000 / 567.197 / 0 (19.8424 hours). Refresh is October 17. These are account snapshots, not fit reservations. SDK duration fields must use `total_seconds()`. No paid rental is admitted; exact CPU-only Vast access remains unresolved and spend is zero.

Local limits remain five aggregate CPU cores, one GPU worker, 5 GB observed VRAM, 17 GB committed-memory target, 18/20 GB high/hard caps and no swap. Fresh memory/accounting checks precede the H384 chain's 6 GB export reservation. New matched fits require exact recipe review and admission; the current job's frozen sources and targets remain unchanged.

## Source records

- [Live architecture frontier](../../campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md) and [H512 independent result review](../reviews/CPU_STUDENT_FRONTIER_H512L1_RESULT_REVIEW_20261015_R1.md).
- [Native Q8 grade receipts](../../campaign_2026-10-14/cpu_student_frontier_r1/native_grades_r1/README.md) bind the H256×2 and H256×1 serving conversions and official full-F0 scores.
- [Independent teacher/student ancestry review](../reviews/CLEAN_T3_TEACHER_STUDENT_BOTTLENECK_REVIEW_20261015_R1.md) verifies the direct parents, target treatment and comparison limits.
- [Teacher trio and fixed half-student blend grade](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json).
- [Current seed-3 student grade](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2/f0_grade_r1/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1.grade.json) and [historical direct-parent grade bank](../../campaign_2026-10-06/bandit_continuation/prediction_bank/GRADE_FOLD0FULL_20261010T004533Z_f3ec9c4efd4d7305.json). The independent reviewer checked matching split, fitting/evaluation membership, normalization and exact initializer checkpoint hash.
- [Older Z4 matched parent/student paired grade](../../campaign_2026-10-06/bandit_continuation/runtime/receipts/oldz4-r5-f0-s3-student-export/F0_GRADE_PAIRED_R5.json).
- [Prior target amplitude packet](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/README.md) and [first-results analysis](../../campaign_2026-10-13/strategy_execution_r1/first_results_analysis_r1/README.md).
- [Collected F3 teacher and pending exports](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/DRAFT_STATUS_R5.md).

No local delta in this note is a public-score forecast. All reported uncertainty conditions on the fitted endpoints unless explicitly described otherwise.

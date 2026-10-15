---
kind: review
status: complete
created_utc: 2026-10-15T16:33:00Z
updated_utc: 2026-10-15T16:33:00Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_CP3_CP5_PAIRED_GRADE_REVIEW_R1.json
  - research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_CP1_COMPONENT_BOOTSTRAP_R1.json
  - research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_POSTFIT_EXPORT_GRADE_COMMANDS_R1.json
  - artifacts/runs/student_f0_warm_s3_alpha0_r1_p3_envr1/provenance.json
  - artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json
---

# Alpha-zero warm-parent CP3 and CP5 grade review

## Question or claim

Does removing the current clean-T3 distillation coefficient improve the matched seed-3 warm-parent continuation at the four- and six-epoch checkpoints? On the exposed fold-0 panel, alpha-zero is lower than the alpha-0.9 control at CP1, CP3, and CP5. This supports the current cache target for this specific continuation; it does not establish a cold-start KD effect or identify a general teacher or student ceiling.

## Method

I executed the exact frozen official grade commands and the unchanged `score_fold0_panel.py` paired component bootstrap for CP3 and CP5. I did not load checkpoints, fit models, or change the grader. CP1 results were already present and are included as trajectory context. The outputs cover all valid sequences with FP32 predictions of shape `[1873, 20000, 2]`; the official metric is `global_wp_clip2_v1` on the frozen fold-0 evaluation panel, 468 groups / 141 components / 808,863 selected rows. The split SHA is `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`.

The arms use the same H256×2 seed-3 warm parent checkpoint `c20261003_combo_s3_f0@ep3` (SHA `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`), fitting membership, clean-T3 cache, propensity source, normalization buffers, per-epoch sample-order identities, model, optimizer, six-epoch OneCycle schedule, batch/chunk geometry, precision, and auxiliary losses. Provenance arg comparison at the same stage differs only in `distill_alpha`, run name, and the arm-specific resume-state path. The actual treatment is current homogeneous T3-cache alpha `0.0` versus `0.9`; both models retain the same older distilled-parent ancestry. This is not a label-only-from-scratch comparison.

## Evidence and interpretation

| Checkpoint | Alpha 0 WP | Alpha 0.9 WP | Alpha 0 minus 0.9 | Conditional 95% component interval |
|---|---:|---:|---:|---:|
| CP1, epoch index 1 / 2 epochs | 0.67636773 | 0.68031459 | −0.00394686 | [−0.00845201, −0.00010596] |
| CP3, epoch index 3 / 4 epochs | 0.67222032 | 0.68096406 | −0.00874373 | [−0.01832483, −0.00146275] |
| CP5, epoch index 5 / 6 epochs | 0.66684254 | 0.68115536 | −0.01431282 | [−0.03116723, −0.00112208] |

CP5 was the predeclared primary endpoint; CP1 and CP3 are descriptive trajectory views. Each paired interval resamples the same 141 fixed evaluation components (2,000 replicates, seed 271828), conditional on these exact predictions. The three checkpoint comparisons reuse one seed, fit lineage, and exposed panel, so they are correlated and are not training replications. An interval below zero describes component-sampling uncertainty for this fixed comparison; it is not a universal rule to stop other alpha values or teacher families.

Both arms attempted 44,760 optimizer updates across six epochs. FP16 AMP skipped 23 alpha-zero updates and 19 alpha-0.9 updates, leaving 44,737 and 44,741 completed steps. At CP3 the totals were 29,840 attempts with 15 versus 13 skips. The histories bind matching sampler RNG identity and identical 211-feature mean/std buffers at every epoch. This small completed-step difference is recorded; it does not change the intended target-only contrast.

The trend is consistent with loss of useful signal when current T3 distillation is removed during this warm continuation: alpha-zero falls from 0.67637 at CP1 to 0.66684 at CP5, while alpha-0.9 remains near 0.681. That narrows the next question toward initialization and student optimization, but does not answer it. The inherited parent was already distilled, auxiliary losses remain active, and the comparison does not measure random-initialized student capacity or teacher quality independently. The stronger old-Z4 student is a different teacher/parent/recipe lineage and is not a matched counterexample.

## Limits and validity

All grades use the repeatedly inspected fold-0 development panel. The bootstrap does not include training-seed variation, and public-transfer uncertainty is not estimated. The alpha-zero branch still inherits earlier distillation from its common parent and retains the same auxiliary losses; it therefore tests removing the current cache coefficient, not removing all teacher information. No teacher-family rejection or public score forecast follows.

Each grade and bootstrap ran serially with a 1-CPU, 1.5-GB, 120-second bound; all five commands exited 0. Before grading, the direct user slice showed 9.327 GB current memory, a two-core quota, 18-GB high / 20-GB maximum, no swap, and an idle GPU at 75 MiB. Root's fresh whole-allocation projection was 15.42 GB against the 17-GB working target. A prior controller status display contained stale active-attempt/reservation fields; root verified the relevant units inactive, and direct live-unit/slice observations were used without stopping or reconciling unrelated services.

## Decision and next step

For this already-distilled seed-3 warm parent and current clean-T3 target, retain alpha 0.9 over alpha 0 as the better-supported continuation. Do not promote this to a teacher-family or overall quality conclusion. To distinguish target benefit from student capacity or initialization, the useful follow-up is a matched fresh-initialized or stronger-parent continuation comparison under the same target, fitting membership and update schedule, followed by another source-qualified fold/seed for any promising endpoint.

## Source records

- [Machine-readable grade and ancestry receipt](../../campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_CP3_CP5_PAIRED_GRADE_REVIEW_R1.json), [frozen command contract](../../campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_POSTFIT_EXPORT_GRADE_COMMANDS_R1.json), [CP3 paired bootstrap](../../campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_CP3_COMPONENT_BOOTSTRAP_R1.json), and [CP5 paired bootstrap](../../campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_CP5_COMPONENT_BOOTSTRAP_R1.json).
- [CP3 alpha-zero grade](../../campaign_2026-10-15/student_quality_tradeoff_r1/f0_grade_alpha0_r2/ep4/student_f0_warm_s3_alpha0_r1_p2_envr1.grade.json), [CP3 alpha-0.9 grade](../../campaign_2026-10-15/student_quality_tradeoff_r1/f0_grade_alpha09_r2/ep4/clean_t3_f0_control_s3_a9_t3r5_r2_p2_envr1.grade.json), and [existing CP5 alpha-0.9 grade and parent comparison](../../campaign_2026-10-15/student_quality_tradeoff_r1/CURRENT_T3_PARENT_PAIRED_GRADE_R1.json).
- [Alpha-zero P2/P3 provenance and histories](../../../artifacts/runs/student_f0_warm_s3_alpha0_r1_p3_envr1/provenance.json) and [alpha-0.9 P2/P3 provenance and histories](../../../artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json); raw history hashes are recorded in the machine-readable receipt.
- [Earlier clean-T3 teacher/student bottleneck review](CLEAN_T3_TEACHER_STUDENT_BOTTLENECK_REVIEW_20261015_R1.md) provides the teacher scores, parent ancestry and cross-lineage limits that these matched grades do not resolve.

## Follow-up: in-sample versus exposed evaluation diagnostic

The bounded [streaming diagnostic](../../campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_FIT_EVAL_DIAGNOSTIC_R1.json) joined the frozen `fit.valid` list (1,405 sequences) and excluded evaluation list (468 sequences) to `valid_targets.npz` by its explicit row-group IDs. It used the unchanged official `GlobalAccumulator`, fixed mask and 99-row warm-up, streamed all six FP32 prediction files, recomputed their byte hashes, and reproduced each saved evaluation grade within `2.3e-16` WP. The [diagnostic script](../../campaign_2026-10-15/student_quality_tradeoff_r1/diagnose_alpha0_fit_eval_r1.py) pins the split, target cache, prediction sidecars and metric implementation. No fitting or bootstrap was run.

Alpha-zero minus alpha-0.9 official WP on the fitting-valid sequences is `−0.000871`, `+0.007670`, and `+0.012247` at CP1/CP3/CP5, while the exposed evaluation differences are `−0.003947`, `−0.008744`, and `−0.014313`. At CP5, the raw `|prediction|>2` fraction is higher for alpha-zero (fit-valid T0/T1 `0.0507%/0.0484%`; eval `0.0436%/0.0402%`) than alpha-0.9 (fit-valid `0.00022%/0.00026%`; eval `0.00012%/0.00087%`), though exceedances remain rare. Evaluation target RMS is also higher than fit-valid (`1.077/1.146` versus `1.020/1.032` for T0/T1), so distribution shift is a plausible alternative to an overfitting-only account. This is descriptive evidence from one warm-parent seed on exposed F0; it does not isolate the cause or establish a generalization claim.

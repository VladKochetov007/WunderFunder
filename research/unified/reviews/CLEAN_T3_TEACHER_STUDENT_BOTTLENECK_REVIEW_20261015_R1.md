---
kind: review
status: complete
created_utc: 2026-10-15T11:50:09Z
updated_utc: 2026-10-15T11:50:09Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json
  - research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2/f0_grade_r1/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1.grade.json
  - artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json
  - cache/clean_t3_f0_h390_t3r5_a9_r2/metadata.json
  - research/campaign_2026-10-13/strategy_execution_r1/first_results_analysis_r1/RESULTS_R1.json
---

# Clean T3 teacher and student quality bottleneck review

## Question or claim

Do the clean T3 F0 grades identify teacher weakness, student capacity, or insufficient student fitting as the main limit? The available evidence does not isolate one bottleneck. In particular, the six-epoch T3 student is warm-started from a previously distilled student, so its score cannot by itself establish that the new T3 teacher target rescued a weak student or that the teacher and student have similar ceilings.

## Mechanism and method

I compared primary grade and provenance JSON only; I did not load prediction arrays, cache payloads or model weights, and ran no fit. Scores use `global_wp_clip2_v1` on the repeatedly inspected fold-0 panel: 468 evaluation groups, 141 components, 808,863 selected rows, the frozen split SHA `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`, row mask `is_scored AND row >= 99`, prediction/label clipping at ±2, and original clipped-label magnitude weights.

The clean T3 cache is an un-clipped FP32 arithmetic mean of the three seed-3/4/5 epoch-2 teacher predictions multiplied by `0.8125`. Although its metadata lists source coefficients of `0.1` each, the homogeneous consumer takes the mean and applies the separate `0.8125` multiplier. During student training the loaded teacher is scaled, clipped to ±2, then mixed as 90% teacher and 10% clipped label; the WMSE loss is weighted by clipped-label magnitude and normalized per chunk. This is distinct from official evaluation clipping and weighting.

## Evidence and interpretation

| Endpoint | WP | Interpretation |
|---|---:|---|
| Clean T3 teacher, seed 3, epoch 2 | 0.67898918 | Individual direct teacher grade. |
| Clean T3 teacher, seed 4, epoch 2 | 0.67836691 | Individual direct teacher grade. |
| Clean T3 teacher, seed 5, epoch 2 | 0.67969849 | Individual direct teacher grade. |
| Equal prediction trio | 0.68090199 | +0.00188379 versus the mean of the three individual scores; distinct-seed variation is descriptive on this one F0 fold. |
| Seed-3 warm parent `c20261003_combo_s3_f0@ep3` | 0.68094121 | Same fold-0 score for the exact checkpoint used to initialize the new seed-3 student. |
| Clean T3 student, seed 3, fixed epoch 5 | 0.68115536 | +0.00021415 versus its warm parent and +0.00025337 versus the equal teacher trio, as raw point differences only. |
| Clean T3 student, seed 4, fixed epoch 5 | 0.68125842 | Its same-seed `c20261003_combo_s4_f0@ep3` warm parent scores 0.68102938; raw difference +0.00022904. |
| Separate old-Z4 student, seed 3, fixed epoch 1 | 0.68314181 | Its old-Z4 warm parent scores 0.68282334; paired delta +0.00031847, 95% component interval [−0.00132008, +0.00179901]. This is a different target and parent lineage. |

The clean T3 student is not fresh-initialized. Its provenance names `c20261003_combo_s3_f0@ep3`; the initialization hash equals that parent's epoch-3 checkpoint SHA `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`. Parent and student provenance have identical fold split, ordered train/valid-fit group hashes, 468 evaluation group IDs and normalization-buffer hashes. The new student therefore starts from an already-distilled H256×2 model. That parent itself was a four-epoch FP16/Muon student with 29,828 completed updates, B32/C1000, alpha 0.5 on two older since-targets and auxiliary weights 0.3/0.3. The new child runs six epochs, fixed epoch 5, with 44,741 completed updates and 19 AMP skips, B16/C2000, AdamW/FP16, alpha 0.9 on the homogeneous clean T3 cache and the same auxiliary weights. Its nominal rows per update remain 32,000, but it receives 1.5 times as many optimizer updates. This is a continuation with a changed target and optimizer, not an equal-budget isolated KD ablation.

The parent and student have identical ordered 468-group provenance lists (SHA256 `6effdddc309b1b0007448f4be34f65cad0f595f3ec088e8d66397d49ef5d2010`). The parent prediction sidecar binds its epoch-3 checkpoint SHA above, prediction SHA `aa38f384907e1ad1122e410c2f5a6fb9353da387bbab788b3e9978b05f646d19` and the same split SHA. The historical grade bank does not record inference precision for this parent prediction, so this is not a fully documented precision-matched comparison. No paired parent-versus-student component interval was found. The student grade is the fixed CP5 FP32 export; the small positive raw parent/student deltas should not be called statistically established gains.

The separate old-Z4 result is also not a clean T3 comparator: its provenance initializes from `oldz4_h230_parent_f0_s3_r3@ep3`, uses a distinct eight-source family cache (coefficient sum 0.8125), alpha 0.7, family weight 0.1 and auxiliary weights 0.3/0.3. Its parent and student are directly paired in the historical grade, but that comparison cannot identify the effect of the newer three-teacher homogeneous T3 target.

Restoring the clean T3 cache multiplier from 0.8125 to 1.0 changed the scale by 16/13 while preserving the rest of the seed-3 recipe. Its score was 0.68119497 versus 0.68115536, paired delta +0.00003961 with 95% component interval [−0.00022472, +0.00030143] over the same 141 F0 components. This one-seed result gives no clear evidence that the reduced cache amplitude is the principal bottleneck; it is not a universal rejection of scale variants.

No official student grade at epoch 2 or an intermediate checkpoint was found. Fit histories state evaluation was withheld; only fixed epoch-5 outputs were scored. Thus the existing artifacts do not provide a measured two-versus-six-epoch quality curve.

## Limits and validity

All evidence is from repeatedly inspected F0 and conditions on trained endpoints. The three teacher fits are distinct seeds but share the same fold population; the two clean T3 students share one teacher cohort and the same evaluation panel. The student-vs-trio score gap is confounded by warm initialization and additional optimization. The student-vs-parent delta additionally changes optimizer, batch/chunk geometry, distillation target and fitting budget. There is no same-parent, same-budget label-only continuation control, and no clean random-initialized student with the six-epoch budget. Therefore teacher quality, student capacity, and the causal contribution of new KD remain unresolved. The teacher trio and student endpoints are close in observed F0 WP, but this is not evidence that their score ceilings are equal; it is one exposed-panel comparison between different fitted lineages and budgets.

The stronger old-Z4 model should remain visible as a separate historical endpoint. Its paired estimate versus its already-strong parent is uncertain and its lineage/target differs; neither it nor the clean T3 student licenses a statement that one teacher family is intrinsically weaker.

## Decision and next step

Prioritize a small, GPU/TPU-budgeted attribution comparison over more CPU-hosting or native-infrastructure work: from the same seed-specific warm parent, run a matched six-epoch label-only continuation and the current clean T3 KD continuation with identical optimizer, sampler, rows/update and precision. Preserve the exact warm-parent checkpoint as a scored endpoint and grade both children with paired component resampling. If resources permit, add a fresh-initialized KD child as a separate initialization contrast; do not conflate it with the warm-start contrast. Keep the scale-1 result as a near-null, exposed-F0 observation. Confirm promising outcomes on another source-qualified fold and distinct fitting seed before promotion.

## Source records

- [Equal clean T3 teacher trio grade](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json) and [single seed-3 teacher grade](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_s3_terminal_r1/F0_FIRST_GRADE_R1.json).
- [Historical F0 grade bank containing the exact clean student warm-parent score](../../campaign_2026-10-06/bandit_continuation/prediction_bank/GRADE_FOLD0FULL_20261010T004533Z_f3ec9c4efd4d7305.json), [parent provenance](../../../artifacts/runs/c20261003_combo_s3_f0/provenance.json), and [parent prediction sidecar](../../../cache/pred_c20261003_combo_s3_f0_valid.json).
- [Clean T3 seed-3 student grade](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2/f0_grade_r1/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1.grade.json), [seed-3 student provenance](../../../artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json), [seed-4 student grade](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/seed4_confirmation_r1/f0_grade_r1/clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1.grade.json), and [seed-4 provenance](../../../artifacts/runs/clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1/provenance.json).
- [Clean T3 cache metadata](../../../cache/clean_t3_f0_h390_t3r5_a9_r2/metadata.json), [cache target construction](../../../src/wnn/final_distillation_cache.py), and [trainer scale/clip/blend/loss implementation](../../../scripts/train_rnn.py).
- [Paired target-amplitude analysis](../../campaign_2026-10-13/strategy_execution_r1/first_results_analysis_r1/RESULTS_R1.json) and [scale-1 student grade](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/grade_r1/clean_t3_f0_amplitude1_s3_a9_t3r5_r3_p3.grade.json).
- [Separate old-Z4 student and warm-parent paired grade](../../campaign_2026-10-06/bandit_continuation/runtime/receipts/oldz4-r5-f0-s3-student-export/F0_GRADE_PAIRED_R5.json) and [old-Z4 student provenance](../../../artifacts/runs/oldz4_h230_student_f0_s3_r3/provenance.json).

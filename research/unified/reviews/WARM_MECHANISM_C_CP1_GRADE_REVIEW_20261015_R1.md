---
kind: review
status: measured_exposed_endpoint
created_utc: 2026-10-15T17:46:54Z
updated_utc: 2026-10-15T17:46:54Z
hypothesis_ids: [H-FROZEN-PARENT-ANCHOR-20261015, H-KD-CONTINUATION-VALUE-20261015]
experiment_ids: [E-WARM-MECHANISM-C-CP1-20261015-R1]
evidence_paths:
  - research/campaign_2026-10-15/student_quality_tradeoff_r1/WARM_MECHANISM_F0_S3_C_R3_CP1_PAIRED_GRADE_R1.json
  - research/campaign_2026-10-15/gru_parent_growth_r1/C_FIT_TERMINAL_STATE_GATE_R1.json
  - research/campaign_2026-10-15/gru_parent_growth_r1/C_CP1_EXPORT_RESULT_R1.json
---

# Warm-mechanism C checkpoint-1 grade review

## Question or claim

This review records the official exposed-F0 grade for the two-epoch C continuation and checks its checkpoint, update counters, feature-only export, and paired comparisons. It does not isolate the causal value of T3 targets: C changes the original parent through continued fitting with T3 targets and auxiliary losses.

## Evidence and interpretation

The candidate is `warm_mechanism_f0_s3_c_r3@ep1`, where `ep1` is the zero-based second-epoch checkpoint from a fixed two-epoch run. It starts from `c20261003_combo_s3_f0@ep3`, uses the homogeneous clean-T3 F0 cache with alpha 0.9, and inherits that checkpoint's preprocessing. The run uses seed 3, B16/C2000, FP16 training, AdamW at `1e-4` with OneCycle, propensity weighting, and `aux_next=0.3` plus `aux_struct=0.3`. The prediction export is FP32. The fixed CP1 checkpoint was not selected by validation.

The terminal trainer counters show 14,920 optimizer attempts, 14,914 completed updates, and six AMP-skipped updates, split as 7,460 attempts / 7,457 completed / three skips in each epoch. The source-bound terminal state gate confirms the training state points to `next_epoch=2`. The candidate provenance records the same frozen F0 split and fitting membership as the full continuation parent; evaluation rows are excluded from direct fit membership. The export receipt binds the checkpoint SHA `78357eca2aec188e227544d46887935265720c35e99a1437269b952d01eb88c2`, FP32 prediction SHA `c45c8d260d011fcc0c5900a4d11ee10d8261799716775ee6b26dd2cb6dd84482`, full-valid shape `[1873,20000,2]`, and ordered fold-0 valid row groups 0–1872. The unchanged official grader accepted the checkpoint, prediction sidecar, split, and 468-group evaluation membership.

On `global_wp_clip2_v1`, C scored 0.6812265132 over 808,863 scored rows. Against its exact original starting checkpoint `c20261003_combo_s3_f0@ep3` (0.6809412068), the paired component-bootstrap delta is +0.0002853064 with 95% interval [-0.0022089120, +0.0024805067] over 141 components and 2,000 resamples (seed 271828). The interval crosses zero. This is a comparison of the parent with a two-epoch continuation, not a matched T3-versus-label-only contrast.

As historical context, C is +0.0000711529 against `clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1@ep5` (0.6811553603), with interval [-0.0014604201, +0.0015390783]. That A9 reference has a different continuation horizon and recipe, so this secondary comparison is not a matched control. Its prediction sidecar does not record inference precision; the C export is FP32.

## Decision and limits

Keep this as an exploratory F0 endpoint and as the C arm reference for the frozen-parent S comparison. The point estimate does not show a reliable gain over the original parent, and it cannot establish that T3 distillation caused the change. The matched alpha-zero trajectory, a distinct fitting seed, another fold, and serving validation remain separate evidence. This panel is repeatedly exposed development evidence, not a fresh holdout or public-score forecast.

## Reproduction

The grader ran unchanged `scripts/score_fold0_panel.py` at SHA `0f67ac06cd769a582901b854737528c89e9a7738c3057acf68d7f99eddbfa2d0` under the campaign limiter with one CPU, a 1.5 GB memory cap, and a 120-second deadline. It scored the candidate against both the exact original parent and the explicitly non-matched A9 CP5 reference. The machine grade is linked above.

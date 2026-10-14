---
kind: review
status: complete
created_utc: 2026-10-14T02:33:16Z
updated_utc: 2026-10-14T02:33:16Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-STUDENT-FRONTIER-F0S3-H256L2-FIT-R2-TERMINAL-20261014, E-STUDENT-F0S3-H256L2-GRADE-20261014-R3]
evidence_paths:
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h256_l2_r2_fit_validation.json
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h256_l2_r2_fold0.json
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h256_l2_r2_fold0_validation.json
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/results/control_prediction_grade_r3/README.md
  - ../../campaign_2026-10-06/bandit_continuation/runtime/receipts/oldz4-r5-f0-s3-student-export/F0_GRADE_SINGLE_R5.json
---

# H256×2 F0 seed-3 control grade review

## Question or claim

Does the first H256×2 seed-3 grade bind the frozen consumer recipe, clean ancestry, full-valid prediction export and official fold-0 scoring path? What does its comparison with the stronger historical F0 score support?

## Findings

**Pass as the first fixed-budget control grade; no source, membership or metric blocker was found. It is exposed fold-0 development evidence and does not promote this student over the stronger historical result.** The result records `global_wp_clip2_v1` WP **0.6697776418601018** (`t0=0.661477105018867`, `t1=0.6780781787013365`) on **468 evaluation groups, 141 connected components and 808,863 selected rows**. It scores fixed candidate `student_frontier_f0s3_h256_l2_r2@ep0`, at weight 1.0, with no reference contrast. The validation receipt pins the same result, split `09263ad4…f9316a3d`, target archive `dbb7a1a4…5622955`, and grade report `524080a7…e98fdde`.

The grade's checkpoint SHA `1baa5835…7f006e6` matches the independently validated fit receipt and the prediction sidecar. The full-valid prediction file was streamed and hashed under the 500 MB reviewer cap: SHA `ede80d9f…96d34da1`; its NumPy header is little-endian FP32 with shape `(1873, 20000, 2)`. The sidecar binds run, epoch 0, checkpoint, split and prediction hash; its ordered row groups are exactly 0–1872. Recomputed ordered-group SHA `008f12e2…9b2acdf7` matches the grade report. The profile/export receipts describe FP32 causal inference, fresh state per sequence and carried state only within each sequence. The complete export is not a one-batch profile.

The frozen consumer contract SHA is `9476373f…a02e768`; the reviewed fit receipt SHA is `1042e31a…fecdf43`. They bind the H256×2 one-epoch FP32 recipe, B8/C2000, fixed checkpoint 0, 14,920 completed updates, no AMP skips, 14,920 recorded gradient-stat steps with zero nonfinite entries, and 16 finite terminal tensors. Four of 11,940 source-qualified fitting groups are dropped by the declared `drop_last`; 11,936 sequence exposures remain. Per-step loss finiteness is not serialized and is not claimed.

The clean teacher target is the contract's homogeneous F0 T3 target: epoch-2 teacher checkpoints for seeds 3, 4 and 5, unweighted FP32 mean, coefficient multiplier 0.8125, and training target `0.1 * clipped_label + 0.9 * clipped_teacher`. The fit receipt binds cache metadata SHA `1e4c5c49…f2be3dc8`, the source-qualified train/valid-fit arrays (10,535 and 1,405 groups), and the 468-group F0 evaluation split. The split is defined on source-qualified connected components and declares no evaluation groups in fitting. The propensity source is `c20260929_prop_f0/model.pt`, SHA `b243b9d4…30b517a7`, with provenance SHA `f6d6cf68…79e812`; the propensity cache and sidecar hashes are recorded in the fit receipt. The initial identity check compared the separate `model_ep3.pt` artifact; that was the wrong path, not an ancestry mismatch. Normalization comes from the pinned clean F0 H256×2 seed-3 epoch-5 checkpoint, and only its 211-feature mean/std buffers were copied; no learned weights were loaded.

The grade receipt's source hashes match the pregrade manifest for `score_fold0_panel.py`, `evaluate_bandit_continuation.py`, `gmetric.py`, `predict_run.py` and the frozen student predictor. The scorer checks the fold-0 split, fit/component separation, prediction sidecar/checkpoint/split bindings, full-file digest, row order, shape and finite predictions before invoking the official `global_wp_clip2_v1` implementation. The 12-file delivery manifest rehashes cleanly. I did not rerun the full scorer: the original grade attempt reports 1,352,044,544 bytes peak RAM, above this review's 500 MB cap. Reviewer work was limited to small JSON/source inspection and bounded streaming hashes; this is an artifact review, not an independent numerical regrade.

The grade records **468 groups and 141 components**. The frozen grader-input manifest (`GRADER_INPUTS_F0_R2.json`, SHA `c64ba6a3…e609e5`) has a wording error in its basis string, which calls the panel “468-component.” Its pinned split, scorer check and actual grade receipt all use 468 evaluation groups; the actual number of connected components is 141. Preserve the frozen manifest and use the correct units in summaries.

The historical `oldz4_h230_student_f0_s3_r3@ep1` grade is **0.6831418133898598**, about **0.0133642** higher, on the same metric, split, target payload, 468 groups, 141 components and selected rows. Keep it visible as the stronger local reference. It is not a matched architecture/control comparison: its provenance says H256×2 despite the legacy `h230` run label, and records two epochs, B16, FP16, a `bd26_h230_w384_family_v1` source cache and distillation alpha 0.7. The new control is one epoch, B8, FP32 and the homogeneous clean-T3 target above. The historical run attempted the same 14,920 optimizer updates but completed 14,913 with 7 AMP skips; the new fit completed all 14,920. No paired interval or matched treatment effect is reported here.

## Evidence and interpretation

| Item | New control | Historical stronger reference |
|---|---|---|
| Metric and panel | `global_wp_clip2_v1`, fold 0 | Same |
| Evaluation units | 468 groups / 141 components | 468 groups / 141 components |
| Score | 0.6697776418601018 | 0.6831418133898598 |
| Architecture from provenance | H256×2 | H256×2 (legacy run name contains `h230`) |
| Fit recipe | 1 epoch, B8, FP32, clean homogeneous T3 target | 2 epochs, B16, FP16, source-family cache, alpha 0.7 |
| Update accounting | 14,920 attempts and completions | 14,920 attempts; 14,913 completions, 7 AMP skips |

These are repeatedly inspected development-panel values, not a public forecast or pristine holdout. The new score anchors the current student ladder's fixed-budget control. It does not establish a cross-shape ranking, a gain over the historical student, convergence, or deployment-precision quality. The historical result remains a valuable stronger endpoint; shared fold exposure means the same panel cannot provide fresh confirmation. No confidence interval was calculated; a component-level comparison should resample the 141 connected components, not treat the 468 groups as independent units.

## Limits and validity

This review checked identities and scoring semantics, not the model's predictions against labels independently. The existing grade was produced by the pinned scorer and passed its postgrade identity validator. The task's 500 MB support limit ruled out replaying a scorer whose recorded peak is approximately 1.35 GB. All fold-0 results are exposed and repeatedly inspected. Other student shapes, distinct fitting seeds, broader fold evidence and any candidate-selection decision remain separate work.

## Decision and next step

Accept the artifact as the first H256×2 matched-ladder control grade. Keep the historical 0.6831418 endpoint visible and do not select the new control from this result alone. Compare complete fixed endpoints across the authorized shapes, then use the broader source-qualified fold/seed evidence required by the current submission instruction. Any future summary should call this a 468-group, 141-component development panel.

## Source records

- [Frozen student recipe contract](../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json) and [independent R2 consumer review](CPU_STUDENT_FRONTIER_CONSUMER_R2_REVIEW_20261014_R1.md).
- [Fit validation receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h256_l2_r2_fit_validation.json), [grade](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h256_l2_r2_fold0.json), [grade validation](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h256_l2_r2_fold0_validation.json), and [prediction/grade delivery README](../../campaign_2026-10-14/cpu_student_frontier_r1/results/control_prediction_grade_r3/README.md).
- [Pregrade input check receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/results/GRADER_INPUTS_CHECK_F0_R3.json), which pins the frozen grader-input manifest by SHA and the current scorer/predictor source hashes.
- [Current scorer](../../../scripts/score_fold0_panel.py), [official metric](../../../src/wnn/gmetric.py), and [historical grade](../../campaign_2026-10-06/bandit_continuation/runtime/receipts/oldz4-r5-f0-s3-student-export/F0_GRADE_SINGLE_R5.json).
- [Experiment ledger](../../experiments.jsonl): `E-STUDENT-FRONTIER-F0S3-H256L2-FIT-R2-TERMINAL-20261014` and `E-STUDENT-F0S3-H256L2-GRADE-20261014-R3`.

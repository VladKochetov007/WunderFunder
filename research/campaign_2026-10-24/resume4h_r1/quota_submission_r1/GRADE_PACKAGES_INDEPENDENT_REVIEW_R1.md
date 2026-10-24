---
kind: review
status: complete
created_utc: 2026-10-24T12:49:47Z
updated_utc: 2026-10-24T12:49:47Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/grade_packages.py
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/GRADE_PACKAGES_RECIPE_DRAFT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/REPLAY_PLAN_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/SELECTION_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/CROSS_REPLAY_GATE_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/baseline_REPLAY_RECEIPT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/s3_REPLAY_RECEIPT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/s3_s4_REPLAY_RECEIPT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/F0_EXPLORATORY_RECIPE_R2.json
  - src/wnn/gmetric.py
  - src/wnn/evaluation.py
---

# Independent static review — quota package grade R1

## Question and scope

Does the final grader and recipe score the three frozen TL3 package replays against the same 64 source-qualified feature-only inputs, using the official metric and fixed comparisons, while enforcing identity checks before target values are read?

Reviewed [grade_packages.py](grade_packages.py) at SHA256 `d01b4e21ae5ffb6aa6974e4c43209de0f47b51f4b30c9db0886865f86d5a73b1` and [recipe](GRADE_PACKAGES_RECIPE_DRAFT_R1.json) at SHA256 `ab4c66354e33e9d13b82af63f0c0df4f534e6aa2de12852788aace6310d251cc`. Read the admitted replay plan, deterministic selection, cross-replay gate and three small replay receipt JSON files. Compared score construction with the frozen F0 scoring basis and canonical `gmetric.py` / component-bootstrap implementation.

No prediction arrays, target cache, validation labels, model weights or native libraries were opened. I did not run imports, the grader, tests or native code. This is a source/metadata review, not a quality grade.

## Findings

**Scoring and prelabel gate: pass.** The recipe pins the unchanged `global_wp_clip2_v1` contract: clip targets and predictions to ±2, weight by absolute clipped target magnitude, and select the official score mask with row index ≥99. The grader independently requires this exact metric dictionary and mask, and scores F0 and F2 separately. Its mask is `official_mask & (rows >= 99)`; the replay source also checks that `need_prediction` is false exactly for rows 0–98 and true thereafter.

Before opening the target cache, `prelabel_gate()` checks the recipe SHA, Python 3.11 executable, source pins, frozen selection, admitted replay plan and split hashes, package ZIP/native hashes, each replay receipt, all three prediction headers and payload hashes, identical ordered feature/need hashes across all 64 sequences, and a separate root grade-admission hash. It then checks the target-cache size and SHA. During scoring it rechecks the target SHA and hashes all three prediction payloads again before writing a create-only result. The cross-replay gate and current receipts report identical ordered input signatures; the grader repeats that comparison from the receipts.

The selection is the frozen deterministic 32-component sample from F0 plus 32 from F2, one full 20,000-row sequence per component. The code verifies split membership, component/sequence identity and uniqueness. It emits exactly the two primary contrasts (S3−baseline and S3/S4−baseline) and the secondary S3/S4−S3; no coefficient search or fit is allowed. The canonical `GlobalAccumulator` performs pooled clipped weighted Pearson scoring. Component intervals use the canonical centered paired moments/bootstrap within each fold, 2,000 draws, fixed seed, and are correctly labelled conditional on fixed replay predictions. The equal-weight fold mean is descriptive only.

The target reader opens only the compact row-group index in the `.npz`, extracts `y.npy` and `mask.npy` to physical scratch in 1-MiB chunks, validates NPY headers/shapes, and memory maps those arrays. Static inspection of the grader and its scoring imports found no Torch import. This design is consistent with a bounded 400,000,000-byte worker; actual peak RSS and runtime still require the admitted execution receipt.

Exposure is carried into the output recipe: the supplied validation set is repeatedly exposed, H390 S3/S4 were fit on train plus validation, and original TL3 ancestry is unknown. F0/F2 are samples, not clean holdouts. The recipe claims zero new fits, no competition upload, no public forecast, and no coefficient selection. These limits prevent the grade from being presented as clean CV or public-transfer evidence.

## Execution condition and disposition

The grader validates resource values in the recipe/admission but does not create its own CPU, memory, swap or wall-time cgroup. The run must use `scripts/limited` with the root-approved 400,000,000-byte memory cap, one CPU, zero swap and 120-second runtime. Root confirmed it will use that wrapper. The README's original direct-Python example should be replaced with the wrapped invocation before handoff.

The separate root `GRADE_ADMISSION_R1.json` and its SHA must be present at launch; the current recipe intentionally stores the admission path and takes the expected admission SHA as a CLI argument to avoid a hash cycle. With that admission and the stated resource wrapper, I found no scoring or identity blocker. No target labels were read during this review, and no quality claim follows from it.

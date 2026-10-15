---
kind: review
status: complete
created_utc: 2026-10-15
updated_utc: 2026-10-15
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h512_l1_r2_fit_validation.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h512_l1_r2_fold0.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h512_l1_r2_fold0_validation.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h512_l1_r2_prediction_profile_validation.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H512L1_R6_SEQUENCE.json
---

# H512×1 F0 seed-3 result review

## Question or claim

Does the completed H512×1 screen support advancing this shape over the matched H256×2 F0 seed-3 control? This is a source/receipt audit, not an independent numerical regrade or a winner/serving certification.

## Mechanism and method

The frozen R2 consumer contract is `9476373fc3c32aea2ae48e9a2c48d211f0f152d13d086fa68efd1e8b1a02e768`; the exact training source is actor commit `f0b5b6eb469d960708b590e7dd4814e3bd2daa19`, with `train_rnn.py` SHA-256 `952939f15e1dfa84bdee20295a4c64c1fbe4fc174d72e76a6bf574d2fdaeba01` and `student_frontier.py` SHA-256 `a747431d9bc2acc25f3f053bdfe7b5cfd64c4a211ecdd75002faa54785062e07`. The worktree was clean at review; the latter module hash also matches the current main-tree file.

The recipe fixes FP32, seed 3, F0, one epoch, fixed epoch-0 checkpoint, batch 8 and 2,000-row training chunks. It records 1,492 batches, 11,936 exposed sequences, four dropped tail sequences and 14,920 completed chunk updates, with zero AMP skips and zero nonfinite recorded gradients. The contract binds the homogeneous clean-T3 cache and its exact metadata/payload hashes, split and source-qualified memberships, three H384 epoch-2 T3 teachers (seeds 3/4/5), propensity sidecar, and the H256×2 epoch-5 control used only for common input mean/std buffers. The cache target is `0.8125 × mean(three epoch-2 teacher outputs)`; the student target is `0.1 × clipped label + 0.9 × clipped homogeneous teacher`. Training uses the frozen chunk-normalized weighted MSE and propensity/true-mask weights. The fit loads no teacher or control learned weights. Metadata review found the 468 evaluation row groups disjoint from the 1,405 fit-valid groups for the student and all three teachers; all carry the same F0 split-manifest hash.

The B16/C4000 FP32 resource profile covered 20,000 rows and reported finite output, zero OOM counters, exit 0, 1.10 GB CUDA allocated and 1.35 GB reserved. Its observed-VRAM sample is only 79.7 MB and should not be represented as a true peak. The complete valid export contains 1,873 sequences; the frozen official grade uses the 468-group excluded F0 panel. The fit validator checked finite terminal tensors and recorded gradient statistics, but the trainer does not serialize a per-step loss-finiteness check.

## Evidence and interpretation

The official `global_wp_clip2_v1` grade is 0.6732377615 (t0 0.66541725, t1 0.68105827), versus 0.6697776419 for the matched H256×2 control. The paired difference is +0.00346012; a 2,000-draw component bootstrap over 141 components reports 95% interval [+0.00200307, +0.00490448]. The candidate and reference use the same split-manifest SHA `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`, target SHA, evaluation membership and official metric. Their checkpoint, full-prediction and report hashes are recorded in the grade and validation receipts; no large model or prediction payload was opened or rehashed for this review.

The receipts support advancing H512×1 as a **first-screen candidate** over this matched H256×2 control on the exposed F0 seed-3 panel. The bootstrap interval is conditional on these fixed predictions; it does not cover training-seed or teacher variation, architecture-screen selection, fold shift, or public transfer. This is one student fit seed on previously inspected development evidence, not independent confirmation. The grade files label the panel “fold0 excluded-468”; this review uses that canonical label.

## Limits and validity

The lineage and consumer contract pins agree across the fit validator, run provenance, export profile and grade. The candidate uses the same exposure, target treatment and training-update budget as the matched screen control. The historical H230×2 0.68314181 result remains visible but is not a matched comparator for this recipe and is not displaced by this result. No native Q8 grade, cross-fold result, distinct student-seed replication, package/ABI check, or measured competition-CPU serving cost is established here. No independent metric recomputation was run.

## Decision and next step

Keep H512×1 in the active candidate set for confirmation; do not call it the overall winner or a submission-ready endpoint. The next quality evidence should include a distinct student fitting seed and another source-qualified fold with compatible fold-specific teacher/cache/normalization ancestry, while keeping this exposed F0 result labeled as development evidence. Measure the complete deployed endpoint's native quality, package constraints and competition-CPU runtime before any serving promotion.

## Source records

- [Frozen R2 consumer contract](../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json)
- [H512×1 fit validation](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h512_l1_r2_fit_validation.json)
- [H512×1 F0 grade and paired contrast](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h512_l1_r2_fold0.json)
- [Grade validation](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h512_l1_r2_fold0_validation.json)
- [B16 prediction-profile validation](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h512_l1_r2_prediction_profile_validation.json)
- [R6 sequence and root admission](../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H512L1_R6_SEQUENCE.json), [admission receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/ROOT_H512L1_R6_SEQUENCE_ADMISSION_R1.json)
- Exact frozen trainer source: `/home/vlad/development/wunder/worktrees/student-frontier-sequence-20261014/scripts/train_rnn.py` at commit `f0b5b6eb469d960708b590e7dd4814e3bd2daa19`.

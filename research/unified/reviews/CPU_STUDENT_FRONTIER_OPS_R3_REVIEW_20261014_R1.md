---
kind: review
status: complete
created_utc: 2026-10-14T02:07:00Z
updated_utc: 2026-10-14T02:07:00Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-STUDENT-F0S3-H256L2-20261014-R2]
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H256L2_R3_SEQUENCE.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/PREDICTION_PROFILE_F0_EIGHT_SHAPES_B16_R2.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/R3_SEQUENCE_STATIC_PREFLIGHT_FINAL.json
---

# F0 student-frontier R3 post-fit operations review

## Question or claim

Does the frozen R3 operations packet safely bind each admitted student shape from the fixed F0 seed-3 fit through a resource-checked full-valid export and the official exposed-fold grade, while reusing the exact H256×2 control attempt?

## Findings

**Source and sequence review: GO.** The previous R2 sequence had only an H256×2 inference profile before exporting the other seven shapes. The R3 successor gives each shape its own B16 profile, terminal profile validator and profile-validation receipt as the prerequisite contract for its full-valid export. Every plan stops after one shape; root retains its planned per-shape admission control.

The frozen source is operations commit `6987476ed7c2b4f08cc6c12923bea19918ba4e46` (clean worktree), with trainer/consumer source commit `f0b5b6eb469d960708b590e7dd4814e3bd2daa19` (clean worktree). The H256×2 plan references the active `student-frontier-fit-f0s3-h256l2-b8-c2000-fp32-r2` attempt and identical worker command; the sequence runner reuses that attempt record rather than duplicating its fit. Scientific trainer settings are common across the cohort; the other seven fit commands change only run name and allowlisted hidden/layer dimensions. The active control has its existing timeout/resource environment preserved.

The fit assertion checks the fixed epoch-0 checkpoint and provenance, exact split and source-qualified memberships, sampler/order and dropped-tail accounting, cache/teacher/target and propensity ancestry, shared normalization-source identity, source hashes, 14,920 completed updates, 11,936 exposed sequences, zero AMP skips, zero nonfinite recorded gradient-stat steps, and finite floating-point terminal tensors. It correctly says that per-step loss finiteness is not separately serialized; no claim is made that this gate checks it.

Each profile uses the production predictor at B16, 20,000 rows, C4000, feature-only inputs and FP32 without autocast; it writes a distinct profile-only output for the first 16 valid target groups. The validator binds that output to the shape, epoch-0 checkpoint, frozen sources and split, checks row order, float32 shape `(16, 20000, 2)`, byte count and finiteness, then requires terminal controller success, known observers, observed VRAM below 5 GB, allocator reservation within 3.5 GB and zero OOM-counter deltas. The next full export uses that validator receipt as its controller contract. Full exports are separate FP32 feature-only files for all 1,873 valid groups; the MAIN scorer consumes `RUN@ep0` under `global_wp_clip2_v1`, with no reference for the control and the fixed H256×2 control as reference for the other seven. The grade validator preserves the exposed-development label.

Resource settings are internally consistent: 5 GB host reservation for fit/profile, 6 GB for full export, 1.5 GB for fit/scorer validation and 0.5 GB for bounded profile/input checks. Each export reserves 310 MB output space for the 299,680,000-byte prediction payload plus header and requires 5 GB free disk after that step's output reserve. Eight retained predictions add about 2.4 GB; the per-step free-space guard rechecks before every launch and fails closed if the reserve no longer fits. These reservations are admission bounds, not evidence that a full export has already completed within them.

## Evidence and interpretation

The independently rerun `check_student_frontier_r3_sequences.py` passed under the 1 CPU / 500 MB task cap and found all eight plans, exact trainer/profile argv, separate profile and quality outputs, a matching profile-receipt dependency for every export, and the 5/6 GB reservations. A separate bounded SHA-256 pass confirmed the current `cache/valid_targets.npz` matches the frozen grader-manifest hash `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`. The fold-0 split and MAIN predictor, scorer, evaluator and metric hashes also match the manifest. No GPU, fit, export, grade or provider action was run by this reviewer.

Key pins: H256×2 plan `0dfc4b090967a68942d16142cd79054f0937a065cb7858263b4a22e4ce977ab4`; shared B16 profile contract `ed4ba5be2b74c4487d7f773d9e8ac0b86c334b611654b127a2e91484a67019e2`; static preflight `e39a79a3b6add53176d6ce38e426bcc17e43b699d93d593cf8a3ad464f946305`; fit validator `ccaf5b030f53eb2ff9b6aedec8243b2190d9e2e98d711b921d5ed419d635347a`; prediction-profile validator `1f978b71d9a2ad459dbec511531e8362a8b980cd9402ab43ecce509a8b813ad2`.

## Limits and validity

This is a source/recipe review, not a terminal resource or quality result. At review time the H256×2 fit was the already-running attempt; its terminal counters, actual B16 profile, full export and grade remain for root to inspect. The static preflight does not load model/data payloads. The profile validator checks the target archive's declared hash binding and byte size; this review independently hashed the current archive, and the official MAIN scorer rehashes it at grading. Fold 0 is exposed development evidence, not a fresh holdout or public-score forecast.

## Decision and next step

Root may use the reviewed R3 plan as the source-bound fit→assert→profile→export→grade sequence, subject to its separate live resource admission and terminal checks. A passed profile admits only the matching shape's export; it does not admit the next shape or establish model quality. The user-authorized eight-shape campaign remains the scientific scope; this review adds no user-approval gate.

## Source records

- Frozen [H256×2 R3 sequence](../../../../worktrees/student-frontier-ops-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H256L2_R3_SEQUENCE.json) and [eight-shape B16 profile contract](../../../../worktrees/student-frontier-ops-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/PREDICTION_PROFILE_F0_EIGHT_SHAPES_B16_R2.json), from the clean operations commit above.
- [Static preflight receipt](../../../../worktrees/student-frontier-ops-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/R3_SEQUENCE_STATIC_PREFLIGHT_FINAL.json), [fit assertion](../../../../worktrees/student-frontier-ops-20261014/scripts/validate_student_frontier_fit.py), [profile assertion](../../../../worktrees/student-frontier-ops-20261014/scripts/validate_student_frontier_prediction_profile.py), and [sequence checker](../../../../worktrees/student-frontier-ops-20261014/scripts/check_student_frontier_r3_sequences.py).
- Frozen [R2 consumer contract](../../../../worktrees/student-frontier-sequence-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json); official [grader-input manifest](../../../../worktrees/student-frontier-ops-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/GRADER_INPUTS_F0_R2.json).

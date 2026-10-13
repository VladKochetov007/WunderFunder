---
kind: review
status: complete
created_utc: 2026-10-13T12:14:00Z
updated_utc: 2026-10-13T12:14:00Z
hypothesis_ids: [H-TEAM-NATIVE-20261013]
experiment_ids: [E-TEAM-MEMBER-BANK-20261013-R1]
evidence_paths: [GPU_PROFILE_ADMISSION_R1.json, GPU_PROFILE_ADMISSION_R2.json, MEMBER_BANK_ADMISSION_R1.json]
---

# Member exporter review and measured admission

## Question or claim

Can the imported individual NPZs be exported faithfully through our FP32 GPU model path for diagnostic diversity analysis? This is a different numerical endpoint from native dynamic-int8/Q8 serving.

## Mechanism and method

Actor `team_gpu_member_builder` wrote the exporter in an isolated worktree, commit4aea6128. Independent `team_score_critic` reviewed actual imported NPZ shapes and source without running CUDA. The model uses112 raw plus99 causal event features, stored normalization, two-layer/one-layer GRUs as declared, and separate projection-skip/GELU heads. Merged512-wide heads are reconstructed before strict state loading. Complete sequences reset event and recurrent state; hidden state carries across500-row chunks. The first99 outputs are zeroed after recurrence advances, and grading separately joins official masks.

## Evidence and interpretation

Eight focused tests, Ruff checks and bounded CLI help passed in the actor workspace. The critic found no functional blocker for actual-shape profiling. It found missing helper/imported-source hashes in the original receipt; commit2916a834 corrected this before execution. Main integrations are12bcdf13 and202c659a. Imported model/feature/data files matched main byte-for-byte. Root subsequently extended batch choices to16 after a measured B1 check; source digests, rather than those commit labels alone, bind the profiles.

| Actual merged-head profile | Complete groups | Forward/export seconds | Peak cgroup GB | Observed VRAM GB |
|---|---:|---:|---:|---:|
| B1, chunk500 | 2 | 1.427 | 2.702 | 0.339 |
| B16, chunk500 | 16 | 1.834 | 2.972 | 1.173 |

Measured B16 process peak RSS is1,510,104 KiB; CUDA allocated/reserved peaks are616,432,640/916,455,424bytes. Both attempts exited0. Initial hashing/import/loading time is excluded from the exporter timer but included in the durable attempt. These are short resource/throughput observations, not full-job ETAs.

On the first two matched complete sequences, B1 versus B16 maximum absolute prediction difference is9.09e-7, RMSE1.36e-7; sequence/input hashes agree and outputs are finite. This is expected FP32 batching roundoff, not native quantizer parity. The reviewer's verbal599 MB full-array estimate was an arithmetic error:1873×20000×2×4=299,680,000 payload bytes, plus the NPY header.

## Limits and validity

The NPZ does not encode activation names; this diagnostic assumes the inspected project GELU architecture and disclaims native fast-activation parity. Final-fit/unknown validation overlap still prohibits clean-CV ranking. The reported memory cap contains cache as well as working arrays; admission uses actual parent charges plus unused reservations and an operating margin.

## Decision and next step

Root admitted fourteen serial, deduplicated B16 member exports under one GPU lock, one CPU and3.6 GB RAM after the actual B16 profile. [Admission](MEMBER_BANK_ADMISSION_R1.json) records a16.574 GB conservative parent total, below17 GB, and freezes the plan/source. [Plan](MEMBER_BANK_PLAN_R1.json) declares nine fixed diagnostic mixtures; it does not optimize weights on this fitted data. Independent grader implementation and terminal collection remain owned. No new ZIP follows from this profile.

## Source records

- [B1 receipt](profiles/E_weights_2_fp32_b1_profile_r1/RESULT.json), [B16 receipt](profiles/E_weights_2_fp32_b16_profile_r2/RESULT.json), [batching comparison](GPU_BATCH_PROFILE_COMPARISON_R1.json).
- [B1 durable attempt](../../campaign_2026-10-06/bandit_continuation/runtime/attempts/team-fp32-e-w2-b1-profile-r1-20261013.json), [B16 durable attempt](../../campaign_2026-10-06/bandit_continuation/runtime/attempts/team-fp32-e-w2-b16-profile-r2-20261013.json), [member bank attempt](../../campaign_2026-10-06/bandit_continuation/runtime/attempts/team-fp32-member-bank-r1-20261013.json).
- [Exporter](../../../scripts/export_team_member_fp32.py), [loader](../../../scripts/team_member_export_lib.py), [bank index](member_bank/README.md).

---
kind: profile
status: preflight_passed_gpu_admission_pending
created_utc: 2026-10-14T00:00:00Z
updated_utc: 2026-10-14T00:00:00Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-STUDENT-F0-S3-B16-LADDER-PROFILE-20261014-R1]
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_B16_LADDER_R1_CONTRACT.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_B16_LADDER_R1_PREFLIGHT.json
---

# Source-qualified B16 student-ladder profile

This packet verifies a bounded resource and update-rate profile for the fixed eight-shape F0 student ladder. It does not fit a full epoch, grade predictions, select an architecture, or claim quality. The profile harness can select a contracted subset of shapes and a resource batch prefix; the matched training recipe remains the explicit B16, one-epoch, 7,460-update recipe in the contract.

The immutable clean cache is `cache/clean_t3_f0_h390_t3r5_a9_r2`, metadata SHA256 `1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8`. Its historical producer contract remains H256×2, B16/C2000, six epochs, fixed epoch 5 and alpha .9. The separate consumer recipe pins all eight shapes, one common 211-feature normalization copied only from the H256 control mean/std, FP32 with TF32 disabled, the same propensity/mask/label weighting and homogeneous teacher targets, AdamW, OneCycleLR, and one fixed one-epoch/7,460-update comparison budget for later fits. No producer metadata is changed.

The [source-qualified correction](PROFILE_SOURCE_QUALIFICATION_CORRECTION_R1.md) retracts the original B1 sampler-order claim. This packet validates seed 3, epoch 0, all 11,940 fitting target rows under source-local `(source, row_group)` keys, the first 16 source-qualified Parquet identities, and both full target payload hashes. The fixed B16 epoch retains 746 batches (11,936 sequence exposures) and drops the same four tail rows for every shape: train groups 3449, 2425, 879 and 7224.

The [CPU preflight receipt](PROFILE_B16_LADDER_R1_PREFLIGHT.json) passed the unchanged strict producer validator, source/cache/teacher/propensity/normalization pins, full sampler digests, and source-qualified row mapping. It made no fit or quality evaluation. The loaded production B16 tensors are 150,080,000 bytes. Cgroup memory was 1,478,656,000 bytes under the 1.5 GB CPU cap; process max RSS was 1,777,696,768 bytes.

B16 resource admission is still pending: H640×1 and H448×3 need actual B1 capacity measurements before a larger common fit batch is chosen. The separate [staged largest-shape B1 packet](PROFILE_H640L1_H448L3_B1_C2000_FP32_RESOURCE_R1_README.md) contains the first source-qualified batch row, exact resource command, and its own CPU preflight. The B16 ladder command below remains prepared, unlaunched, and is not evidence that the largest shapes fit this envelope.

The B16 worker is configured for a 3 GB host reservation, a 3.5 GB PyTorch CUDA allocator cap, and a 5 GB observed-VRAM ceiling. It records CUDA allocated/reserved peaks, observed VRAM, chunk update times, and one-sequence causal export time. Three measured chunk updates per shape are resource probes, not architecture-quality evidence.

The result file is [PROFILE_B16_LADDER_R1_RESULT.json](PROFILE_B16_LADDER_R1_RESULT.json) and will be written only after root admission. Root owns the GPU lock and launch; this actor has not invoked CUDA for this profile.

The exact B16 profile command is:

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-frontier-20261014
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_B16_LADDER_R1_CONTRACT.json"

"$MAIN/scripts/bandit_resources.py" --root "$WT" \
  --state "$MAIN/research/campaign_2026-10-06/bandit_continuation/runtime" \
  launch --attempt student-frontier-profile-b16-c2000-fp32-r1 \
  --kind gpu --cpus 1 --reserve-bytes 3000000000 \
  --profile 'clean-F0 source-qualified first B16 batch, 150.08 MB returned tensors, all 8 shapes; 3 FP32 chunk updates and one B1 causal export each; allocator cap 3.5 GB; observed VRAM ceiling 5 GB' \
  --contract "$CONTRACT" --python "$MAIN/.venv/bin/python" -- \
  timeout --signal=TERM --kill-after=10s 10m \
  /usr/bin/env WNN_PROJECT_ROOT="$MAIN" PYTHONPATH="$WT/src" \
  WNN_GPU_MEMORY_BYTES=3500000000 \
  "$MAIN/.venv/bin/python" "$WT/scripts/profile_student_frontier_batch.py" \
  --data-root "$MAIN" --contract "$CONTRACT" \
  --output "$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_B16_LADDER_R1_RESULT.json" \
  --profile-batch-size 16 --shape-indices 0 1 2 3 4 5 6 7 --execute
```

The [profile contract](PROFILE_B16_LADDER_R1_CONTRACT.json) SHA256 is `c6cc21c25ffec2996ed6a42368cdcd5751d0223060df0e799e1e7b87be4bf515`; the source-qualified batch harness SHA256 is `0bb78a9a3ec449d214d3ce293f051ae7e4528f8dfe065901ff27cbf02e0a0d5b`. A source change requires a regenerated preflight before admission.

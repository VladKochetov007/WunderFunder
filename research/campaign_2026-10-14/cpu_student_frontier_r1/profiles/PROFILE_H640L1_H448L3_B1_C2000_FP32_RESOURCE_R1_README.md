---
kind: profile
status: preflight_passed_gpu_admission_pending
created_utc: 2026-10-14T00:00:00Z
updated_utc: 2026-10-14T00:00:00Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-STUDENT-F0-S3-LARGEST-B1-RESOURCE-PROFILE-20261014-R1]
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H640L1_H448L3_B1_C2000_FP32_RESOURCE_R1_CONTRACT.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H640L1_H448L3_B1_C2000_FP32_RESOURCE_R1_PREFLIGHT.json
---

# Largest-shape B1 resource profile

This is a resource-only CUDA profile for H640×1 and H448×3, selected to measure both the largest hidden width and the largest approximate recurrent activation volume before choosing a common training batch size. For each shape it makes three FP32 updates over 2,000-row chunks from one real sequence, then measures one B1 causal export. It does not fit a full epoch, estimate quality, rank the two shapes, or extrapolate a full-fit duration.

The profile uses the first source-qualified row of the exact seed-3, epoch-0 B16 sampler batch: `(valid, row_group=37, seq_ix=44346154)`. The harness maps all 16 sampler positions to `(source, row_group, seq_ix)`, checks the source-specific target mapping for every fit row, invokes the unchanged strict clean-T3 cache validator, and binds the three teacher models and F0 propensity model/caches. It copies only the 211-feature mean/std from the fitting-only H256 control checkpoint. Normalization identity is the shared mean SHA256 `bb85c72b87656979f97c1786400aec6bc9e43acd337078000eb482c09c5a2da8` and std SHA256 `b730600851e52e246fc638230844d6e4f2828e4972b45db1f0128cce4e4454c5`.

The [CPU preflight](PROFILE_H640L1_H448L3_B1_C2000_FP32_RESOURCE_R1_PREFLIGHT.json) passed strict producer payload hashes, exact source-qualified sampler and Parquet sequence identities, cache ancestry, and common normalization checks. It made no CUDA update. Cgroup memory was 1,499,860,992 bytes under the 1.5 GB CPU support cap; process max RSS was 1,766,436,864 bytes. A preceding [H640×1-only preflight](PROFILE_H640L1_B1_C2000_FP32_RESOURCE_R1_PREFLIGHT.json) passed the same checks before H448×3 was added to the contracted profile scope.

The B1 profile worker reserves 3 GB host memory, sets `WNN_GPU_MEMORY_BYTES=3500000000`, and stops above 5 GB observed VRAM. It loads the actual first B16 batch on host, but transfers only its first source-qualified sequence to each model. Root owns the GPU lock and admission; no CUDA measurement exists yet. The result and copied terminal controller receipt will be indexed here after the admitted job finishes.

The exact root-owned command is:

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-frontier-20261014
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H640L1_H448L3_B1_C2000_FP32_RESOURCE_R1_CONTRACT.json"

"$MAIN/scripts/bandit_resources.py" --root "$WT" \
  --state "$MAIN/research/campaign_2026-10-06/bandit_continuation/runtime" \
  launch --attempt student-frontier-profile-b1-h640l1-h448l3-c2000-fp32-r1 \
  --kind gpu --cpus 1 --reserve-bytes 3000000000 \
  --profile 'source-qualified first row of clean-F0 B16; profile B1 H640x1 and H448x3 only; 3 FP32 chunk updates plus one B1 causal export per shape; no quality or full-fit extrapolation; allocator cap 3.5 GB; observed VRAM ceiling 5 GB' \
  --contract "$CONTRACT" --python "$MAIN/.venv/bin/python" -- \
  timeout --signal=TERM --kill-after=10s 10m \
  /usr/bin/env WNN_PROJECT_ROOT="$MAIN" PYTHONPATH="$WT/src" \
  WNN_GPU_MEMORY_BYTES=3500000000 \
  "$MAIN/.venv/bin/python" "$WT/scripts/profile_student_frontier_batch.py" \
  --data-root "$MAIN" --contract "$CONTRACT" \
  --output "$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H640L1_H448L3_B1_C2000_FP32_RESOURCE_R1_RESULT.json" \
  --profile-batch-size 1 --shape-indices 5 7 --execute
```

The profile contract SHA256 is `f8175f31f4747bcc78b8ad01f01ba904987ea056f8188fd9f7bb1b47c141304b`; the shared batch-profile source SHA256 is `0bb78a9a3ec449d214d3ce293f051ae7e4528f8dfe065901ff27cbf02e0a0d5b`. Any source or contract change requires a new preflight before launch.

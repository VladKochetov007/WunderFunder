# CPU student frontier actual-data profile review R1

## Decision

Conditional GO for one bounded H192×2, batch-1, 2,000-row FP32 update as a resource and finite-update smoke only. Do not treat it as a fit, quality comparison, capacity result for batch 16, or clearance for the eight-shape ladder. Root admission must independently enforce the declared 3,000,000,000-byte host reservation, one CPU thread, no swap, a finite runtime, the 3,500,000,000-byte PyTorch allocator cap, and the single local-GPU reservation.

## Frozen input reviewed

The review examined the uncommitted student delivery worktree at `7800d30473e3810e12d9ba22596f9fd6df5626ca`:

- `scripts/profile_student_frontier.py`
- `research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H192L2_B1_C2000_FP32_CONTRACT_R1.json`
- `research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H192L2_B1_C2000_FP32_PREFLIGHT_R1.json`
- production data, split, sampler, feature, model and resource primitives called by the script

The profile contract SHA256 is `70bee8ee2a34b53c02680b931360a521eec5a4ade6ac81666fe9c675bf6a59fc`. All ten source hashes in the contract recompute exactly, including the profile script. The CPU preflight receipt repeats the contract and source hashes and records one sampled row group, setup time and memory; it is not GPU or bounded-worker evidence.

## Checks that pass

- Seed 3 first permutes the clean F0 train membership, appends valid fitting membership, then applies `EpochOrderSampler` at epoch 0. That matches the production trainer's epoch-0 order for `train+valid`; the sampled group 37 is in the source-qualified train fit set, while held-out evaluation membership is excluded.
- The sampled teacher row is selected through the producer's ordered row-group index. The script binds the exact split and metadata hashes, historical producer `student_contract`, target arithmetic and declared target-file digest, teacher checkpoint/provenance hashes, and the H256×2 epoch-5 control checkpoint/provenance hashes.
- The copied normalization vectors are only the reference model's mean/std buffers; their hashes match the contract. The student is otherwise freshly initialized at seed 3.
- The update uses the existing 211-wide `RNNModel` path, recomputed event features, zero initial state, FP32 with TF32 disabled, 99-row warmup, label-magnitude weighted clipped-target MSE, gradient clipping and one AdamW step. It checks finite loss/gradients/parameters and records CUDA allocator and host memory observations. No evaluation, predictions or checkpoint are written.

## Limits and execution conditions

This standalone profile checks the pinned metadata and target manifest claims, array shape/dtype/size, row-group ordering, and finiteness of one sampled target row. It does **not** call the unchanged legacy `validate_distillation_cache` path or recompute the target array's full 1,685,600,128-byte SHA256. That is acceptable only for this explicitly non-quality resource smoke; the later trainer consumer must pass the strict legacy validator and verify the full target payload before any fit. Do not use a successful profile to certify cache integrity.

The profile is batch 1 and one shape, not the earlier proposed batch-16/eight-shape cohort. Its VRAM and host results cannot be extrapolated to those shapes or full training. The CPU preflight reported peak RSS about 1.595 GB and cgroup current charge about 1.407 GB, but its receipt says `cuda_profile_requested: false`; neither value is evidence that the GPU update fits under a 3 GB host cgroup.

Before launch, compare the input contract SHA to the frozen value above, set `WNN_GPU_MEMORY_BYTES=3500000000`, enforce the 3 GB `MemoryMax`/no-swap/one-thread bounded runtime, and monitor total device memory and parent aggregate headroom. The script should add a stat-size limit before hashing and parsing `metadata.json` (current file: 323,404 bytes) so a replaced oversized file fails before processing. No profile, GPU action, fit, score or runtime claim was produced by this review.

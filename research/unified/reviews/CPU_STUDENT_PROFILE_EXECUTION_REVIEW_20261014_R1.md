# CPU student frontier actual-data profile review R1

## Decision

Conditional GO for one bounded H192×2, batch-1, 2,000-row FP32 update as a resource and finite-update smoke only. Do not treat it as a fit, quality comparison, capacity result for batch 16, or clearance for the eight-shape ladder. Root admission must independently enforce the declared 3,000,000,000-byte host reservation, one CPU thread, no swap, a finite runtime, the 3,500,000,000-byte PyTorch allocator cap, and the single local-GPU reservation.

## Frozen input reviewed

The review examined the uncommitted student delivery worktree at `7800d30473e3810e12d9ba22596f9fd6df5626ca`:

- `scripts/profile_student_frontier.py`
- `research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H192L2_B1_C2000_FP32_CONTRACT_R1.json`
- `research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H192L2_B1_C2000_FP32_PREFLIGHT_R1.json`
- production data, split, sampler, feature, model and resource primitives called by the script

The first profile contract SHA256 was `70bee8ee2a34b53c02680b931360a521eec5a4ade6ac81666fe9c675bf6a59fc`. It was superseded during review by a frozen, hash-consistent profile packet; the final packet is recorded below.

## Checks that pass

- Seed 3 first permutes the clean F0 train membership, appends valid fitting membership, then applies `EpochOrderSampler` at epoch 0. That matches the production trainer's epoch-0 order for `train+valid`; the sampled group 37 is in the source-qualified train fit set, while held-out evaluation membership is excluded.
- The sampled teacher row is selected through the producer's ordered row-group index. The script binds the exact split and metadata hashes, historical producer `student_contract`, target arithmetic and declared target-file digest, teacher checkpoint/provenance hashes, and the H256×2 epoch-5 control checkpoint/provenance hashes.
- The copied normalization vectors are only the reference model's mean/std buffers; their hashes match the contract. The student is otherwise freshly initialized at seed 3.
- The update uses the existing 211-wide `RNNModel` path, recomputed event features, zero initial state, FP32 with TF32 disabled, 99-row warmup, label-magnitude weighted clipped-target MSE, gradient clipping and one AdamW step. It checks finite loss/gradients/parameters and records CUDA allocator and host memory observations. No evaluation, predictions or checkpoint are written.

## Limits and execution conditions

This standalone profile checks the pinned metadata and target manifest claims, array shape/dtype/size, row-group ordering, and finiteness of one sampled target row. It does **not** call the unchanged legacy `validate_distillation_cache` path or recompute the target array's full 1,685,600,128-byte SHA256. That is acceptable only for this explicitly non-quality resource smoke; the later trainer consumer must pass the strict legacy validator and verify the full target payload before any fit. Do not use a successful profile to certify cache integrity.

The profile is batch 1 and one shape, not the earlier proposed batch-16/eight-shape cohort. Its VRAM and host results cannot be extrapolated to those shapes or full training. The CPU preflight reported peak RSS about 1.595 GB and cgroup current charge about 1.407 GB, but its receipt says `cuda_profile_requested: false`; neither value is evidence that the GPU update fits under a 3 GB host cgroup.

Before launch, compare the input contract SHA to the final frozen value below, set `WNN_GPU_MEMORY_BYTES=3500000000`, enforce the 3 GB `MemoryMax`/no-swap/one-thread bounded runtime, and monitor total device memory and parent aggregate headroom. No profile, GPU action, fit, score or runtime claim was produced by this review.

## Frozen delta re-review

The post-review packet is frozen in student worktree commit `1ebd3991c601585834549eb35cdbb4cb902974a8`. Its script SHA256 is `2ab9985b711fe687743b9676fba4899191d398fbbfb9c266c209113a7b8b9b01`, contract SHA256 is `ae88afc3dac2969600811d239544e33fa32e009f896f30052471b6b37a6982f4`, and regenerated CPU-only preflight SHA256 is `a9510e15d52167413e2ce37778ebf6c467f0bb860c3589146f69ede2e338c542`. The contract pins the exact script hash, and the preflight receipt repeats both current hashes.

The delta adds a 512,000-byte pre-read bound for the current 323,404-byte cache metadata file, verifies the sampled parquet `seq_ix` against the source-qualified split manifest, and checks `nvidia-smi` usage before and after the update as well as rejecting PyTorch reserved allocator memory above 3,500,000,000 bytes or observed GPU use above 5,000,000,000 bytes. These checks preserve the conditional GO for the one-sequence smoke. They do not verify the full distillation target file or support broader capacity claims.

Residual bounded-reader note: the profile and existing `load_campaign_split` reader read the pinned 1,388,846-byte split JSON without a pre-read size cap. This is not a blocker for the exact current source and this narrow smoke, but the size should be explicitly capped in a later hardening pass if strict bounds are required for every metadata read.

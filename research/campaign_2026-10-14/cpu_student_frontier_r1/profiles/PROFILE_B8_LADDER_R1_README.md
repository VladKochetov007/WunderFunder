---
kind: profile
status: cpu_preflight_passed_root_gpu_admission_pending
created_utc: 2026-10-14T00:00:00Z
updated_utc: 2026-10-14T00:00:00Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-STUDENT-F0-S3-B8-LADDER-PROFILE-20261014-R1]
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_B8_LADDER_R1_CONTRACT.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_B8_LADDER_R1_PREFLIGHT.json
---

# Source-qualified B8 student-ladder resource profile

This packet is a bounded actual-data resource/update-rate profile for all eight requested GRU shapes. It runs three FP32 C2000 optimizer updates and one B1 causal prediction per shape. It does not train a full epoch, grade outputs, rank architectures, or claim quality. Root owns the single-GPU lock and launch.

The profile contract binds the immutable clean homogeneous cache metadata (`1e4c5c49…f2be3dc8`), exact teacher and propensity ancestry, the source-qualified F0 split, the 211-feature mean/std copied from the clean H256×2 seed-3 control, the producer's historical H256×2/B16/C2000/six-epoch/epoch-5/alpha-.9 contract, and the current code hashes. The historical producer metadata and payloads are not rewritten. The harness calls the existing strict cache validator and requires both full target payload hashes in `verified_files` before profiling.

The fixed sampler is seed 3, epoch 0, with 11,940 fitting groups from the train groups followed by valid-fit groups. B8 retains 1,492 full batches, 11,936 sequence exposures and 14,920 chunk updates; the same four tail groups are omitted for all shapes. The actual first 16 sampler positions are checked against `(source, row_group, seq_ix)` identities, then the first eight source-specific cached target rows are checked against the production loader. This explicitly preserves the overlapping numeric row-group ID 37 in `train` and `valid`; the first actual batch begins with `valid/37`, `seq_ix=44,346,154`.

The [CPU preflight](PROFILE_B8_LADDER_R1_PREFLIGHT.json) passed the strict target hashes, cache/teacher/propensity ancestry, source-qualified order and target-row checks, and common normalization identity. It performed no optimizer update or quality evaluation. Its recorded cgroup charge was 1,499,885,568 bytes under the 1.5 GB support-task cap; returned B8 batch tensors total 75,040,000 bytes. The provisional allocator admission basis scales the measured largest-shape B1 reservation of 327,155,712 bytes linearly by eight, giving 2,617,245,696 bytes against the explicit 3.5 GB allocator cap. This leaves 882,754,304 bytes of allocator headroom; the actual B8 profile must measure the real peak before any full fit is admitted.

The profile order is H192×2, H256×1, H256×2, H384×2, H512×1, H640×1, H512×2, H448×3, chosen to probe increasing recurrent capacity before the two- and three-layer models. The user's eventual fitting order remains control H256×2, compact pair H256×1/H192×2, then the other five; the probe order does not select models. The run records allocated/reserved CUDA peaks, observed VRAM, host charge, finite update evidence, update timings and one-sequence causal export timings. Any linear duration estimate is update-only and excludes the full epoch's data loading/export overhead.

The exact profile launch packet is:

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-frontier-b8-20261014
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_B8_LADDER_R1_CONTRACT.json"

"$MAIN/scripts/bandit_resources.py" --root "$WT" \
  --state "$MAIN/research/campaign_2026-10-06/bandit_continuation/runtime" \
  launch --attempt student-frontier-profile-b8-c2000-fp32-r1 \
  --kind gpu --cpus 1 --reserve-bytes 3500000000 \
  --profile 'clean-F0 source-qualified first B8 batch, 75.04 MB returned tensors, all 8 shapes; 3 FP32 chunk updates and one B1 causal export each; allocator cap 3.5 GB; observed VRAM ceiling 5 GB' \
  --contract "$CONTRACT" --python "$MAIN/.venv/bin/python" -- \
  timeout --signal=TERM --kill-after=10s 10m \
  /usr/bin/env WNN_PROJECT_ROOT="$MAIN" PYTHONPATH="$WT/src" \
  WNN_GPU_MEMORY_BYTES=3500000000 \
  "$MAIN/.venv/bin/python" "$WT/scripts/profile_student_frontier_batch.py" \
  --data-root "$MAIN" --contract "$CONTRACT" \
  --output "$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_B8_LADDER_R1_RESULT.json" \
  --profile-batch-size 8 --shape-indices 0 1 2 3 4 5 6 7 --execute
```

The contract SHA256 is `4f2b59b7579e3387c0379860c665adfab2b4622acd8cbf09605202222064a6d5`. The batch harness SHA256 is `07bf5efa592ce4c599c7653d41b31cc4668a7b2d657af5eac7e3b0e02d698679`; its bound `train_rnn.py` SHA256 is `3ef9f2faae8eab69f3e31bf84a7411e5eda0aaab1110066a0264a26f79294d2d`. The profile is valid only for these frozen hashes and the declared B8 resource scope.

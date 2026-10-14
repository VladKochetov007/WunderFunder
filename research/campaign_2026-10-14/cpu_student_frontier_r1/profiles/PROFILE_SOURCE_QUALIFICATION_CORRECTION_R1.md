---
kind: incident
status: complete
created_utc: 2026-10-14T00:18:16Z
updated_utc: 2026-10-14T00:18:16Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H192L2_B1_C2000_FP32_GPU_R1.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_H192L2_B1_C2000_FP32_ATTEMPT_R1.json
---

# B1 student profile source-qualification correction

## Question or claim

The successful H192×2/B1 CUDA probe is valid as a one-sequence resource and finite-update smoke, but its original claim that it used the first seed-3 sampler row is retracted.

## Mechanism and method

The frozen B1 harness converted the sampler's combined train-plus-valid dataset positions to bare row-group integers, then classified the selected integer as `train` whenever it appeared in `split.train_groups`. Train and validation Parquet row-group namespaces overlap. For seed 3, epoch 0, batch 16, `EpochOrderSampler` emits positions `[10562, 8725, 6305, 1586, 2134, 11477, 8128, 1560, 2922, 8340, 4913, 2805, 4517, 9259, 2632, 2203]`. The first combined position is **10,562** and maps to **(`valid`, 37, `seq_ix=44346154`)**. The harness instead loaded **(`train`, 37, `seq_ix=7598721`)**, because 37 also occurs in the train namespace. This was a source-qualification error in the profile harness, not a cache producer mutation.

The exact first 16 source-qualified identities, checked against the split manifest and each source Parquet `seq_ix`, are: `(valid,37,44346154)`, `(train,9839,1997333359)`, `(train,1639,342781553)`, `(train,7977,1629092802)`, `(train,7548,1546977674)`, `(valid,1263,1466505621)`, `(train,3788,774192677)`, `(train,9480,1930716618)`, `(train,2478,514227336)`, `(train,1094,230774826)`, `(train,7529,1544636377)`, `(train,10227,2077965454)`, `(train,2131,444849898)`, `(train,9134,1864890861)`, `(train,1832,382374263)`, and `(train,5224,1074328294)`. The B16 `drop_last` tail is the four fitting rows `(train,3449)`, `(train,2425)`, `(train,879)`, and `(train,7224)`; every ladder shape must use that same seeded order and omit that same tail.

## Evidence and interpretation

Keep the original [B1 result](PROFILE_H192L2_B1_C2000_FP32_GPU_R1.json), [terminal controller receipt](PROFILE_H192L2_B1_C2000_FP32_ATTEMPT_R1.json), [worker log](PROFILE_H192L2_B1_C2000_FP32_ATTEMPT_R1.log), and frozen source unchanged. The update actually used train group 37 (`seq_ix=7598721`), which is a valid F0 fitting sequence in its own right. Its 1.86 s update and measured memory remain evidence for that declared one-sequence resource smoke. It does not prove batch membership/order, batch-16 capacity, or architecture quality.

The [B1 contract](PROFILE_H192L2_B1_C2000_FP32_CONTRACT_R1.json) and preflight also contained the incorrect exact-order association. Their bytes remain historical evidence; this correction supersedes that interpretation. The corrected batch-16 harness maps every sampler position to a `(source, row_group)` pair, checks target-row indexes separately per source, verifies all 16 `(source, row_group, seq_ix)` identities and the first16-position vector against this actual F0 order, and checks the entire cache row set under its source key before profiling B16.

## Limits and validity

The B1 resource result is not invalidated as a one-sequence update: it loaded a real clean-F0 fitting sequence, used the pinned shared H256 normalization, and completed one finite parameter-changing FP32 update. Its source/order linkage is not valid. Numeric row-group IDs are not globally unique across `train.parquet` and `valid.parquet`; future code and receipts must carry the source with every row-group ID.

## Decision and next step

Do not use this smoke as recipe-order verification or extrapolate its resource envelope to B16. The corrected B16 harness now passes the source-qualified first-16 test, complete source-local cache-row mapping, and unchanged strict cache payload validation in its [CPU preflight](PROFILE_B16_LADDER_R1_PREFLIGHT.json). This confirms the consumer boundary only; the B16 CUDA profile remains unlaunched pending a measured resource envelope for H640×1 and H448×3. The [staged B1 profile packet](PROFILE_H640L1_H448L3_B1_C2000_FP32_RESOURCE_R1_README.md) covers both shapes under root admission and makes no quality claim.

## Source records

The immutable profile artifacts are listed in [the profile index](README.md). The original profile was committed as `1ebd3991c601585834549eb35cdbb4cb902974a8`, with the terminal attempt indexed in `4fae363377658488e95d50d916d6bb95240d2136`.

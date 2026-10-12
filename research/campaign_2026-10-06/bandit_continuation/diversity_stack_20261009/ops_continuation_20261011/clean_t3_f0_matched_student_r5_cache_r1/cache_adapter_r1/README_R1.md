# Clean F0 T3 row-group cache adapter R1

This independent adapter fixes the current `_r5` producer interface for the matched clean F0 student. It leaves the frozen R3 cache builder and predictor untouched. The frozen consumer already has the narrow `cleanfold_homogeneous_strong_t3_v1` cache profile; this adapter emits that metadata shape.

The teacher fit and prediction orders have different meanings. Each seed's `provenance.json` has its own randomized `fit_row_groups.train` order and fingerprint. Each prediction sidecar has the actual array's `row_groups` order. The adapter validates each order against its own binding, requires the source-qualified member sets to equal F0 FIT_TRAIN, and joins predictions by row-group ID. It never treats a `train.parquet` ID as a `valid.parquet` ID.

For valid predictions, the inputs are the existing full-valid feature-only arrays with 1,873 source groups. The adapter verifies their sidecars and selects exactly the 1,405 F0 FIT_VALID IDs from `valid_targets.npz`'s `row_group` member; all 468 evaluation IDs are rejected from outputs. It reads no target-label archive member. Training order comes from the analogous `train_targets.npz` `row_group` member.

The cache target is computed in bounded chunks as float32 addition of seed 3, then seed 4, then seed 5; division by float32 3; then multiplication by float32 0.8125. It applies no target clipping. Inputs are read-only NPY memmaps and outputs use `open_memmap`; a unique `.partial` directory is retained on failure, metadata is written last, and the directory is renamed only after all outputs are complete.

## Current evidence

`CACHE_ADAPTER_CONTRACT_R1.json` binds the student recipe, split, frozen consumer, unchanged legacy builder, three teacher checkpoints/provenance files, and the validated s3 FIT_TRAIN export. At this revision, actual s3 FIT_TRAIN plus all three full-valid sidecar/array pairs are available. FIT_TRAIN exports for seeds 4 and 5 are still pending. The actual input-validation receipt hashes the available producer arrays; it is not a built teacher cache.

The F0 split has 10,535 FIT_TRAIN groups from `train.parquet`, 1,405 FIT_VALID groups from `valid.parquet`, and 468 evaluation groups from `valid.parquet`. FIT_VALID and evaluation are disjoint. The s3 terminal receipt confirms all 421,400,000 values in its completed FP32 `[10535,20000,2]` export are finite and binds its output hash to the actual predictor sidecar.

No teacher target cache has been materialized and no student fit has been launched. Do not run the adapter with `--build` until all three FIT_TRAIN exports are present and root supplies a release receipt with schema `wnn.clean-t3-f0-cache-adapter-release.r1`, `cache_materialization_authorized: true`, the exact adapter-contract SHA256, and output root `cache/clean_t3_f0_h390_t3r5_a9_r1`.

## Bounded validation commands

Run the synthetic and actual producer tests with the explicit project CPU/RAM boundary:

```sh
WNN_T3_ARTIFACT_ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack \
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=500000000 WNN_RUNTIME_MAX=120s \
scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/cache_adapter_r1/test_clean_t3_f0_homogeneous_cache_r1.py
```

Revalidate and record current producer hashes before any future cache materialization:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=500000000 WNN_RUNTIME_MAX=120s \
scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/cache_adapter_r1/build_clean_t3_f0_homogeneous_cache_r1.py \
  --contract research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/cache_adapter_r1/CACHE_ADAPTER_CONTRACT_R1.json \
  --project-root /home/vlad/development/wunder/wnn_connectome_starterpack \
  --receipt research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/cache_adapter_r1/ACTUAL_INPUT_VALIDATION_R1.json
```

These are inspection commands. The unique cache output root is `[10535,20000,2]` train and `[1405,20000,2]` valid, both FP32. Materialization remains a later, separately admitted action.

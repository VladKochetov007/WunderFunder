---
kind: operations
status: complete
created_utc: 2026-10-17T01:21:41Z
updated_utc: 2026-10-17T01:21:41Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - STORAGE_RETENTION_AUDIT_20261017_R1.md
  - cache_prune_manifest_20261017_r1.json
  - ../submissions/README.md
  - ../models/huggingface/README.md
---

# Old prediction-cache prune — 2026-10-17

## Result

Removed 463 regular prediction arrays from the checkout's top-level `cache/` directory. The selection was exact: `cache/pred_*.npy` files modified before 2026-10-10 UTC, excluding a small retained set for the baseline, the H390 seed pair in the October 12 submission candidate, the local Z4 and clean-T3 controls, one F3 teacher, and one heterogeneous-teacher example. One old sparse incomplete `.partial` output was also removed. No source Parquet, model checkpoint, submission ZIP, research record, score sidecar, Kaggle object, or Hugging Face object was deleted.

The removed arrays represented **187,431,899,264 bytes (187.43 GB) of logical file size** and **187,433,500,672 bytes by the sum of per-file allocated-block metadata**. `/home` had 25,922,260,992 bytes free before the operation and 137,162,244,096 bytes at the post-cleanup check: **111.24 GB more available by `df`**. The difference from summed file blocks is expected on this Btrfs volume, where shared extents, hard links, and retained references can prevent one deleted path from reclaiming its apparent per-file allocation. The post-operation `df` value is the practical result; the prior audit could not inspect Btrfs snapshot references.

The exact path, size, modification time, inode, and link-count manifest is [cache_prune_manifest_20261017_r1.json](cache_prune_manifest_20261017_r1.json). It records the retained prediction paths, every removed array, the one incomplete file, and the execution recovery. There were no remaining deletion errors. The exact retained set is:

- `pred_baseline_train.npy` and `pred_baseline_valid.npy` as reference controls.
- `pred_vt_all_h390_ds_homogeneous_s3_r3_export_r1_valid.npy` and seed 4's corresponding output as distinct submitted-endpoint seeds.
- `pred_oldz4_h230_student_f0_s3_r3_valid.npy` as a strong local Z4-family result.
- The seed-3 and seed-4 clean-T3 F0 control outputs as the independent-seed comparison.
- `pred_clean_t3_f3_muonall_h384_e3_s3_r5_valid.npy` as a cross-fold example.
- `pred_h230_hetero_s3_ep1_20261008_v1_valid.npy` as one distinct heterogeneous-family example; its existing symlink alias remains valid.

Eight of the nine retained arrays still have adjacent JSON sidecars; the legacy baseline training prediction has none. Where a removed prediction had a JSON sidecar, that small sidecar was retained, along with the archived score and provenance records. The original source files `datasets/train.parquet`, `datasets/valid.parquet`, and `datasets/valid_mask.parquet` remain present and unchanged. Frozen submission packages and the private model archive remain available; the archive is documented in the [HF model README](../models/huggingface/README.md). The [submission index](../submissions/README.md) keeps the public outcomes and package ancestry.

## Retention interpretation

The deleted files were old derived prediction outputs, not original competition data. They were outside the currently retained strong endpoint examples and were not inputs to an active local WNN job at cleanup time. Their small metadata sidecars and archived score/provenance records were left in place. This is a storage-pruning decision, not a claim that every older prediction was scientifically useless or that every one can be regenerated without locating its exact checkpoint and recipe. The immutable package and source-data protections from the [storage-retention audit](STORAGE_RETENTION_AUDIT_20261017_R1.md) still apply.

The cleanup ran with low CPU and I/O priority. The first pass removed 118 files, then an over-strict equality check between nanosecond filesystem mtimes and microsecond-formatted manifest timestamps skipped unchanged files. Recovery verified each remaining path against its frozen inode, byte size, and age before unlinking it. The final manifest preserves the skipped-path list and records completion; no retained prediction or non-cache file was part of that recovery set.

No Kaggle deletion, Hugging Face mutation, or upload occurred.

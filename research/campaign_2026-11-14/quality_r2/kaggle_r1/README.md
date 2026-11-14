# HOLD — legacy Q14 Kaggle packet (R1)

**Not admitted; do not push or launch.** The frozen PR49 Muon factory does not pass `strict_fp32=True` for `--precision fp32`; the optimizer state uses BF16 Newton–Schulz. The in-flight archive is preserved as unlaunched legacy evidence. R2 requires root's corrected source pin and new six-fit matched panel.

The previous launch instructions below are historical only and are disabled by `push_private.sh`.

Frozen four-fit panel: alpha `.9` control versus alpha `.9` plus `--distill-boot-h 8`, matched seeds 5 and 6, using the exact PR49 source tree at `/home/vlad/development/wunder/research_tmp/pr49_wt` (commit recorded in [BUILD_RECEIPT_R1.json](BUILD_RECEIPT_R1.json)). Training arguments are copied from the root quality-r2 `run_fit.sh` / teacher-track `recipe_args.sh`; only run name, seed, and the declared treatment differ. Fixed checkpoint is epoch 3 of 4. Full official valid is evaluation and remains exposed development evidence.

## Portability and lineage

- The unchanged chronological split is SHA-256 `6d23a1b76f66d8001dac805d2b218a3b139e4eebe555f0c9b96fe1263ce186f0` (10,325 train / 282 excluded / 1,873 valid). Its eight source identities and original footer metadata are sealed in `portable_inputs.json`; the worker requires byte, full-source SHA-256, parquet footer, and original split identity to match before fitting.
- Raw `train.parquet` and `valid.parquet` mount from private `vladkochetov07/wunder-funder`; full hashes are train `4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81` and valid `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`.
- The current `cache/train_targets.npz` (1,909,515,814 B; SHA `d56140eb6946107cfc4fc18af75e57ce8e1d1444939c2a2d3188afe051314a4a`) and exact `tt29rep_fbc1k_s5` train prediction cache, sidecar, `model.pt`, and run provenance are sealed into the private packet. The PR49 producer binding passes locally.
- `cache/valid_targets.npz` mounts from private `wnn-c20261004-fold0-assets` only after exact SHA verification (`dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`). That asset's train-target cache (`ead1a05e…`) is stale and is deliberately not mounted.
- The immutable trainer and model modules come from the PR49 worktree. The pinned `kaggle_traced_train.py` support file is included from the root checkout. [kaggle_strict_traced_train.py](kaggle_strict_traced_train.py) applies strict FP32 compatibly: SM 7.5 T4 uses legacy TF32-off flags where the newer cuDNN RNN control is absent, newer RNN IEEE control is set when available, unexpected devices without that control fail closed, and autocast is explicitly disabled around the trainer. It exports each worker's device capability and effective flags as a declared artifact.

## Resource and launch boundary

Each worker is limited to 2 CPU cores, 8 GB host RAM, 3 GB operating GPU memory, 5 GB GPU stop, and at most 2 hours. Those per-worker timeouts do not enforce the total quota by themselves: the sealed kernel bootstrap also has one shared 13,500-second wall deadline from setup through artifact finalization. With two T4s the absolute device-time bound is 7.5 GPU-hours; with one T4 it is 3.75. The live Kaggle quota was 30.00 GPU-hours before launch.

The generated private bundle and upload directory are in `/home/vlad/development/wunder/research_tmp/q14_kaggle/upload_dataset`. Build command:

```bash
python research/campaign_2026-11-14/quality_r2/kaggle_r1/build_packet.py
```

Publish the private bundle and start its four workers with one command:

```bash
bash research/campaign_2026-11-14/quality_r2/kaggle_r1/push_private.sh
```

Kaggle's runtime split loader rechecks parquet footer identity, including the frozen source mtime, in addition to the full content hash. If the Kaggle mount changes that identity, workers fail before fitting; the local source identity is not weakened to accommodate a mount mismatch.

## Current status

Preparation started from [the scratch inventory](../../../../../research_tmp/q14_kaggle/INITIAL_PROGRESS_R1.md). Build and launch receipts are recorded separately; no score is claimed until worker outputs are collected and graded locally.

# Kaggle R2 strict-FP32 panel

Private six-fit full-data packet, prepared from the clean root-pinned source commit `48776fba8af26b16b7e6813def7ccf571f56c1ac`. The only recipe correction is forwarding `strict_fp32=(precision==fp32)` through both ordinary Muon factory branches; canonical orthogonalizer code is unchanged. Root reports CUDA validation across both branches (six steps, 24 Newton–Schulz calls); this packet builder does not rerun fits or CUDA tests.

## Panel and execution order

The sealed `spec_paths` order is `q14s_a50_s3`, `q14s_a90_s3`, `q14s_a50_s4`, `q14s_a90_s4`, `q14s_boot8_s3`, `q14s_boot8_s4`. The first four are the alpha controls, so both alpha baselines are queued before boot fits. Alpha .9 controls are shared exactly across the alpha and bootstrap contrasts for each seed. Six cloud fits total.

All other training arguments match the quality_r2 root recipe and teacher-track `recipe_args.sh`: PR49 teacher `tt29rep_fbc1k_s5`, chronological train-only membership and split SHA `6d23a1b76f66d8001dac805d2b218a3b139e4eebe555f0c9b96fe1263ce186f0`, hidden 256, 2 layers, Muon RNN scope, FP32, 4 epochs, batch 32/chunk 1000, fixed checkpoint policy. Full validation is evaluation only and remains exposed development evidence. The stale target cache in the existing Kaggle asset dataset is not referenced; current local `train_targets.npz` is packaged and hashed. Raw official parquets and valid targets use the already verified external mounts in `portable_inputs.json`.

The private bundle declares 3,000,000,000-byte GPU operating and 5,000,000,000-byte stop limits, 7,200-second per-worker timeout, and one kernel-wide 13,500-second deadline. The push command also passes the Kaggle outer timeout of 13,500 seconds. At two T4 lanes, that caps summed device time at 7.5 GPU-hours. T4 numerical guards are in `kaggle_strict_traced_train.py`; they disable autocast and TF32 and record capability/effective flags. No torch install override is added.

## State

Root verified all sealed hashes and the six paired arguments ([preflight](ROOT_PREFLIGHT_R2.json)), uploaded the private1.58GB bundle and pushed kernel version1 with the outer13,500-second timeout ([push receipt](PUSH_R2.log)). Initial provider status is [QUEUED](STATUS_LAUNCH_R2.txt), not a completed fit or a score. Dataset files are visible with matching sizes ([listing](DATASET_FILES_R2.csv)); the separate status endpoint returned403 ([error](DATASET_STATUS_ERROR_R2.txt)) despite successful private creation and kernel acceptance. No retry or public/competition upload was performed.

Packet construction and local preflight only. `push_private.sh` is the reviewable root launch command and is intentionally not run here. R1 is preserved as a separate unlaunched legacy recipe and is not an input to this packet.

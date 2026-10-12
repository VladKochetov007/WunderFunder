# Clean T3 s3 FIT_TRAIN actual-device profile

Status: one bounded 16-sequence profile completed. The complete FIT_TRAIN export is prepared as a root-admission successor and has not been launched.

## Frozen lineage and membership

The profile used the clean F0 H384 Muon-all seed 3, epoch 2 checkpoint `model_ep2.pt` (`201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582`). Predictor and trainer were imported from the frozen H390/R3 source snapshot with hashes `30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0` and `a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6`. The bound source manifest is `41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa`.

The actual selector preflight passed before CUDA: 10,535 FIT_TRAIN groups, 1,405 FIT_VALID groups, 468 evaluated valid groups, and zero source-qualified fit/evaluation ancestry overlap. The profile decoded the first 16 selected FIT_TRAIN groups, IDs `0–14,16`, representing 320,000 sequence rows. Their ordered int64 membership hash is `46b5ac7530d3f294736d746f217bcf3c069b1df85a7af4c92c7dbcaa4cbca663`; full FIT_TRAIN membership hash is `8ebb217415c8cdc8c468bb081a7fa815e90b661fd1714f6d05da434a2fa3bbda`. Split hash: `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`.

Input identity is bound to the existing R3/R5 full-content SHA and selector preflight; point-of-use file size, mtime, and footer/length/PAR1 identity matched. The 29.24 GB train parquet was not rehashed.

## Actual-device profile

Attempt `clean-t3-f0-s3-fit-train-profile-r2-20261012` ran under the shared `wnn-campaign.slice` admission and GPU lock. Limits were one CPU, 5.5 GB process RAM, 4.3 GB CUDA allocator, 5 GB observed-VRAM ceiling, no swap, zero loader workers, 20-second lock/admission wait, and a 270-second command timeout within the 300-second profile cap.

The frozen predictor processed FP32 input shape `[16,20000,112]` with recurrent chunk length 4,000 and wrote `[16,20000,2]` (320,000 sequence rows) in 2.954 seconds. Measured phases: split selection 0.041 s; checkpoint load 0.006 s; CUDA setup 0.027 s; model build/device transfer 0.135 s; feature-only decode of 16 groups 0.963 s; prediction 0.679 s. CUDA allocator API peaks were 926,202,368 bytes allocated and 1,140,850,688 bytes reserved.

The controller recorded worker RAM peak 1,981,784,064 bytes and parent-slice peak 12,339,228,672 bytes. Attempt and parent OOM/high/max event deltas were zero. Controller NVML sampling recorded only the 46,137,344-byte baseline; the run was short and this is not evidence of peak device VRAM. Post-run `nvidia-smi` identified an NVIDIA GeForce RTX 3060 Laptop GPU with 6,144 MiB total. The CUDA allocator peaks are API-side measurements, not NVML device-wide proof. `PROFILE_CONTRACT_ERRATUM_R1.json` records that the original contract's display label `RTX 4060 Ti` was a metadata typo and preserves the original consumed contract hash.

The distinct profile output is 2,560,128 bytes at `cache/pred_clean_t3_f0_muonall_h384_e3_s3_r5_local_fit_train_profile_20261012_r1_train.npy`, with a separate 967-byte sidecar. The array file SHA256 is `79e3e4577b1298697d69004e49dd7d9ec3404cd4f007d1cba16d374f94173d4c`; the sidecar SHA256 is `8223731a50bc174f4179172a982c66bfd174148506870b9d2dbe29d972083513`. This subset is resource evidence only, not a complete teacher cache or an evaluation score.

The first attempt, `clean-t3-f0-s3-fit-train-profile-r1-20261012`, is preserved in the canonical attempt ledger and log. It failed before source import because the submitted command had `_202611` in two source paths; worker peak RAM was 75,603,968 bytes, with no CUDA allocation or OOM event. The corrected retry used a new ID and completed. The executed wrapper hash was `a5f16bc4f66f935c780c75ef79d975097274a0fbb709bbb6d6090626e09ecffb`; the committed wrapper has only import-order/lint cleanup (`6c005b7a0d7718b500222490af479bf0d8afaba021123ca64be9f672872f14bb`). Both attempts and logs remain in the shared runtime directory.

## Full-export successor

`FULL_EXPORT_SUCCESSOR_R1.json` and `R1_FULL_EXPORT_CONTRACT_PENDING_ROOT_ADMISSION.json` bind a future complete 10,535-group FP32 export to the same CP2 checkpoint, frozen producer source, split, data identity, and exact output paths. Expected output is `[10535,20000,2]`, 1,685,600,128 bytes including NPY header. A simple linear phase extrapolation from this first batch is about 1,081 seconds; the proposed worker command timeout is 3,600 seconds. It keeps the measured 1 CPU / 5.5 GB RAM / 4.3 GB allocator / 5 GB observed-VRAM / no-swap / zero-loader profile. Root must recheck current aggregate capacity, GPU-lock availability, and conditional-start authority at launch. This successor is not itself authorization to export.

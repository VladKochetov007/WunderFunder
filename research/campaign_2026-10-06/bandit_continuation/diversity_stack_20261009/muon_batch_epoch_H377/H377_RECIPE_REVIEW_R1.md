# H377 Muon batch/LR refill packet, R1

This packet stages one actual-device batch-64 profile and four fresh fold-1 recipe comparisons. It does not edit the frozen trainer, start a model fit, mutate a provider queue, replace the refill daemon, or claim a profile receipt. The controller must review the profile output and current admission state before adding any of the four full fits to the queue.

## Frozen execution source and inputs

The completed source recipe is `vt_f1_muonall_prop0_e4_s3`, built from scratch with seed 3 on the current Vast Muon-all × prop-power-0 recipe. H377 changes the epoch schedule to three epochs and saves fixed epoch 2; it does not resume or initialize from that teacher. The frozen live source snapshot is `/root/wnn/lr377_source_r1`, with 58 files and 638,829 source bytes. Its receipt is [LR377_SOURCE_MANIFEST_R1.json](LR377_SOURCE_MANIFEST_R1.json), SHA-256 `382beb68d556dd63215a9f7467bda09a23626ec3e1a2195739cffcab6819d9cc`; the four project-root symlinks are bound by [LR377_PROJECT_ALIASES_R1.json](LR377_PROJECT_ALIASES_R1.json). The job preflight checks the remote `SOURCE_MANIFEST.json` bytes, all 58 file hashes and exact file set, all four aliases, the fold-1 split hash, and the portable-input registry hash before each profile or fit.

Commands set `WNN_PROJECT_ROOT=/root/wnn`, `PYTHONPATH=/root/wnn/lr377_source_r1/src`, and `WNN_PORTABLE_DATA_MANIFEST` to the bound fold-1 portable-input manifest. Both trainer and predictor paths resolve inside `lr377_source_r1`; cache, dataset, artifact and research accesses resolve through the four verified aliases to `/root/wnn`. The current live Vast source is not the older frozen v6 snapshot, and no v6 source bytes are substituted or edited. Training-state provenance will record the actual source and dataset hashes.

Fitting membership is the exact source-qualified fold-1 set: 10,533 train row groups plus 1,416 fit-valid row groups, 11,949 total. Evaluation uses the fixed, repeatedly exposed set of 457 validation row groups from 149 components. [MEMBERSHIP_R1.json](MEMBERSHIP_R1.json) records both memberships and their digests. The portable input registry binds the supplied train and validation Parquet sources and target caches. Full data hashing is deferred to the trainer's existing provenance path; preflight hashes only the small registry and split receipts.

## Preregistered comparisons

All five jobs retain GRU-256, two layers, linear encoder, no dropout, seed 3, fold 1, `train+valid`, chunk 1,000, WMSE with chunk normalization, propensity object `c20260929_prop_f1` with power 0, default label/class weights, auxiliary weights 0.3/0.3, `since-weight=32`, Muon `scope=all`, OneCycle, fp16, fixed cycling beta-1, no fit-time validation, and a fixed epoch-2 checkpoint. The optimizer remains a hybrid: Muon handles qualifying matrix groups while vector and bias parameters use AdamW. Every group shares the LR schedule, so each LR contrast also changes the AdamW fallback's LR.

| Job | Sequence batch | Peak LR | Weight decay | Contrast |
|---|---:|---:|---:|---|
| `h377_b32_control` | 32 | 0.001 | 0.01 | Fresh same-recipe control |
| `h377_b64_sqrt_lr_only` | 64 | 0.0014142135623730952 | 0.01 | Exact square-root LR scaling; decay dose changes |
| `h377_b64_sqrt_decay_dose_matched` | 64 | 0.0014142135623730952 | 0.014180787422138722 | Square-root LR with matched nominal decay dose |
| `h377_b64_linear_lr` | 64 | 0.002 | 0.01 | Linear LR scaling |

At 11,949 fit sequences, drop-last and 20 chunk updates per sequence batch, the three-epoch OneCycle schedule has 22,380 B32 attempts and 11,160 B64 attempts. The B32 LR integral is 11.216132400000001. The exact square-root B64 integral is derived by scaling the audited B64 schedule sum at LR 0.0014 by `sqrt(2)/1.4`; the matched decay uses `WD × sum(attempted LR)` and the audited B32 value. This matches a nominal integral only; it does not match the exact multiplicative retention product, and skipped AMP updates can change realized decay. The LR-only and linear arms are not pure-Muon contrasts and do not validate an SDE rule.

Four older B64 scale jobs belong to a different Muon-RNN/no-propensity recipe. At the recorded 2026-10-11 10:26 UTC snapshot, its linear arm was running and square-root arm pending. H377 preserves both and is qualified as a separate current-lead recipe comparison.

## Profile and artifact delivery

The standalone profile job is `vt_f1_h377_muonall_prop0_b64sqrt_profile_s3`, using batch 64, exact square-root LR, five warmup batches and three measured profile batches. It removes `--save-training-state` and returns from the trainer's profile path before checkpoint writing; it cannot stand in for a fit artifact. Its training command has explicit `timeout --signal=TERM --kill-after=10s 600s` and matching 600-second job-step metadata. The profile command is recorded in [p00_h377_b64_profile_s3.json](staging_refill_jobs_r1/p00_h377_b64_profile_s3.json); no actual-device profile has yet run in this packet.

Each fit job runs identity preflight, fit, `predict_run.py valid --epoch 2`, then a bounded export validator. GNU `timeout` is explicit because the refill daemon ignores per-step timeout metadata. Worst-case step timeouts plus four 10-second kill graces total 5,315 seconds: preflight 15, fit 4,300, export 900, validation 60. The post-export check verifies the NPY shape `(1873, 20000, 2)`, float32 dtype, exact `valid_targets.npz` row-group order, epoch-2 checkpoint hash, prediction hash, split hash, and portable-input manifest hash. The predictor writes a JSON sidecar; the refill runner's terminal success check also requires it.

The primary endpoint is `global_wp_clip2_v1` on the fixed exposed fold-1 evaluation groups, with each recipe compared to the fresh B32 control. The conditional mixture is preregistered at exactly 50/50 against the frozen student endpoint `vt_f1_dsq_dm_t3_a9_e6@ep5`. The immutable pre-grade receipt [LR_FIXED_STUDENT_ENDPOINT_R1.json](LR_FIXED_STUDENT_ENDPOINT_R1.json), SHA-256 `ab92b3b417e11fce4aefd9fc99701199f0f293fed096f6a739c0b172fc562a2b`, binds its checkpoint (`c8e08b33a316f3539d4381429233b90403b76de83bda6dbecc84d7952f8212ad`), prediction (`bf1d4a52a67ccd530b0fff80976a6759d6a6265b7c699f77a73cfca9e044d0ad`), sidecar (`2323e59cde003aec195b6216a6367c1c9e9c3e893605b9575165f930ea48cefc`) and standalone local score `0.671031473`. Blend weights are fixed and will not be tuned. Fold 1 is exposed; component-resampling uncertainty and training-seed uncertainty remain distinct, and no public-score forecast is claimed.

## Static validation status

The deterministic builder and offline validator use the standard library and do not import the trainer or PyTorch. `STATIC_VALIDATION_R1.json` reports only `PASS_STATIC_ONLY`; the remote source preflight, actual B64 hardware profile, fits, prediction downloads, and conditional endpoint receipt remain controller/operator gates.

# Clean F0 matched student pair — preparation R1

This packet prepares one fixed-seed parent comparison. Both student arms use the same clean F0 split, homogeneous clean-T3 teacher target, H256 student recipe, six-epoch OneCycle schedule, fixed epoch-5 endpoint, and evaluation. The only planned change between arms is the own parent checkpoint: `c20261003_combo_s3_f0@ep3` versus `kg_h382r3_f0_muonall_prop0_e4_s3@ep3`.

The recipe and command vectors are copied from the frozen `clean_t3_f0_matched_student_r5_cache_r1` recipe. The new run IDs are unique. The cache formula is `T=float32(0.8125 * float32_mean(seed3_ep2, seed4_ep2, seed5_ep2))`, with no cache-time clipping; student targets use alpha 0.9. Fitting uses 10,535 `train.parquet` groups and 1,405 `valid.parquet` FIT_VALID groups. The 468 F0 evaluation groups remain excluded from teacher fitting and cache construction. F0 has been repeatedly exposed during development, so this is not fresh-holdout evidence.

Preparation is blocked. At capture, the S3 and S4 FIT_TRAIN exports had actual passing terminal receipts and the S5 FIT_TRAIN export was running. The S5 array and terminal hash/order binding were not available in this preparation. No homogeneous student cache has been materialized, no full-shape profile has run, no private Kaggle cache/source version is bound, and root has not released or admitted this student cohort. The earlier 16-sequence target fixture is not a substitute for the complete cache required by the frozen trainer.

The manifest binds the exact student recipe, split, source snapshot, propensity model, teacher and parent checkpoints/provenance, and current export receipts. The command templates preserve all science arguments and staged optimizer/scheduler/scaler/model/loader/RNG continuation. Their absolute source path is still the local frozen source snapshot; root must bind the actual Kaggle source and project-root aliases before making an upload-ready kernel. No provider mutation or fit is authorized by this packet.

After S5 completes, the cache owner must freshly verify all three FIT_TRAIN arrays and full-valid sidecars against their actual producer receipts, map by source-qualified row-group ID, materialize the exact train and FIT_VALID targets once, and obtain independent review plus root cache release. Then run the prepared H256/B16/C2000 profile on one actual Kaggle device serially, using the full cache and five warmup plus 25 measured complete batches. This preparation makes no speed or runtime claim. Root must review actual host/VRAM/resource results and current quota before either fit.

If released, run the control and treatment as separate serial GPU jobs with three saved/resumed OneCycle stages each. Preserve selected CP5, training state, history, provenance and full-valid feature-only FP32 predictions. Score both singles and their paired treatment-minus-control delta with the unchanged `global_wp_clip2_v1` scorer on the fixed 468 exposed F0 groups. Do not search blend weights or treat a local delta as a public forecast.

## Bounded offline check

The readiness validator is standard-library only. It hashes small source/checkpoint/provenance/receipt files, checks producer sidecar metadata, and reads prediction-file sizes without rehashing large arrays. The cache adapter remains responsible for full array hashes and finite-value checks during its separately released materialization. The current check is expected to report blocked and cannot launch training:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=650000000 WNN_RUNTIME_MAX=120s \
scripts/limited python3 \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r6_pair_20261012/validate_pair_preparation_r1.py \
  --receipt research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r6_pair_20261012/ACTUAL_PENDING_PREFLIGHT_R1.json
```

Run the focused tests with the same limits and interpreter. They verify actual current parent/source/producer receipts, command derivation, and refusal while the cache/root release is missing. They are not a cache-build or fit test.

# E450 full supplied-VALID fixed-model report

`score_seed4_full_valid.py` scores three preregistered, fixed transformations: seed-3 parent/head at 0.5/0.5, seed-4 parent/head at 0.5/0.5, and an equal seed bag with four members at 0.25 each. It verifies exact FP32 cache, sidecar, checkpoint, fixed-epoch, shape, dtype, and all-VALID row-order identities before reading predictions. The weight maps are checked against `checks/e450_screen_seed4_bag_fp32_v1.json`; the script has no fitting or tuning operation.

It uses the unchanged `scripts/validation_scorecard.py` validation logic and metric. Predictions are accumulated by parquet row group with the frozen `need_prediction AND is_scored` mask and reported for all 1,873 supplied VALID sequences, the 1,505 sequences in the final fit, and the 368 sequences excluded from that fit. The excluded group is historically exposed validation data, not an untouched test set. This is a descriptive supplied-VALID result, not a public score or native-runtime result.

The reproducible CPU draft is `specs/e450_full_valid_fixed_models_v1.draft.json`. It binds the wrapper, frozen metric/mask sources, preregistered weights, parquet, all four FP32 caches and sidecars, and all four epoch checkpoints and provenance records. Limits are one CPU, 2 GiB RAM, no GPU, and 1,800 seconds. The root sealer/controller owns admission and execution.

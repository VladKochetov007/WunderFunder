# Clean T3 F0 H256 profile preparation R1

`fixture_r1/teacher_target_fit_valid16.npy` contains only the source-qualified first 16 F0 FIT_VALID sequences in `valid_targets.npz` row-group order. It is FP32 `[16,20000,2]`, built from the actual seed 3/4/5 full-valid feature-only predictions using the frozen float32 mean and `0.8125` scale. The receipt pins the target IDs and all three prediction, sidecar, checkpoint and provenance hashes. No evaluation group was selected and no target-label member was loaded by the fixture builder.

The planned H256 profile uses the pending recipe's own control parent `c20261003_combo_s3_f0@ep3`, batch 16, recurrent chunk 2,000, FP16 autocast and the alpha 0.9 teacher/label blend. The stock trainer's profile flags cannot consume this fixture: its frozen campaign-split and cache validators require the entire F0 FIT_TRAIN and FIT_VALID arrays. No validation was bypassed, no complete cache was made, and no GPU command was launched.

The fixture can be rebuilt with the bounded command recorded in `PROFILE_PREP_R1.json`. Root can proceed after complete caches land using the stock trainer profile, or review a separately scoped geometry-only helper. Any geometry-only measurement must remain explicitly separate from full trainer/optimizer parity and scientific grading.

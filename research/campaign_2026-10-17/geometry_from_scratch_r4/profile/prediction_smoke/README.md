# R4 evaluation prediction smoke

The files in this folder are a predictor compatibility artifact from a one-batch, epoch-0 diagnostic GRU fit. They verify that the newly added `causal_geometry_discrete` checkpoint prior is loaded by `scripts/predict_run.py` and that the CampaignSplit evaluation-only selector exports the right rows. They are not an R4 result and must not enter the paired score analysis.

- [Fold-0 prediction](fold0_seed42000_epoch0.npy) — finite FP32 `(468, 20000, 2)` output for the fixed component-held-out evaluation groups.
- [Prediction sidecar](fold0_seed42000_epoch0.json) — checkpoint and prediction hashes, row-group order, fold manifest hash and `evaluation` export scope.
- [Admission receipt](../profile_success_001.json) — records the command, scope, resource ceiling and smoke outcome.

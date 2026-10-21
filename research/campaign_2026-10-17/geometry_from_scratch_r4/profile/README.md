# R4 bounded profiles

This folder records the pre-fit resource admission checks for E-GEOMETRY-FROM-SCRATCH-20261017-R4. The profile uses the same feature path, batch and chunk dimensions as the registered recipe, runs within the 5,000,000,000-byte task cap and one CPU core, and leaves the shared GPU lock free afterward. Training and prediction checks are separate because CampaignSplit predictions are exported after fitting.

- [Preflight failure 001](preflight_failure_001.json) and its [log](train_profile_f0_s42000.log) preserve the first command; it stopped at argument validation because `--no-eval-during-fit` was missing and made zero optimizer updates.
- [Profile admission evidence](profile_success_001.json) records the five measured training batches, CUDA and cgroup peaks, finite gradient/update checks, no new OOM events and the runtime estimate for the complete matrix.
- [Corrected short training profile](train_profile_f0_s42000_r2.log) completed the corrected CampaignSplit preflight and two training batches.
- [Resource profile](train_profile_f0_s42000_r3.log) completed one warmup plus five measured full training batches, with the profile resource snapshot embedded in the log.
- [Train-and-predict smoke](train_predict_smoke_f0_s42000.log) and [prediction log](predict_smoke_f0_s42000.log) record a one-batch epoch-0 diagnostic fit followed by inference over all 468 held-out fold-0 groups. These are resource/compatibility checks, not a scored model result.
- [Smoke prediction artifacts](prediction_smoke/README.md) preserve the finite `(468, 20000, 2)` FP32 output and its row-order/split sidecar; the checkpoint is under `artifacts/runs/geometry_r4_predict_smoke_f0_s42000/`.

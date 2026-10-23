# Native route for an age-readout endpoint

**Status: source-only route map.** No packaging code, tests, fits, native replay, or provider action was performed for this note. The age-screen recipe is `research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json`, SHA256 `0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd`.

## Model and callback contract

Each selected age checkpoint is a 323→64 tanh→2 residual readout, not a recurrent checkpoint. Its output is the exact parent’s first two outputs plus `0.1 × residual`. It consumes normalized canonical 211 columns (`raw112 + event99`) followed by 112 capped reset-local log ages scaled by `log1p(10000)`. The model and feature contracts are in `src/wnn/training/age_readout_model.py`, `age_readout_features.py`, and the recipe.

The shortest functional serving path is a small `PredictionModel.predict(data_point)` wrapper around the exact parent checkpoint in the existing native `wnn_native.Predictor`. The callback receives raw112 in `data_point.state`; native `Predictor.step(raw112)` computes its event99 stream and emits the parent output. The wrapper must separately reproduce canonical `raw112 + event99` for the readout, apply the parent mean/std saved in the age checkpoint, update the selected `StreamingFieldAgeFeatures` clock from raw112, and return `parent[:2] + 0.1 * head(x)`. Reset both recurrent and feature state when `seq_ix` changes, process warmup rows to advance state, and return `None` for those rows. The callback interface and raw112 input are defined in `utils.py`.

Recipe-pinned parents:

- F0: `artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt`, SHA256 `e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`.
- F1: `artifacts/runs/oldz4_h230_parent_f1_s3_r4/model_ep3.pt`, SHA256 `be2273cc96027295c19133504354a262cd1a2066fa5ae3df7d6be64ed0a1ad8e`.

The six planned age artifacts are under `/home/vlad/development/wunder/research_disk/b300-fast-20261023/runs/age_readout_{f0_s1001|f1_s1002}_{a|g|f}_r1/`, with `model_ep2.pt`, `prediction.npy`, `prediction.json`, and `provenance.json` per run. These are only candidate files once root admission and execution produce them.

## Packaging constraints

`scripts/package_submission.py` exports standard recurrent checkpoint members. `scripts/package_tl3_h512_member.py` only accepts a 211-input H512 GRU and replaces the fixed `weights_s5.npz` slot in the TL3 archive. Neither packages this residual MLP directly. A complete age endpoint needs a create-only builder that binds the exact parent NPZ, residual-head weights, wrapper, and the three native ABI libraries. No Rust change is required for a first implementation; Python per-row event/age bookkeeping and two small affine layers add unmeasured callback cost. The FP32 PyTorch cache parent outputs may differ from native parent outputs, so compare native parent predictions and wrapper outputs before using cache-head grades as serving evidence.

## Complete-endpoint comparison assets

- Original TL3 archive: `submissions/team_tl3_original_20261018.zip`, SHA256 `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`.
- Existing F0 TL3 prediction cache: `cache/pred_tl3_unknown_ancestry_f0_valid.npy`, SHA256 `3a7456f0a2e294a852618535b9e3f2551fc15c5b70fd033824bddb10be83bbe3`; sidecar `cache/pred_tl3_unknown_ancestry_f0_valid.json`, SHA256 `b9d8b7371dbff016d48e155da154cd87d85c887cb08921e86caabba082059910`.
- F0 source split: `research/campaign_2026-09-29/artifacts/splits/fold0.json`, SHA256 `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`. The fixed endpoint screen covers 468 groups, 141 connected components, and 808,863 scored rows. Targets/masks are mapped through `cache/valid_targets.npz`, SHA256 `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`.
- Reusable replay/grade structure: `research/campaign_2026-10-20/submission_delivery_r1/tl3_h512_endpoint_screen_r1/{endpoint_contract.py,replay_h512_tl3_endpoint.py,grade_h512_tl3_endpoints.py,endpoint_score_inputs.py}`. Its contract is pinned to an H512 replacement and F0; adapt to age-head identity and F1 rather than reusing it unchanged.
- Full-host replay reference: `research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_REFERENCE_REPLAY_R1.json`; original TL3 took 1,587.6 seconds total, including 1,445.2 seconds in the callback, over 468 groups on one CPU. Treat this as a planning measurement for that host and endpoint, not an age-wrapper runtime estimate.

Before promotion, compare each chosen **complete native parent-plus-head endpoint** against its exact parent and the original TL3 on identical source-qualified rows; check order, warmup, resets, output parity, package size, native runtime, and `global_wp_clip2_v1`. The old TL3 local F0 result is an exposed fitted diagnostic, not clean CV: it scored 0.70181812, but its two retained members have unknown ancestry. The reported public TL3 result is 0.66832 (`1lmehfi3`, 39m12s). The two-parent exploratory package was reported at 0.66771; the H512 replacement at 0.66667 publicly and 0.69445462 on the exposed complete F0 native panel. These are historical references, not forecasts for age features.

The F0/F1 screen uses different parent models and seeds. Candidate head scores remain conditional on those bases; a cached-parent improvement alone does not establish a better complete native endpoint or a submission result.

# F1 Core/HFT Consumer Review R1

**Verdict: BLOCKED before freeze.** The HFT recipe is faithful to the completed F0 producer, but the current untracked F1 packet fails its first import/preflight check on a source-binding lookup. No F1 crop, profile, fit, export, or model evaluation was run. This is a runnable-consumer defect, not negative evidence about the F1 model direction.

## Reviewed bytes and scope

The reviewer worktree is `codex/f1-core-hft-review-20261012`, based on `5b62d9d18cceffefac545e602a0f577cc5769f99`. The builder worktree was at `6de33f78d6831e37a85f1745aef29ceddd5b2260`; its entire packet directory was untracked at review time, so that commit does **not** contain the reviewed packet. The packet file hashes below identify the exact draft contents inspected. The MAIN worktree was at `a1d4234dc93757130076a93cf0a94f2b00e33d53` and had unrelated changes; none were edited.

The frozen source set binds the reviewed c337 trainer and predictor, the F0 crop producer, and the R4 F1 inputs. The relevant producer evidence in MAIN is `artifacts/runs/oldz4_h230_hft_f0_s3_r5/provenance.json` and `ops_continuation_20261011/oldz4_student_core_hft_f0_r5/{HFT_FIT_CONTRACT_R5.json,run_hft_stage_r5.py}`. The actual F0 provenance records `train_source=valid`, `mask_weight=true`, fixed checkpoint epoch 0, and inherited `aux_struct=aux_next=0.3`. The F0 fit contract/argv also establish one-epoch GCorr, AdamW, OneCycle, batch 16/chunk 2000, fp16, `inp,gru` frozen, `lr=0.0005`, and `wd=0.01`.

The F1 plan correctly distinguishes ancestry from direct fitting: the student/core ancestor has source-qualified membership of 10,533 train plus 1,416 inner-valid groups; the HFT stage directly fits only the 1,416 fold-eligible valid groups with `--mask-weight`, no train groups, and excludes all 457 outer evaluation groups. The F1 HFT argv and expected provenance match the F0 HFT semantics, including inherited `.3/.3` auxiliaries (the flags are intentionally omitted). This is not the prior `PARENT|HEAD` parameter-soup endpoint; any output-level mixture is a separate, unevaluated decision.

## Blocking consumer defect

`run_f1_core_hft_stage_r1.py::verify_packet_bindings` validates the frozen F0 crop source manifest, assigns its correct primitive hash to `crop_source_sha`, then compares it with `sealed.get("f0_crop_source_sha256")`. That top-level key is absent. The binding is nested at `sealed["core_crop"]["r5_crop_source_sha256"]`, where it matches the source-manifest value `c805c4f8647abbaa434a575870199b558b411e412a3ceaff450a36e4a657161a`. The smallest correction is to compare with the nested `crop_pins` value; then publish a new packet/source-binding identity and revalidate it before any admission.

The bounded `validate-import` command exited 1 at this comparison with `ValueError: F0 crop primitive differs from the prepared F1 contract` (runner line 166 in the reviewed snapshot). It failed before trainer/predictor imports and before split-membership validation; it did not access parquet rows, initialize CUDA, or mutate artifacts. Therefore the downstream checks below are static review only, not runtime validation.

## Remaining stage gates and consumer checks

The crop branch is structured to consume actual F1 student CP1, call the pinned `derive_core`, compare the derived checkpoint against the independent R5 `verify_crop` result, enforce the checkpoint-size bound, and create a write-once crop receipt. Its required CP1 terminal gate and F1 crop-admission receipt were not present/run in this review.

Profile and fit stages require separate root-owned receipts. The fit gate also checks actual-CUDA finite loss, at least one completed update, zero nonfinite-gradient updates, and zero OOM delta. F0’s earlier strict profile with an AMP skip and its separate resource-only release do not satisfy the F1 clean-profile gate; F1 must obtain its own measured profile and resource admission. No profile, fit, resource, or CP0 terminal evidence is established here. The fit contract says `fit_authorized: false` but also records `fit_admission_decision: PASS`; that metadata is contradictory and should be made unambiguous before freeze. The stage runner still requires the exact separate root admission files, so the stale field alone does not release the fit.

The export branch binds the terminal CP0 and its provenance/checkpoint hashes, verifies the valid-only direct-fit/evaluation groups, initializer checkpoint and normalization identity, then calls the pinned predictor and checks its sidecar, `[1873, 20000, 2]` float32 output, byte size, and finiteness. The predictor uses the pinned c337 `make_model` plus `load_state_dict` path; this source has no `make_model_keys` hook. `import_predictor` requires its `train_rnn` module to be the exact imported trainer and resets both module roots to MAIN. Because preflight stopped before imports and no CP0 exists for this stage, predictor/profile compatibility is not runtime-proven. Root’s terminal review must bind the full fit provenance/command, since the exporter itself checks only a subset of fit hyperparameters.

## Exact pins and next gate

Builder draft hashes:

| File | SHA-256 |
|---|---|
| `F1_STAGE_PLAN_R1.json` | `6ce53ac201aa526234ab18856b1bba1504ab514c43ecbfe433e34d17d345f9c8` |
| `HFT_FIT_CONTRACT_R1.json` | `1101b2adcd49e65bc0d3c3cad126770a7a679d5dfeb1f01740515bffc8a63a1a` |
| `HFT_PROFILE_CONTRACT_R1.json` | `d384fdcc7cedc770703fadcab26e87eca072efed0377378dbaa6ffbea95a1c87` |
| `HFT_EXPORT_CONTRACT_R1.json` | `21a09eb5c0737e34bf6bfa7ea541c843096a6e7fd0c73fdb922de3efa28bbec2` |
| `CORE_CROP_CONTRACT_R1.json` | `614caab058b96d60c9bb76c4601cc26bd9869ae4eeef78fb36def6c96d68e50b` |
| `SOURCE_INPUT_BINDINGS_R1.json` | `316b108b30c076fade74ae42a09d028e433f14c37500d946df1faa594ad01ae8` |
| `run_f1_core_hft_stage_r1.py` | `a59a74387feaf89fe93f17fc487a08c84c6403424cfe260fdc8b9eaa2578db32` |

Key MAIN producer/source hashes: F0 provenance `bf6105bfb1d137345bf775b09f06f432b5d993c61cf019aa927c2a896ebddac2`; F0 HFT contract `64df5f235242042a3e9848eb260321d4a0b46979862aa9299dfa5f401059b541`; F0 HFT runner `395a603a9d7bd961a8ec0b45374f2a345243b915aa80d5783b1548887128f418`; F0 R6 crop consumer `203ea796322d7ecbd238111b1339d144aec4caa2981ee9da6d4b62b481384d9b`; F0 R6 actual consumer verifier `e6226f05276dccf25f635aff1e6321a00b3b868086ec7bf45bca8b53949b63e8`; c337 `train_rnn.py` `5b135ea90480dde8b530c4543e1d587a044efa1f98dc6a3327b9dca6fcd40187`; c337 `predict_run.py` `afc5aaaef7edd78f4b719a6be450b10d82aefcbf497b4515f684f51930e229f9`.

Next, the builder must fix the nested lookup, produce a newly identified committed packet, and pass bounded `validate-import` against that exact commit. Only then can root review the CP1 terminal/crop, actual-device profile and resource evidence, fit admission, and CP0 export. This review does not authorize those stages or promote a submission.

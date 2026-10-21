# H512 B16 conditional F1 r2 recipe

## Root conditional admission — October 21

[Final independent review](INDEPENDENT_STATIC_REVIEW_R2.json) passes within static scope. Root independently refreshed [provider quota](ROOT_QUOTA_PRELAUNCH_R2.json): 17,006.922 seconds unreserved, zero reserved. The [actual admission](root_admission.json) permits one fresh B16 profile and a cold six-epoch fit only if measured speed fits the actual deadline, 15,500-second outer cap and 1,200-second quota reserve. The final bundle still needs exact byte verification before upload. No R2 model or quality result exists. Earlier placeholder statements below are historical preparation.

This code-only packet prepares one resource-adapted H512 F1 endpoint from the clean canonical source checkout 68ff682977b822f38351aac0f621146f5f13b440. The six worker helpers were transplanted from B8 packet commit ef11a7d07 and changed for batch 16, the R2 run and artifact names, the new update count, and the reduced wall cap. The canonical trainer and src/wnn sources are unchanged.

The exact trainer argv, source identities, code closure and fail-closed gates are in [the proposal](H512_F1_B16_CONDITIONAL_PROFILE_FIT_PROPOSAL_20261021_R2.json) and [execution spec](KAGGLE_F1_B16_CONDITIONAL_EXECUTION_SPEC_R2.json). The run names are r01_h512_f1_s1004_kaggle_profile_r2, r01_h512_f1_s1004_kaggle_fit_r2, and r01_h512_f1_s1004_kaggle_swa345_r2. The worker receipt path is conditional_worker_receipt_r2.json with the existing wnn.kaggle-h512-f1-conditional-worker-receipt.v1 schema.

The fit membership remains 10,533 train plus 1,416 valid sequences (11,949 per epoch). At batch 16 with drop-last, that is 746 batches per epoch, 13 dropped sequences per epoch, 4,476 batches over six epochs, and 44,760 chunk updates. Batch 8 dropped five sequences per epoch and used 89,580 updates. The B8 worker completed a 40-update engineering profile and skipped fitting at its cost gate; its reported provider charge was 270.730 seconds. It provides no quality result. R2 requires a fresh B16 profile with one warmup and three measured batches (40 finite chunk updates). Its measured B16 PROFILE.step_wall feeds the worker exact cost gate. The approximately 2.18-second per-batch threshold is planning context only. The B16 batch and optimizer-clock changes make this a resource adaptation, not a matched width or seed replication.

The recipe retains six epochs, fixed CP5, seed 1004, 211 causal inputs, chunk-normalized original-label WMSE, strict FP32, AdamW at 1e-4, OneCycle warmup fraction 0.05, clipping at 1.0, and an equal-parameter CP3/4/5 SWA endpoint. The conditional full fit uses 4,476 B16 batches. The worker wall cap is 15,200 seconds, with a 300-second finalizer and 15,500-second outer cap; it preserves a 1,200 provider-second safety reserve and records a separate two-device allocation bound. No R2 profile or fit is admitted. The direct provider snapshot reported by the root at 00:48:28 UTC is retained for context only; root admission must use a fresh poll and reservation check.

The copied fold-1 view and portable input registry retain their source hashes (548b42f0… and 2dcd5cee…). Structural-cache size/SHA metadata is in private_cache_bundle/assets-manifest.json; the private archive remains in the original B8 worktree at the exact path pinned in the proposal and is not copied here. Stage and hash-check those two cache files before canonical split preflight. Raw train/valid files remain external mounts. KAGGLE_BUNDLE_PREPARATION_R2.json records the exact 170-file code closure but explicitly defers bundle construction until root admission.

The root-owned paired grader is being prepared separately in research/campaign_2026-10-21/kaggle_h512_b16_f1_endpoint_grade_r2. Root must freeze that grader and its terminal acceptance contract before any fit. Its fixed screen remains exposed F1 robustness evidence; it cannot establish a clean holdout or be pooled with identical-arm fold estimates. No bundle was built, uploaded, or launched while preparing this packet, and no Torch import or array load occurred.

## Final source and evaluation handoff

Clean execution source is commit `33588ef26f1bcfc065f6c546a062327ce992eb6b`, based on unchanged canonical68ff trainer. Independent review caught two preparation path mismatches (proposal filename and required worker receipt); both were corrected before any launch and the170file closure was resealed. The [separate B16 fixed grader](../kaggle_h512_b16_f1_endpoint_grade_r2/README.md) pins the final proposal and44,760update contract, and requires root acceptance of actual downloaded checkpoints and exports. It preserves the unused B8grader unchanged. Root owns fresh quota review, private bundle sealing and any admission; none is implied by these source files.

## Exact bundle verified; one upload/launch authorized

Root independently checked the [bundle handoff](B16_BUNDLE_HANDOFF_R2.json) and wrote [exact-byte verification](ROOT_BUNDLE_VERIFICATION_R2.json): 841,659-byte archive SHA366d5271a70e367f226b18400189c5d184a3ddb2926afcedae60b1db1fe78c6f, all170codefiles and9localinputs, real admission, runtime-spec identity and private T4bootstrap with15,500secondcap. The two rawparquets remain external. Operator is authorized exactly one private dataset upload, remote hash verification and kernel push; no retry, paid compute or competition submission. Provider state and finite profile/fit progress remain to be observed.

## Transport failure and explicit correction

The [upload-source diagnosis](B16_UPLOAD_SOURCE_DIAGNOSIS_R2.json) and [version1 terminal receipt](B16_OPERATOR_TERMINAL_R2.json) record a concrete operator staging error: only `bundle.wnn` was copied to the upload folder, omitting `bundle-manifest.json`. The bootstrap stopped after1.29seconds, before profile/fit; provider charge7.673seconds. This is transport evidence, not a model result.

Root issued a [separate one-attempt transport repair admission](TRANSPORT_REPAIR_ADMISSION_R3.json) after fresh direct quota showed16,999.249seconds/zero reserved. Scientific code/recipe/archive remain unchanged. The corrected private dataset must contain both byte-verified files, with remote listing and download hashes checked before a new push; an early push deadline preserves embedded admission freshness. This does not authorize a blind retry.

## Corrected version2 running

The [local upload file receipt](B16_DATASET_V2_LOCAL_FILES_R3.json) and [remote dataset verification](B16_REMOTE_V2_VERIFICATION_R3.json) establish both required files in datasetversion2 with exact reviewed hashes. [Version2 operator snapshot](B16_V2_OPERATOR_SNAPSHOT_R3.json) binds the one corrected kernelpush and providerRUNNING state at01:11:10.367UTC. Live logs reached inputvalidation; profile and fit results remain unestablished. No recipe/code/archive changes or further retry authorization.

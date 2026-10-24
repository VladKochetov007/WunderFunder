---
kind: review
status: complete
created_utc: 2026-10-24T14:08:30Z
updated_utc: 2026-10-24T14:08:30Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/PUBLICATION_INVENTORY_R1.md
  - research/campaign_2026-10-24/resume4h_r1/publication_source_r1/SYNC_RECEIPT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/publication_source_r1/README.md
  - research/campaign_2026-10-24/resume4h_r1/publication_source_r1/SOURCE_FILE_HASHES_R1.json
  - research/campaign_2026-10-24/resume4h_r1/PR_REFRESH_INTERPRETATION_R3.md
---

# Publication review — October 24 campaign

## Question or claim

Does the campaign-evidence publication worktree contain the reviewed campaign scope, resolve its local evidence dependencies against the intended research-team base, and preserve safe metadata without publishing scientific payloads?

## Method

Read-only review of worktree /home/vlad/development/wunder/research_tmp/resume4h-publication-20261024 at base d32be666a10c6928a100a91a021337225520ce32. I checked its changed-path scope, whitespace, source-only trainer patch, sync and URL-redaction receipts, and Markdown relative targets. Path checks covered 76 Markdown files under the campaign (excluding frozen imported PR source snapshots), the selected root/current-state and unified README indexes, and 1,652 distinct local target paths. Availability was checked against both the publication worktree and its base Git tree. I did not run imports, code, models, payload reads, tests, provider calls, or Git staging/commit operations.

## Evidence and interpretation

### Scope and evidence integrity

The worktree has 12 modified tracked documentation/index/ledger/current-state files plus untracked campaign, wrapper, and October 23 review material. No canonical src/ path is changed in the publication diff; git diff --check is clean and the index has no staged changes. The campaign source snapshot contains a three-file trainer patch with a source-hash manifest. Its README identifies the patch as relative to source execution commit c3e208b387ead0ec967f9abc4b14275eb8ba94c5 and warns against blindly applying it to the diverged research-team base. It is an evidence snapshot, not a canonical trainer edit. The campaign PR and recipe describe zero new scientific fits, no competition upload, and the validation/ancestry limits of the exploratory ZIPs.

The sync receipt records 491 campaign files copied, seven unreferenced full GitHub file-list dumps omitted, and no scientific payloads copied. The seven omissions are not linked elsewhere in the campaign and are deliberate. The older CAMPAIGN_COPY_DRIFT_R1 receipt predates this sync and is historical, not evidence of current file drift.

Nine of the ten previously checked changed/source-only paths match byte-for-byte between the source checkout and publication worktree. The remaining difference is CAMPAIGN_SYNTHESIS_R1.md: the source copy adds a “Later remote refresh” section recording the third PR refresh; the publication copy lacks that addition. The source SHA-256 begins 0e7442038670 and the publication copy begins 728327892ade. Final sync should port this late section and then recheck the file.

### URL redaction

The root campaign redaction receipt records 41 query values redacted in 17 metadata JSON files and zero remaining credential query values. All 17 sanitized-file SHA-256 values in that receipt match the corresponding publication-worktree files. A separate nonprinting check scanned 438 publication text files: it found 41 raw.githubusercontent.com URL occurrences, all with placeholder query values, and zero nonplaceholder sensitive query values. No URL or query value is reproduced here.

At review time, publication_source_r1/sanitize_publication_urls.py and URL_QUERY_REDACTION_R1.json exist in the source checkout but are absent from the publication worktree. Include the final redaction receipt with the publication evidence; after any final resync, rerun the sanitizer before staging because a sync could restore raw metadata copies. Keep the original source metadata and its original hashes private as the receipt describes.

### Local links that do not resolve on the publication base

The scan found 62 non-payload targets that are physically present in the source checkout but absent from both the publication worktree and its base: 14 Markdown files, 38 JSON metadata/result files, and 10 Python modules. These are real publication link dependencies, not missing files in the source checkout. Resolve each by including the specific small document/metadata dependency or changing the link to a stable source-revision URL. Do not copy canonical source modules over research-team files without a source comparison.

The ten Python targets are all canonical package modules (nine under src/wnn/training and src/wnn/online_relationship.py), not campaign adapters. The trainer patch manifest covers only three other source files. Any source link should identify the commit that produced the reviewed artifacts.

Exact missing Markdown targets:
- research/campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md
- research/campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/README.md
- research/campaign_2026-10-18/online_features_r1/sequence_oracle_r1/README.md
- research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/RESULTS_F0_R1.md
- research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md
- research/campaign_2026-10-21/next_cycle_r1/i1/README.md
- research/experiment_drafts/b300-fast-local-kaggle-20261023/NATIVE_ROUTE_R1.md
- research/experiment_drafts/b300-fast-local-kaggle-20261023/README.md
- research/unified/ideas/FUTURE_INPUT_DYNAMICS_AUXILIARY_20261021_R1.md
- research/unified/ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md
- research/unified/ideas/TEACHER_STUDENT_SUBMISSION_COMBINATION_20261018_R1.md
- research/unified/operations/RESEARCH_STOP_20261018_R1.md
- research/unified/reviews/I1_DYNAMICS_NATIVE_SERVING_COMPATIBILITY_REVIEW_20261022_R1.md
- research/campaign_2026-10-21/next_cycle_r1/weight_mean_projected_r1/README.md

Exact missing JSON targets:
- artifacts/qat_renewable_r1/final_blend_grade_r4/result.json
- artifacts/qat_renewable_r1/final_native_self_s3_r2/fp32_native_q8/result.json
- artifacts/qat_renewable_r1/final_native_self_s3_r2/fp32_ptq_w4/result.json
- artifacts/qat_renewable_r1/final_native_self_s3_r2/learned_w4_qat/result.json
- artifacts/qat_renewable_r1/final_release_r1/BASELINE_CALLBACK_RUNTIME_R3.json
- research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/recipe.json
- research/campaign_2026-10-21/next_cycle_r1/CYCLE_ACCOUNTING_POSTBOOT_R1.json
- research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_PROFILE_RESULT_R2.json
- research/campaign_2026-10-21/next_cycle_r1/NATIVE_GRADE_R3.json
- research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json
- research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_PRECISION_PROBE_R2.json
- research/campaign_2026-10-21/next_cycle_r1/WARM_REMAINDER_COST_DEFERRAL_R1.json
- research/campaign_2026-10-21/next_cycle_r1/WEIGHT_MEAN_PROJECTED_GRADE_R1.json
- research/campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json
- research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
- research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/F0_CONTROL_EXPORT_ACCEPTANCE_R4.json
- research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/PREFLIGHT_F0_RESULT_R3.json
- research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/PREFLIGHT_F1_ORDER_RESULT_R1.json
- research/campaign_2026-10-22/quota_retained_pair_r1/FIXED_SAMPLE_GRADE_R2.json
- research/campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/ACTOR_SOURCE_SHA256_R3.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/INDEPENDENT_SOURCE_REVIEW_R3.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/KAGGLE_PROFILE_TERMINAL_R2.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/LOCAL_CACHE_VERIFICATION_R1.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/PROFILE_BUNDLE_PREPARE_R2.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/RESOURCE_AND_DELIVERY_AMENDMENT_R2.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/SOURCE_OVERLAY_MANIFEST_R3.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/SOURCE_OVERLAY_PREPARE_R3.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json
- research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json
- research/experiment_drafts/dynamics-aux-20261021/recipe.json
- research/experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json
- research/experiment_drafts/online-relationships-20261022/GRADE_SPEC_R1.json
- research/experiment_drafts/online-relationships-20261022/MEAN_ONLY_LAG_IDENTITY_R1.json
- research/experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json
- research/experiment_drafts/online-relationships-20261022/ROLLING_COST_REVIEW_R2.json
- research/experiment_drafts/online-relationships-20261022/TERMINAL_COLLECTION_R1.json
- research/experiment_drafts/online-relationships-20261022/recipe.json

Exact missing Python targets:
- src/wnn/online_relationship.py
- src/wnn/training/age_readout_cohort.py
- src/wnn/training/age_readout_model.py
- src/wnn/training/age_readout_portable.py
- src/wnn/training/age_readout_portable_builder.py
- src/wnn/training/age_readout_portable_identity.py
- src/wnn/training/age_readout_portable_validate.py
- src/wnn/training/online_relationship.py
- src/wnn/training/online_relationship_fit.py
- src/wnn/training/online_relationship_sources.py

### Deliberately external or preserved targets

Eight additional source-tree targets are binary/runtime payloads, not documentation failures. The publication inventory and receipts intentionally keep them out of Git: three prior prediction NPY files, the direction-probe moments NPZ, the Kaggle CUDA overlay WNN file, and three ZIP submissions (one October 14 ZIP and the two October 24 quota packages). Preserve their existing source/hash receipts and do not add them to this publication set.

Fourteen relative targets in the frozen imported PR source snapshots also remain unresolved locally. Those snapshot READMEs intentionally preserve original source-relative links; pr_audit_25_40/README.md explains that they resolve in the upstream source branch directories identified by the GitHub metadata. I excluded those immutable snapshots from this campaign-doc dependency count, so they are not conflated with the 62 current publication dependencies.

## Limits and validity

This was a metadata and navigation audit of the current sparse publication worktree, not a review of model validity, result payloads, or external-link availability. Source-tree presence shows a file can be ported or referenced; it does not prove it belongs in the publication PR. The source module links need a reviewed immutable revision, and the 62 local references must be resolved before readers on research-team can follow them.

## Decision and next step

The scientific-content scope is appropriately bounded and the publication files scanned contain no unredacted credential query values. The snapshot is not link-complete yet. Before staging, sync the late campaign-synthesis section and URL-redaction receipt, rerun the sanitizer after any content sync, then resolve the 62 targets by selective dependency inclusion or stable source-revision links. Keep ignored scientific payloads external and preserve imported PR snapshots as source evidence. Root owns the final sync, navigation integration, and publication decision.

## Source records

- [Publication inventory](PUBLICATION_INVENTORY_R1.md) and [campaign README](README.md).
- [Sync receipt](publication_source_r1/SYNC_RECEIPT_R1.json), [source patch note](publication_source_r1/README.md), and [source hash manifest](publication_source_r1/SOURCE_FILE_HASHES_R1.json).
- [Current synthesis in source checkout](CAMPAIGN_SYNTHESIS_R1.md) includes the post-sync PR refresh; the publication copy must be refreshed.

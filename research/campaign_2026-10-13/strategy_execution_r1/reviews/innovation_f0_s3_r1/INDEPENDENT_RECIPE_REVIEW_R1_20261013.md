---
kind: review
status: fail-no-launch
hypothesis_ids: [H-STRATEGY-20261012-01, H-238]
experiment_ids: [E-STRATEGY-INNOVATION-F0-S3-20261013-R1]
evidence_paths: [RECIPE_R1.json, MOMENTS_CONTROL_COMMANDS_R1.json, INNOVATIONS_COMMANDS_R1.json]
---

# Independent recipe review: S09 R1 is not runnable

**Verdict: FAIL R1; do not launch either profile or fit command.** The pair is well matched on its declared scientific contrast and the actual parent/cache ancestry is clean, but the frozen trainer's cache validator rejects the actual cache metadata before it opens target indexes or cache arrays. This is an operational compatibility failure, not evidence against the innovation hypothesis.

## What passed

The candidate is frozen at source commit `c6e774edc8d5e0c63715a83287cf36fd3969a779`; its worktree is clean and all 391 production-source hashes in `RECIPE_R1.json` match. The parent is the actual own checkpoint `c20261003_combo_s3_f0@ep3`, SHA `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`. Its recorded fitting groups match the fold-0 fit manifest (10,535 train and 1,405 valid); 468 evaluation groups/components are disjoint. The parent has fresh weights (`init=""`) and records its two own independently trained seed-3/seed-4 teachers and propensity checkpoint. Their actual checkpoint hashes match the parent's provenance. Teacher prediction payloads were not reread.

The two arms use the same seed, fold, parent, cache, target-scale CLI value, alpha, batch, chunk, preprocessing inheritance and six-epoch schedule. The sole selector change is zeroing the final 16 innovation coordinates for the moments control. Stage argv agree after normalizing arm names, resume paths and that selector. P1/P2/P3 stop after inclusive epochs 1/3/5, with absolute P2/P3 predecessor-state paths; the fixed checkpoint is epoch 5. Fit, profile, export and grade authority fields are false. The actor's actual-parser receipt reports ten parser checks passing but explicitly stops before post-parse work, so it does not exercise cache validation.

The feature prior uses 32 source-qualified fitting groups and a fixed seed, with no evaluation groups in the sampling set. The independent parent-extension check verifies the 211-to-291-input expansion and exact short-fixture output preservation. Earlier separate implementation evidence covers a 20,000-row causal feature path. Those checks support recipe intent and initialization only; they make no model-quality claim.

## Blocking producer/consumer mismatch

Both fit and profile worker vectors pass the real cache root `cache/clean_t3_f0_h390_t3r5_a9_r2` with variant `homogeneous`. Frozen `scripts/train_rnn.py` calls `validate_distillation_cache` before constructing its datasets; the recipe binds `src/wnn/distillation_cache.py` at SHA256 `403fd74db91f08bb255db147fb50f7f6b6b9702280350aaa2479c01a7f972f19`. Against the actual cache metadata, the validator raises:

> `distillation family-head order is not the declared four-family order`

The bounded independent check reconstructed `CampaignSplit` from the hash-bound fold-0 manifest, passed it to the frozen validator, and installed an `np.load` spy. The cache's fold and exact ordered fit/evaluation membership passed the checks preceding the exception. The spy recorded **zero** array-load attempts. Thus neither profile reaches its measured CUDA steps, and neither fit reaches training.

The actual metadata sidecar is 323,404 bytes, SHA256 `1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8`. It declares the same fold-0 manifest SHA as the packet, 10,535 train groups, 1,405 fit-valid groups and 468 evaluation groups. But this cache is a homogeneous strong-teacher cache: `family_order=["strong"]`; three own epoch-2 strong teachers each have coefficient 0.1, sum 0.3; the homogeneous target multiplier is 0.8125; and its student convention is six epochs, fixed CP5. The frozen validator requires the four-family order, at least one teacher from each family, source coefficients summing to 0.8125, and a two-epoch fixed CP1 student convention. The first listed failure is decisive; the later coefficient and epoch checks are additional contract mismatches, not separately reached runtime errors.

The cache declares `main_homogeneous_train.npy` with shape `[10535,20000,2]`, float32, 1,685,600,128 bytes and SHA256 `6a19f7ac45c6e8a02d3b89c0ac4986ffab4885ab7a4ebc5fb4599b3047f4e7b3`; `main_homogeneous_valid.npy` is `[1405,20000,2]`, float32, 224,800,128 bytes, SHA256 `d1cd78537f3b6aaa830861ac771cf5c4621f1f2c1f19a73f6affb813c33ca3a2`. These are metadata-declared values only; the payloads were not read or rehashed. The cache's ordered row-group metadata hashes equal its own ordered fit-group hashes. Since validation fails first, the validator's comparison with `train_targets.npz`/`valid_targets.npz`, the file shape/dtype/hash checks, and the finite-value scan are not reached.

If those checks passed, `Sequences` would map each selected `row_group` to its metadata-listed array position, then open `cache_root / f"main_{variant}_{split}.npy"`. For this actual cache, those generated paths match the metadata-declared paths. The training loop multiplies loaded targets by the command's `--distill-cache-scale=1.0`; therefore the stored 0.8125 target multiplier remains the effective teacher amplitude. The packet's `fit_schedule.scale=1.2307692307692308` conflicts with `target_scale=1.0` and the actual CLI value; it is a stale/ambiguous summary field, not a second runtime scale. R1 should not be used to infer a different target amplitude.

The row-group hashes establish only metadata-list consistency. Because the validator stops before reading target-index ordering or cache arrays, this review does not claim end-to-end row-to-target alignment, payload integrity, or finite values. A successor must bind a consumer path that accepts this exact homogeneous producer contract and demonstrate membership/order/path/scalar behavior against the actual producer metadata and appropriate bounded payload checks. Do not treat disabling or bypassing validation as a repair.

## Scope and resource record

The planned device profile is profile-only, batch 16 × 20,000 rows × 291 inputs, 2,000-row microchunks, five warmup and five measured steps, with a 6,000,000,000-byte worker reservation and 1,200-second bound. This is a proposed envelope, not measured evidence. No GPU command, profile, fit, prediction or grade was launched in this review.

The actual cache-consumer check ran with `scripts/limited`, one CPU, 400,000,000-byte memory cap and 120-second limit. It completed in 0.17 seconds, maximum RSS 78,636 KiB, with zero swaps. The root-provided conservative admission bracket for the earlier bounded verifier was parent 10,841,759,744 bytes, measured active worker 4,671,320,064 bytes against a 5,500,000,000-byte reservation (unused 828,679,936 bytes); plus a 1.2 GB verifier, 700 MB operator, 400 MB root task and 1 GB uncertainty margin, total 14,970,439,680 bytes below the 17 GB committed-memory target. The bounded verifier itself peaked at 657,256 KiB; this review did not alter caps or shared admission state.

The source/membership and ancestry checks are in `VERIFY_R1_RESULT.json` and `ANCESTRY_R1_RESULT.json`; command checks and exact packet digests are in `COMMAND_REVIEW_R1_RESULT.json`; actual cache metadata and the no-array-read validator result are in `CACHE_METADATA_R1.json`. `CACHE_METADATA_R1.json` is a compact summary, not a copy of the 323 KB metadata sidecar. `review_receipt.json` gives the machine-readable verdict and resource records.

R1 failure is operational only. The experiment remains scientifically untested; no model-quality outcome or negative update is justified.

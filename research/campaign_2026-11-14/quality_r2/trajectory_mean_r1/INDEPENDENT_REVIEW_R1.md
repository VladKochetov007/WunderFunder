# Independent implementation review R1

**Scope.** Read-only review of `derive_mean.py`, `grade_local.py`, `export_local_r2.sh`, `export_predictions_r2.py`, `LOCAL_CONFIRMATION_R3.json`, the canonical prediction exporter and the local scoring helper. The full-valid export was active during review. No prediction arrays were opened, no `torch`/`numpy` was imported, and no fits or tests were run. No derived scores were inspected.

## Findings

### 1. The grader does not bind row-group IDs to their source — blocks trustworthy grading

`grade_local.py` checks that the metadata has 1,873 row-group IDs and that their **numeric set** equals `valid_targets.npz` IDs. It does not check `metadata["split"]`, `campaign_split_manifest_sha256`, `split_manifest_sha256`, or `evaluation_membership_sha256`. The producer writes all four source/membership fields; `campaign_prediction_groups()` also requires the requested split to be `valid` and binds the campaign manifest to the checkpoint. The grader currently checks only the self-reported `campaign_export_scope` string and numeric IDs. Since row-group numbers are source-local, equal numbers do not prove equal source membership. A stale or incorrectly sourced sidecar/prediction can therefore pass the grader if its IDs collide with the valid IDs.

**Fix:** require `split == "valid"`; bind both manifest hashes to the frozen `chronological_primary.json`/checkpoint split hash; recompute and compare the membership digest; retain the exact valid-group comparison. Treat `(source, row_group)` as the identity, not the integer alone.

### 2. The grader does not reject a valid-fitted checkpoint — blocks the train-only interpretation

The grader reads run provenance but never checks its training-source or fit-membership fields. It can grade and trigger seed-5 work for a run whose provenance says it included held-out valid groups in fitting. The inspected parent provenance records `train_source="train"` and `include_heldout_valid=false`; that is reassuring evidence about these parents, but the grader does not enforce it. `full_valid=true` is also present for monitoring, so it must not be mistaken for fit membership or used as the sole leakage check.

**Fix:** validate the serialized args (`train_source == "train"`, `include_heldout_valid is false`) and source-qualified `fit_row_groups`/membership against the frozen split. Reject any valid-source fit membership. Keep evaluation/monitoring exposure labelled separately; this valid panel is already exposed.

### 3. The checkpoint hash check compares sidecar claims, not checkpoint bytes — stale or wrong checkpoint can pass

`grade_local.py` asserts that the export metadata's `checkpoint_sha256` equals the value claimed by `provenance.json`, then hashes the provenance file. It never hashes `RUNS/run/model_ep3.pt`, and it does not check `checkpoint_file`. If the checkpoint is replaced after export while its provenance remains stale, the grader will still score the old predictions as though they belong to the current checkpoint. This breaks the stated prediction/checkpoint binding.

**Fix:** hash the actual `model_ep3.pt`; require that digest to equal both the export metadata and provenance entry, and require `checkpoint_file == "model_ep3.pt"`. For derived endpoints, also verify the recorded derivation parent/checkpoint hashes against the derivation receipt.

## Checks that passed by inspection

- The derivation selects epochs 2 and 3 from each *same named parent run*, checks both checkpoint hashes against parent provenance, requires identical args, epoch labels 2/3, FP32 finite tensors and matching layouts, and performs a fixed 0.5/0.5 arithmetic mean. It preserves equal `mean` and `std` buffers instead of averaging them. The small derived receipts bind each mean to its parent and list no new fit or quality result. I found no mathematical or ancestry defect in this path.
- The export wrapper pins IEEE FP32 recurrent inference and disables autocast/TF32. The producer uses the valid parquet source for full-valid feature-only export and emits source, scope, checkpoint, provenance and membership fields; the weakness is that the grader does not verify several of them.
- The seed-5 trigger code matches R3's stated thresholds and at-most-two-fit cap by inspection. It only proposes runs; it does not launch them. Seed-3/full-valid comparisons remain exposed single-seed screening evidence, not confirmation or a public-score estimate.

## Limits

This review establishes implementation-level gaps, not that any active export is wrong. The exporter’s own source checks and the inspected parent provenance support the declared current lineage. No output predictions or derived scores were inspected, and no run was independently replayed. The grading gaps should be fixed before relying on its scores to admit confirmation fits.

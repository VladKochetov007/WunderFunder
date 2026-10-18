# Independent review request: F3 result collection R1

Review the exact `PREPARATION_MANIFEST.json` and this packet's source files. Please do not run remote commands, load prediction/checkpoint payloads, or run the collector against the campaign.

Check that the collector binds the unmodified F3 plan/fold/split metadata, all seven original v1 attempt records and source manifests, both stage-1 and terminal checkpoint receipt pairs, exact fit/evaluation order and counts, checkpoint ancestry, actual arrival seal and both exact PASS review files, fixed epoch-5 sidecars/exports, and the existing paired grade. Inspect source path and create-only publication safety, bounded memory/disk behavior, and snapshot evidence inventory. Run only the tiny offline unit tests if useful.

The review receipt should state explicit `status: "passed"` and `decision: "PASS_STATIC_REVIEW_READY_FOR_ROOT_GATED_COLLECTION"` or report blockers, and bind the exact `manifest_sha256` plus test-receipt SHA. Do not claim F3 quality or recommend admission; root retains admission and any execution.

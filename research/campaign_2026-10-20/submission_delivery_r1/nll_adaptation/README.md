# Local NLL adaptation

This folder contains the frozen two-fit recipe and the prediction/scoring handoff for the H256×2 seed-3 F0 continuation. The fit recipe compares chunk-WMSE with diagonal Gaussian NLL from the same parent CP5, exact MainT3 cache target, precision, optimizer, schedule, and fixed CP1 endpoint. See [recipe.json](recipe.json) for the exact profile and fit argv, hashes, and target-scale contract.

The [fold and production asset catalog](FOLD_ASSET_CATALOG_R1.md) records reusable F1/F2 continuation parents, target/propensity cache availability and the all-data production lineage. It supports a later focused follow-up if F0 warrants it; its metadata-only ancestry findings preserve the exposed-fold and missing-cache limitations.

Root reports both bounded profiles exited 0 with finite updates, 60 chunk updates per arm, and about 1.68 seconds per full B16 training batch. Fit execution and result collection remain root-owned. No export or grading result is claimed in this preparation note.

## Prediction and scoring

- [evaluation_manifest.json](evaluation_manifest.json) pins the fold0 manifest and lists all 468 source-qualified `(valid, row_group)` evaluation groups with their component IDs. It records all 1,873 valid-source prediction rows and labels the other 1,405 rows as fitting diagnostics; only the 468 evaluation rows are scored.
- [score_plan_r1.json](score_plan_r1.json) freezes the exact predictor argv, cache identities, source pins, runtime limits, byte checks, and scoring command. It uses the pinned 68ff predictor's existing no-grad B16/C4000 path. The first live batch is monitored under the root-admitted RAM and allocator caps.
- [score_screen_f0_r1.json](score_screen_f0_r1.json) fixes seven endpoints: reused parent CP5; WMSE and NLL CP1; the fixed 50/50 parent/NLL blend; a unique-name FP32 re-export of old Z4 CP1; and fixed 75/25 old-Z4/WMSE and old-Z4/NLL blends. There is no coefficient search. The existing frozen `screen_score.py` uses the 2,000-replicate, seed-271828 paired component bootstrap.

The parent prediction cache has been byte-hash checked and its sidecar has all 1,873 row groups in order. The historical old-Z4 FP16 prediction cache is also byte-checked but remains an untouched historical reference; all new Z4 blends use the separately exported FP32 cache stem. Its prior F0 score was 0.683142 with paired delta +0.000318 (95% component-bootstrap interval −0.001320 to +0.001799) against the older parent; see the pinned receipt in `score_plan_r1.json`. An independent critic cleared its source-qualified F0 ancestry (13 run records, including eight teachers) before the two fixed Z4 blends were added.

Prediction export reads features only. It selects all valid row groups from `valid_targets.npz` for row-order compatibility, then `FeatureOnlySequences` reads `valid.parquet` with `targets=false` and `mask=false`. The scorer joins prediction rows by the target cache's row-group index and applies `global_wp_clip2_v1` only to the frozen 468-group evaluation panel. F0 is an exposed development panel, not a pristine holdout or a public forecast.

The new mean/logvariance checkpoints retain 23 pre-variance outputs (main and auxiliary heads) and append two logvariance outputs, for 25 total head rows. Prediction exports only the first two mean columns. The old `profile-only` predictor gate is specific to H384; no source changes or profile-only run are part of this handoff.

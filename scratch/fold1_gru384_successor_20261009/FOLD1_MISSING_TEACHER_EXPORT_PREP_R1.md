# Fold1 teacher export preparation

Status: preparation only. This does not authorize prediction export or cache construction. The two new baseline checkpoints are still being produced, so their checkpoint and provenance hashes must be bound in a successor packet after both fits pass the terminal gate.

The missing artifacts are eight FP32 prediction arrays for the existing fold1 four-family distillation cache:

| Source checkpoint | Output split | Fixed epoch | Expected shape | Expected file bytes |
|---|---|---:|---:|---:|
| `c20261003_combo_s3_f1` | train | 3 | `(10606, 20000, 2)` | 1,696,960,128 |
| `c20261003_combo_s4_f1` | train | 3 | `(10606, 20000, 2)` | 1,696,960,128 |
| `c20260929_ft_since_s3_f1` | train | 0 | `(10606, 20000, 2)` | 1,696,960,128 |
| `c20260929_ft_since_s4_f1` | train | 0 | `(10606, 20000, 2)` | 1,696,960,128 |
| `bd26_w384raw_s3_f1` | train | 3 | `(10606, 20000, 2)` | 1,696,960,128 |
| `bd26_w384raw_s3_f1` | valid | 3 | `(1873, 20000, 2)` | 299,680,128 |
| `bd26_w384raw_s4_f1` | train | 3 | `(10606, 20000, 2)` | 1,696,960,128 |
| `bd26_w384raw_s4_f1` | valid | 3 | `(1873, 20000, 2)` | 299,680,128 |

Total new array bytes are 10,781,121,024. All eight final arrays, `.partial` files, sidecars and per-run stale `seqsums_*.npz` files were absent at the static check. Recheck every one immediately before each export; `predict_run.py` can replace an existing final path and removes `seqsums_<run>.npz` after export.

The fold1 manifest hash is `31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae`. The fit pool contains 10,533 train groups and 1,416 valid groups. The 457 evaluated valid groups form 149 components and exclude 73 linked train groups. The complete target indexes used by `predict_run.py` have 10,606 train rows and 1,873 valid rows. Their ordered group digests are recorded in `FOLD1_MISSING_EXPORT_STATIC_PREFLIGHT_R1.json`.

The standard predictor exports full target-index order. That includes excluded groups in the raw arrays; it does not mean those groups enter a teacher fit or the student distillation cache. `scripts/build_distill_cache.py` first validates each teacher's exact fold1 ancestry and full prediction row order, then selects only `CampaignSplit.train_groups` and `CampaignSplit.valid_groups` when writing fit targets. Its ancestry closure uses `CampaignSplit.assert_fit_groups`, which rejects any fitted component that overlaps the 457 evaluation groups. The future comparison remains a historical fold1 robustness check, not a fresh holdout or a public-score forecast.

The four existing strong/full-finetuned valid arrays and the four existing raw train/valid arrays were checked without loading their payloads into RAM: NPY headers report the expected FP32 shapes, their sidecar `row_groups` exactly match the full target-index order, and sidecar run/epoch/checkpoint/split identities match the direct source checkpoint and frozen fold1 manifest. The source checkpoints and every initializer/propensity ancestor were also checked by the actual `build_distill_cache._audit_ancestor_closure`; see `FOLD1_TEACHER_ANCESTRY_PREFLIGHT_R2.json`. Full prediction payload SHA and finiteness scans remain for the final consumer preflight because this static check intentionally did not read several gigabytes of NPY data.

The four current source checkpoint SHA values are:

- `c20261003_combo_s3_f1@ep3`: `b0dcc761751e3f807dba42d55650f06f392b2565fb51e4ba787d3603c13b3cc4`
- `c20261003_combo_s4_f1@ep3`: `456114b04d1371b994ed5d7fbfe9b39ee96f3c460286644baf161c8f3fbcd610`
- `c20260929_ft_since_s3_f1@ep0`: `1a76c748624e7c9a438675791c251ed142b7965879dfed691c210ae1881a1d7d`
- `c20260929_ft_since_s4_f1@ep0`: `814f6106f5422c962ff175e98bba15995b2cc45dc1fc0d172e88020784842330`

The raw teacher and propensity ancestry checkpoint hashes are bound in the ancestry preflight. New baseline checkpoint/provenance hashes are intentionally unresolved until the admitted seed3/seed4 fits terminate successfully.

Commands for the eventual eight exports use the existing in-repo predictor and exact epoch arguments. They omit `--output-name`, so the canonical `pred_<run>_<split>.npy` filename and sidecar `run` stay aligned with the downstream consumer. `--inference-precision checkpoint` is made explicit to match the current default; arrays remain FP32 storage. No output names are aliased. Do not run these commands until root binds actual completed baseline checkpoint hashes, freezes a bounded guarded export queue, and admits its disk/resource allocation.

`predict_run.py` sidecars record run, split, epoch, checkpoint hash, split-manifest hash, prediction hash, precision and row groups. `build_distill_cache.py` requires exact run/split/epoch/checkpoint/split identities, exact full row-group order, matching prediction SHA, shape `(row_count, 20000, 2)`, float32 dtype and finite values. It does not require an `inference_precision` field from older sidecars; those existing sidecars omit it. The future export consumer preflight must still run the actual cache builder's input validation over every source, including full file-hash/finiteness scans, before any fold1 cache build. Cache building and score evaluation are outside this export allocation.

Reviewed producer/consumer sources and exact SHA pins are listed in `FOLD1_MISSING_TEACHER_EXPORT_PREP_R1.json`. The current predictor, trainer and consumer source files are unchanged. No export, cache build, score or new fit was run for this preparation.

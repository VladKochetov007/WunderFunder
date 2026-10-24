# Independent review — F1 mean-shrink R2 metadata repair

## Finding

**PASS to root admission for the R2 label-free preflight. No blocking R2 contract defect found.** This does not authorize a quality grade by itself. Grade remains gated on a new root admission bound to the exact R2 recipe, source manifest and preflight receipt. I did not run preflight/scoring or read prediction/target arrays.

## Frozen identities checked

- Recipe: [F1_MEAN_SHRINK_RECIPE_R2.json](F1_MEAN_SHRINK_RECIPE_R2.json), SHA-256 `3bfe93978314a6448754467a6f2106220a4f51ca0d87e4fb0130cb2bad74e243`.
- Source manifest: [F1_MEAN_SHRINK_SOURCE_MANIFEST_R2.json](F1_MEAN_SHRINK_SOURCE_MANIFEST_R2.json), SHA-256 `287371e5a8bc42c38a37d32388e70e2c9125ac98a97b44e4125643161d07e7ee`.
- R2 preflight: [f1_grade_preflight_r2.py](f1_grade_preflight_r2.py), SHA-256 `5651d3aeca1d6e74f21edaf3ba55c8293daa1f3c29ae3c86cd7fa2869cbb3634`.
- R2 runner: [grade_f1_mean_shrink_r2.py](grade_f1_mean_shrink_r2.py), SHA-256 `9e06a2fcd495e1d82f89429c20a4601d2f08eeb54d10f92eab077b3c77f71038`.
- Unchanged scorer: [f1_grade_scoring.py](f1_grade_scoring.py), SHA-256 `ca539f151c565d939d4002bfdb3473ba0c761ededede97be3228fd6c8fb493f0`.
- Recipe and manifest hashes match. All 86 files listed in the R2 source manifest exist and match their SHA-256 pins. The preserved R1 snapshot's five files also match its snapshot receipt.

## R1 failure and R2 correction

The [preserved R1 failure record](PREFLIGHT_EXECUTION_R1.json) and log show a 0.232-second failure at the teacher-cache provenance-path assertion, before candidate/parent prediction hashes or target-archive hash. No labels were decoded. The producer's actual eight `cache.sources` records contain `run`, checkpoint, fit counts/digests and prediction metadata, but no `provenance_path` field.

R2 fixes this at the correct boundary. For each source run named by the pinned cache coefficients, it resolves the repository's canonical `artifacts/runs/<run>/provenance.json`, hashes that file, and requires the digest to match the frozen ancestry run pin. It then checks the provenance fit rows against the cache's fit counts and ordered hashes, and preserves checkpoint, membership, split, prediction-sidecar and recursive ancestry checks. The R2 change also records the resolved path and verified digest in the ancestry report. It no longer requires a producer field that does not exist.

The R1 and R2 recipes differ only in metadata/plumbing: R2 output, preflight and admission paths; the R2 manifest path/hash; the ancestry-check record; status; and a revision note. The experiment, fixed float32 alpha `0.1`, endpoint definitions, folds/seeds, membership, metric, contrasts, bootstrap and resource limits are unchanged. The scoring implementation is unchanged.

## Ancestry-only evidence reviewed

The [R2 ancestry-only checker](f1_ancestry_metadata_check_r2.py) uses the R2 recipe and preflight ancestry verifier. Its recorded [receipt](F1_ANCESTRY_METADATA_CHECK_R2.json), SHA-256 `12b74c7ae26a7ee9c30cf37c547b0b15ea1911bdeaaf850f1ddd4a7bdd878ad6`, reports success for 14 runs, 27 edges, eight cache sources and 457 evaluation groups. The report says prediction and target arrays were not opened or hashed, target values were not read, and checkpoints were hashed but not imported or deserialized. The separate execution record reports the 150 MiB, one-core, 60-second bound and no retained peak-memory figure.

This supports the repaired ancestry metadata join. It does not replace the R2 full preflight: the full preflight still needs to verify the candidate and parent NPY headers/full digests and the target archive's frozen digest. It will read those files for hashing/header inspection but does not decode target labels. The quality grade must remain behind the exact root admission gate in the R2 runner.

## Scientific and resource contract

The R2 protocol remains the same one exposed F1/A/lag-1/seed-1002 readout-only comparison, with zero new fits. It uses 457 ordered valid groups and 149 connected components; the candidate readout fit subset is 1,243 groups across 314 components, disjoint from evaluation. The parent remains `oldz4_h230_parent_f1_s3_r4` epoch 3; historical FP16 AMP training and the cached CPU strict-FP32 endpoint distinction stay explicit.

The primary contrast remains fixed-shrink minus parent; secondary contrasts remain raw A minus parent and shrink minus raw A. The official metric remains `global_wp_clip2_v1`, with the original clipped-target-magnitude weights, ±2 clipping, and official row mask. The component bootstrap is conditional on the saved endpoints, not training or transfer uncertainty. This remains an exposed, post-F0-designed diagnostic, not independent confirmation or completion of the historical 12-fit panel.

The recipe and runner retain the 400,000,000-byte, one-core, 300-second, zero-swap, physical-scratch bound. Actual preflight/grade memory and runtime have not been established by this review. Root should run the prescribed limited command for preflight and create a separate R2 admission before any target-quality read.

## Limits

The historical teacher-prediction sidecars still lack explicit `scope` and no-target-read fields; the R2 verifier records this and validates run/checkpoint/split/epoch/order/digests and source-qualified fit membership. It does not hash teacher prediction arrays. This metadata limitation remains nonblocking for the fixed saved-endpoint grade, but the ancestry check should not be described as a teacher-payload audit. No R2 full preflight, scoring, fit or target-quality read was performed in this review.

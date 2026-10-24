# Independent review: exploratory F0-only online grade

## Review decision

**Conditional pass on the scientific question; fail on using the current
frozen grader or treating current artifacts as grade-ready.** Comparing the
seven fixed F0/seed1001 endpoints is a useful, bounded exploratory result that
costs no fits. The result would be about those fixed endpoints on an already
exposed F0 evaluation set. It would not complete the old panel, establish
fresh confirmation, estimate seed variation, or predict public transfer.

At the first review snapshot, only `feasibility_scout/REPORT.md` existed; the
author has since added `grade_online_f0.py` and its preflight/scoring modules.
The frozen `F0_EXPLORATORY_RECIPE_R1.json` has not yet appeared. The old
`grade_online_relationships.py` cannot grade the subset:
`validate_recipe` requires exactly 12 admitted fit cells across F0/F1 × lag
1/16 × A/B/C, and `grade_fold` hardcodes 13 old contrasts per fold. Keep the
old recipe, grader, terminal receipt and incomplete-panel status unchanged. A
new frozen F0 recipe, separate adapter and create-only result receipt are
required.

## Frozen scope and comparison accounting

- Endpoints: the exact F0/seed1001 parent prediction plus A, B and C at lag 1
  and lag 16. No new fits. The old 12-fit panel remains incomplete.
- Primary contrasts: `C−B` and `B−A` within each lag, four primary contrasts.
- Retain the full original contrast set as descriptive secondary results:
  `C−A`, A/B/C versus the parent within both lags, and `lag16−lag1` for each
  arm. That is 13 old contrasts plus the two new B−A contrasts, or 15 total
  fold-local contrasts. Label them and their uncertainty scope in the recipe
  and result; do not let code ordering make every row appear primary.
- Use `global_wp_clip2_v1`, its unchanged ±2 clipping and clipped-target
  magnitude weights, the original score mask, both target correlations and
  the frozen four position bins. The score is pooled across selected rows,
  not a mean of sequence scores. Primary bootstrap intervals are for overall
  WP only; point deltas for the other targets/bins remain descriptive.
- F0 contains 468 evaluation groups and 141 dependency components. Preserve
  the source-qualified ordered group list and the component mapping from the
  frozen split. All local evaluation folds are exposed. Keep the conclusion
  exploratory and conditional on these fixed fits.

The contrast arithmetic above follows the original recipe's five contrasts
per lag (`C−A`, `C−B`, and A/B/C−parent) plus three lag contrasts. Adding
`B−A` per lag yields 15. C−B remains a primary contrast, while C−A and the
parent/lag contrasts are secondary under the new question.

## Initial adapter code review

The first adapter snapshot was available for source review after the protocol
check. These are concrete issues in that snapshot; recheck the files after
the author records any revisions.

1. **Blocker — terminal source-pin schema mismatch.** In
   `f0_grade_preflight.py`, `_verify_historical_source_pins()` indexes
   `terminal["source_validation"]["runtime"]` and `["grader"]`, then expects
   `expected_sha256` and `observed_sha256` maps. The pinned
   `PANEL_TERMINAL_R1.json` instead has `runtime_pin_count`,
   `grader_pin_count`, `mismatches`, and
   `admission_and_execution_recipe_hashes_agree`. It has no `runtime` or
   `grader` object, so preflight will fail with `KeyError` before validating
   endpoints. Either validate the actual terminal schema or use an explicitly
   pinned source manifest for the historical checks; do not alter the old
   receipt to fit the new reader.
2. **Blocker — ancestry is not recursive yet.** The preflight checks the
   selected parent's checkpoint/provenance file hashes, and candidate
   provenance checks equality to those frozen hashes. It does not parse the
   selected parent's provenance or resolve its `initial_ancestor` and
   teacher/initializer records. That leaves the required chain
   candidate → F0 parent → initial/teacher lineage semantically unverified.
   Inspect the frozen parent provenance and validate the linked artifact
   hashes and fitting-membership exclusions before calling ancestry checked.
3. **Blocker — worker cap is not bound by the grade admission reader.**
   `_require_admission()` rejects shared memory above 1 GiB but does not
   require the requested worker cap of at most 400,000,000 bytes. The root
   receipt and invocation must bind that stricter worker limit while marimo
   is live; the shared cap alone does not prove compliance.
4. **Integrity/reporting gap — target SHA only checked before extraction.**
   `score_prepared_f0()` hashes the target archive before extraction, but
   never rechecks it after streaming/scoring and does not record the computed
   digest/verified-read status in the grade result. Add a post-read hash
   check and report the actual verified hash. The label-free preflight's
   expected hash and byte size are useful but are not themselves a content
   check.
5. **Descriptive output gap — bin deltas are omitted.** The new scorer keeps
   all 15 contrast definitions and reports endpoint scores for the fixed
   bins, but its contrast objects contain deltas only for overall WP/T0/T1.
   The old grade contract planned target/bin point deltas for every contrast.
   Emit those deltas for the 11 secondary contrasts too, or explicitly mark
   the fixed-bin comparisons as omitted rather than implying the old
   descriptive output was retained.

The adapter does correctly define four primary contrasts and 11 secondary
contrasts (15 total), uses the canonical `GlobalAccumulator`, builds
per-sequence centered moments, and passes the same component mapping and
fixed bootstrap seed/draw count to both sides of each paired interval. It
streams compressed `y` and `mask` NPY members to scratch-backed maps after a
pre-score hash check. These are source-level findings only; no synthetic test
or campaign payload was run/read.

## Blocking preflight requirements

Do not open `cache/valid_targets.npz` or score any label until every candidate
and the parent has passed the artifact, order and provenance gate.

1. **Bind all seven complete prediction files.** Parse the NPY headers; require
   exactly `(468, 20000, 2)`, little-endian FP32, C order and exact header plus
   payload length. Compute each full-file SHA-256 in bounded memory and match
   it to its frozen sidecar/recipe identity. Reject truncated, trailing,
   non-finite or mismatched arrays. The six candidate prediction sidecars and
   the parent prediction manifest must each name the exact ordered F0
   `valid`/row-group references from the frozen split. Reject duplicates,
   subsets, supersets and permutations; do not accept just matching row
   counts. The old terminal receipt only checked candidate headers/byte
   counts: it did not verify prediction values/hashes, exact row-group
   membership, or the parent payload.
2. **Verify the parent endpoint.** Bind its prediction to the selected F0
   parent cache, checkpoint and provenance hashes, source/split identity,
   normalization identities and strict-FP32 evaluation endpoint. Do not use
   in-sample parent fit predictions as an evaluation endpoint. The historical
   FP16 training precision of the parent is not an FP16 prediction claim;
   record training and endpoint precision separately.
3. **Walk ancestry rather than trusting copied metadata.** For each of the
   six heads, require exact arm/lag/seed/run/epoch, candidate checkpoint,
   provenance and prediction hashes; bind the same F0 parent checkpoint and
   provenance, normalization, subset manifest and selected fit-membership
   digest, split, canonical propensity checkpoint/provenance, cache payloads
   and sidecars, and the source code hashes used for fitting/export. Confirm
   that all six use the frozen matching F0/1001 cell and that the fit subset
   excludes its 468 evaluation groups. Inspect the parent's own provenance
   and its initial-ancestor records/checkpoint hashes recursively through the
   declared teacher/initializer lineage; check the corresponding membership
   records for evaluation exclusion. A candidate provenance field that merely
   repeats a parent hash is not by itself a recursive ancestry audit. If an
   ancestry link cannot be checked, fail admission or state exactly which
   link is unverified; do not imply clean ancestry.
4. **Bind target transport and membership.** Before label access, verify the
   target archive SHA against the frozen recipe and split source record. Then
   inspect the member names, NPY headers, `row_group`, target and mask shapes,
   dtypes and full archive byte coverage. Require unique row-group IDs and
   align targets/masks by the exact source-qualified F0 order, not by assumed
   NPZ order. Resolve every evaluation group to exactly one dependency
   component and verify the frozen count of 141 components. Recheck the
   target archive hash after streaming/scoring. A cache filename or sidecar
   claim alone does not establish payload identity.
5. **Preserve causal order/reset semantics.** Candidate sidecars must bind the
   exact prediction row order to the split order. Pin and check the original
   producer code/provenance for causal input construction, state reset at
   every sequence boundary, and the lag-1/lag-16 maturity convention. Do not
   infer independent checkpoint-to-prediction parity: the old grader verifies
   exported prediction bytes and provenance but does not rerun candidate
   checkpoints. Label this as a limitation unless a separate authorized
   parity check is done.
6. **Compute the official metric and paired uncertainty.** Use the canonical
   metric implementation or a reviewed equivalent with the same mask and
   clipping/weights. Pool all selected rows for each target, then average the
   two target correlations. For each contrast, use the same 141-component
   resample multiplicities for candidate and reference and recompute pooled
   WP from stable centered moments. Freeze the bootstrap draw count and seed
   in the new recipe (the old code uses 2,000 draws and seed 271828 if reused
   without changing its contract). Report point delta direction explicitly
   (`candidate − reference`) and label intervals conditional on fixed
   predictions; they do not quantify training-seed or fold variance. Do not
   pool F0 with F1 or use these exposed labels to tune/choose variants.

## Initial adapter code review

The first adapter snapshot was available for source review after the protocol
check. Recheck these files after the author records revisions.

1. **Blocker — terminal source-pin schema mismatch.** In
   `f0_grade_preflight.py`, `_verify_historical_source_pins()` indexes
   `terminal["source_validation"]["runtime"]` and `["grader"]`, then expects
   `expected_sha256` and `observed_sha256` maps. The pinned
   `PANEL_TERMINAL_R1.json` instead has `runtime_pin_count`,
   `grader_pin_count`, `mismatches`, and
   `admission_and_execution_recipe_hashes_agree`. It has no `runtime` or
   `grader` object, so preflight will fail with `KeyError` before validating
   endpoints. Validate the actual terminal schema or use an explicitly
   pinned source manifest for these historical checks; do not alter the old
   receipt to fit the new reader.
2. **Blocker — ancestry is not recursive yet.** The preflight checks the
   selected parent's checkpoint/provenance file hashes, and candidate
   provenance checks equality to those frozen hashes. It does not parse the
   selected parent's provenance or resolve its `initial_ancestor` and
   teacher/initializer records. That leaves the chain
   candidate → F0 parent → initial/teacher lineage semantically unverified.
   Inspect the frozen parent provenance and validate linked artifact hashes
   and fitting-membership exclusions before calling ancestry checked.
3. **Blocker — worker cap is not bound by the grade admission reader.**
   `_require_admission()` rejects shared memory above 1 GiB but does not
   require the requested worker cap of at most 400,000,000 bytes. The root
   receipt and invocation must bind that stricter worker limit while marimo
   is live; the shared cap alone does not prove compliance.
4. **Integrity/reporting gap — target SHA only checked before extraction.**
   `score_prepared_f0()` hashes the target archive before extraction, but
   never rechecks it after streaming/scoring and does not record the computed
   digest/verified-read status in the grade result. Add a post-read hash
   check and report the actual verified hash. The label-free preflight's
   expected hash and byte size are useful but are not themselves a content
   check.
5. **Descriptive output gap — bin deltas are omitted.** The new scorer keeps
   all 15 contrast definitions and reports endpoint scores for the fixed
   bins, but its contrast objects contain deltas only for overall WP/T0/T1.
   The old grade contract planned target/bin point deltas for every contrast.
   Emit those deltas for the 11 secondary contrasts too, or explicitly mark
   the fixed-bin comparisons as omitted rather than implying the old
   descriptive output was retained.
6. **Freeze the contrast contract in both recipe and scorer.** The scorer
   currently builds its 15 contrasts in code while the preflight report copies
   primary/secondary labels from the recipe. Validate that the frozen recipe's
   complete contrast list and roles exactly equal the scorer's list, so the
   report cannot describe a different primary analysis from the one executed.
7. **Record candidate inference precision from the producer contract.** The
   candidate prediction sidecar records `<f4`, and the pinned producer source
   uses FP32 feature/prediction tensors, disables autocast for fitting and
   disables CUDA TF32. The sidecar/provenance does not explicitly record the
   actual evaluation device or endpoint precision. Preserve that source-level
   basis in the grade's endpoint identity and label the result as fixed
   exported FP32 predictions; do not treat NPY dtype alone as proof of the
   entire inference path.

The adapter does correctly define four primary contrasts and 11 secondary
contrasts (15 total), uses the canonical `GlobalAccumulator`, builds
per-sequence centered moments, and passes the same component mapping and
fixed bootstrap seed/draw count to both sides of each paired interval. It
streams compressed `y` and `mask` NPY members to scratch-backed maps after a
pre-score hash check. These are source-level findings only; no synthetic test
or campaign payload was run/read.

The recursive ancestry needed to close finding 2 is available in small local
metadata: the F0 parent provenance binds initial checkpoint SHA
`d4a1e50b…`, which matches `c20261003_combo_s3_f0@ep3`; that run's provenance
names teacher checkpoint/prediction hashes for `c20260929_since_s3_f0@ep3`
and `c20260929_since_s4_f0@ep3`, whose provenance files declare no further
init/distillation parent. All four records share split SHA `09263ad4…`; each
lists 468 evaluation groups and 1,405 valid fit groups, with no
source-qualified `valid` fit/evaluation overlap. The numeric group IDs overlap
across the distinct `train` and `valid` namespaces, so a recursive check must
retain the split name in every membership reference. This reviewer checked
those JSON fields only; no lineage checkpoint or prediction payload was
hashed or loaded.

## Memory and deadline gate

The user's marimo amendment permits up to ten aggregate CPU cores; it does not
raise the shared 1,073,741,824-byte memory ceiling or remove the no-swap
constraint. The worker cap must be at most 400,000,000 bytes while marimo is
active. Record the actual worker cgroup peak and OOM counters, plus shared
slice usage, against those limits. Put stream/member temporary files only on
physical scratch under `/home/vlad/development/wunder/research_tmp`.

Stream the compressed target members to scratch-backed NPY files or score
them in bounded group chunks. Do not materialize `y`, `mask`, all endpoint
copies or an evaluation-fold duplicate in RAM. Keep prediction arrays as
read-only memmaps and accumulate each endpoint's metric/centered component
moments in bounded chunks. A fixed-endpoint component bootstrap needs only the
component sufficient statistics, not full prediction copies. Measure memory
while the marimo process is actually live, and stop before any limit breach.
The old four-gigabyte worker run is not evidence for this cap; it reached a
3,999,997,952-byte peak under the old terminal receipt.

## Claim limits and provenance of this review

Confirmed source facts: the original recipe declares `global_wp_clip2_v1`,
primary `C−A`, `C−B`, and each arm versus parent, with three lag contrasts as
secondary; the frozen scorer computes the original 13 contrasts per fold;
the recipe declares 468 F0 evaluation groups and 141 components; and the
terminal record says the original panel timed out with seven of twelve heads
exported and no quality grade. The original terminal inventory and independent
metadata review explicitly did not validate candidate payload hashes, parent
coverage or complete fourteen-endpoint coverage.

This reviewer read the resume and CPU amendment, the feasibility report, the
original recipe/spec/terminal records, grader source modules and official
metric/bootstrap implementations. No prediction or target arrays were
opened, no score or synthetic test was run, and no fit/profile/import/provider
call was performed. The old full-panel result remains `INCOMPLETE`; the
proposed F0 grade is not admitted by this review until its separate recipe and
adapter exist and pass the checklist above. Root retains admission and
integration authority.

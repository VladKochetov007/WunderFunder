# F1 mean-only fixed-shrink follow-up

The [completed R2 result](RESULTS_R2.md) is essentially flat against the parent:
−0.000004696 WP, with conditional interval spanning zero. It removes most of
the raw A loss but does not reproduce F0's small gain. The [grade](F1_MEAN_SHRINK_GRADE_R2.json)
finished in 15.036 seconds under the admitted 400,000,000-byte, one-core,
zero-swap cap after the repaired full preflight. The [independent scalar result
review](RESULT_REVIEW_R2.md) confirms the reported arithmetic, coverage and
hash stability; it did not open prediction or target arrays or rerun scoring.
Historical preparation and failed R1 records below are preserved.

This folder contains a new zero-fit follow-up for the saved F1, seed-1002,
lag-1 mean-only readout. It compares the existing raw A prediction, its frozen
parent prediction, and `parent + float32(0.1) * (raw A - parent)`. The primary
contrast is shrink minus parent; the two secondary contrasts are raw minus
parent and shrink minus raw. The coefficient is fixed, with no scale search.

This was designed after the exposed F0 shrink diagnostic and is not an
independent confirmation. F1 is previously exposed development evaluation. It
does not complete the original 12-fit panel. The A arm forces its relationship
slope to zero but continues its decayed mean state updates; it is not a
no-history or raw-only control. Its 1,243 readout-fit groups come from 314
selected components (1,071 train and 172 valid), disjoint from the 457 held-out
valid evaluation groups and their 149 connected components.

The parent is `oldz4_h230_parent_f1_s3_r4`, epoch 3. Its historical training
precision was FP16 AMP; the saved evaluation prediction used by this packet is
CPU strict FP32. The ancestry verifier pins the fold-1 split and checks the
candidate fit subset, the parent's 10,533 train plus 1,416 valid fit groups,
all parent/teacher/propensity edges, the two saved resume-state edges, and
source-qualified membership against the fold's matching train/valid namespaces.
The teacher-cache producer schema has no explicit `scope` or no-target-read
flag for its historical teacher prediction sidecars. It does bind each output's
run, checkpoint, epoch, source split, ordered row groups and prediction digest;
the cache fit memberships are checked against the parent provenance and the
fold-1 fit namespaces, which exclude all 457 evaluation groups. The independent
review judged this metadata limitation nonblocking for this fixed-endpoint
exploratory grade.
No `.npy`, target, or prediction payload was opened while preparing this
packet. The R1 label-free preflight was attempted, then stopped before
prediction/target payload checks at the producer-schema mismatch described
below. R1 source files remain frozen in [r1snapshot](r1snapshot/).

## Frozen files

- [Recipe](F1_MEAN_SHRINK_RECIPE_R1.json) freezes the exact saved artifacts,
  457 ordered evaluation refs, 149 bootstrap components, source-qualified
  ancestry, unchanged official metric, three contrasts, and a 400,000,000-byte,
  one-core, five-minute, zero-swap worker. Its SHA-256 is
  `78537f892ba969a6f6b61be00a557c93e88de8bf61339a8ef52d9e06f490c35b`.
- [Source manifest](F1_MEAN_SHRINK_SOURCE_MANIFEST_R1.json) pins the scorer,
  preflight, runner, historical F1 producer metadata, all 27 source files named
  by candidate provenance, the ancestry provenance files, and the fixed F0
  context, plus 16 teacher-prediction sidecars. Its SHA-256 is
  `1b4f4f7c21aedea95e5bbc287438f230994b9b470dc502604d54b0d03e88b87f`. Prediction
  and target payloads are digest-checked by preflight and are not copied into
  this folder.
- [Preflight adapter](f1_grade_preflight.py) validates the recipe and source
  manifest, exact source-qualified order, source subset and component
  separation, parent-cache contracts, candidate provenance, recursive F1
  ancestry and full prediction/target byte identities. It does not decode
  target labels.
- [Scorer](f1_grade_scoring.py) reuses the original streaming target-map
  helper, official `GlobalAccumulator`, and component-bootstrap implementation.
  It scores one group at a time and rechecks target and prediction identities
  after scoring.
- [Runner](grade_f1_mean_shrink_r1.py) has create-only output paths and requires
  root admission bound to the exact recipe, source manifest, preflight receipt,
  metric and worker limits before label reads.

The parent-folder README is maintained by the resume controller. The R1 files
above preserve the initial review state; R2 admission, full preflight and grade
records are now included below. The admitted grade performed one target-quality
read for these fixed endpoints and introduced no new model fit.

The [independent static review](INDEPENDENT_REVIEW_R1.md) found no contract
blocker for root admission and confirmed the frozen recipe/source pins. It did
not run preflight or read arrays/labels; the historical teacher-sidecar scope
limitation and unmeasured worker peak are recorded there.

## R1 preflight failure and R2 correction

The [R1 execution record](PREFLIGHT_EXECUTION_R1.json) shows a 0.232-second
failure before prediction or target payload checks, with no decoded labels and
no OOM/swap. The verifier expected
`cache.sources[].provenance_path`, but the actual producer schema has no such
field. Each cache source binds its `run`, checkpoint, `fit_group_counts`,
`fit_row_group_hashes`, prediction records, and `ancestor_closure`. Recursive
provenance is separately pinned by run ID and SHA. The R1 code asked for a
field the producer never emits.

R2 preserves these joins. It resolves each source's canonical
`artifacts/runs/<run>/provenance.json` from the source run ID, verifies that
file against the frozen run provenance hash, then retains the existing
checkpoint, fit-membership, row-order, sidecar, split, and ancestry checks. It
also records the derived provenance path and source pin in the ancestry report.
The original R1 source packet is byte-pinned in
[r1snapshot/SNAPSHOT_RECEIPT_R1.json](r1snapshot/SNAPSHOT_RECEIPT_R1.json).

- [R2 recipe](F1_MEAN_SHRINK_RECIPE_R2.json), SHA-256
  `3bfe93978314a6448754467a6f2106220a4f51ca0d87e4fb0130cb2bad74e243`.
- [R2 source manifest](F1_MEAN_SHRINK_SOURCE_MANIFEST_R2.json), SHA-256
  `287371e5a8bc42c38a37d32388e70e2c9125ac98a97b44e4125643161d07e7ee`, with
  86 pins. It replaces the R1 preflight/runner with R2 files and includes the
  ancestry-only metadata checker; the scoring implementation and the other
  source pins are unchanged.
- [R2 preflight adapter](f1_grade_preflight_r2.py), SHA-256
  `5651d3aeca1d6e74f21edaf3ba55c8293daa1f3c29ae3c86cd7fa2869cbb3634`;
  [R2 runner](grade_f1_mean_shrink_r2.py), SHA-256
  `9e06a2fcd495e1d82f89429c20a4601d2f08eeb54d10f92eab077b3c77f71038`.
- [Ancestry-only checker](f1_ancestry_metadata_check_r2.py) runs the actual
  recursive ancestry verifier without opening prediction or target arrays.
  Its [result](F1_ANCESTRY_METADATA_CHECK_R2.json), SHA-256
  `12b74c7ae26a7ee9c30cf37c547b0b15ea1911bdeaaf850f1ddd4a7bdd878ad6`, passed
  all 14 runs, 27 ancestry edges, eight teacher-cache sources, and the 457
  evaluation groups. It verified source metadata and checkpoint hashes only;
  prediction/target arrays were neither opened nor hashed.
- [Execution record](ANCESTRY_METADATA_EXECUTION_R2.json) records the successful
  150 MiB, one-CPU, 60-second-limited run. Peak memory was not retained by the
  cleaned-up systemd scope.

The R2 full label-free preflight completed, followed by the separately admitted
quality grade. The preflight verified prediction and target payload hashes but
did not decode target labels. The grade read the authorized target labels to
score the three fixed endpoints, then verified the target and prediction hashes
again. Its execution receipt records 15.036 seconds, peak memory 399,998,976
bytes, and zero OOM or swap.

## Historical R2 preflight command and completed grade

The ancestry-only metadata check was run successfully with the exact R2 recipe:

```sh
systemd-run --user --scope --unit=f1-ancestry-meta-r2-20261024 \
  --property=MemoryMax=150M --property=CPUQuota=100% --property=RuntimeMaxSec=60s -- \
  timeout 55s .venv/bin/python \
  research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/f1_ancestry_metadata_check_r2.py \
  --recipe research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/F1_MEAN_SHRINK_RECIPE_R2.json \
  --recipe-sha256 3bfe93978314a6448754467a6f2106220a4f51ca0d87e4fb0130cb2bad74e243 \
  --output research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/F1_ANCESTRY_METADATA_CHECK_R2.json
```

The R2 full preflight command that completed is preserved for provenance. It
reads hashes and NPY headers for the parent/candidate predictions and target
archive; it does not decode labels:

```sh
WNN_MEM_MAX=400000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_RUNTIME_MAX=300 \
  scripts/limited .venv/bin/python \
  research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/grade_f1_mean_shrink_r2.py \
  --recipe research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/F1_MEAN_SHRINK_RECIPE_R2.json \
  --recipe-sha256 3bfe93978314a6448754467a6f2106220a4f51ca0d87e4fb0130cb2bad74e243 \
  --output /home/vlad/development/wunder/research_tmp/resume4h-r1/f1-mean-shrink-preflight-r2.json \
  --preflight-only
```

The [R2 admission](F1_MEAN_SHRINK_ADMISSION_R2.json) binds the frozen recipe,
source manifest and preflight receipt. The [grade execution
receipt](GRADE_EXECUTION_R2.json) records the admitted run and resource
counters. This R2 quality grade has been completed; no new fit was run.

The [independent R2 source review](INDEPENDENT_REVIEW_R2.md) passed the
metadata-path repair for preflight and admission, confirmed all 86 source pins
and the unchanged estimand, and retained the teacher-sidecar scope limitation.
The separate [scalar grade review](RESULT_REVIEW_R2.md) confirms the completed
result arithmetic and reported identities without accessing arrays.


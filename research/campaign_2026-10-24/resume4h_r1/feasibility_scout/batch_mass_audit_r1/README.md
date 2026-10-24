# F0/F1 fitting batch-mass audit

This packet records a descriptive audit of the existing online relationship
fitter's target-weight measure. It does not fit models or grade an endpoint.
The admitted runner streamed the exact five-column `(y0, y1, row_weight,
parent0, parent1)` suffixes from the original in-sample fit caches and verified
their hashes. No evaluation labels were read.

## Frozen inputs and source contract

The [recipe](BATCH_MASS_RECIPE_R1.json) freezes both folds, seeds, payload
hashes, producer sidecars, selected source-qualified groups, dependency
components, evaluation-group exclusions, original cache/source hashes, and
batch formulas. The [source manifest](SOURCE_MANIFEST_R1.json) has 54
code, recipe, metadata, receipt, and payload-identity records. It includes the
portable transport, its export receipt and attestations, and the prepared
bundle manifest; the expected suffix hashes agree across those records. The
payload hashes were not recomputed during preparation.

The [source receipt](SOURCE_RECEIPT_R1.json) records the metadata-only
preparation check:

- F0: 1,236 sequences from 1,118 train and 118 fit-valid groups, with 315
  selected components and 468 disjoint evaluation groups.
- F1: 1,243 sequences from 1,071 train and 172 fit-valid groups, with 314
  selected components and 457 disjoint evaluation groups.
- Both producer source maps match the current source files. The cache sidecars,
  fit-subset manifests, split manifests, ordered `row_refs`, selected-component
  closure, evaluation membership, and parent/propensity metadata all passed.
- The runner's NPY header parser read only each 128-byte header. F0 is
  `<f4`, C-order, `(1236, 20000, 5)`, 494,400,128 bytes; F1 is
  `<f4`, C-order, `(1243, 20000, 5)`, 497,200,128 bytes.

The later [admitted result](BATCH_MASS_RESULT_R1.json) streamed and hashed both
suffixes; its verified SHA-256 values match the frozen transport records. This
is now distinct from the earlier metadata-only preparation and did not read
the evaluation-label payload.

The [synthetic receipt](SYNTHETIC_CHECK_R1.json) records the bounded math
fixture. Ruff format/check and the fixture passed under 150 MB, one CPU core,
and a 60-second cap. No raw feature arrays, evaluation arrays or labels were
read, and no model or Torch module was imported.

The independent [R1 code review](CODE_REVIEW_R1.md) checked the frozen source
and manifest pins, primary loss reduction, transported column layout, row
ordering, component measure interpretation, and synthetic/header receipts. It
finds no source/schema/math blocker, while making no claim about unscanned
suffix values, optimizer parity, or scientific quality. The later
[independent result review](RESULT_REVIEW_R1.md) cross-checks the admitted
suffix digests against their pins and recomputes the saved component and loss
aggregates from JSON; it does not read suffix values or targets.

## Measured summaries and interpretation

The source fitter gathers complete sequences in exact cache order and batches
rows across sequence boundaries in chunks of 8,192, retaining the last partial
batch. `primary_pair_wmse` clips targets to ±2, uses `abs(clipped target) ×
row_weight`, compares against the raw prediction, and clamps each batch's
denominator at `1e-8`.

The completed R1 audit reports per-batch target-weight mass, target shares,
zero-mass batch counts, and `(total mass / batch count) / batch mass`
distortion factors. For the frozen parent it compares a pooled-row descriptive
loss with the source fitter's equal-batch normalized loss. It also reports
dependency-component concentration under the pooled measure and the
source-clamped batch measure. All sums use float64; this is not a claim of
Torch FP32 reduction or optimizer trajectory parity. See the result review for
the numerical summaries and limits on their interpretation.

The fitted targets are exposed development data. Fold and seed change together
between F0 and F1. The result can describe how the original objective weights
its minibatches, but it cannot establish a clean validation effect or select a
new endpoint. Pooled loss is a diagnostic alternative measure, not the
optimized source objective or official WP metric.

## Root-admitted command

The one-core worker cap was 400,000,000 bytes and 120 seconds; the two payloads
total 991,600,256 bytes and were streamed in bounded chunks. The command below
is the root-admitted command used for R1. It writes a create-only result under
physical scratch:

```sh
WNN_MEM_MAX=400000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_RUNTIME_MAX=120 \
  scripts/limited uv run --no-sync python \
  research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/audit_batch_mass.py \
  --recipe research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/BATCH_MASS_RECIPE_R1.json \
  --source-manifest research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/SOURCE_MANIFEST_R1.json \
  --admission /home/vlad/development/wunder/research_tmp/batch-mass-audit-admission-r1.json
```

R1 completed in 6.942 seconds at the worker-wrapper boundary with return code
0, a 399,998,976-byte peak, zero OOM events, and zero swap. The [result
review](RESULT_REVIEW_R1.md) confirms the aggregate arithmetic and exact
transport identities. The source/execution receipts preserve the earlier
metadata-only preparation separately from this completed suffix stream.

## Completed audit

- [Findings and limitations](RESULTS_R1.md) report relative row-weight multipliers of0.109–9.743/F0 and0.104–8.291/F1, lower fixed-parent loss under the equal-batch measure, and less concentrated component weights. These are descriptive objective differences, not proof of harm or an official quality gain.
- [Full result](BATCH_MASS_RESULT_R1.json) includes verified suffix identities and every component share. [Admission](ADMISSION_R1.json), [execution receipt](EXECUTION_R1.json) and [empty success log](EXECUTION_R1.log) record one bounded400MB/one-core run with zero OOM/swap.

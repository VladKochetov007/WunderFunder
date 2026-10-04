# E426 fold-0 event-count CPU audit draft

**Prepared, not launched.** This audit is a bounded input-only preflight before
the E426 GPU fits. It does not train, predict, score, or touch the GPU. It uses
the real fold-0 manifest, the staged event-count implementation, and the same
TRAIN permutation / fit-VALID concatenation / epoch-0 sampler used by the
production E426 trainer.

## Resource envelope and command

Seal as a CPU job with one CPU, 2 GiB memory, no swap, and a 600-second hard
timeout. Set `CUDA_VISIBLE_DEVICES=""`, `WNN_THREADS=1`,
`WNN_LOADER_WORKERS=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `ARROW_NUM_THREADS=1`, and
`POLARS_MAX_THREADS=1`. Run once with a fresh, non-overwriting output path:

```text
.venv/bin/python research/campaign_2026-10-04/staging/e426/scripts/audit_event_counts.py \
  --root . \
  --manifest research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --output research/campaign_2026-10-04/checks/e426_event_count_audit_f0_s3.json
```

The script rejects an existing output. Run it under the capped resource unit;
do not invoke it in an unconstrained shell.

## Checks performed

The script imports staged `wnn.event_ema` and frozen `wnn.splits` plus
`wnn.training_sampling`, validates the real manifest against repository data
identities, and requires fold 0. It reproduces the production fit order:

1. Seed-3 permutation of all 10,535 manifest TRAIN groups (`--train-seqs 10607`
   includes the full set).
2. The 1,405 manifest fit-VALID groups appended once (`--valid-repeat 1`).
3. The actual `EpochOrderSampler` at seed 3, epoch 0, batch 32,
   `drop_last=True`.

Before opening a Parquet source, it appends forbidden TRAIN group 528 to the
real TRAIN fit IDs and requires the real split guard to reject that source
list. It then verifies group 528 is absent from the valid source and eligible
TRAIN group 529 is present. Only after those checks does it count endpoints.
This negative check uses the real manifest and the same group-membership
contract; it is independent of the earlier synthetic fixture's metadata-only
hole assertion.

The staged `endpoint_events_by_update` reads only `a6` from one Parquet row
group at a time, computes 20 counts per full 20,000-row sequence for
`update_rows=1000`, and releases each group before reading the next. It never
loads labels or scoring columns and keeps only the small `[11,940, 20]`
int64 count matrix, sampler indices, and update schedule in memory. The first
epoch has 373 complete batches (11,936 retained sequences, four dropped) and
7,460 optimizer updates. It checks that update counts sum to the retained
sequences' endpoint total.

It materializes the actual constant, aligned, and shuffled schedules with
base decay 0.999 and shuffle seed 86173. It verifies each schedule has exactly
7,460 updates, aligned/shuffled schedules use the observed per-update count
multiset, and the product of teacher decays equals `0.999 ** 7460` within
floating-point tolerance. The output includes the ordered `(split,row_group)`
list and hash, manifest hash, the manifest-pinned source identity records for
both Parquet files (byte size, mtime, and recorded footer hash), count-matrix
hash/shape, total and retained event counts, dropped sampler rows, negative-528
rejection reason, and all three per-update count/decay vectors and hashes. The
ordered source-list hash binds the exact scan order; the manifest source records
are reported as the repository's existing identity contract rather than
recomputing a whole-file SHA-256 over the 35-GB Parquet sources.

## Acceptance and limits

Accept only exit 0, verified job teardown, elapsed time below 600 seconds,
peak RSS below 2 GiB, and a complete output JSON whose manifest is the pinned
fold-0 hash and whose membership assertions pass. Preserve the audit output and
runner log immutably. A failed membership check, forbidden group acceptance,
empty event epoch, count-total mismatch, schedule-product mismatch, timeout,
or resource violation blocks the full E426 fits.

This checks the real event-count scan and schedule construction but does not
validate GPU training memory or predictive value. The separate full-length GPU
smoke remains necessary for resource viability; the three-arm fit launch still
requires a separate decision after both preflights pass.

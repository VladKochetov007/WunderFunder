# P2 innovation-clock source audit — October 18, 2026

## Finding

The proposed input-only P2 innovation-clock screen has no safe, already
materialized full-sequence feature stream to consume. The fresh-context
data-sentinel audit was read-only: it inspected manifests and artifact
provenance without opening feature, target or scoring-mask payloads, and did
not retry the earlier Parquet path.

## Available artifacts

- The warmup cache at
  `/home/vlad/development/wunder/research_tmp/online_features_r1/warmup_gate_r1/extract_r2/warmup_features.npy`
  is 73,136 bytes with SHA-256
  `849e0a93…ffffc25`. Its producer receipt identifies a 468 × 39 float32
  matrix of summaries from only the first 99 rows of F0 validation sequences.
  It cannot supply the 112-channel row stream needed to calculate innovations
  throughout each sequence.
- The Kalman projection saved a 580,569-byte plot, SHA-256
  `d83e3df8…9e5d2f`, after inspecting four selected channels over eight groups
  (160,000 rows). It did not retain row-level feature arrays.
- The trusted causal-proxy recipe records `datasets/train.parquet` as
  29,240,219,011 bytes with a 60,437,285-byte footer, mtime
  `1577826000000000000`, and footer SHA-256
  `071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb`.
  A prior metadata-only profile succeeded under a 1.5 GB reservation, reading
  metadata for 212,140,000 rows / 10,607 groups with 1,036,624 KiB process
  peak. This is a possible future bounded source route, not a reusable feature
  cache.

## Safe follow-up conditions

The earlier innovation-clock attempt used an unbounded PyArrow footer parse at
a 1 GB child cap and was OOM-killed before reading a row group or target/mask
payload. The incident is recorded separately in the
[metadata OOM note](LOCAL_TRAIN_PARQUET_METADATA_OOM_20261018_R1.md); that
command must not be retried as written.

A new source builder would first verify the train file's size and nanosecond
mtime, read and hash only the predeclared footer byte range, compare it with
the trusted recipe, and only then parse Parquet metadata under a measured
reservation. It would process one source-qualified row group at a time with
targets and masks disabled. Before any score claim, it must preserve causal
prefix invariance and sequence-state reset checks. No source or feature fit was
admitted by this audit. A later, separately admitted bounded feature-only
screen used this verified footer path on 32 fit-side components; its results
and limitations are in the [innovation-clock packet](../../campaign_2026-10-18/online_features_r1/innovation_clock_r1/README.md).
That follow-up does not change the audit's original read-only scope or the
no-retry disposition for the unbounded command.

## Provenance and navigation

This is the result of the read-only `p2_data_sentinel` assignment for
`C-ONLINE-FEATURES-20261018-R1`. It contributes source-availability evidence
to P2, not a negative scientific result. The related proposal is in the
[independent proposal set](../../campaign_2026-10-18/online_features_r1/independent_proposals/PROPOSALS.md),
and the earlier boundedness incident is in
[the operations record](LOCAL_TRAIN_PARQUET_METADATA_OOM_20261018_R1.md).

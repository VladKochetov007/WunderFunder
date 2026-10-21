# Independent review: all-channel innovation-clock screen

## Scope

The MFT Data Sentinel independently re-audited the source, code identity,
membership, causality checks, runtime receipt, and compact summary outputs for
the [all-channel causal innovation-clock screen](../../campaign_2026-10-18/online_features_r1/innovation_clock_r1/README.md).
The review was read-only; it did not rerun the experiment or inspect targets.

## Findings

- The durable attempt's `screen.py` hash matches the run record. The
  `innovation_features.py` hash matches the result file, and its recorded
  modification time precedes worker start. The worker checked the pinned
  train-file size, mtime, trailer, footer size and footer hash before opening
  Parquet metadata. This pins the metadata identity, not a full-file payload
  hash.
- The 32 selected source-qualified components and sequence IDs match the
  frozen `fold0.fit.train` membership. The sequence reader requested
  `targets=False` and `mask=False`, then projected only `seq_ix` and the raw
  feature columns. No evaluation group was used.
- Warmup-only scaling, pre-update innovations, prefix/future-perturbation
  equality, and A-B-A sequence reset checks support extractor-level causality
  and reset. They do not verify integration into the competition serving
  callback.
- Execution completed in 15.07 seconds under one CPU, a 1.5 GB cap, no swap
  and no GPU. Worker peak was 1,351,356,416 bytes; worker `memory.high` delta
  was 87 and parent-hierarchical `high` delta was 117. Worker `max`/OOM/
  OOM-kill and parent-local OOM/OOM-kill deltas were zero. The copied receipt
  and both summary CSV hashes match their recorded values; the reviewer
  recomputed the phase summaries and matched the result.

## Reporting correction

The original [result JSON](../../campaign_2026-10-18/online_features_r1/innovation_clock_r1/result_r1.json)
describes pooled top/bottom semantic-group lists as the groups ranking highest
or lowest “in each later bin.” The CSVs show that, individually, the top group
in each of the three later bins is `i1_ask_price_like`, and the bottom group
is `i0_trade_price`. The listed three groups are the pooled top and bottom
three across those bins. The machine-readable
[correction addendum](../../campaign_2026-10-18/online_features_r1/innovation_clock_r1/review_correction_r1.json)
preserves this distinction without rewriting the original result artifact.

## Claim ceiling

The strongest supported claim is that, in one selected sequence from each of
32 exposed `fold0.fit.train` components, innovation summaries were
nondegenerate and their component means rose across four time bins (1.79,
2.37, 2.51, 2.63); feature groups differed, and the scalar clock's median
component p95 reached its cap by rows 500–2,000. This supports iterating the
representation. It supplies no target association, endpoint score, quality
gain, clean validation, or public-transfer evidence.

## Source records

- [Experiment packet](../../campaign_2026-10-18/online_features_r1/innovation_clock_r1/README.md)
- [Durable attempt](../../campaign_2026-10-06/bandit_continuation/runtime/attempts/innovation-clock-feature-20261018-r1.json)
- [Feature-only receipt](../../campaign_2026-10-18/online_features_r1/innovation_clock_r1/feature_only_receipt_r1.json)
- [Review correction](../../campaign_2026-10-18/online_features_r1/innovation_clock_r1/review_correction_r1.json)

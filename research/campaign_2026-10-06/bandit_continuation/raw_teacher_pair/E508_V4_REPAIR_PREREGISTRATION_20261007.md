# E-508 v4 split-namespace repair profile

Prepared 2026-10-07, Europe/Kyiv. This is an execution repair for the existing
raw-only baseline-teacher lineage, not a new scientific recipe or a score
experiment.

## Parent evidence and hypothesis

E-508 v3 verified the pinned trainer, fold/assets manifests, audit-only source
manifest and checkpoint, and both full raw parquet hashes. It then failed before
GPU allocation in `exact_group_sets` with `corrupt train row group entered fit
membership`. Its terminal receipt records zero allocated CUDA bytes and no
profile or fit.

The split manifest stores exclusions with a split-qualified key
(`excluded.corrupt_train: [528]`). `CampaignSplit` likewise validates train and
valid members in separate namespaces (`src/wnn/splits.py`). The v3 worker
incorrectly rejected integer 528 in either set. V4 requires train row group 528
to remain absent and permits valid row group 528, while retaining exact train,
valid and evaluation counts. No split, fitting membership, target, recipe or
checkpoint ancestry changes.

## Frozen execution boundary

The packet is `packet_v4/`, kernel
`vladkochetov07/wnn-raw-teacher-pair-e508-20261007-v4`. Its kernel and metadata
are SHA-256-bound in `packet_v4/review_packet.json`; it embeds the reviewed
production trainer commit and the current worker hash. Kaggle inputs remain
private and internet-disabled. The packet checks full train/valid parquet
SHA-256, Parquet footer identity, fold/assets/source hashes and exact split
membership before profiling.

This packet can run one short seed-3 profile and then exits for controller
review. It cannot start either scientific teacher fit. Worker hard timeout is
3,600 seconds; the provider watch cap is 4,200 seconds. The next admissible step
is a separately identified fit packet, only after a measured profile is
reviewed. This attempt makes no score or public-transfer claim.

## Resource snapshot

The most recent authenticated Kaggle read-only poll was
`2026-10-07T03:35:03Z` (`cloud_preflight/provider_poll_log.jsonl`): account GPU
quota used 40,009.628 of 108,000 seconds, no reservation, and retained campaign
headroom 53,818.209 seconds under the existing 56,780.516-second campaign cap.
No paid rental is used. Refresh quota and all kernel statuses immediately
before push. A 4,200-second provider-watch bound leaves over 49,600 campaign
GPU seconds in the worst case using the current snapshot.

Pre-push checks: packet source hashes and metadata; regression suite for the
split-local ID guard; independent read-only packet critique; and live Kaggle
quota/status. The upload and run are not authorized by this preregistration
alone if any of those checks fail.

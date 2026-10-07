# E-508 v5: supported CUDA selector repair

Prepared 2026-10-07, Europe/Kyiv. This is a new, profile-only execution attempt
after the distinct v4 command failure. It does not change the scientific recipe
or claim a score. Parent evidence and the concrete failure are preserved in
[`E508_V4_FAILURE_20261007.md`](E508_V4_FAILURE_20261007.md) and
[`e508_profile_terminal_v4.json`](../../cloud_preflight/e508_profile_terminal_v4.json).

## One declared change

The v4 worker passed `--device cuda:0`, while the pinned trainer parser permits
only `cuda` and `cpu`. V5 passes `--device cuda`, which selects the default CUDA
device (device 0). Internal tensor placement and prediction remain on
`torch.device("cuda:0")`. The trainer is still pinned to commit
`69b2331825e193a070015ca4182872724595209a`; its parser is inspected directly at
that commit. No target, data source, membership, architecture, optimizer,
precision, epoch/checkpoint rule, profile shape, or output order changes.

## Frozen execution boundary

Packet: `packet_v5/`, ID `E508_RAW_BASELINE_V5`, kernel
`vladkochetov07/wnn-raw-teacher-pair-e508-20261007-v5`. The new packet binds
the v5 worker bytes, embedded source archive and metadata independently of the
immutable v4 packet. Inputs remain private; internet and TPU are disabled.
Exact parquet, footer, source, assets, target-cache and split checks remain in
place.

This attempt runs only the short seed-3 profile and exits for controller review.
It cannot launch either teacher fit. Worker timeout remains 3,600 seconds and
provider-watch cap remains 4,200 seconds; no automatic OOM retry is enabled.
Before push, independently review the materialized packet, verify the v5
command and hashes, refresh GPU/TPU quota and all tracked kernel states, and
confirm reserved campaign GPU headroom can absorb the full provider cap while
preserving the account reserve.

## Success and failure interpretation

Success requires actual finite profile-step timing, correct T4 identification,
resource headroom, and the frozen pair/export runtime projection under the
declared cap. A provider `RUNNING` state alone does not pass the gate. A repeat
CLI argument failure is an implementation failure; OOM or another hardware
failure must be diagnosed before any retry. No scientific fit, official score,
or public-gain claim is part of this packet.

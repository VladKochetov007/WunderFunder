# Local train Parquet metadata OOM, 2026-10-18

## Status

Contained to one 1 GB transient systemd scope. The feature-only feasibility
screen is stopped; no retry or further Parquet read is authorized by this
receipt. This is an operational failure, not scientific evidence for or
against the innovation-clock idea.

## Attempt and impact

At 2026-10-18 06:07:59 Europe/Kyiv, a bounded Python command attempted to
inspect row-group counts for `datasets/train.parquet` and then
`datasets/valid.parquet`. The command began with `pyarrow.parquet.ParquetFile`
on the 29,240,219,011-byte train file. It did not print output, did not reach
the validation file, and did not read a row group. No output file was written.
The command requested 1 CPU, 1,000,000,000 bytes of memory, zero swap, and a
120-second runtime through `scripts/limited`; scratch pointed at the approved
physical `research_tmp` filesystem.

At 06:08:00 the kernel OOM-killed Python inside
`run-p3081242-i3058961.scope`. `systemd` reports `Result=oom-kill`,
`MemoryMax=1,000,000,000`, `MemoryPeak=999,997,440`, `MemorySwapMax=0`, and
668,198,000 ns CPU time. The kernel reports 974,228 KiB anonymous RSS at kill.
This is a real child-scope OOM. The parent campaign slice's hierarchical
`oom`/`oom_kill` counters moved from the prior recorded 3/3 to 4/4; its local
counter remained 0/0. Do not describe this as “no OOM” merely because the
desktop survived or the child had a memory cap.

The failed operation attempted Parquet-footer metadata parsing only; it never
selected targets, masks, or input row values. Immediately beforehand, a
separate validation-only check successfully read the footer and `seq_ix` /
`step_in_seq` from five validation row groups. That success did not establish a
safe memory bound for the much larger train file.

## Cause and prevention

I assumed the train Parquet footer would be small without first measuring or
bounding its metadata length. The file's payload size was known, but its
row-group count and footer size were not. `ParquetFile` materialized unknown
metadata under a hard cap, which contained the failure but did not prevent an
OOM.

Do not repeat a full `ParquetFile(train.parquet)` open from this screen. Any
future train-file metadata inspection must first use a trusted producer
manifest or a bounded footer-length read that rejects metadata above an
explicit pre-admitted size. If no such bound is available, use existing
source-qualified manifests and cached artifacts, or leave the screen
unrun. Keep the task stopped until a safe metadata route is established.

## Provenance

- Campaign: [`online_features_r1`](../../campaign_2026-10-18/online_features_r1/README.md)
- Scope: `run-p3081242-i3058961.scope`
- Kernel event: `journalctl -k`, 2026-10-18 06:08:00 Europe/Kyiv,
  `oom_memcg=/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice/run-p3081242-i3058961.scope`
- systemd result: `systemctl --user show run-p3081242-i3058961.scope`
- Related direction: P2 input-only innovation clock in the
  [independent proposal packet](../../campaign_2026-10-18/online_features_r1/independent_proposals/PROPOSALS.md).

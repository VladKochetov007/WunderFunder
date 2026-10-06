# Campaign operations

The aggregate local boundary is `wnn-campaign.slice`: 10 CPU cores, 20 decimal GB
RAM, 18 GB memory-high, no swap. GPU workers use the shared main-checkout runtime
directory and its single GPU lock, including workers launched from worktrees.
Observed total device memory must remain below 5 decimal GB; the PyTorch allocator
cap is 4.3 GB. Kaggle has separate recorded budgets and headroom.

Persist the parent properties with `systemctl --user set-property`, without
`--runtime`, then verify them after `systemctl --user daemon-reload`. Inspect both
`systemctl --user cat wnn-campaign.slice` and `systemctl --user show` when changing
unit configuration. A runtime observation alone did not establish persistence:
the E502 incident restored older saved 40 GB limits when the Kaggle poller was
installed. The corrected values now survive reload. Retain that incident and
its failed guard receipt even when the completed model passes artifact review.

Existing wrappers have historical defaults. Always supply `WNN_SLICE`,
`WNN_CPU_QUOTA`, `WNN_THREADS`, and `WNN_MEM_MAX` explicitly for support work.
Check aggregate admission before training. Page cache counts toward RAM;
`memory.events:max` alone is not an OOM. Inspect `oom`, `oom_kill`, and
`oom_group_kill` separately, plus worker exit status and checkpoint completeness.

The local job guard samples about every five seconds. The Kaggle timer
`wnn-campaign-kaggle-poller.timer` writes quota/status observations each minute;
it does not resubmit jobs or imply their payload ran. Provider timeouts have not
bounded startup reliably. Preserve provisional quota and terminal reconciliation
separately. Never substitute a kernel version number for an actual cancellation
session ID.

Frozen sequence files are immutable once started. A failed attempt stops later
admission and needs a recorded diagnosis. Resume through a new plan/receipt,
with an explicit artifact acceptance record when a completed fit was affected
only by monitoring. Do not rewrite a failed receipt or rerun a completed fit
solely to produce a cleaner status. Never automatically retry an OOM.

An explicit user stop closes new admission and lets healthy workers finish.
Emergency observed resource violations may stop the offending worker. Any
reopening requires a written review of the actual cause and effective limits.

### Completed-export cache handoff — 2026-10-06T22:45:24.060042+00:00

The strong seed3 training prediction export finished exit0 in 566.2s with no OOM kill. The next export was refused before launch because parent charged RAM plus the unchanged8GB reserve exceeded18GB. Original sequence and refused attempt remain preserved. New immutable `TEACHER_EXPORT_SEQUENCE_v2.json` (SHA002a23b6aaf86a1b18ad7b6da5eb912473ac1210bf4a342e428a0ccd1b6a3312) inserts bounded CPU cleanup before each remaining scientific export.

Controller source a50a1adb: `reclaim_bandit_export_cache.py` requires a successful completed export, validates current parent limits, takes the shared GPU lock without waiting, requests at most6GB of file-only reclamation (`swappiness=0`), retains at least6GB file cache, and refuses admission unless observed parent RAM is at most10GB. No limit is changed. Four safety tests passed (active GPU, failed export, unchanged limits/file-only mode, insufficient reclaim). An isolated worktree freezes helper code; scientific exports still use69b23318.

First cleanup requested 5335776000bytes, reducing charged parent RAM from 13835776000 to 8494985216bytes. Sequence service `wnn-bandit-teacher-exports-20261007-v2.service` is running, with the8GB export reserve unchanged. Failure stops for review without rerunning a failed scientific worker.

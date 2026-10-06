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

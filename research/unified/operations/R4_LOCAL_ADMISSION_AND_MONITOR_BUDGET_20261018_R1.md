# R4 local admission and monitor budget — 2026-10-18

## Finding

The R4 matrix is authorized as one local GPU worker with one CPU core, a
5,000,000,000-byte hard RAM cap, no swap, and scratch on physical
`research_tmp`. The aggregate parent remains capped at two CPU cores and
20,000,000,000 bytes, with an 18 GB high watermark.

Attempt `geometry-r4-matrix-20261018-r3` was rejected before service creation.
At that instant the five-minute read-only local/Kaggle monitor was active with
`CPUQuota=100%`, so the controller conservatively reserved one full core for
the monitor, another 0.2 cores for support services and launch scope, and one
core for R4. The total exceeded the two-core aggregate. No model work or GPU
allocation occurred.

The monitor is a bounded one-shot checker. Its preceding tick used 3.69 CPU
seconds over 49 seconds, with a 512 MiB memory cap. To preserve the timer while
leaving admission headroom for R4, its service quota was changed from 100% to
20% in both the tracked unit template and the installed user unit. The
five-minute timer remains enabled; this change does not stop the monitor or
alter its Kaggle/local checks. With the monitor at 0.2 cores, the expected
R4-time reservation is 1.0 core for the worker plus 0.2 for the monitor, 0.1
for the research watcher, 0.1 for the file-cache guard and launch support,
within the two-core parent.

## Evidence and status

- The monitor run properties showed `CPUQuota=100%`, a 3.69 CPU-second total,
  and an elapsed time of 49 seconds. It had exited successfully before this
  adjustment.
- The local user unit was reloaded with `systemctl --user daemon-reload`; systemd
  reports an effective 20% CPU quota and the five-minute timer remains active.
- The first R4 launch attempt failed before Python startup because the script
  was not executable. The second rejected a mistyped `train.parquet` hash; the
  runner stopped before fitting. The third was rejected at CPU admission while
  the monitor tick was active. All three attempts have zero scientific fits.
- Corrected input identities and source pins are recorded in [R4 admission
  R3](../../campaign_2026-10-17/geometry_from_scratch_r4/admission_r3.json).
  The [R4 runtime ledger](../../campaign_2026-10-17/geometry_from_scratch_r4/runtime/README.md)
  preserves each admission result. Attempt R4 passed fresh admission and began
  the fold-0 seed-42000 control fit at 2026-10-17 21:27:24 UTC; no score is
  available yet.

This note records an operational resource change, not evidence about model
quality. The service timer remains the owner of periodic status checks and will
continue to run alongside the single admitted GPU worker.

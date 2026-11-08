# Fixed4E B300 progress

Separate PR70 reporter for labels/mixture_corr × seeds101101/101102. Only actual
hostb300 new fit directories count. Historical30E/old seeds/profiles do not alias.
Source88ad6a..., manifestd9f9ba..., all four raw/normalized config hashes and
official whole-sequence membership are pinned. Native completion requires4E,
3280updates, producer checkpointSHA, all four1873/3528273 full-valid events and
same-forward standalone z metrics. Final mixture grade additionally binds the
canonical score's source/config/checkpoint/reference composition and fixedalpha.1.

Reuse the committed L4 publisher/API helper without modifying its files. Account
verification, pending marker reconciliation, uncertain POST no-repeat, local
receipts and env-only credentials are identical. Keep B state in a separate
directory; shared helper retains its `l4-wave-*` body/receipt filenames there.

ROOT copies actual native receiver-plan bytes to
`C:/Users/admin/Documents/Codex/artifacts/objective-fix-20261108/reporting/receiver-plan.json`.
Expected native planSHA is
`02ce50018009b093a03564be4e02c3cf4ca027655bc8c2297923c064354767e5`.

From NEW_W with `PYTHONPATH=src;.` and all OMP/MKL/OPENBLAS thread limits2:

```powershell
$report = 'research/experiment_drafts/generalization-fix-20261108/reporting/b300/report_b300_wave.py'
$local = 'C:/Users/admin/Documents/Codex/artifacts/objective-fix-20261108/reporting'
$live = 'C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard'
& $python $report --mirror-config "$live/mirror-config-generalization-objective-r1.json" --receiver-plan "$local/receiver-plan.json" --receiver-plan-sha256 02ce50018009b093a03564be4e02c3cf4ca027655bc8c2297923c064354767e5 --pr 70 --state "$local/b300-wave-state.json" --interval-seconds 1800 --poll-seconds 30 --max-seconds 14400 --dry-run --once
```

ROOT-only live launch omits `--dry-run --once` and inherits `WNN_GITHUB_BASIC` from
the process environment. This helper opens no SSH and queues no RPC. It consumes
the existing mirror. Cached timestamps remain explicit; stale data is never
treated as process death. Producer hashes do not claim downloaded weight archives.
Every scientific interpretation/promotion stays with ROOT. No network/fit was
started by the critic during implementation/testing.

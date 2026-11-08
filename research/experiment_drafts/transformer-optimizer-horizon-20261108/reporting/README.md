# Six-cell progress and grade reporting

This adapter reads only ROOT's local mirror under
`C:/Users/admin/Documents/Codex/artifacts/transformer-optimizer-horizon-20261108/live-dashboard/native`
and the exact frozen receiver plan. It does not own the training controller,
RPC stream, signals, model/data files or GitHub authorization. The stale coarse
receiver `status.json` is intentionally not used: phase and freshness come from
producer `monitor_progress.json`, observer events and rank0 training records.

## Evidence levels

- Per-epoch full-valid same-forward raw-z monitor values are explicitly
  provisional. A monitor event is admitted only with the frozen source/config,
  full 1,873-sequence / 3,528,273-row coverage and the exact raw-z WP alias.
- A fixed grade is shown only when ROOT's readout has a completed4E/3,280-update
  terminal receipt and an official score for all valid rows, with matching
  source/config/data identity and terminal/score-producer checkpoint hashes.
  It is labelled `grade producer verified; final byte audit pending` until the
  frozen comparison binds the score-file SHA and `model.pt` SHA.
- The final byte-audit label requires the complete six-cell comparison output,
  with each score SHA and checkpoint SHA matching its readout. Missing or
  inconsistent grade files stay pending. Paired seed deltas appear only from
  this validated comparison artifact.

## Bounded read-only readout collection

`collect_readouts.py` is standard-library-only. ROOT may execute it through the
existing ROOT file RPC when the local mirror lacks the terminal/score files or
the root-level `comparison.json`. It first checks the fixed plan SHA
`b40ba65c7c2495f1a08a42331800ce8090bf9bbe63e7cb5514af401e784cf1c0`, then reads
only each of the six literal cells' `fit/terminal_status.json` and
`fit/score.json`, plus `R/comparison.json`. It refuses symlinks, changed files,
files above4MiB, or more than16MiB aggregate data. It never reads weights, input
rows, source code, control state or process identities and writes nothing on the
native host. ROOT keeps RPC request/result ownership and executes the collector
source in memory with the frozen runtime and plan paths; no collector script is
installed or invoked on the native host. ROOT saves the returned JSON as the
local `readout-bundle.json`; the reporter checks its plan/source binding and
byte bound before use.

The R2 envelope follows actual producer sizes: the first completed cell has a
724,777-byte terminal record and a1,870,045-byte score record. Six equal pairs
already require15,568,932 bytes before the small comparison; the original8MiB
aggregate could not retain the complete panel. The13-file test verifies the
larger bounded collection, while the4MiB individual-file rejection remains.
The original eight-test receipt is retained as `focused-reporting-tests.xml`;
the nine-test R2 receipt is `focused-reporting-tests-r2.xml`.

The progress signature changes on a new validated canonical score, final byte
audit, or producer failure. Otherwise the existing pending-marker publisher
posts hourly. An uncertain GitHub POST is reconciled before any retry. The
publisher verifies `SemyonEpanov` and PR72 through the existing
`sequence_progress_github.py` client. Credentials stay in ROOT's process
environment; no token file is part of this packet.

## ROOT dry run and launch

Set paths and the one-thread limits in ROOT's local PowerShell session:

```powershell
$A = 'C:/Users/admin/Documents/Codex/artifacts/transformer-optimizer-horizon-20261108'
$W = 'C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108'
$PY = 'C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$reportArgs = @(
  "$W/research/experiment_drafts/transformer-optimizer-horizon-20261108/reporting/report_horizon.py",
  '--mirror-config', "$A/live-dashboard/mirror-config.json",
  '--receiver-plan', "$A/frozen/fit-operator-r1/plan.json",
  '--receiver-plan-sha256', 'b40ba65c7c2495f1a08a42331800ce8090bf9bbe63e7cb5514af401e784cf1c0',
  '--authority', "$A/frozen/authority-r2.json",
  '--readout-bundle', "$A/live-dashboard/readout-bundle.json",
  '--shared-worktree', 'C:/Users/admin/Documents/Codex/reviews/WunderFunder-generalization-fix-20261108',
  '--pr', '72',
  '--state', "$A/live-dashboard/horizon-pr72-state.json",
  '--interval-seconds', '3600',
  '--poll-seconds', '30'
)
& $PY @reportArgs --dry-run --once
```

ROOT reviews the preview before starting the same command as an owned bounded
publisher, omitting `--dry-run --once`. The loop stops at the unchanged
ROOT-authorized cohort deadline. It does not resume, stop or change any fit.

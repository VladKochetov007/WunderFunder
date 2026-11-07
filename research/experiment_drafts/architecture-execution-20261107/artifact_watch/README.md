# Immutable per-epoch graph watcher

This local CPU-only watcher reads the actual bridge mirrors in
`A/live-dashboard/runs` and their unchanged native counterparts in
`A/live-dashboard/native`. It opens no SSH connection, loads no model, runs no
training/evaluation and writes no live metric. ROOT starts it after review.

Every15seconds it finds every complete producer-native full-valid epoch, including
multiple epochs already present when it starts. Each event must have exactly
1,873sequences and3,528,273selected rows, sequential epoch identity and finite
pooled/t0/t1WP. A partial mirror line waits. Missing main-journal evaluation bytes
wait for the bridge rather than inventing a snapshot.

For each new epoch it publishes, once:

```text
A/snapshots/<host-run>/epochN-<producer-UTC-time>-<eventSHA>/
A/snapshots/<host-run>/epochN.json
A/plots/<host-run>/epochN/
```

The snapshot contains exact native byte prefixes through that evaluation.
Rank-zero native and existing derived rows are bounded by the actual update
clock; observer rows use the recorded evaluation wall clock. This prevents later
epochs appearing in an earlier graph during delayed replay. Available metadata,
progress, epoch-index and terminal JSON are copied unchanged as explicitly
labelled capture-time observations. No checkpoint weights are read or copied.
The manifest binds every copied file, the recorded canonical source/data identity,
the actual host/source-directory owner and native collection receipt. Existing
snapshots and plots are verified and reused; changed old epoch bytes are rejected.

Only one plotting child runs at a time. It invokes the existing
`scripts/plot_sequence_monitor.py` on the immutable snapshot with
OMP_NUM_THREADS=2, MKL_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=2. PNG/SVG exports are
first produced in a new scratch directory, validated against the helper's native
receipt and snapshot hashes, then published by a create-only directory rename.
Every figure must have both formats. Failed scratch evidence stays separate; the
watcher does not repeatedly retry a failed job during the same invocation.

ROOT start command, using the prepared local plotting environment:

```powershell
$artifactRoot = 'C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107'
$sourceRoot = 'C:/Users/admin/Documents/Codex/reviews/WunderFunder-architecture-execution-20261107'
$plotPython = "$artifactRoot/local-plot-env/Scripts/python.exe"
$watcher = "$sourceRoot/research/experiment_drafts/architecture-execution-20261107/artifact_watch/watch_epoch_plots.py"
$env:PYTHONPATH = "$sourceRoot/src"
Start-Process -WindowStyle Hidden -FilePath $plotPython -WorkingDirectory $sourceRoot `
  -ArgumentList @('-u', $watcher, '--artifacts', $artifactRoot, '--source-root', $sourceRoot, '--plot-python', $plotPython, '--max-seconds', '86400') `
  -RedirectStandardOutput "$artifactRoot/epoch-plot-watcher.log" `
  -RedirectStandardError "$artifactRoot/epoch-plot-watcher.err" -PassThru
```

The isolated CPU environment was provisioned entirely offline from existing UV
cache: Python3.13.15, matplotlib3.11.2 and numpy2.5.4. No shared environment was
changed. The watcher is bounded to86,400seconds and uses a cross-process lock.
Its mutable status is `A/epoch-plot-watcher-status.json`. If no full validation
event exists, no scientific graph is fabricated. `--once` performs one discovery
pass and drains its pending plot jobs; it remains subject to the same wall bound.

Validation: replay/immutability and wrong-mask rejection tests pass. The existing
plot helper also completed a real CPU smoke test against clearly synthetic
fixtures, producing PNG/SVG pairs; one PNG was visually inspected. All3tests
passed in9.17seconds, and Ruff format/check passed. Test artifacts live only under
`A/watcher-selftest-20261107-r1`; they are not model results. No watcher was started
against actual mirrors by the implementation agent.

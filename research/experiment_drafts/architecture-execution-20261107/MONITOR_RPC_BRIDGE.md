# Two-host native monitoring mirror

Stage `scripts/collect_sequence_monitor.py` in each declared runtime's `control/`
directory, outside every frozen trainer capsule. This stdlib helper supports the
receiving Python3.9. `collect_runs(run_root, host, cursors)` returns unchanged source
bytes in complete-line deltas capped at256KiB/file and8MiB/response. Root owns
staging, existing SSH daemons and actual service start. No process, GPU, model,
credential, scientific configuration or checkpoint payload is accessed.

The concrete local config has been prepared at
`C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/mirror-config.json`.
It names the actual L4 `R/runs`, B300 `R/b300` and both staged `R/control/` helpers.
From the architecture execution worktree, after Root stages both collectors:

```powershell
$env:PYTHONPATH = 'src;.'
python scripts/mirror_sequence_rpc.py --config C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/mirror-config.json --max-seconds 86400
```

In a separate process:

```powershell
python scripts/research_dashboard.py --runs-root C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/runs --port 8766
```

The existing read-only dashboard is <http://127.0.0.1:8766/>. Use hidden windows
when launching detached background helpers. The mirror queues one read delta per
host every15seconds and opens no SSH connection. `--once` waits for one successful
response from each host. All mirror activity ends at its explicit wall limit.

Each source run gets its own `l4-` or `b300-` prefix. Original run metadata,
events, evaluations, observer, progress, raw rank-zero and checkpoint-index JSON
are preserved with their original bytes and source mtimes. SHA256/offset checks
precede application. Snapshot JSON becomes public only after complete reception
and parsing. An incomplete RPC result JSON waits for completion; a rejected reply
is recorded in bridge state. Producer journal resets retain old local prefixes.
Missing reads do not create fake fresh events or imply a dead worker.

Both owned transport filenames remain present: removing request/results could
race the SSH daemon's already enumerated directory list. Only this bridge's
consumed result content is atomically compacted to a hash/byte-count
acknowledgement, after durable native-byte/cursor publication. Operator requests
and results are untouched. Scientific source files are never rewritten.

B300 raw `rank0.jsonl` is additionally converted into a separate local
`monitor_rank0.jsonl` chart stream. Derived epoch positions use actual fitting
membership and config/world-size padded schedule metadata. Raw rank rows have no
per-update wall timestamps. Their derived points appear on epoch/step axes,
are omitted from elapsed-time axes, and never affect run liveness. Original raw
rows remain unchanged. Native observer and evaluation timestamps drive health.

Use `scripts/plot_sequence_monitor.py` locally on an immutable collected run
snapshot to write PNG/SVG files and a source/artifact hash receipt. This does not
load or evaluate a model. Node-side matplotlib is unnecessary. Preserve every
evaluation and final graph bundle before releasing any owned worker.

Focused local tests cover bounded complete-line deltas, same-inode truncation,
old-prefix retention, idempotent response replay, corrupt SHA256, cross-host
rejection, both queue replies, incomplete result writes and unchanged source-time
health with derived rank-zero curves. These are synthetic schema/protocol tests,
not receiving-host, scientific metric or GPU-utilization evidence.

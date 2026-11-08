# Fixed-wave progress reporting

This separate local reporter covers only the six new L4 cells,6E/38,760 updates.
It preserves the old PR68/69 reporters and never starts training or opens SSH.
It uses the existing GitHub API helper: fixed repository, actual authenticated
`SemyonEpanov` login before writes, no redirects, credential only from process
environment `WNN_GITHUB_BASIC`. Never put that value in arguments/files/logs.

Every1800seconds and each observed terminal/readout change, it batches all cells
into one PR71 comment. Durable pending markers prevent automatic repeat POSTs
after an uncertain response or restart. State, body, comment IDs, errors and
publication receipts are retained in its separate directory. Cached source UTC
and ages remain explicit. Stale events never establish process death.

Completion requires the exact fixed source and normalized config inventories,
10325train/1873valid membership,6E/38,760 updates, producer checkpointSHA and
all six full-valid events. The fixed24 grades additionally require exact readout
source0042a76f..., qualified reference1863e5...,1873sequences/3,528,273rows,
producer identity, E6checkpoint hash or fixed equalSWA3/4/5/6 parent/index hashes.
Copied receipts establish producer records, not downloaded checkpoint/array bytes.

ROOT must copy the actual admitted readout JSON to a local file, for example
`A/generalization-fix-20261108/reporting/ROOT-readout-admission.json`. Required
fields are `approved:true`, `reviewer:"ROOT"`, exact `source_sha256` and
`reference_manifest_sha256`; this reporter creates no approval.

Run from NEW_W with `PYTHONPATH=src;.` and all OMP/MKL/OPENBLAS thread limits2:

```powershell
$report = 'research/experiment_drafts/generalization-fix-20261108/reporting/report_l4_wave.py'
$local = 'C:/Users/admin/Documents/Codex/artifacts/generalization-fix-20261108/reporting'
$live = 'C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard'
& $python $report --mirror-config "$live/mirror-config-generalization-r1.json" --pr 71 --state "$local/l4-wave-state.json" --readout-root "$local/readouts" --readout-admission "$local/ROOT-readout-admission.json" --dry-run --once
```

ROOT-only live launch uses the same arguments without `--dry-run --once`, adding:

```text
--interval-seconds 1800 --poll-seconds 30 --max-seconds 14400
--rpc-root C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/l4-rpc
--native-run-root /data2/lc-epanov_s/tabdata/runtime/generalization-fix-20261108-r1/runs
```

The optional collector submits only read-only Python to the existing persistent ROOT RPC
queue, with one pending request at a time. It reads only the12 exact
`<cell>/readouts/{swa3456,epoch6}/readout.json` paths; maximum1MiB/file and8MiB
response. Local copies retain actual bytes/source timestamps and verify hashes;
conflicting existing receipts are refused. Non-atomic partial RPC result JSON is
retried using the same pendingID. No token is included in any RPC request.

`--dry-run` renders local evidence without credentials, API calls or RPC requests.
Omit the optional readout arguments for fit-only reporting; fixed grades then
remain pending. Unit tests mock publication and use temporary local RPC folders.

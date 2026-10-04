# October 4 campaign runner

This runner uses new state under `research/campaign_2026-10-04/runner`. It never
reads, removes, or reactivates historical queue pause markers. The controller
initializes it after independent review and resource preflight. The implementation
does not configure or widen existing systemd slices.

The controller must keep **all** local research, transfers, marimo server/kernel,
and build processes under `wnn-campaign.slice`: CPU quota <=1200%, RAM <=36 GiB,
swap zero. The adapter checks these effective properties at admission and during
execution. It admits one GPU service (<=8 CPU/24 GiB) and two CPU services
(each <=1 CPU/2 GiB). Reserve a further 1 CPU/2 GiB for marimo. Docker-daemon jobs
and commands that launch detached scopes are outside this adapter's containment
contract; arrange their own verified parent cgroup before using them.

The portable JSON schema is implemented by `wnn.campaign_spec.ExperimentSpec`:

```json
{
  "version": 1,
  "name": "unique_scientific_run",
  "kind": "gpu",
  "argv": ["{python}", "scripts/train_rnn.py", "--name", "unique_scientific_run"],
  "code": [{"path": "scripts/train_rnn.py", "size": 123, "sha256": "64 lowercase hex characters"}],
  "code_sha256": "hash of canonical sorted code descriptors",
  "inputs": [{"path": "datasets/train.parquet", "size": 123, "sha256": "64 lowercase hex characters"}],
  "artifacts": ["artifacts/runs/unique_scientific_run/model.pt"],
  "limits": {
    "cpus": 8,
    "ram_bytes": 25769803776,
    "timeout_seconds": 7200,
    "gpu_memory_bytes": 4200000000,
    "gpu_stop_bytes": 4800000000
  },
  "environment": {"WNN_LOADER_WORKERS": "3", "WNN_LOADER_RAM_GB": "3"}
}
```

Every directly or indirectly imported scientific source and every required data,
teacher, propensity, split, and initial-checkpoint dependency belongs in its
manifest. Paths are relative to the worker checkout; absolute paths, escaping
symlinks, duplicate paths, and outputs overwriting inputs are rejected. Files bind
size and full SHA-256, not mtime. `code_sha256` hashes UTF-8 JSON of path-sorted
descriptors with `sort_keys=True, separators=(',', ':')`. `{python}` expands only
as argv[0], to the adapter's explicitly chosen interpreter. All other arguments
are literal. CPU specs set both device-memory fields to zero. Remote specs may
declare larger resources; the local adapter independently rejects those limits.

An unsealed JSON uses lists of path strings for `code` and `inputs` and omits
`code_sha256`. Seal once in a bounded CPU scope; outputs never overwrite a prior
spec. Content sealing is the only stage that needs an initial complete data read.
The local worker then verifies full input hashes once per attempt in its bounded
preflight; the launcher only rechecks source hashes.

```bash
.venv/bin/python scripts/run_campaign.py seal unsealed.json sealed.json
.venv/bin/python scripts/run_campaign.py init \
  --contract research/campaign_2026-10-04/EXECUTION_CONTRACT.md --block local001
.venv/bin/python scripts/run_campaign.py launch sealed.json --attempt run001
.venv/bin/python scripts/run_campaign.py status
```

`--root` and `--state` are global options placed before the subcommand. `launch`
also accepts `--python /absolute/interpreter`. No shell interpolation occurs.
Existing required artifacts reject a launch; a corrected retry needs a fresh
attempt ID and fresh output identity. Existing run directories with extra files
should likewise be avoided by the controller.

Each attempt runs as `wnn-c04-ATTEMPT.service`, with its monitor and command in
one cgroup, an independent timeout (at most 12 hours), no automatic restart and
control-group termination. GPU allocation uses only `WNN_GPU_MEMORY_BYTES` when
explicitly set. The default 4.2-billion-byte allocator limit leaves headroom for
context and other device users. Observed total memory is checked every iteration
against <=4.8 billion bytes; failed observation pauses the campaign. This is a
sampled monitor, not a hardware reservation of all device allocations.

A block contains 86,400 GPU-job seconds. Admission reserves the job timeout plus
930 seconds (900-second source/input preflight, 20-second runtime guard and
10-second systemd stop grace). Completed and failed jobs charge measured worker
wall time, including preflight and teardown. Uncertain launches and lost monitors
charge the entire reservation, never refunding unknown work. CPU jobs do not
consume GPU allowance. Renewals retain previous block/attempt records:

```bash
.venv/bin/python scripts/run_campaign.py block local002 --review 'Outcomes reviewed; updated priorities recorded in ledger'
```

Durable evidence includes exact spec/resolved argv/runtime hashes/environment,
heartbeats, peak RAM/total VRAM, cgroup OOM counters, log, exit status, artifact
hashes, and incident records. Systemd owns lifetime if a monitor disappears.
Before new admission, live services must match active records and running
heartbeats must be recent. Normal occupied slots return `AdmissionBusy` without
pausing. The controller waits for service teardown before subsequent work.

```bash
.venv/bin/python scripts/run_campaign.py pause --reason 'Diagnosing resource incident'
.venv/bin/python scripts/run_campaign.py reconcile
.venv/bin/python scripts/run_campaign.py acknowledge INCIDENT_ID --review 'Diagnosis and correction; fresh attempt and outputs selected'
```

Reconciliation records a vanished service's systemd result and conservatively
charges it. It does not terminate an unknown live worker or automatically retry.
Acknowledgement requires no active attempts and a diagnosis bound to the current
incident; it refreshes reviewed runner hashes while retaining historical ones.
Renew a contract through a fresh state directory rather than altering old evidence.

Tests mock process/systemd/CUDA interactions. A controller-owned real CPU smoke
and independent runner review are still required before scientific fits.

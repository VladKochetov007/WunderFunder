# Bounded local GitHub progress reporter

`scripts/report_sequence_progress.py` reads the existing mirror config and its
source-native cached fit directories. It creates no SSH connection, reads no
profile as a fit, loads no model, runs no evaluator and uploads no weights.
Root supplies one/two PR numbers and the existing process-only
`WNN_GITHUB_BASIC` environment credential (base64 username:PAT). The helper never
stores or logs that value, decoded credentials, request headers or HTTP error
bodies. Dry runs do not construct the API client or read the credential.

The REST client is restricted to `api.github.com` and
`VladKochetov007/WunderFunder`. Redirects and environment proxies are disabled.
It verifies `GET /user` returns exactly `SemyonEpanov`, verifies the selected PR,
and verifies the account again immediately before each write. The API surfaces
are the official [authenticated user endpoint](https://docs.github.com/en/rest/users/users#get-the-authenticated-user)
and [PR conversation issue comments](https://docs.github.com/en/rest/issues/comments#create-an-issue-comment).

From the architecture execution worktree, first render the actual cached bodies
with Root's real PR numbers:

```powershell
$env:PYTHONPATH = 'src;.'
python scripts/report_sequence_progress.py --mirror-config C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/mirror-config.json --l4-pr <L4-PR> --b300-pr <B300-PR> --dry-run --once
```

This writes `github-report-pr<ID>-dry-run.md` and `github-report-dry-run.json`
next to the mirror config. Actual no-network preview bodies have also been
generated at `github-report-l4-preview.md` and `github-report-b300-preview.md` in
that external artifacts directory; those previews do not assume PR numbers.

After Root has supplied the credential only to the reporter process environment:

```powershell
python scripts/report_sequence_progress.py --mirror-config C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/mirror-config.json --l4-pr <L4-PR> --b300-pr <B300-PR> --interval-seconds 1800 --poll-seconds 30 --max-seconds 777600
```

Use `--once` for one live due/reconciliation cycle. Root owns actual start and all
posts; development/testing performed no authenticated requests or posts. Run a
background instance in a hidden window. The helper is single-threaded, sets
OMP/MKL/OpenBLAS limits to2 and caps both its loop and request timeouts at the
declared wall limit (at most9days for the newly authorized B300 envelope).
L4 reporting remains capped at24hours. The original reporter start clock is
persisted and reused on restart; restarting does not silently reset these limits.

## Evidence and completion labels

Expected panels are the8declared L4 fits (12epochs) and4B300 fits (30epochs).
Whole source-bound metadata must contain10325train fitting and1873valid members,
the official split SHA and consistent source/config/data identities. L4 update
budgets are calculated from actual config/member/world-size padded schedules;
B300 additionally requires exactly774600updates.

`fit complete` requires the canonical `terminal_status.json` producer manifest:
completed status, exact completed epochs/updates, coherent identity and a valid
64-hex `model_sha256`. This records a producer checkpoint SHA. It does not claim
an archive was downloaded, uploaded or independently byte-verified. A final model
SHA is not required to equal an epoch snapshot SHA: the files can be serialized
with different payloads. Epoch indexes remain independently retained history.

Final scientific endpoint completeness is separate. L4 requires the actual final
epoch12full-valid event at its complete update count,1873sequences and3528273
scored rows. B300 requires native `fit/score.json` with the same full coverage,
official metric, exact candidate checkpoint/config and an explicitly hash-bound
fixed0.9reference+0.1candidate grade. Per-epoch B300 reports remain reports until
that grade arrives. No winner, public-score forecast or TL3 delta is generated.
Clipping is labelled as observed tail samples; unavailable nonfinite counters
remain explicitly unrecorded. Source UTC/age comes from cached native producer
observations, separately from the comment's generation timestamp. A source clock
ahead of local time is shown explicitly.

CollectorR1 already includes terminal manifests and epoch indexes. Score receipts
require the separate nontraining `scripts/collect_sequence_monitor_r2.py`:
SHA256 `797794ca537ab96eb76e903c705b16e5e7d303b087108bb380b4282bceb9145b`.
Root stages it as each `R/control/collect_sequence_monitor_r2.py`, swaps both
config `collector_path` values, and restarts the local bridge to load the new
local whitelist. Existing source/cursor names are preserved. R1 remains byte
unchanged; no frozen trainer or active controller file is modified.

## Restart and uncertain-write behavior

The state is `github-report-state.json`; bodies and append-only comment/error
receipts use the `github-report-` prefix. An OS file lock prevents concurrent
instances. Each PR gets a batched update on terminal-state change or every30min,
with no per-epoch posting. Comment IDs, URLs, source identities and body hashes
are retained.

A pending body/marker is persisted before any HTTP write. After restart, the
helper searches that exact marker under the verified account before creating a
comment. The POST attempt itself is durably recorded immediately before the
write. If its response is lost and no marker is visible, the helper does not
repeat that POST automatically; it performs read-only reconciliation every5min
and records the uncertainty for Root. This preserves the no-duplicate boundary.
Root can resolve an outstanding uncertain write using the marker/receipts; do
not discard pending state blindly. No token belongs in any archive or state file.

Focused tests use synthetic receipt fixtures and mocked HTTP only: exact-budget
completion, profile exclusion, full-grade coverage, cached/future source clocks,
wrong-account refusal, pending-marker recovery after a lost response, no-repeat
uncertain writes, dry-run isolation, API-origin/error redaction, and collectorR2
cursor/score preservation. These tests do not establish live API permissions.

## Explicit optimized B300 panel

The original B300 panel keeps its strict774600-update rule. A separately authorized
optimized recipe uses `--panel-manifest <frozen.json>`, with this exact schema:

```json
{
  "schema": "wnn.sequence-progress-panel.v1",
  "panels": {
    "b300": {
      "run_prefix": "b300-opt",
      "epochs": 30,
      "source_sha256": "<actual producer source SHA256>",
      "split_sha256": "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c",
      "cells": {
        "labels_s101001": {"config_sha256": "<canonical config SHA256>", "expected_updates": 0},
        "labels_s101002": {"config_sha256": "<canonical config SHA256>", "expected_updates": 0},
        "complement_s101001": {"config_sha256": "<canonical config SHA256>", "expected_updates": 0},
        "complement_s101002": {"config_sha256": "<canonical config SHA256>", "expected_updates": 0}
      }
    }
  }
}
```

The placeholders and zero budgets above must be replaced before use. Each config
SHA is `canonical_sha(parsed_config)` from the producer identity, rather than a
newline-sensitive file-transport digest. Each positive update budget must equal
the padded schedule derived from that cell's actual native metadata. Exact30epochs,
official whole-sequence10325/1873membership, source and split remain mandatory.
The reporter reads the manifest once, retains its exact byte SHA256 in comments
and receipts, and reads only `b300-opt-<cell>-fit`. It never falls back to the
historical `b300-<cell>-fit` directories. A source/config/budget mismatch produces
an evidence-unavailable row and cannot produce a completion claim.

Root appends this third logical panel to the existing bridge `hosts` array:

```json
{
  "host": "b300-opt",
  "rpc_root": "C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/b300-rpc",
  "run_root": "/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/throughput-engineering-r1/runs",
  "collector_path": "/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/control/collect_sequence_monitor_r2.py"
}
```

The physical B300 RPC queue is reused; unique requests, cursors, ownership and
dashboard IDs remain qualified by `b300-opt`. Existing `l4`/`b300` entries and
bridge state stay present. Root restarts only its local bridge to load this
nontraining revision; no frozen trainer or active scientific process is edited.

After Root freezes the selected recipe identities, render before restarting only
Root's existing reporter with its process environment credential:

```powershell
python scripts/report_sequence_progress.py --mirror-config C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/mirror-config.json --l4-pr <L4-PR> --b300-pr 69 --panel-manifest C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/b300-opt-panel.json --interval-seconds 1800 --poll-seconds 30 --max-seconds 777600 --dry-run --once
```

Remove `--dry-run --once` for the bounded live reporter. Keep the same state path:
an older pending write is reconciled first; prior comment receipts and original
panel's latest comment remain retained, while the manifest digest makes the new
recipe transition explicit. The persistent original start clock is preserved.
This mapping does not approve any profile or fit, or promote historical partials.

## Two explicitly mapped L4 GRU continuations

The same optional frozen panel file accepts one additional top-level key,
`l4_continuations`, with exactly these two entries:

```json
"l4_continuations": {
  "gru_s101001": {
    "run_id": "l4-gru_s101001_monitor_opt-fit",
    "source_sha256": "<actual continuation producer source SHA256>",
    "config_sha256": "46364a0f68eefea9464918e848bf2af5166af4e2d65fbfb78d16d7c71d969325",
    "expected_updates": 77520,
    "epochs": 12
  },
  "gru_s101002": {
    "run_id": "l4-gru_s101002_monitor_opt-fit",
    "source_sha256": "<actual continuation producer source SHA256>",
    "config_sha256": "fccc9a5bd2bd10eac158c6a647a5feea5368e18e88c4d0ed62bf3d71188d875b",
    "expected_updates": 77520,
    "epochs": 12
  }
}
```

Root supplies actual producer source identities from native R3 receipts. The
620-file inventory digest is a different identity and must not be substituted.
Keep existing `panels.b300` unchanged, or use `"panels": {}` for a file mapping
only L4. This requires `--l4-pr`; no new command-line flag or collector revision
is needed. The other six L4 rows remain unchanged. Missing or inconsistent new
continuation evidence never falls back to an older completed row. Only these
exact two host-qualified run IDs are accepted; arbitrary aliases are refused.

Reports retain previous comment history and explicitly describe the preserved
S1 epoch/log prefix followed by S3 updates. They do not attribute all12epochs to
S3 or infer a quality gain. A final scientific endpoint requires the actual
full-valid epoch12event at77520updates bound to the final S3 terminal identity.
The separately published transition proof remains the lineage authority.

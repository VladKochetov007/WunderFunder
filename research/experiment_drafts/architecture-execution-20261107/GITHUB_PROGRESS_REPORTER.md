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

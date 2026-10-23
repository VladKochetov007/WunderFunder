# ROOT score-result acceptance helper V1

This local standard-library helper validates one future score result after the
reviewed V4 bounded reader has captured it. Its input API takes the exact
`RESULT.json` bytes, the raw reader stdout, raw score `ADMISSION.json` and
`LAUNCH.json`, the already accepted export acceptance and exact-stop receipts,
the frozen export binding, and the captured export manifest. It verifies the
embedded base64 byte joins, the score unit and invocation, the strict native
`active/exited`, PID `0`, code `1`, status `0` success tuple, and the fixed
export/source/membership joins before it returns a compact acceptance object.

The report contract is the frozen scorer V5 output: five fixed arms and four
paired contrasts, the `global_wp_clip2_v1` metric, `is_scored AND (t >= 99)`
mask, source and split pins, fixed model/checkpoint identities, registered
source metadata, and the exact export-manifest/prediction/sidecar pins. Numeric
score and bootstrap fields must be finite JSON numbers; booleans are rejected
where numeric values are expected. Arm names, contrast names, bootstrap sample
counts, seed, component counts, intervals, and metric IDs are checked. The
helper only validates the scalar report; it does not interpret a gain or make
a public-transfer claim.

`build_root_acceptance()` keeps the exact original report bytes intact and
returns ROOT's compact receipt with the original result path, byte count, and
SHA-256. `write_root_acceptance_exclusive()` creates only the new receipt with
`O_EXCL`/`O_NOFOLLOW`, then fsyncs the file and parent directory. It never
rewrites or copies `RESULT.json`. The receipt uses
`fg19.first-cell-score-root-acceptance.v1` /
`ACCEPTED_FIXED_FIRST_CELL_DEVELOPMENT_SCORE_REPORT` and remains at
`PENDING_EXACT_STOP` until ROOT performs its separate exact-owned stop.

The code reads no files itself, imports no scorer or model code, does not use
NumPy, and has no B/systemd/network path. Its fixture uses the local approved
export acceptance/stop, binding, and captured manifest as bounded metadata
inputs plus synthetic scalar score rows. No score result or quality conclusion
exists in this preparation packet. See [LOCAL_CHECKS.md](LOCAL_CHECKS.md) for
the exact focused validation record.

## ROOT score-result caller V4 — local source preparation only

`ROOT_ACCEPT_SCORE_RESULT_V4.py` is the narrow V3 successor. It detects both
existing output paths and symlinks, including a dangling
`ROOT_ACCEPTANCE_BINDING.json` symlink, before any helper call or receipt write.
The exact V3 rejection remains preserved in
[`ROOT_CALLER_SOURCE_REVIEW_V3.json`](ROOT_CALLER_SOURCE_REVIEW_V3.json); V4
does not change scoring, source validation, result interpretation, or receipt
schemas. [`ROOT_CALLER_V4_PREFLIGHT_PROOF.json`](ROOT_CALLER_V4_PREFLIGHT_PROOF.json)
records the exact source delta, and
[`test_root_score_acceptance_v4_preflight.py`](test_root_score_acceptance_v4_preflight.py)
uses the guard expression with synthetic paths to check absent, existing, and
dangling-symlink cases.

The caller is one-shot. If any write fails, preserve every output already
present and have ROOT inspect the exact files, source, and raw inputs read-only;
never rerun acceptance or blindly retry this caller. The score acceptance and
auxiliary binding are separate create-only files and cannot be made atomic as a
pair. If the acceptance is durable but binding creation fails, ROOT may use a
separately reviewed binding-only recovery if needed; it must not repeat score
acceptance. Generic disk failures do not add another preflight gate.

The V4 proof and three focused fixture cases passed, and scoped Ruff checks
passed. This packet does not execute the caller, read score or array payloads,
query B, or create an acceptance receipt. Independent source review passed in
[`ROOT_CALLER_SOURCE_REVIEW_V4.json`](ROOT_CALLER_SOURCE_REVIEW_V4.json),
7,666 bytes / SHA-256
`5ebc3e35cfa135ebb105d3fdce62a85aef2b3a40b61cf452425f375023329209`, with
status `PASS_SOURCE_ONLY_ROOT_SCORE_ACCEPTANCE_CALLER_V4`. That review validates
the local caller source and its fixed helper/reader/status chain; it is not an
actual score-result acceptance or exact-stop receipt.

## ROOT score-result caller V5 — next local record and body pins

`ROOT_ACCEPT_SCORE_RESULT_V5.py` preserves the reviewed V4 symlink-aware
preflight and one-shot write policy. Its only five literal changes target the
separate local `score_v2` record and the V12 admission body/review path, schema,
and status. [`ROOT_CALLER_V5_LITERAL_REBASE_PROOF.json`](ROOT_CALLER_V5_LITERAL_REBASE_PROOF.json)
proves the inverse AST equality to V4 after restoring those five literals.
Independent source review is recorded in
[`ROOT_CALLER_SOURCE_REVIEW_V5.json`](ROOT_CALLER_SOURCE_REVIEW_V5.json),
5,812 bytes / SHA-256
`6f35785364ca4734183c44850562981e0d3624814ae53d731254e0a93006b267`, with
status `PASS_SOURCE_ONLY_ROOT_SCORE_ACCEPTANCE_CALLER_V5`. This rebase does not
execute the caller or create a score acceptance; ROOT owns any eventual
read-only capture, acceptance, and stop.

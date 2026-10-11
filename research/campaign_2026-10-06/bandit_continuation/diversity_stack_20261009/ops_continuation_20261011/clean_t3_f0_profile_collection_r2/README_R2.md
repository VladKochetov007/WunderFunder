# Clean T3 F0 profile terminal collector R2

R2 is an additive, read-only terminal-collection successor to the preserved R1
profile packet at commit `9080f2ae21780595bbed0d163f6f562eb475a34e`. It binds
only the root-admitted kernel
`vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1-runner`, pushed once as
version 1 with the READY private dataset version 1 and source archive SHA-256
`0ba1b1f7b99dc5dabd892c7d99aa030733465aebc73bb652455c9bc527e0d65f`. The
copied admission, push, and READY receipts are byte-pinned. R2 does not poll
other targets, dispatch, cancel, upload, retry a kernel, or run a fit.

## Inventory interface and completion rule

The installed Kaggle CLI 2.2.4 `kernels output` command consumes
`list_kernel_session_output`, not the different `kernels files` endpoint. The
installed implementation submits `user_name` and `kernel_slug`, leaves
`version_label` unset, and writes the first-page `response.log` to
`<kernel-slug>.log` in the output directory. Its session-output descriptors
expose only `file_name` and a URL. R2 validates safe normalized relative POSIX
names, paginates all pages, and never persists descriptor URLs or raw page
tokens. The receipt records the first-page response-log byte count and SHA-256
so the SDK-generated sidecar can be verified before collection is called
complete. The endpoint exposes no per-file remote size or checksum; R2 records
the downloaded byte count and SHA-256 without claiming remote-byte identity.

The session-output request is current-slug scoped, not API-version pinned. R2
binds this unique slug to the exact root push v1 receipt and the returned runtime
spec/source hashes. Do not reuse it after pushing another kernel version. The
separate `kernels files --format json` query and its versioned and unsuffixed
empty responses are retained as diagnostics only; an empty result there cannot
block fetching logs or establish that session outputs are empty.

Before `kernels output`, R2 checks the authoritative session-output names for a
collision with `<kernel-slug>.log`. The CLI writes that sidecar after download,
so a collision aborts before the command can overwrite provider output. A
valid empty session-output listing is allowed; logs are still fetched, and an
empty output payload closes only when its sidecar matches the response-log
metadata. `collection_complete` requires successful low-level logs/output
collection, complete session-output pagination, exact equality between
provider descriptor names and downloaded payload names, and exact local
response-log size/SHA agreement. Missing pages, invalid descriptors, name
collisions, missing files, unexpected files, symlinks, absent log metadata, or
sidecar mismatch leave it false. `profile_success` remains false for the
observed provider ERROR, independent of collection completeness.

The earlier raw R1 capture already contains the terminal logs and 84 local
output files. A read-only session-output listing found 83 descriptors and the
same 83 payload names, plus the SDK log sidecar. That R1 inventory did not
record `response.log` size/SHA, so it is useful name-set evidence but does not
itself satisfy the stricter R2 completion rule. R2 will write its fresh exact
inventory and handoff into immutable `collection_run_NNN/` receipts. The
top-level `_LATEST.json` files are replaceable pointers only. Each invocation
gets a new run directory; page files and summary receipts use create-once
writes. Before the shared collector replaces an incomplete low-level receipt,
R2 copies its exact bytes into a numbered history file. The
`already_collected` path resolves `terminal_collection.attempt_directory` from
the pinned receipt, preserving retries beyond `attempt_001`.

The captured terminal outcome is an execution failure, not a scientific result.
The two seed-3 T4 profiles had identical training arguments apart from unique
run names. At 277.9 seconds the worker reported
`RuntimeError: Timeout` for profile 1 against its 240-second worker timeout;
the outer runner then exited with `CalledProcessError`. The result summary
records no finished jobs or artifacts. Captured resource records show peak
host RSS of 581,152,768 and 587,087,872 bytes and sampled peak GPU allocation
of zero bytes; these counters do not establish successful GPU work. The
profile-only command produces no finite-gradient counter or checkpoint reload
evidence. See `RAW_TERMINAL_CAPTURE_REPORT_R1.md` for receipt pins, exact
commands, captured hashes, quota snapshot, and collection caveat. No profile
passed, no checkpoint was produced, and these timeouts are not a negative model
quality result.

## Bounded invocation

After independent packet review, the exact wrapper command is stored in
`OPERATOR_STATUS_HANDOFF_R2.json`. It runs under 25% CPU, one thread, 700,000,000
bytes RAM, and 720 seconds. Downloads are capped at 8 GiB per attempt, with a
4 GiB free-space floor applied globally. Earlier attempts are retained; the
per-attempt bound is not presented as a total-retention cap. The code uses the
main checkout's reviewed Kaggle CLI and collector sources, but writes only in
this isolated worktree. A status-poll row with no real `status` is reported as
`provider_status_unavailable` and exits nonzero rather than being misread as a
nonterminal job.

Focused offline validation command:

```sh
cd /home/vlad/development/wunder/wnn_kaggle_t3_f0_packet_builder_20261011_r1 && WNN_SLICE=wnn-campaign.slice WNN_CPU_PCT=25 WNN_CPU_QUOTA=25% WNN_THREADS=1 WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python -m unittest discover -s /home/vlad/development/wunder/wnn_kaggle_t3_f0_packet_builder_20261011_r1/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_profile_collection_r2 -p 'test_collect_profile_r2.py' -v
```

R2 contains 16 focused tests for CLI file-page parsing, valid empty session
output, nested safe filenames, session-output pagination, URL/token secrecy,
response-log closure, malformed inventory, collision handling, terminal
status errors, immutable retry receipts, and resolving `attempt_002`.

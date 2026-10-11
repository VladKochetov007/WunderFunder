# R6 bootstrap-only successor (prepared, not admitted)

R5 version 1 stopped in bootstrap before external inputs were mounted or the
teacher driver started. Its terminal trace is sealed in the parent packet.

This identified successor keeps the R5 kernel ID and private dataset version,
with an intended kernel version 2. It changes only the kernel bootstrap's
external-mount logical-path guard. The archive-member allowlist, dataset
archive/manifest, source, worker specs, data identities, teacher recipes,
resource limits, and embedded R5 science authority remain byte-identical.

The repair uses a distinct external-mount key allowlist derived from the exact
`EXPECTED_EXTERNAL_MOUNTS` mapping, which is already compared for exact equality
to the mounted manifest. It keeps the archive whitelist separate and rejects
absolute paths, parent components, backslashes, and undeclared keys.

The actual bootstrap entry point was exercised against the unchanged R5
archive in a temporary local filesystem fixture. Eight tests cover a valid
mapping, missing/extra/wrong-valued mappings, absolute/traversal/backslash
paths, and an actual archived member omitted from the fixture manifest's
archive allowlist. The fixture replaces Kaggle mount paths with temporary
directories and placeholder input files, and intercepts the driver launch;
it does not validate real mounted input contents, run a profile, or start a
worker. The R5 coverage gap was that earlier archive and authority checks did
not execute the external-mount loop, so this bootstrap defect escaped them.
The nine declared prediction arrays contain exactly 6,630,240,000 float32
payload bytes before NPY headers and sidecars. The proposed terminal budget is
8 GiB total, leaving 1,959,694,592 bytes for headers, sidecars, checkpoints,
receipts, and logs; the exact remote inventory must fit that cap before any
download. Hashing is one-megabyte streaming, and the existing read-only NPY
validator maps one array at a time and scans eight 20,000-by-2 rows per chunk.
The packet proposes a 4 GiB free-disk reserve and bounded 1,800-second
download ceiling, both pending a root terminal-collection admission. No
network throughput is inferred. See `R6_BOOTSTRAP_VALIDATION_R1.json` and
`R6_TERMINAL_COLLECTION_BUDGET_R1.json` for exact output paths and hashes.

The packet remains pending independent review and a fresh root release bound
to the corrected bootstrap and kernel version 2. It does not authorize a
dataset update, kernel push, or fit.

The latest read-only provider refresh in `R6_CURRENT_R5_QUOTA_POLL_R1.jsonl`
queried only the existing R5 kernel target at 2026-10-11 23:57:33 UTC. It
remained terminal `ERROR`; account GPU usage was 63,277.613 seconds, provider
reservation was zero, and campaign headroom was 30,550.224 seconds. This is
not a status query for a version-2 R6 attempt, which has not been pushed. The
root reported that the R5 logical hold was closed in
`ROOT_R5_TERMINAL_RESERVATION_CLOSE_R1.json` (SHA-256
`42a8ca0bd0ca90273cc5ee9d215c54ea12be53e95cc0d3cfea2a5053c0f19aa3`); that
authority receipt is maintained by root and is not copied into this packet.

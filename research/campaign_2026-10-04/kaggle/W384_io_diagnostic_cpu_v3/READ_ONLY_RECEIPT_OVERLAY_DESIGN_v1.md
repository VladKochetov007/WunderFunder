# Same-runtime read-only full-hash receipt overlay: design only

Status: source-only proposal. It changes no shared verifier, kernel, dataset, or frozen packet, and has not been approved or run. The v3 diagnostic remains the strict-hashing baseline.

## Goal and boundary

The current frozen runner can read the same fold input bytes in three places: controller `verify_files` for each spec, and `verify_portable_source` in each of two child processes. A follow-up may test whether one exact Kaggle kernel process can hash an identical mounted file once, then let its own children reuse that verified identity. Reuse is limited to one process tree, one kernel run, and one exact attached private dataset version. Receipts must never be stored for another run or used by the production trainer, grader, or shared source tree.

The overlay would live only in a separate diagnostic kernel bundle. Frozen `scripts/kaggle_common.py`, `src/wnn/portability.py`, split manifests, worker specs, and production verification remain byte-for-byte unchanged. The overlay may wrap the diagnostic controller/child call sites only after it has created and validated the receipt. Every receipt rejection falls back to the unchanged full content hash or aborts; it never converts a failed check into acceptance.

## Receipt creation

For each unique source object needed by controller descriptors or the portable registry:

1. Resolve the expected descriptor and registry SHA-256 values first; reject disagreement. Open the resolved mounted file with `O_RDONLY | O_NOFOLLOW` and retain the file descriptor until every child exits. Require an exact provider dataset slug/version identity, a read-only mount, and a path that is inside that mounted dataset.
2. Capture path/lstat and descriptor fstat before reading: resolved path, device, inode, byte length, `mtime_ns`, and `ctime_ns`. Stream the entire file through SHA-256 from the held descriptor. Compare the observed digest to every bound spec/registry expected digest. Capture fstat and path/lstat again; require all identity fields to match and the pathname to still resolve to the descriptor's same device/inode.
3. Make a canonical receipt containing kernel identity, attached dataset slug and exact version, logical and resolved paths, descriptor SHA, observed SHA, file-stat tuple, mount identity/read-only result, and held-FD identity. Do not treat a filename, size, mtime, external manifest, or claimed hash as a substitute for the actual full read.
4. Put the receipt bytes in a sealed read-only memfd (`F_SEAL_WRITE|GROW|SHRINK|SEAL`) and pass both the receipt FD and corresponding held source FD to each child. Each child checks the seals, `F_GETFL` read-only access, `fstat`, `/proc/self/fd/N` target, current path resolution/stat, mount read-only state, exact kernel/dataset-version fields, and expected SHA before allowing the diagnostic-only receipt wrapper to return the already measured digest.
5. Keep the source descriptors open through child completion. Recheck descriptor and pathname stats, mount identity, and read-only state after child use. Any mismatch invalidates the receipt and marks that run ineligible; it cannot be repaired by trusting the previous filename or size.

The receipt wrapper is an isolated optimization around the diagnostic call sites; no environment switch, cache, or function replacement is installed in shared project code. The ordinary `verify_portable_source` still checks registry schema, logical identity, and size before the wrapper can answer its digest request. Controller descriptors must match the receipt's expected digest as well. Exact dataset-version identity is mandatory; if Kaggle does not expose it inside the runtime, the receipt path is disabled and the baseline full hashes remain required.

## Adversarial invalidation fixtures required before any probe

A pure-stdlib harness must prove rejection or strict-hash fallback for: modified bytes after receipt creation; same-size content replacement; file replacement with a new inode; symlink retarget or path escape; changed mtime/ctime/device/inode; writable or remounted input; writable/unsealed/tampered receipt memfd; missing/closed/writable source FD; descriptor target not matching the expected path; wrong dataset version or kernel ID; disagreement between spec SHA and registry SHA; duplicate logical names mapped to distinct inodes; and a failed/short initial full read. Positive reuse requires a real opened descriptor, a full digest match, stable before/after identity, a sealed receipt, and read-only mount. No fixture may accept on size or filename alone.

## Compare against simpler policies

- **Strict duplicate hashing (baseline):** simplest trust model and always valid. Keep it if v3 shows duplicate reads are not a material part of elapsed time.
- **Copy plus full hash:** read the read-only input and copy to kernel-owned local disk, then hash the copy and check source/copy identities. This still incurs the copy write and hash read, but can be useful if measured local-disk throughput materially exceeds mounted-input throughput on subsequent repeated passes. Require free space for the complete owned copy plus at least 20% headroom; copy only into a fresh diagnostic-owned directory, never alter mounts, and clean only exact owned files. Current v3 records capacity without copying.
- **Same-runtime receipt overlay:** can remove repeated reads after one complete content read, but has the strictest provenance and descriptor-lifetime requirements. Consider only if v3 establishes repeated full reads dominate startup and exact mounted dataset versions are observable. Compare end-to-end elapsed time, bytes read, CPU, and failures against both baseline and the measured copy alternative. Do not project savings from filenames or declared sizes.

## Decision gate

Wait for v3 worker and provider artifacts. Preserve provider status separately from worker summary. Do not admit an overlay or GPU repeat unless v3 completed enough per-file telemetry to show an avoidable duplicate-read cost, receipt eligibility prerequisites were observed, all invalidation fixtures pass, and the root separately reviews a fresh immutable diagnostic identity and whole-kernel wall/quota budget. This document reports no I/O metric and is not evidence that receipt reuse is safe or faster.

## Frozen descriptor overlap (identity arithmetic, not runtime measurement)

Read-only inspection of the current W384-v3 archive and the portable registry found 256 code plus 30 input descriptors in each probe spec. Each spec declares 42,193,078,009 bytes; all 286 path/SHA/size identities are identical between the W256 and W384 probe specs. The controller therefore requests 84,386,156,018 bytes of full reads across those two specs, of which 42,193,078,009 are exact cross-spec duplicates. The portable registry has six sources totaling 37,103,450,341 bytes; every path/SHA/size identity overlaps a controller input. The two portable children request 74,206,900,682 bytes in total. Summing controller and child declared full reads gives 158,593,056,700 bytes before archive/manifest reads. These are descriptor-size arithmetic only: they do not establish bytes actually read, cache behavior, measured throughput, or elapsed time. v3 telemetry must determine actual read completion and timing before any performance claim or receipt experiment.

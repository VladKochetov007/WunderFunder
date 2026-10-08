# Kaggle terminal collector independent review — `cdb594df`

**Verdict: HOLD installation pending receipt-file hash verification.** The committed collector is narrowly scoped to the root-pinned smoke4 and TPUv7 kernels, uses read-only Kaggle operations, preserves attempt directories, and has bounded retries/runtime/disk use. Its current smoke4 terminal receipt is compatible with the existing poller and its recorded file inventory matches the actual collected bytes. A stale or corrupted receipt-listed file can nevertheless be accepted as already collected because neither the collector nor poller rechecks the recorded SHA-256/size.

## Frozen inputs

- Commit: `cdb594df391349cf9a7a84b8ff83589fe771ca30`
- `collect_terminal_outputs.py`: `3c5839553d905a959577c4b559d2d1be7316c7a23d1a7bd18a4a335469fd950c`
- target config: `d4f38864f23508a409ead8b48d55b0f04653b5067b87ee33276545cbfc5b16c8`
- service unit: `3ae06479c4ee7d30cfcd11df7b37b74ce02ad19187dcc72aed328c3bcb44b801`
- timer unit: `24ab64ebb299cb38acafa006ddb243a551724f9bb8ccb1da62265a8c6f089bc1`
- tests: `6fab5bb085d2d300aa24d75e4950bf3bc8bc274dc2eec6d46753b421ed0a97b3`

The snapshot is under `scratch/kaggle_terminal_collector_cdb594df/` in this isolated review worktree. The worktree also had unrelated untracked runner diagnostics; none were staged or changed.

## Checks that passed

- The two configured kernels exactly match the hardcoded slug/metadata, root-admission paths and SHA-256 values, push receipt paths, and admitted packet hashes. The actual root admissions hash to `79adf2e9…86a23` (smoke4) and `dbb32828…c4bd4` (TPUv7), as pinned. Both actual push receipts explicitly record the expected packet hashes, canonical slug, one push, and version 1.
- Execution is limited to `kernels logs` and `kernels output` downloads plus the existing poller's quota/status reads. The code has no push, cancel, stop, create, or launch path. The command uses the canonical kernel slug; version 1 and the one-push receipts bind the provider's default latest-version source for these targets.
- Attempts use incrementing `attempt_NNN` directories and fail on directory collisions. Per-target retries are capped at 3. Command deadlines are 800 seconds; the support wrapper caps a service run at 850 seconds and systemd at 900 seconds. The config caps each command at 8 GiB and requires at least 4 GiB free disk throughout. The 8 GiB limit is per command (logs and outputs separately), so one attempt can approach 16 GiB before the global free-space guard stops it.
- The actual smoke4 receipt at `research/campaign_2026-10-06/bandit_continuation/kaggle_warm_cohort/smoke4_20261008/KAGGLE_SMOKE4_TERMINAL_OPERATOR_RECEIPT_R1.json` is recognized by both the committed collector's completion check and the actual `poll_provider._terminal_evidence`. Recomputed inventory matches all 37 receipt entries byte-for-byte and hash-for-hash; 31 are provider output files. Its status remains `KernelWorkerStatus.ERROR`, so successful downloading does not imply a successful fit. Empty provider output is separately marked as insufficient to infer kernel execution.
- The collector suite passed: 13 tests in 2.77 seconds using the service `.venv` Python. Python 3.11 compilation passed for the collector and tests. The collector service timer is currently not-found/inactive; this review did not install it, execute the collector, or contact Kaggle.

## Required correction

`collection_is_complete()` and `poll_provider._terminal_evidence()` only require receipt-listed paths to exist. They ignore the saved `size_bytes` and `sha256`. A scratch test changed a listed file after writing its receipt, and both functions still returned “collected”; the poller would stop status monitoring and the collector would skip another download. Reproduction: `scratch/kaggle_terminal_collector_cdb594df/tests/test_receipt_hash_recheck_repro.py`, SHA-256 `b75782309bd31a97de6a557907e59f24e0b01ae98d2aa537f56c1306836ed84d`. Verify every listed file's relative path, size, and digest before treating evidence as complete, and add a corruption test.

The collector also normalizes permissions across every file/directory beneath the target artifact root after a collection. It keeps prior attempt bytes and names intact, but chmods older attempts too. This is a metadata side effect to retain in mind when the target directory contains earlier evidence.

No TPUv7 terminal output receipt was present at review time, so live TPU receipt compatibility remains unobserved; the common schema and fixed target wiring were reviewed statically.

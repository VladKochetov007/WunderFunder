# Native minGRU callback prototype R1

This isolated code-only branch adds a native minGRU model and binding using the root Python reference state dict. The callback reuses the existing causal 112-to-211 feature builder and leaves the shipped GRU callback, trainer and active fit unchanged. No fit or grade was run.

- [Implementation report](REPORT.md) records the recurrence, full-sequence numerical errors, reset/chunk checks, size accounting, CPU timings and claim limits.
- [Build receipt](BUILD_R1.json) binds the root reference commit, source hashes, compiler, extension features and debug binary SHA-256.
- [Focused test receipt](TESTS_R1.json) records the full 62-pass Rust library suite, 15 Python tests and Ruff result.
- [20,000-row runtime receipt](RUNTIME_PROBE_R1.json) preserves deterministic input/model hashes, full raw callback timings, exact/fast/Q8 errors, storage, one-core host and peak RSS.
- [Source manifest](SOURCE_MANIFEST_R1.json) lists modified and reused code hashes, reference/input provenance and built binary identity.
- [Native implementation](../../../../native/src/min_gru.rs), [Python binding](../../../../native/src/python.rs), [focused tests](../../../../tests/test_min_gru_native.py) and [reproducible probe](../../../../scripts/probe_min_gru_native.py) are the runnable implementation.
- [Python minGRU reference](../../../../src/wnn/min_gru.py) was introduced in root commit `750701ce96807c2fc6921c7e12512893f8ed33c1`; this branch is based on `b6c38e685826375dd42d8b2aa7faa747ab8b184d` plus that reference commit.

## Preserved command diagnostics

- [Initial build failure](CARGO_BUILD_FAILURE_R1.json): exact command/source hash and the fixed ambiguous `usize` sum type.
- [Initial Python environment path failure](PYTEST_ENV_FAILURE_R1.json): the sparse worktree has no local ignored `.venv`; the retry used the existing MAIN interpreter by absolute path.
- [Workspace formatter failure](FORMAT_CHECK_FAILURE_R1.json): broad `cargo fmt --all` reports legacy differences in untouched files.
- [Focused rustfmt diagnostic](RUSTFMT_SOURCE_FAILURE_R1.json) and its [captured output](RUSTFMT_SOURCE_DIFF_R1.txt): the added binding hunks were corrected; unrelated old regions were preserved.
- [Ruff import-order failure](RUFF_FAILURE_R1.json): corrected and followed by a passing final check.
- [Toolchain path failure](TOOLCHAIN_PATH_FAILURE_R1.json): `/usr/bin/rustc` and `/usr/bin/cargo` are absent; the rustup toolchain under `/home/vlad/.cargo/bin` built the extension successfully.
- [Sparse staging failure](SPARSE_STAGING_FAILURE_R1.json): the first staging command hit omitted sparse-checkout patterns for the new receipt subtree, probe and test; diagnosis preceded the sparse-pattern correction.
- [Staged whitespace-check failure](STAGED_DIFF_CHECK_FAILURE_R1.json): the exact staged check found trailing spaces in blank lines of the preserved raw rustfmt output; that artifact remains unchanged, and the remaining paths receive the normal check.

The experiment is indexed as `E-MINGRU-NATIVE-CALLBACK-20261014-R1`. It is deterministic synthetic transport evidence only: no trained weights, competition predictions, release package, architecture selection or speed claim.

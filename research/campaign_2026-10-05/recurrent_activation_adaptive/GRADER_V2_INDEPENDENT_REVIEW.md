# E479 v2 scorer independent review

**Decision: GO for a controller-owned grade using the exact scorer hash below.** This review does not score evaluation predictions or make a public inference. I verified fit-only receipts and checkpoint payloads, then stopped before the first test-role parquet read.

## Exact source and verification

- Scorer `score_recurrent_adaptive_v2.py`: SHA-256 `c9edc310bc8aefe9ddc2df43538a9fac45335ad8d5a9923d93bcf08398ff3965`.
- Synthetic v2 tests `test_score_recurrent_adaptive_v2.py`: SHA-256 `08012f4fe0a55eec4485f3c3fde27a416417612075c33ed7d8df81a8fe8ef6a6`.
- Fit seal `e479_sixteen_fit_v2.sealed.json`: SHA-256 `18ad9aa867f25424d80f3e1e5e8e81c2a6ae67e62e07ad037f7d9ace00494d42`.
- Runner attempt `e479-learned-candidate-sixteen-fit-v2.json`: SHA-256 `e9431f06b4415ddcf0bd334edbd7885bcd0223993abe4582847a43730775776b`.
- Actual fit manifest: SHA-256 `9f6b4f9ed3994a857511a67254eff1f5de01e062e51d95b05da8fd9b1b964317`.
- Actual E479 v2 seal and attempt were passed explicitly; their names, attempt ID, source/input identities, runner state, successful unit result, teardown and worker/parent OOM counters validated.
- `--verify-fit-only` verified all 16 checkpoints against the manifest hashes, exact tensor names/shapes, finite values, identity normalization buffers, model parameter counts, fit orders, update/cadence telemetry and fit-only normalization. The command returned `all_fit_receipts_and_checkpoints_verified_test_rows_unread`, `arms: 16`, and `test_rows_read: false`.

The verification ran under `scripts/limited` with one logical CPU and a 2 GiB memory cap. The only data rows read to independently recompute normalization were the 24 fit-role sequences; the verifier stops before calling the test-role reader. Synthetic-only checks passed (3 tests), `py_compile` passed, and Ruff passed after the B008 cleanup.

## Reviewed fix and remaining scope

The failed v1 grade expected B=4 and chunk=256 fields on every per-arm record, while the actual sealed fit actor records them once at manifest scope. V2 requires the manifest-level values and rejects contradictory values if arm-level fields are present, matching the actual fit format without weakening cadence validation. The per-arm receipt still binds all 474 updates, seed/optimizer/condition identity, exact shuffled fit order, telemetry positions, effective-alpha bounds and finite learned-alpha telemetry. The new fit-only preverification loads and strictly checks every checkpoint before any test-role rows are read.

V2 retains the inherited independent recurrence and scorer logic. It uses all 16 fixed arms and 28 registered contrasts with the shared 16-component bootstrap mapping; conclusions remain descriptive on already exposed components and separate by seed. No fitting or scoring happened in this review. Any source change invalidates the exact hash reviewed here.

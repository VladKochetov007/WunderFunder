# E488 v5 actual fit-only gate review

I independently ran the complete `validate_full_evidence` path against the
actual E488 fit and probe attempts, specs, probe report, manifest and
checkpoints. The test-row loader was patched to raise if called. The run
completed successfully without reaching it.

## Frozen identities

- `score_duration_independent_v5.py`: SHA-256
  `4fcbf4065c985be6a636e920769e9e2e11d9523570ac9ac698d2a7be56df4d9c`
- `test_score_duration_independent_v5.py`: SHA-256
  `ad7c0f3e8c8c7b1b29e5559e30b26a7ec09063ea7357fccd0db491ebe9e86240`
- E488 fit/probe run receipts were taken from their registered sealed paths;
  the scorer reported 24 validated snapshots and 24 fit-role components.

## Actual gate execution

The command ran under `scripts/limited` with CPUQuota 100%, MemoryMax 2 GiB,
and swap disabled. It called `validate_full_evidence` directly using the
actual sealed fit/probe specs, actual controller attempt JSONs, the fit run,
and the actual probe report. `_selected_test_rows` was patched to throw.

The gate returned `all_fit_only_evidence_passed_test_rows_unread`, verified
all 12 trajectories and 24 snapshots, and reported the probe-plus-fit charge
as 126.02973189299519 seconds. It derived the normalization only from the 24
fit-role feature groups. The 16 test component identities came from the
role manifest; no test parquet row groups or labels were read.

The v5 probe validation preserves the registered clipping behavior. It
requires clipping-factor values to match the configured formula in `[0,1]`
for every actual-shape cell, while requiring active clipping only in the two
native-reference parity checks. This accepts the observed inactive GELU
cell (`clip_factor == 1`) without dropping the finite loss, gradient,
parameter-update, state, source, artifact, resource, or teardown checks.

The final v5 test suite also includes one integrated fixture that runs the
complete fit/probe/spec/attempt validation path under a Parquet wrapper that
permits only the 24 fit-role feature columns and rejects other row groups or
columns. It also patches the held-out loader to fail. This fixture passed with
the other 12 tests.

## Disposition

**GO for the root's actual v5 `--mode verify-only` retry.** The direct actual
evidence gate passed, and the test-row loader remained unreachable. Score
mode remains a separate explicit action after root admission.

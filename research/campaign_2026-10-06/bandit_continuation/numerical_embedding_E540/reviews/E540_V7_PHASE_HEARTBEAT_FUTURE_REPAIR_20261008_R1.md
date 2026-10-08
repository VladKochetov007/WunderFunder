# E540 V7 future phase-heartbeat repair

This code-only change is isolated on branch `research/e540-phase-heartbeat-future-20261008`, based on the reviewed V7 source commit `1b327b4175199b36f09f852e6d11a143f8a62d95`. It does not change the live V7 source, packet, recipe, objective, worker, or provider attempt. It is not production-approved and requires independent review before use in a future packet.

The outer GPU runner now projects E540 V7's atomically replaced `cohort.json`, arm receipt, and profile-gate receipt into each live resource heartbeat. It emits only an enumerated stage, `linear`/`ple` arm ID, enumerated statuses, bounded elapsed/profile timings, and bounded completed-update counts. It reads only fixed E540 receipt paths for the exact V7 spec name; files and parent directories cannot be symlinks, and each JSON file is capped at 64 KiB. Malformed, stale, oversized, nonfinite, unknown-schema, or unknown-enum receipts produce `status: unavailable`. Unknown fields—including paths, arguments, error text, hashes, and credential-like values—are never projected.

Validation ran under one CPU and a 600 MB memory cap, with no data reads, CUDA use, provider calls, or fitting:

- `tests/test_kaggle_gpu_runner_phase_heartbeat.py`: 11 passed
- Existing E540 V7 and packet tests: 32 passed
- Ruff check and Python byte-compilation passed

The bounded tests cover allowed profile/gate summaries, generic-job behavior, malformed/oversized/symlinked receipts, unknown schema/status/phase, nonfinite and oversized numeric values, and secret-field non-disclosure. The change only makes phase receipts visible in the outer heartbeat; it does not make those receipts a scientific result or alter the runner's worker lifecycle.

---
kind: review
status: complete
created_utc: 2026-10-22T22:59:50Z
updated_utc: 2026-10-22T23:03:55Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - scripts/thermal_guard.py
  - tests/test_thermal_guard.py
---

# Optional CPU package thermal guard builder receipt

## Scope and status

Prepared an opt-in CPU package temperature check for future use. This is a local proposal only: it was not integrated into the root checkout, enabled on any process, or run against the active experiment. No command, worker, recipe, grader, training code or native code was changed in the root checkout.

The existing guard file was absent at the start of the isolated worktree (`experiment/online-relationships-20261022`, HEAD `273eed267dd71f639272b6e55ada74c36b4133e2`). Its root source SHA-256 was `f62e983eb0364d1d6042b87a29d013d695ea91dac62cda4aad17bb62c1ac2c16`; the unchanged root smoke3-guard test SHA-256 was `f7775d1390363e6c9a7450133768dd60bd0d265fd8966cd782e423ba1db381d6`. Only that exact guard source was copied as the patch base; no unrelated root commits were imported. The resulting guard SHA-256 is `d6de8170d4e19b71323332e31d2aceca889140ca318e4246b35dccbf66aa6a3b`; the focused test SHA-256 after the review additions below is `310421bbed9ba9beacba018d3ac3be00737d5b8eb390c05d9dd8e897487cbb60`. The exact unified diff against the root source is at `/home/vlad/development/wunder/research_tmp/thermal_guard_builder/thermal_guard_cpu_package.patch` (SHA-256 `8f1c4aa760d1f8182263d4951f31d46a72639c116d2eb6fc0824886994e5da2a`).

## Change

`--cpu-package-temperature-guard` explicitly enables a sysfs-only read of all readable `temp*_input` values whose matching `temp*_label` starts with `Package id `; the hottest package reading is used. At 97°C while running the guard pauses the child; while paused it remains paused above 90°C and can resume at or below 90°C. Missing or unreadable enabled sensors raise a clear error before the child starts, with a `guard_failed` receipt. GPU temperature, available-memory and PSI checks retain their existing thresholds and order. The CPU sample is included in events/heartbeats only when enabled, so invocation and output defaults remain compatible.

## Checks and limits

Final checks ran separately under `systemd-run --user --scope` limits of CPUQuota=5%, MemoryMax=350M, MemorySwapMax=0 and RuntimeMaxSec=60. Pytest temporary files used physical scratch under `/home/vlad/development/wunder/research_tmp/thermal_guard_builder`; plugin autoload and bytecode writes were disabled.

- `ruff format scripts/thermal_guard.py tests/test_thermal_guard.py`: 2 files left unchanged.
- `ruff check scripts/thermal_guard.py tests/test_thermal_guard.py`: all checks passed.
- `pytest -p no:cacheprovider --basetemp=/home/vlad/development/wunder/research_tmp/thermal_guard_builder/pytest-tmp tests/test_thermal_guard.py`: 10 passed in 2.60s. Since the initial seven-test run, added checks that enabled missing-sensor safe-start writes `guard_failed` without spawning a child, and that a cool CPU does not clear either a hot available-memory or memory-PSI pause reason. The focused suite imports only the guard's standard-library dependencies and pytest; it does not import torch or project data.

No live sensor availability or guard runtime was measured, and the new guard was not launched. Sensor discovery currently requires Linux hwmon labels beginning `Package id `; other CPU driver label conventions fail closed and would need an explicit, source-grounded extension before opt-in use. The thresholds are the requested proposal, not evidence of thermal safety on other systems. Root owns review and any later admission.

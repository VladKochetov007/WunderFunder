---
kind: review
status: complete
created_utc: 2026-10-22T22:59:50Z
updated_utc: 2026-10-22T23:07:46Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - scripts/thermal_guard.py
  - tests/test_thermal_guard.py
---

# Optional CPU package thermal guard builder receipt

## Scope and status

Prepared an opt-in CPU package temperature check for future use. This is a local proposal only: it was not integrated into the root checkout, enabled on any process, or run against the active experiment. No command, worker, recipe, grader, training code or native code was changed in the root checkout.

The existing guard file was absent at the start of the isolated worktree (`experiment/online-relationships-20261022`, HEAD `273eed267dd71f639272b6e55ada74c36b4133e2`). The exact root guard source SHA-256 is `f62e983eb0364d1d6042b87a29d013d695ea91dac62cda4aad17bb62c1ac2c16`; only that source was used as the patch base, with no unrelated root commits imported. The resulting guard SHA-256 is `d6de8170d4e19b71323332e31d2aceca889140ca318e4246b35dccbf66aa6a3b`; the focused test SHA-256 is `310421bbed9ba9beacba018d3ac3be00737d5b8eb390c05d9dd8e897487cbb60`. The committed [combined git-apply patch](CPU_PACKAGE_THERMAL_GUARD_PATCH_20261023_R1.diff) contains that guard change and the new focused test file (SHA-256 `4917719b5b3bdd59201741179a7b1028b39f615cc2272197261ab04e5469de6d`). `git apply --check` succeeded against a physical-scratch copy with the exact root-source hash; applying there produced guard and test hashes matching the proposed files. No root file was changed during verification.

## Change

`--cpu-package-temperature-guard` explicitly enables a sysfs-only read of all readable `temp*_input` values whose matching `temp*_label` starts with `Package id `; the hottest package reading is used. At 97°C while running the guard pauses the child; while paused it remains paused above 90°C and can resume at or below 90°C. Missing or unreadable enabled sensors raise a clear error before the child starts, with a `guard_failed` receipt. GPU temperature, available-memory and PSI checks retain their existing thresholds and order. The CPU sample is included in events/heartbeats only when enabled, so invocation and output defaults remain compatible.

## Checks and limits

Final checks ran separately under `systemd-run --user --scope` limits of CPUQuota=5%, MemoryMax=350M, MemorySwapMax=0 and RuntimeMaxSec=60. Pytest temporary files used physical scratch under `/home/vlad/development/wunder/research_tmp/thermal_guard_builder`; plugin autoload and bytecode writes were disabled.

- `ruff format scripts/thermal_guard.py tests/test_thermal_guard.py`: 2 files left unchanged.
- `ruff check scripts/thermal_guard.py tests/test_thermal_guard.py`: all checks passed.
- `pytest -p no:cacheprovider --basetemp=/home/vlad/development/wunder/research_tmp/thermal_guard_builder/pytest-tmp tests/test_thermal_guard.py`: 10 passed in 2.60s. Since the initial seven-test run, added checks that enabled missing-sensor safe-start writes `guard_failed` without spawning a child, and that a cool CPU does not clear either a hot available-memory or memory-PSI pause reason. The focused suite imports only the guard's standard-library dependencies and pytest; it does not import torch or project data.

The root owner independently called the read-only CPU package sensor function; it returned 92.0°C at 2026-10-22 23:03:39 UTC. No child or guard was launched. This is one host observation, not a runtime measurement or evidence about other systems. Sensor discovery currently requires Linux hwmon labels beginning `Package id `; other CPU driver label conventions fail closed and would need an explicit, source-grounded extension before opt-in use. The thresholds are the requested proposal, not evidence of thermal safety on other systems. Root owns review and any later admission.

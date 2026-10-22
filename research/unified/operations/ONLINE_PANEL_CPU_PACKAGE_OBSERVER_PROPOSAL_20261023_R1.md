---
kind: operations
status: proposed
created_utc: 2026-10-22T23:37:23Z
updated_utc: 2026-10-22T23:38:48Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - scripts/observe_online_panel_cpu_temperature.py
  - tests/test_observe_online_panel_cpu_temperature.py
---

# Single-invocation CPU package observer proposal

## Scope and status

This is a local operations proposal for the already-running panel worker. It was not launched, and no live systemd query, freeze, thaw, worker restart, quota or deadline change was made. Model, recipe, grader and original guard bytes were left unchanged. Root owns review, indexing and any later launch.

The observer is hardcoded to `wnn-online-panel-r1-20261023.service` and InvocationID `1e64dd46f5294caba4977275e6c91feb`. It has no target override or job-admission path. Each one-second poll reads the hottest sysfs sensor labeled `Package id ` and obtains `InvocationID`, `ActiveState` and `FreezerState` for that unit. A mismatch or inactive unit ends observation. Before every control command it repeats the identity/state query; its only control commands are `systemctl --user freeze` and `thaw` for this exact unit.

At 97°C or higher, if the unit is running and not already owned/frozen, it requests a whole-unit freeze, which also stops the service's inner GPU guard. It tracks a freeze as its own only when the unit was running and its freeze command succeeded. It requests thaw only at or below 90°C and only when this process owns the freeze and the current state is `frozen`. A unit already `frozen` or `freezing` is left unclaimed, so a later cool sample cannot thaw another owner's freeze. The observer also relinquishes ownership if it sees the unit running or thawing.

If the CPU sensor is missing or unreadable, the observer rechecks the exact active invocation and attempts a fail-closed freeze. It leaves an already frozen/freezing unit unchanged; if the unit is active and its freezer state changes between query and action, it retries the verification up to three times. It then exits with a clear error. A failed systemd query/action also exits; if identity cannot be verified, it does not issue a unit control command.

## Verification

The observer and its tests are local proposed files. SHA-256 values:

- `scripts/observe_online_panel_cpu_temperature.py`: `193b8e07254c0dd0a44cbc346a8800624ba61844c5c6b48da62de40692ac8054`
- `tests/test_observe_online_panel_cpu_temperature.py`: `bb960739eaa52222a165ce45b627f11b668e7765786166e3d988199c04086ad0`

Ruff format left both files unchanged; Ruff check passed. The focused mocked suite passed **9 tests in 2.20s** under CPUQuota=5%, MemoryMax=150M, MemorySwapMax=0, RuntimeMaxSec=30. Tests cover package-label selection and missing-sensor reporting, the 97°C freeze boundary, the 90°C thaw boundary, external-freeze ownership, fail-closed sensor failure, target mismatch/inactivity, and invocation change immediately before an action. The suite mocks all systemd commands; it did not import model/data code or launch the observer.

## Limits for root review

This was not exercised against this host's systemd manager or a live service, so the property output and whole-unit freezer behavior still need root's read-only compatibility review before any launch. The identity query and `systemctl freeze/thaw` are separate calls: the observer checks immediately before each action, but systemd's unit-name command does not atomically bind the action to an InvocationID. A restart in that narrow interval could retarget the command. Also, systemd exposes no owner token for a frozen cgroup; this code avoids adopting a freeze already present when observed, but cannot distinguish a later external freeze layered onto a freeze it initiated. If the CPU sensor fails, the observer exits after requesting freeze and intentionally does not auto-thaw; an operator must inspect the exact service before recovery.

---
kind: review
status: complete
created_utc: 2026-10-12T23:08:16Z
updated_utc: 2026-10-12T23:08:16Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - specs/adr/0034-strategy-execution-full-gpu-allowance.md
  - research/campaign_2026-10-13/strategy_execution_r1/CONTRACT.json
  - /home/vlad/development/wunder/wnn-agent-worktrees/strategy-resources-20261013
---

# Independent review of local GPU resource admission R1

## Question or claim

Does commit `b185e7048785e1c6db60303dbd228c8ce803bd3a` safely support the approved strategy run through the existing durable local runtime, while preserving the 5 CPU, 17 GB committed RAM, 18/20 GB cgroup, no-swap, and one-GPU-worker constraints? The review is limited to resource admission, CUDA allocator setup, historical-record compatibility, environment validation, and the old watchdog/queue paths.

## Mechanism and method

Reviewed the candidate worktree at `/home/vlad/development/wunder/wnn-agent-worktrees/strategy-resources-20261013`, parent `fda03869747eb1a0938ebf9676c20d8db1dd1310`, and the accepted authority in ADR0034 plus the strategy contract. Traced `scripts/bandit_resources.py` through reservation, systemd environment construction, worker admission/monitoring, and the `state/gpu.lock`; traced `src/wnn/resources.py` through `configure_torch`; checked `src/wnn/campaign_spec.py` reserved-environment validation; inspected the running watchdog and local-queue unit definitions and their referenced paths.

Executed targeted tests with the main checkout's Python interpreter and candidate `src` on `PYTHONPATH`, limited to one CPU, 1.2 GB RAM, and 120 seconds. The command was:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s \
PYTHONPATH=/home/vlad/development/wunder/wnn-agent-worktrees/strategy-resources-20261013/src \
scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
-m pytest -q tests/test_bandit_resources.py tests/test_campaign_allocator.py tests/test_campaign_runner.py \
-k 'gpu_service_is_directly_bounded or gpu_memory_safety_reserve or observed_vram_above_withdrawn or available_fraction or explicit_bytes_override or configurable_reserve or spec_rejects_identity_and_environment_bypasses or gpu_lock_rejects'
```

Read the existing shared runtime ledger with the candidate's `job_records()` parser. No fit, CUDA initialization, provider request, or source edit in the candidate worktree was performed.

## Evidence and interpretation

### Confirmed issue: combined RAM reservations are undercounted

`active_reservations()` sums the full RAM reservations of active attempts, while `verify_capacity()` separately checks `active_ram + new_ram <= 17 GB` and `measured_parent + new_ram <= 17 GB` (`scripts/bandit_resources.py:383-398, 435-459`). It omits the unused part of each active reservation, although current parent usage already includes the memory those workers are consuming.

Concrete counterexample: the parent currently uses 10 GB, including an 8 GB active worker and 2 GB of other parent use. That worker has a 12 GB reservation, leaving 4 GB of admitted headroom. A new 4 GB bound creates a possible committed total of `10 + 4 + 4 = 18 GB`. The current checks pass because `12 + 4 = 16 GB` and `10 + 4 = 14 GB`, both below 17 GB. This violates the campaign rule to combine current parent memory, unused admitted-worker reservation, and the new complete bound without double-counting. This is a pre-existing admission-accounting gap exposed by the lowered 17 GB target; fix it before admitting concurrent support or worker reservations. Unknown active-worker cgroup usage should not be treated as zero headroom.

The issue is not a blocker for the first solo fit under the observed state: the shared runtime has 350 parseable historical attempts and zero active/unresolved records, the historical admission barrier is still closed pending the authorized resume, and the campaign cgroup was at 1,634,570,240 bytes with the 18 GB high, 20 GB max, and zero-swap properties verified. The single-worker launch must still fit its measured reservation inside the resulting 17 GB check.

### GPU policy and one-worker serialization pass for the shared runtime

In the candidate durable path, launch and worker preflight still record NVIDIA-SMI usage, but the former 5,000,000,000-byte start/monitor/exit rejection and 4,300,000,000-byte allocator injection are removed (`scripts/bandit_resources.py:559-699, 742-820, 920-960, 1000-1150`). Observed VRAM remains telemetry. The service explicitly receives the configured safety reserve and does not receive either legacy `WNN_GPU_MEMORY_BYTES` or `WNN_VRAM_GB` overrides. `configure_torch()` reads CUDA free/total bytes once at startup, subtracts the nonnegative reserve, and sets the PyTorch allocator fraction from the remaining free bytes (`src/wnn/resources.py:47-110`). A depleted reserve fails closed. This is a startup snapshot plus reserve, not a continuously rebalanced CUDA limit; actual-device shape profiling remains necessary before a long fit.

The controller's GPU lock is `state/gpu.lock`. Root confirmed the plan to resume and reuse the same historical runtime after the candidate code is integrated, so future work shares the existing lock and does not create a competing state-scoped lock. The local queue's `STOP` remains present; its timer is pinned to that same historical runtime and cannot submit old pending work. Current attempt ledger inspection found no worker that could race the first launch.

### Historical records and environment validation pass

The candidate `job_records()` parser successfully read all 350 existing records in the shared runtime; zero were active or unresolved. Historical GPU fields are not required by `active_reservations()`, `reconcile()`, or `status()`. Repository consumers of the new telemetry use `peak_observed_vram_bytes`; I found no operational reader that casts the newly nullable `observed_vram_cap_bytes` or `allocator_admission_bytes` fields to integers. Resume should be performed through the reviewed candidate command so the existing stop is moved to `barriers/` and preserved.

`WNN_GPU_MEMORY_RESERVE_BYTES` is added to `campaign_spec.RESERVED_ENV`; campaign specifications cannot inject it as a user-controlled override. Both the controller and CUDA helper accept only ASCII decimal nonnegative values. The durable service environment is explicitly assembled, then sets the validated reserve for the worker.

### Existing 5 GB monitor/watch paths do not stop the shared durable worker

The installed `wnn-watchdog.timer` invokes `scripts/watchdog.py --once --notify`. Its local check reads the historical barrier and queue, reports conditions, and does not signal or kill workers or apply a 5 GB threshold. The active `wnn-codex-research-watch.service` runs `research/campaign_2026-10-03/continuation/watch.py`; that monitor records journal/OOM incidents and writes a pause marker for the separate 2026-10-01 queue, with no path to the shared `bandit_resources` stop marker or worker process. The old `wnn-local-queue-r1.timer` still points at the legacy queue and shared runtime, but the queue's `STOP` is present. Thus these current monitor paths cannot enforce the withdrawn 5 GB threshold against the resumed durable worker.

Separate legacy routes remain capped: `src/wnn/campaign_spec.py` retains its local `VRAM_MAX`, and `scripts/campaign_job.py` carries older GPU policy. The approved fit must enter through the reviewed `bandit_resources` path and call the updated `configure_torch()`; wrapping it in a separate legacy local campaign launcher would retain that route's cap.

## Limits and validity

Independent targeted checks passed: `24 passed, 123 deselected in 0.16s` across `tests/test_bandit_resources.py`, `tests/test_campaign_allocator.py`, and `tests/test_campaign_runner.py`. Coverage included service-environment construction, invalid reserve values, free-memory fraction calculation, 5 GB telemetry-only behavior, same-state GPU-lock exclusion, and rejection of reserved campaign environment keys. The historical ledger parse was an additional read-only check.

The PyTorch allocator behavior was tested with a fake CUDA object, not on the physical device. No model run, actual CUDA profile, watchdog-triggered incident, or cross-state concurrency was exercised. The review supports removal of the fixed cap on the selected durable shared-state path; it does not establish a safe model batch/shape or certify concurrent RAM admission.

## Decision and next step

**Conditional pass for one new single-GPU-worker attempt after integration and reviewed resume, provided its full measured RAM reservation fits the 17 GB rule.** Correct the confirmed combined-RAM accounting before any concurrent worker/support admission. Keep the historical `state` path so the old queue and new worker share `gpu.lock`; preserve its barrier via the candidate `resume --review` flow. No GPU fit should rely on the unit tests as a device profile.

## Source records

- [ADR0034 resource authority](../../../../../specs/adr/0034-strategy-execution-full-gpu-allowance.md)
- [Strategy contract](../../CONTRACT.json)
- Candidate commit `b185e7048785e1c6db60303dbd228c8ce803bd3a`, parent `fda03869747eb1a0938ebf9676c20d8db1dd1310`
- Running timer definitions: `wnn-watchdog.timer`, `wnn-local-queue-r1.timer`, and `wnn-codex-research-watch.service`

# Independent monitor review

Review date: 2026-10-05. Read-only review of `checks/monitor_jobs.py` and the local systemd monitor units. All fixtures called pure functions with mocked provider responses; no Kaggle commands, worker launches, provider writes, or scientific fits were run.

## Current behavior confirmed

Offline fixtures against the current checker verified that it now alerts on an incorrect `MemoryHigh` soft cap, reports a stale RUNNING timestamp (>8 hours), and handles a QUEUED status with no timestamp without raising. The checker sums GPU memory across all observed devices and now preserves nonzero status-command errors as `OBSERVATION_ERROR`. Its pending-status cap is 20 and its catalog fetch is limited to 100 entries. Local checks validate the authorized 27 GB soft cap, 30 GB hard cap, 12-core quota, disabled swap, active attempt service state, and aggregate GPU use. The systemd timer is read-only and runs periodically.

The companion sequence replication script is numerically the same training recipe as `fit_sequence.py`: the complete diff is limited to accepting a seed argument in smoke/run functions, passing it to Torch initialization and NumPy fit-order shuffling, recording the seed, and hashing the replication script's own source name. Model, optimizer, loss, cohort, evaluation, normalization, and all arm definitions are unchanged. The specified replication seeds 271829/271830 therefore change initialization and fit order only.

## Remaining monitor findings

1. **Future provider timestamps suppress hang detection.** A mocked RUNNING kernel with `lastRunTime=2026-10-06T00:00:00+00:00` at the review clock produced no alert. Its computed age is negative. Reject timestamps too far in the future (with an explicit clock-skew alert) and avoid treating them as a valid start time.

2. **Pinned jobs that leave the recent catalog lose their cached start time.** With the registered E463 job absent from a fresh catalog but present in cache with an old `lastRunTime`, the pinned synthetic catalog row contains only `ref`. The mocked status response overwrote its cache and dropped `lastRunTime`, leading to “active run start time unavailable” on both pinned jobs. Merge provider status fields into the prior cached record, or carry `lastRunTime` from the old entry when querying a synthetic pinned row.

3. **Deferred active checks are silent.** A mock containing 25 active kernels returned `status_checks_deferred=7` (the registered pinned jobs also enter the pending set), while `alerts()` emitted no warning. The status cap is sensible, but the persisted dashboard should expose a warning/count for deferred active checks so the owner knows the observation is incomplete.

4. **Successful terminal attempts discard resource evidence.** `local()` skips a terminal attempt with `exit_code == 0` before reading its recorded peak RAM/VRAM fields. A completed process can therefore exceed the 4.5 GB near-VRAM warning and disappear from the live GPU query before the checker sees it. Inspect retained peak fields for recent terminal attempts before skipping them, or persist threshold-crossing events independently.

These are monitor observability gaps; none requires automatic retry or cancellation. The mock reproductions are summarized above and can be repeated by passing equivalent provider/local dictionaries to `alerts()` and mocking `kernel_status()` for the pinned-catalog case.

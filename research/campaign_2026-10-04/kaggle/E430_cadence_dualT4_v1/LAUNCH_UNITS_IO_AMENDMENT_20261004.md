# E430 launch-unit and repeated-I/O amendment

This is a source-only accounting amendment. It does not edit or replace the frozen v1 bundle, upload a dataset, or authorize an E430 run.

## Keep the budgets in separate units

The E430 `source_inventory.json` field `quota_required_for_admission_gpu_hours: 23.34` and its source critique conflate two distinct budgets. The arithmetic 23.34 is a **campaign device-GPU-hour requirement**, not a Kaggle provider-quota requirement:

- Scheduled worker stages: 40 minutes of probes plus 7 hours of paired/solo fit bounds = 7.6667 kernel wall-hours.
- At two allocated T4s, that is 15.3333 device-GPU-hours. The original one-hour wall margin adds 2 device-GPU-hours, giving 17.3333 planned device-GPU-hours.
- Adding the separate 6 device-GPU-hour recovery reserve yields 23.3333 device-GPU-hours. The campaign gate is `cumulative_device_gpu_hours_used + 17.3333 + 6 <= 80`.

Kaggle provider quota is a separate typed API balance. At 2026-10-04 18:55:41Z, it was 27,848.308 used / 108,000 total / 80,151.692 seconds (22.2644 provider hours) remaining, with zero reserved. A dual-T4 W384 v3 observation from 17:47:18.394338Z to 18:02:27.799540Z saw provider usage rise 908.804 seconds during 909.405 seconds elapsed (ratio 0.99934). This is one observed interval, not a universal billing guarantee and not device utilization.

Admission must record two independent checks: (1) provider quota available for the newly frozen maximum kernel provider-time reservation, plus any provider-quota reserve root sets; and (2) the global 80 device-GPU-hour campaign ceiling, including the separate 6 device-hour recovery reserve. Never compare the 23.34 device-hour figure directly to provider quota. Re-read the typed provider balance at admission and reconcile its delta after termination.

## Repeated integrity reads invalidate the current one-hour overhead assumption

The E430 v1 source has six phase launches (two probe phases and two fit phases, with two specs in each paired phase). Its frozen specs declare 41,192,989,386 bytes of code-plus-input dependencies per spec. The current `kaggle_gpu_runner` verifies those dependencies before it spawns workers; across six distinct specs, this is 247,160,961,362 bytes of controller-side full-hash reads.

Each of the six probe/fit workers then calls the unchanged split loader and portable source verifier. The fold-0 portable registry covers 37,103,450,341 bytes per worker, or 222,620,702,046 additional bytes across six workers. Total planned integrity reads are about 469.8 GB, excluding archive extraction, input decoding, and model/optimizer reads. The worker timeouts include the child-side validation, but they do not bound the controller's pre-spawn `verify_files` pass.

In W384 v3, the provider log gap from `validating_inputs` to the first worker heartbeat was 3,440.275 seconds for two specs containing 84,386,156,018 declared code/input bytes: about 24.5 MB/s aggregate. Applying that rate to E430's controller hashes alone gives a planning scenario of 2.80 hours, before other startup work. This is an extrapolation, not a guaranteed throughput bound. W384 v3 workers also timed out before a first update with no GPU bytes observed; the exact per-file phase was not instrumented, so the portable-hash attribution remains the best-supported diagnosis rather than a direct measurement.

Therefore the old 1-hour source/input-verification margin is not supportable as a bound. At the W384 controller rate, 2.80 hours of controller hashing is already required; replacing the old 1-hour margin gives an approximate 10.47-hour kernel wall scenario (7.67 worker-stage hours + 2.80 controller-hash hours), not a safe upper bound. `phase_runner.py` has no whole-session deadline around repeated `kaggle_gpu_runner` startup/hash phases, and per-worker fit/probe timeouts begin after worker launch. E430 remains unadmitted until the CPU I/O diagnostic measures these phases and a separately enforced whole-kernel deadline and provider-quota reservation are frozen. Do not skip or reuse hashes from a receipt unless a separately reviewed exact dataset-version/path/stat/content binding proves it safe; the current v1 verifier is unchanged.

## Frozen identities used by this amendment

See `LAUNCH_UNITS_IO_AMENDMENT_20261004.json` for hashes of the v1 builder, runner, archive, manifest, source inventory, and the W384 timing/quota evidence. All calculations here are read-only derivations from those artifacts.

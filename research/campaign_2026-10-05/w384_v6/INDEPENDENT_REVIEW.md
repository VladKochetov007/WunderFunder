# Independent v6 review — offline source and bundle audit

**Decision: NOT READY for controller push.** Static science and bundle parity
checks pass, and the 8-GiB admission diagnostics are useful. Two operational
defects need resolution before launch: the one-worker admission fallback is
incompatible with the phase runner's two-device contract, and v6 drops v5's
25,180-second whole-kernel watchdog. No provider action was taken.

## Verified

- The four v6 spec JSONs were compared field-by-field with the staged v5 specs.
  The only differences are the intended phase-runner path/hash/size, resource-
  runner hash/size, aggregate code hash, and `limits.ram_bytes` changing from
  10 GiB to 8 GiB. For every spec, `argv`, all `inputs` and their checkpoint,
  data, target, mask and plan identities, artifacts, environment, experiment
  name/version/kind, CPU limit, timeout, GPU allocator limit, and GPU stop limit
  remain identical. The v6 standalone parity test independently asserts these
  invariants.
- The generated bootstrap's `EXPECTED_EXTERNAL_MOUNTS` exactly matches both
  v5's reviewed expected map and the v6 manifest: all 26 external mounts match.
  The manifest and dataset metadata are private. A bounded archive check
  verified the v6 SHA-256, 265 unique regular-file tar members, exact member-set
  closure, all non-external sizes and hashes, and semantic equality of all four
  archived specs to their staged JSON.
- Kernel metadata selects private `NvidiaTeslaT4`, GPU enabled, TPU disabled,
  and internet disabled. The bundle does not encode a provider wall timeout.
- The v6 worker runner keeps the v5 admission formula and resource checks. Its
  diff from the v5 runner only adds admitted/rejected `admission.json` receipts
  containing host and device inventory. On admission error it writes the
  rejection receipt before raising. The phase runner persists both successful
  and rejected phase inventory/admission data into
  `/kaggle/working/results/admission_receipts.json` and the final phase report.
- The host admission budget remains
  `min(0.8 * capacity, available - 0.2 * capacity)` and still rejects a single
  worker whose RAM reservation exceeds it. The 90%-of-capacity aggregate host
  stop is unchanged.
- Per-worker RAM is 8 GiB (8,589,934,592 bytes), above the v5 probe peaks of
  5,317,271,552 and 5,489,156,096 bytes. This remains a sampled RSS stop rather
  than a provider cgroup limit. The 250-ms sample interval, 11-GiB allocator
  setting, 12-GiB GPU operating target, 13-GiB GPU hard stop, and
  preexisting-GPU-memory guard are unchanged.
- Bounded check run:
  `WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=180 WNN_SLICE=wnn-campaign.slice scripts/limited .venv/bin/python research/campaign_2026-10-05/w384_v6/test_admission.py`
  returned `passed 11 resource-admission and rejection-receipt cases; guard unchanged`.
  The independent archive/mount check also passed under a 1-core, 2-GiB cap.

## Required fixes before push

### Admission can fall back to one worker but the experiment requires two

The unchanged `admit_specs()` can return concurrency 1 whenever one 8-GiB
reservation fits but two do not. The new tests explicitly exercise that case.
However, `phase_runner.launch_runner()` still waits for the runner to finish
the whole phase and then requires `summary.concurrency_admitted == 2`. The
worker loop schedules both queued specs one at a time onto device index 0 when
concurrency is 1: after the first finishes, there is no occupied device, so
`next(...)` chooses index 0 again.

As a result, a one-worker probe phase wastes both probes on one GPU and only
then fails the concurrency/distinct-device gate. More seriously, if probes
were admitted at concurrency 2 but fit-stage inventory later admits 1, the
runner can execute both full fits sequentially on GPU 0 before the phase runner
rejects the phase. The fixed fit-to-probe device mapping and paired score are
then never reached. This can spend close to both 12,600-second fit limits
without a usable comparison. A serialization fallback therefore is not safe
with the current two-T4 experiment contract.

Before launch, either fail closed on `concurrency < 2` **before spawning any
worker** in both probe and fit phases, or explicitly implement and review a
serial two-device schedule that assigns each width to its probed physical GPU,
updates the phase checks, and fits within the wall-time budget. In particular,
do not leave a post-run concurrency check as the only guard.

### The whole-job watchdog from v5 was removed

The v5 bootstrap starts an independent watchdog at 25,180 seconds and fails
closed if it cannot start; the reviewed provider timeout was 25,200 seconds.
The v6 generated `bundle/kernel/kernel.py` has no watchdog, deadline, or timeout
around its `subprocess.run`. Its kernel metadata has no timeout field, and the
offline build receipt contains no provider push timeout. The per-worker 12,600-
second limits do not replace a whole-kernel deadline that also covers
preflight, hashing, both probes, fit phases, and exports.

Restore a 25,180-second whole-job watchdog in the v6 bootstrap and have the
controller push with an explicit 25,200-second provider timeout. Verify both
constants in the final source/command receipt before any push. Private/offline
metadata is correct; the hard wall limit is currently missing.

## Source evidence

- v6 staged work: [build receipt](build_receipt.json), [bundle manifest](bundle/bundle-manifest.json),
  [kernel metadata](bundle/kernel/kernel-metadata.json), [phase runner](phase_runner.py),
  [resource runner](runtime/scripts/kaggle_gpu_runner.py), and
  [admission tests](test_admission.py).
- v5 comparison baseline: `research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4_transport/provider_output_v1/experiment/`
  and its reviewed transport bootstrap at
  `research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v5_transport_review/kernel_push/kernel.py`.
- v6 bundle archive SHA-256:
  `ece9798e46681a86f798be2fb135cb259786e6520ebbf4af74d062571ec5405d`.
- v5 whole-job watchdog: `timeout_seconds=25_180` in its bootstrap. The v6
  bootstrap currently contains no corresponding code.

## Resolution review — 2026-10-05 00:11 UTC

**Decision: READY for controller-owned private upload and independent live probe**, subject to a fresh provider inventory and an explicit 25,200-second provider timeout. No provider action was taken in this review. This resolves the two findings above; their original text is retained as the initial review record.

The exact final offline bundle reviewed has SHA-256
`db08107d0df5ee9194959bbba4862385189d1fc0c91324ecd28ed12e8bebdd6e`.
The build receipt and generated files agree: manifest
`f89a77743159357ba735bbccf6385f73e79537715331bb578e2b833a2b6b6c40`,
kernel bootstrap
`c60dd550411bc9efbe858849b1f12a5136867c5599974075c1f3fef7297b496e`,
phase runner
`4f19bd7629dee6c16f6b8b1c1249cbe439c0b46c34284be1393f23f4cf6af656`,
and worker runner
`01bb1846c6825b0af860f86607ef6ecd8acdc1b311c34ba16bab70c0bf328b9d`.
The receipt records `status=built_offline_not_uploaded`, private bundle, and
provider wall cap 25,200 seconds.

Resolution evidence:

- The runner now writes the inventory/admission rejection receipt and refuses
  `concurrency < 2` before any worker process starts. Its bounded test
  exercises a host that can fit one 8-GiB reservation but not two, then asserts
  both `InsufficientPairedCapacity` and zero spawned workers. The same test
  retains the existing host/GPU admission boundaries and exercises observable
  `POSIX_FADV_DONTNEED` partial failure without changing sampled available RAM.
- The independently run admission suite passed all 12 cases under
  `scripts/limited` at 1 CPU / 2 GiB. The repaired mount/archive fixture compiled
  and passed all 10 adversarial cases under the same cap.
- I independently checked the final archive SHA against receipt and manifest,
  265 unique regular archive members against the exact declared non-external
  allow-set, all 26 external mappings, and safe member paths. The final
  bootstrap is byte-for-byte the reviewed v5 bootstrap after substituting only
  the v6 bundle slug and archive SHA. It retains exact external-mount
  validation and `timeout_seconds=25_180`.
- The four-spec parity test passed: only the intended runner/resource-wrapper
  identities and the RAM reservation differ from staged v5. Kernel metadata
  remains private, GPU-enabled T4, TPU-disabled, internet-disabled. The
  scientific command, inputs, masks, checkpoints, and fit/probe budgets remain
  unchanged.

This is an offline readiness receipt, not evidence that a current Kaggle
session can admit two T4 workers or fit both 8-GiB reservations. Before upload,
the controller must refresh live inventory/quota and retain the provider's
explicit `--timeout 25200`. After upload, require the fixed one-batch probes to
complete on two distinct T4s before either fit begins; the job has not yet
passed that live gate.

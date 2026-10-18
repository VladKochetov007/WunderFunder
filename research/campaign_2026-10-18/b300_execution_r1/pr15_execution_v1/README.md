# PR15 execution snapshot and receipts

This packet records root-supplied remote evidence from the single authorized SSH session. The live snapshot is at 2026-10-18 18:20:04.728 UTC; the local update did not open SSH or run remote commands. The snapshot shows the first 16 PR15 fits admitted, with the eight earlier R01 fits preserved. It records 24 campaign-owned CUDA processes, exactly three per GPU, zero completed fits and no scores.

The PR15 queue contains 120 cells: four arms with 30 cells each. The first wave is seed 1001 on folds F0-F3 for all arms. Trainer logs report 2.24 million parameters, three epochs, the fixed epoch-2 checkpoint, no evaluation during fitting, and fold update totals 44,760 / 44,790 / 44,850 / 44,640 for F0-F3. These are execution observations, not model results.

Six strict-FP32 density profiles passed under frozen profile packet SHA256 `c18f1877b8579c03d2a0905534f5ecbd13ed56a54306ba66d63b47d892b665510`. Each had two warmup batches, eight timed batches, 100 finite optimizer attempts and completed updates, finite loss/gradient checks, and zero OOM events. On GPU 7, one added worker measured 4.02 sequences/s and two measured 3.26 + 3.14 (6.40 aggregate, +59.2%). On GPU 2, one measured 4.14 and two measured 3.07 + 3.08 (6.15 aggregate, +48.6%). These are added-worker throughput comparisons; the existing R01 rate was not directly measured.

Five NVSM samples in the live receipt show mean device utilization of 81.6%, 90.2%, 90.2%, 82.2%, 85.4%, 89.6%, 80.8%, and 86.6% on GPU indices 0-7 (85.8% overall). The earlier 17:58 sample was 41-46% across the devices. Device utilization is not process-attributed, and this brief sample does not establish a sustained rate. At the live snapshot, PR15 cgroup memory was 39.37 GiB current (39.64 GiB peak), R01 was 19.45 GiB current (19.72 GiB peak), and all recorded memory OOM counters were zero. Foreign vLLM workloads on GPUs 0, 2, and 4 remained preserved.

A separate fit preflight passed at 18:22:38.754 UTC for the 16 canonical training/prediction argv cells and linked to a completed profile/resource receipt. It explicitly started no GPU fit and produced no quality score. The separate focused PR15 CPU test log reports six passed in 3.98 seconds.

The forecast for the additional series is 46.91 hours under the controller's provisional 1.5 runtime margin and one hour of post-processing per job. It is a projection, not observed elapsed time; contention from R01 was not directly measured. No PR15 fit had completed and no score was available in the live snapshot.

The V1-V3 profile attempts remain part of the audit history: V1 used a seed outside the original matrix and was rejected before CUDA; V2 lacked fresh state and was rejected before updates; V3 included the canonical but profile-incompatible `--save-training-state` flag and was rejected before updates. V4 removed that flag only from the profile invocation; the full-fit recipe/argv remained unchanged and all six density profiles passed.

## Receipt files

| File | SHA256 | What it records |
|---|---|---|
| [`live_receipt.json`](live_receipt.json) | `242df284eb3233428b33ca96c465fce25c9defced6581b6b15c877baa90bf6d3` | 18:20:04.728 UTC live workers, NVSM samples, cgroup memory and OOM counters |
| [`density_pass.json`](density_pass.json) | `ebf4b734b5b928734483e8b4d9a6a52e660a60308161913c33895901d92e18db` | Six V4 profiles, update/finite checks, throughput and resource evidence |
| [`launch_receipt.json`](launch_receipt.json) | `973030b3f72c369b7c077cd47ce14c49c552162081728b89624b71312fbc88b4` | Frozen manifest and systemd admission for the 120-cell queue / first 16 fits |
| [`fit_preflight.json`](fit_preflight.json) | `67060f8bfc2fae038a3521fe24f9c95af0c7892b0a8a3cc8fc72ce8876437e6a` | Sixteen canonical argv checks and profile/resource binding; no fit or score |
| [`forecast.json`](forecast.json) | `5ef3bc9379847c563486b36f56dcff19b4f12622213e9f04dddd95be2e7867bd` | Provisional lane projection inputs |
| [`cpu_tests.log`](cpu_tests.log) | `298fe4532a1513af8f3811648fe33fb7fdb76c22415dc7239d2dcc4b8d9894ce` | Six focused CPU tests passed |

See the campaign's [live status](../LIVE_STATUS.json), [authority record](../AUTHORITY.json), and [execution overview](../README.md). All paths and hashes above refer to the local, root-provided receipt copies.

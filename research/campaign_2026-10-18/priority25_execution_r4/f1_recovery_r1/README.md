# F1 continuation admission recovery, 18 October 2026

At10:08UTC deep stage1 finished successfully. The original sequence then refused
wide continuation before starting a worker: CPU reservations were1 replay +0.1
controller +1 requested worker =2.1 cores, above the two-core campaign limit.
The failed attempt has `not_started/admission_failed`; no continuation output or
optimizer existed. The failure is an orchestration resource-accounting incident,
not a numerical or model result.

- [Rejected original attempt](REJECTED_ATTEMPT_V1.json) and
  [failed sequence](FAILED_SEQUENCE_V1.json) preserve the exact first failure.
- [Recovery plan](F1_RECOVERY_PLAN.json), SHA
  `f1aa897f0be902e114935f0ac9044b0616e2f3e3a889991651d028823938a111`,
  changes only the wide continuation attempt and its launch `--attempt` value
  from `ep25-v1` to `ep25-v2`. Trainer command, run name, outputs, original
  `--contract` and all other steps are unchanged. [Independent review](PLAN_REVIEW_V1.json)
  checked this exact two-field difference.
- [Fresh recovery admission](ADMISSION_V1.json) binds the retained failed attempt,
  absent new attempt/output, parent resources and new command. Reducing the
  existing submission replay to75% CPU leaves0.1 controller +1 worker +0.75
  replay =1.85 cores. The replay is not restarted; its full elapsed time is no
  longer a constant-quota benchmark. Earlier100% CPU timing panels remain valid.
- [Launch](LAUNCH_V1.json) records the new controller starting10:20:06UTC:
  `computer-vision-homework-r01-f1-recovery-v1.service`. Read
  `B/runtime/receipts/RANK01_F1_S1001_RECOVERY_SEQUENCE_V1.json`.
  The old failed controller must not be restarted or its receipts erased.

Both F1 stage1 fits are archived and independently verified;
the [10:43 UTC handoff](../receipts/F1_DEEP_STAGE1_HANDOFF_20261018T1043Z.json)
records the completed deep check and stopped support units. Use existing V2 helpers for stage1. Wide continuation needs
new separately reviewed recovery helper bindings, including original V2 stage1
verification identities; unchanged V2 terminal expectations name the failed v1
attempt. Do not change frozen source, original plan, or prior archives.

[Actual resumed optimizer progress](ACTUAL_PROGRESS_V1.json) observes wide epoch2 batch20/1493, finite loss/LR and a fresh running attempt after recovery. This is real batch progress, not a terminal checkpoint or parity result.

[Wide-terminal V3 helpers](checkpoints/README.md) now have [independent review](CHECKPOINT_REVIEW_V3.json) and [hash-verified remote staging](CHECKPOINT_STAGING_V3.json), with seven local tests passing. They have not verified an actual terminal checkpoint. Both stage1 and deep terminal use V2. The later epoch2 section below records the separate intermediate collector and its completed execution.


## Wide continuation epoch2 collection

The [epoch2 handoff](../receipts/F1_WIDE_CONTINUATION_EP2_20261018.json) records the first successful continuation archive and the live epoch3 worker. Producer counters are44,790 updates/4,479 batches,zero skips/nonfinite; copied files and persisted parent binding pass. This is metadata/byte-copy evidence, not a tensor or quality grade.

- [Collector preparation](intermediate/README.md), [source](intermediate/collect_f1_wide_epoch2.py), [seven focused tests](intermediate/test_collect_f1_wide_epoch2.py) and [frozen manifest](intermediate/PREPARATION_MANIFEST.json) bind the exact v2 attempt and original V2 stage1 parent. The manifest SHA is54be2e28cf42d8a545b07a3ea9d3f2b082dbbfe810d99339951fe86f916bb83f. Its historical preparation wording is superseded operationally by the actual execution below.
- [Independent review](intermediate/REVIEW_V1.json) corrected attempt-read and source-generation races before freeze. Its outer model-SHA acceptance condition passed before launch and against the archived model. [Staging](intermediate/STAGING_V1.json) records the exact remote files; V2/V3/plans/trainer remained unchanged.
- [Prelaunch failure](intermediate/PRELAUNCH_FAILURE_V1.json) preserves a root-only missing `exit_status` key assumption on a running attempt; no unit started. [Corrected launch](intermediate/LAUNCH_V2.json) uses the actual running schema, fresh1.35aggregate CPU admission,physical scratch and hidden CUDA.
- [Collection receipt](../receipts/RANK01_WIDE_F1_S1001_EP2_COLLECTION.json), [execution](intermediate/EXECUTION_V1.json) and [stop receipt](intermediate/STOP_V1.json) bind stable three-file copies,exact model SHA and stage1 ancestry. Collector exited0 in6.299038s under25%CPU/512MB/120s; the own exited unit was captured then stopped. A read-only capture's empty-terminal-cgroup assumption was diagnosed and retained in the execution receipt, without rerunning collection.

Do not repeat epoch2 collection. This helper covers epoch2 only; future intermediate generations need separately prepared support, while reviewed V3 already covers wide terminal. Terminal structural checks still inspect the actual tensor/state payload and both stage1 ancestors.

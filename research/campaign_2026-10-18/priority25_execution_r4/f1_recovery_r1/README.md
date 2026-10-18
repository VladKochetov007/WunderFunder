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

Both F1 stage1 fits are retained. Wide stage1 is already independently verified;
deep stage1 collection/verification remains pending behind the serialized CPU
submission replay. Use existing V2 helpers for stage1. Wide continuation needs
new separately reviewed recovery helper bindings, including original V2 stage1
verification identities; unchanged V2 terminal expectations name the failed v1
attempt. Do not change frozen source, original plan, or prior archives.

[Actual resumed optimizer progress](ACTUAL_PROGRESS_V1.json) observes wide epoch2 batch20/1493, finite loss/LR and a fresh running attempt after recovery. This is real batch progress, not a terminal checkpoint or parity result.

# R6 execution records

This folder preserves launch attempts, source freezes, metadata-only checks, and
maintenance observations for the R6 diagnostic. A terminal launch record is
operational evidence, not a gradient result. The [R6 root README](../README.md)
summarizes the current status and fixed scientific scope.

## Current attempt: V3

[V3 launch intent](probe_v3/LAUNCH_INTENT.json) pins the launcher and binds the
45-file preparation manifest
([SHA-256](../PREPARATION_MANIFEST.json):
`c291687c93689463133f5e4109e7adef116dc79bca8aec08b7ec30d29a28505d`).
[Launch stdout](probe_v3/LAUNCH_STDOUT.json) records unit
`computer-vision-homework-eda19-r6-v3.service`, invocation
`b4c2ac3dc5454fbb8c72180cd2b31fe3`, and the launch-time `running` status;
[launch stderr](probe_v3/LAUNCH_STDERR.txt) records the successful unit start.
[Terminal capture](probe_v3/terminal_capture/README.md), exact extracted
[RESULT](probe_v3/terminal_capture/RESULT.json), [SOURCE](probe_v3/terminal_capture/SOURCE.json),
[DONE](probe_v3/terminal_capture/DONE.json) and [stop receipt](probe_v3/terminal_capture/STOP_RECEIPT.json)
are now saved. The unit exited0at01:11:35UTC and was stopped only after
durable length/SHA validation and exact-invocation recheck. The
[independent scalar audit](../RESULT_AUDIT.json) passes. The output remains
the create-only `research_tmp/nf19-r6-v3` on B; no payload arrays were transferred.
The0.5CPU/1.4GBhigh/1.5GBmax/no-swap/600s caps held; see
[resource reconciliation](../RESOURCE_RECONCILIATION.json). An empty terminal
ControlGroup produced one unscoped root-cgroup capture; it is retained and
excluded from own-unit event inference. Correct parent/populated-F3 snapshots
show no new memory events.

The V3 startup correction is independently reviewed in
[`ROOT_EXEC_STARTUP_V3_REVIEW.json`](../ROOT_EXEC_STARTUP_V3_REVIEW.json); the
source and synthetic tests are reviewed in
[`IMPLEMENTATION_REVIEW_V3.json`](../IMPLEMENTATION_REVIEW_V3.json). The
[startup note](startup_correction_v3/README.md),
[inline launcher](startup_correction_v3/root_inline_v3.py), and
[admission body](startup_correction_v3/remote_admission_body_v3.py) explain
how a reader-sentinel import smoke is followed by `execv` into the clean CLI
process. This fixes startup and loader compatibility only; it does not change
the model, checkpoint, sample, gradient windows, objective, or caps.

## Historical attempts and preserved sources

**V1 startup failure.** The [launch receipts](probe_v1/LAUNCH_INTENT.json),
[reconciliation](probe_v1/LAUNCH_RECONCILIATION.json), and
[terminal capture README](probe_v1/terminal_capture/README.md) identify unit
`computer-vision-homework-eda19-r6-v1.service`. Its separate
[reader-denied import smoke](probe_v1/terminal_capture/ROOT_IMPORT_SMOKE.json)
passed without array, parquet, or checkpoint reader calls or CUDA
initialization. Root then invoked the CLI in the same interpreter; its
pre-import guard rejected the already-loaded `wnn` package. The captured
[failure](probe_v1/terminal_capture/FAILURE.json) occurred before sample-array
access or model/gradient work. V1 is an operational failure, not a result.

**V2 restricted-loader failure.** The [launch receipts](probe_v2/LAUNCH_INTENT.json)
and [terminal capture README](probe_v2/terminal_capture/README.md) identify
unit `computer-vision-homework-eda19-r6-v2.service`. Its
[capture summary](probe_v2/terminal_capture/CAPTURE_SUMMARY.json) records
`weights_only=True` rejecting the `TorchVersion` metadata global before model
construction. V2 had hashed and mapped the 24 selected sample NPYs before
checkpoint loading, but the operator did not transfer their array payloads.
There was no forward/backward pass or gradient window. The exact failure
record and raw capture remain in
[`REMOTE_CAPTURE_RAW.json`](probe_v2/terminal_capture/REMOTE_CAPTURE_RAW.json).

The exact V2
[preparation manifest](loader_freeze_v2/PREPARATION_MANIFEST.json),
[test receipt](loader_freeze_v2/TEST_RECEIPT.json),
[model loader](loader_freeze_v2/model_contract.py),
[probe test](loader_freeze_v2/test_model_probe.py), and
[execution source](loader_freeze_v2/probe_execution.py) are byte-preserved.
The V1 prelaunch freeze is described in its
[README](prelaunch_freeze_v1/README.md), with the saved source and manifest
hashes.

The V2 direct-CLI repair is documented in
[`startup_correction_v2/README.md`](startup_correction_v2/README.md), with its
[correction record](startup_correction_v2/CORRECTION.json), independent
[startup review](startup_correction_v2/ROOT_STARTUP_CORRECTION_REVIEW.json),
[inline source](startup_correction_v2/root_inline_v1.py), and
[admission body](startup_correction_v2/remote_admission_body_v2.py). V3
supersedes that launch path because the source manifest changed for the
restricted-loader fix; neither repair represents a scientific comparison.

## Metadata and host evidence

- [Checkpoint argument observation](CURRENT_MODEL_ARGS_OBSERVATION.json) and
  its [query stderr](CURRENT_MODEL_ARGS_OBSERVATION_STDERR.txt) preserve the
  pinned provenance metadata used by the narrow inactive-`aux_weight=0.5`
  regression test. This is metadata only, not checkpoint deserialization.
- [Static pickle-global observation](CURRENT_MODEL_PICKLE_GLOBALS.json) and
  its [query stderr](CURRENT_MODEL_PICKLE_GLOBALS_STDERR.txt) list four
  pickle globals and zero `STACK_GLOBAL` opcodes. The inspection did not
  unpickle the checkpoint or read tensor storage.
- [F3 health snapshot](F3_HEALTH_20261019T0035Z/README.md) is separate
  maintenance evidence captured at 00:33:11 UTC; it is not R6 model evidence.

The V3 result, final stop receipt, and independent result audit should be
indexed here only after the operator has captured and validated them. Until
then, do not infer completion or gradient findings from the launch record.

[Latest rare F3 health snapshot](F3_HEALTH_20261019T0115Z/README.md), actual01:18:37UTC, supersedes the earlier0035bucket for lifecycle freshness. Controller/deepworker running; recorded optimizer logep4batch1400/1488; parent/F3memoryevents unchanged. No checkpoint payload action or job mutation.

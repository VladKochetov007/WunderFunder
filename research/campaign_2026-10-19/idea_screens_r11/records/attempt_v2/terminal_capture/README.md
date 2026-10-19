# R11 V2 terminal capture

This directory preserves metadata for `computer-vision-homework-eda19-r11-v2.service`, invocation `da0ba17ed86f46b6812fe8b763d06a31`. The exact systemd properties were captured after the unit exited successfully and before the exact owned unit was stopped.

- [CAPTURE_RECEIPT.json](CAPTURE_RECEIPT.json) contains base64 raw transport bytes, file lengths and SHA-256 identities, the terminal unit and parent/campaign cgroup observations, and output inventory.
- [ROOT_CAPTURE_VALIDATION.json](ROOT_CAPTURE_VALIDATION.json) records `PASS_BEFORE_STOP`, complete source/result pair, 32 source pins, and 24 runtime modules.
- [STOP_RECEIPT.json](STOP_RECEIPT.json) records a successful exact stop at `2026-10-19T04:36:19.991005+00:00`; afterward the unit was not-found/inactive/dead.
- [REMOTE_SOURCE.json](REMOTE_SOURCE.json), [REMOTE_DONE.json](REMOTE_DONE.json), [REMOTE_ROOT_ADMISSION.json](REMOTE_ROOT_ADMISSION.json), [REMOTE_ROOT_IMPORT_SMOKE.json](REMOTE_ROOT_IMPORT_SMOKE.json), [REMOTE_ROOT_LAUNCH.json](REMOTE_ROOT_LAUNCH.json), and [REMOTE_RESULT.json](REMOTE_RESULT.json) preserve the selected source/launch/result metadata. [REMOTE_PROGRESS.jsonl](REMOTE_PROGRESS.jsonl) and [REMOTE_PROBE_LOG.txt](REMOTE_PROBE_LOG.txt) preserve the bounded progress/log capture.
- [Resource reconciliation](../RESOURCE_RECONCILIATION.json) and [metadata review](../RESOURCE_METADATA_REVIEW.json) distinguish unit MemoryPeak, parent slice memory, pre-final-evaluation cgroup current, and process RSS. The unit control-group property was empty at terminal, so own-cgroup data is `null`; no host cgroup-root fallback was used.

Systemd properties derive a unit wall duration of 71.249675 seconds from the monotonic start/exit timestamps and 35.389213 seconds of CPU usage from `CPUUsageNSec`. These are distinct from `DONE.json` reported elapsed wall/CPU values (61.675449318 seconds and 30.628683667 seconds); the records preserve both scopes without inferring why they differ.

The `FINAL_MODELS_AND_NORMALIZER.npz` was not transferred, opened, or independently rehashed. Only its output-inventory size and producer-reported SHA-256 are recorded. V1's separate failed startup capture remains preserved under [attempt_v1](../../attempt_v1/failure_capture/FAILURE_CAPTURE.json).

# NLL native exit observation — 21 October 2026, 01:40 UTC

One bounded read-only capture at 01:40:07.650773 UTC found the exact NLL invocation `c833c613fb92426981ab255a5faf4452` exited at 01:26:10 UTC. `computer-vision-homework-fg19-child-nll-fit-v1.service` reports `Result=success`, active/exited, `MainPID=0`, and systemd exit code/status `1/0`.

[Raw response](STDOUT.raw.json) is 79,903 B, SHA-256 `f3efe3369cfed72ef12b57a7e86290edd195867635ac93468427973a2b777d6e`; [transport](TRANSPORT.json) records SSH 0, persistence before parse, no retry, and empty stderr. The frozen [reader](REMOTE.py) is 12,963 B, SHA-256 `cffc39769f4d16212216816498f7d6d6dc83db152cebaf73b0820b123b88beaf`, unchanged from startup. [Admission](ADMISSION.json), [launch](LAUNCH.json), and [unit log](unit.log) are exact base64 joins from the raw response.

The log has six complete FIT and six GRAD_STATS records: 44,760 successful updates, 4,476 batches, zero AMP skips, and zero nonfinite gradient steps; evaluation was withheld. The run directory was inspected by stat only. The capture reads no tensor, optimizer, cache, dictionary, or dataset payload.

This establishes native exit evidence only. It is not tensor integrity acceptance or quality evidence, and this read did not stop the unit or inspect any other job. See the [canonical terminal capture](../terminal_native_capture_v1/README.md), [typed native evidence](../TERMINAL_EVIDENCE.json), [resource reconciliation](../NATIVE_RESOURCE_RECONCILIATION.json), and [metadata review](../terminal_native_capture_v1/METADATA_REVIEW.json).

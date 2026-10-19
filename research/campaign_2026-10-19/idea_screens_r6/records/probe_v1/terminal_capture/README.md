# R6 CPU diagnostic terminal capture

Raw terminal metadata for `25bb02e9cd6e4f3780a4e5e978e5401e` of `computer-vision-homework-eda19-r6-v1.service` was observed at 2026-10-19T00:56:34.050135+00:00. The unit launched at 2026-10-19T00:55:05.294246Z and ended with `ActiveState=failed`, `SubState=failed`, `Result=exit-code`, `ExecMainStatus=1` at `Mon 2026-10-19 00:55:10 UTC`. Requested CPU/memory/swap/runtime cap properties and exact invocation are recorded and validated in `CAPTURE_SUMMARY.json`.

`REMOTE_CAPTURE_RAW.json` preserves the exact remote envelope; every embedded raw payload has a locally checked byte length and SHA-256. Core receipt bytes, when present, are also saved individually. The capture includes the failure/progress/root receipts, full unit status/ExecStart and journal tail, parent and populated campaign cgroup/event state, F3 unit state, and stat-only output inventory. No model, gradient, NPY/NPZ, checkpoint, or array contents were read or transferred.

This preserves an operational failure for diagnosis; it is not a scientific result and the job was not retried. After exact-unit validation and durable capture, `systemctl --user stop` returned 0 for this already terminal unit; the post-stop systemd state remains `ActiveState=failed`, `SubState=failed`, with the same invocation and exit status.

## Exact-unit stop

After validating the durable terminal capture and rechecking the exact invocation, failed terminal state, and requested caps, only this owned exited unit was stopped. The raw systemctl precheck/stop/postcheck bytes are preserved in `REMOTE_STOP_RAW.json`; locally validated receipt summary is `STOP_RECEIPT.json`. No retry or other unit action was taken.

# R6v2 CPU diagnostic terminal capture

This metadata-only capture belongs to unit `computer-vision-homework-eda19-r6-v2.service`, invocation `0d5d1222a4e44816bfc788bdc9bd0ab4`, launched at `2026-10-19T00:59:50.948079Z`. The exact terminal status, cap properties, command/journal bytes, source-root receipts, output receipt bytes, cgroup events, and stat-only output inventory are preserved in `REMOTE_CAPTURE_RAW.json`; all embedded raw payload lengths and SHA-256 values were locally validated before stop eligibility.

R6v2 intentionally omits an in-process `ROOT_IMPORT_SMOKE` receipt and reuses the durable R6v1 PASS; that absence is expected. No arrays, gradients, model/checkpoint data, or NPZ payloads were read or transferred.

The capture records a terminal failure rather than a scientific result. No retry or restart was performed.

## Exact-unit stop

After the terminal capture was durably saved and locally validated, the exact invocation and failed terminal state/caps were rechecked. `systemctl --user stop` returned 0 for this already-exited unit; the postcheck retained the same invocation and terminal failure state. Raw precheck/stop/postcheck bytes are preserved in `REMOTE_STOP_RAW.json`; `STOP_RECEIPT.json` summarizes local validation. No retry, restart, or other unit action was performed.

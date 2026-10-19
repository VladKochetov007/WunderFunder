# R6v3 CPU diagnostic terminal capture

This capture records unit `computer-vision-homework-eda19-r6-v3.service`, invocation `b4c2ac3dc5454fbb8c72180cd2b31fe3`, launched at `2026-10-19T01:10:54.694420Z`. It preserves exact small receipt bytes (`START`, `SOURCE`, `RESULT`, `DONE`, `FAILURE`, `PROGRESS`, root receipts), unit status/ExecStart/journal bytes, cgroup snapshots, and stat-only output inventory in `REMOTE_CAPTURE_RAW.json`. All embedded raw payload lengths and SHA-256 values were validated locally before stop eligibility.

No array, gradient, checkpoint, or model payload was read or transferred. This is an operational terminal capture only; scientific acceptance is separate.

## Extracted receipts and memory scope note

Exact small output/root receipt bytes have been extracted from the validated raw envelope into this directory; `EXTRACTED_PAYLOADS.json` lists their lengths and SHA-256 values. At terminal observation, systemd `ControlGroup` was empty, so the raw `own_cgroup` object resolved to `/sys/fs/cgroup` and is unscoped; it is preserved as a capture artifact but is not R6-specific evidence. Do not infer R6 cgroup events from that object. The systemd unit `MemoryPeak`, probe progress RSS/cgroup-current metrics, and campaign/F3 cgroup snapshots are separate scopes.

## Exact-unit stop and cgroup scope

After the terminal capture and extracted raw receipt bytes were durably saved and locally verified, the exact invocation, successful exited state, and cap properties were rechecked. `systemctl --user stop` returned 0 for this exact unit; postcheck is preserved in `REMOTE_STOP_RAW.json` and summarized in `STOP_RECEIPT.json`. At terminal observation systemd `ControlGroup` was empty; the raw `own_cgroup` path therefore resolved to `/sys/fs/cgroup` and is an invalid unscoped object, not R6-specific evidence. No unit events are inferred from it. Systemd `MemoryPeak`, probe progress RSS/current metrics, and campaign/F3 cgroup snapshots are separate scopes.

# R7v1 terminal capture

This metadata-only capture records `computer-vision-homework-eda19-r7-v1.service`, invocation `a62a2814610a4f69990635856d83682e`. Raw command/status and receipt bytes are preserved in `REMOTE_CAPTURE_RAW.json`; exact small receipts and source `probe.log` tail are extracted locally and listed in `CAPTURE_SUMMARY.json` with byte lengths and producer SHA-256 values.

`FINAL_MODELS_AND_NORMALIZER.npz` is stat-only; its payload was not opened or transferred. `ROOT_STARTUP_REVIEW` remains local and was not read remotely. No array, model, or gradient payloads were transferred.

Exact output/root receipt files have been extracted and verified; `EXTRACTED_PAYLOADS.json` lists their byte lengths and SHA-256 values. `SOURCE_probe.log.tail` preserves the observed empty `support-source/eda19-r7-v1/probe.log` (zero bytes). The NPZ producer SHA and size are from DONE.json, while the NPZ itself was stat-only.

## Exact-unit stop

After durable capture and local raw-payload/extracted-receipt validation, the exact invocation, successful exited state, and caps were rechecked. `systemctl --user stop` returned 0 for this exact R7 unit; precheck, stop and postcheck bytes are in `REMOTE_STOP_RAW.json`, summarized in `STOP_RECEIPT.json`. No other unit action or retry occurred.

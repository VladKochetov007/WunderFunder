# R5 probe terminal capture

Captured at 2026-10-19T00:21:33.314117+00:00 from `/home/grinchenkoeg/computer-vision-homework-20261018`. This is a raw metadata/receipt capture for exact unit `computer-vision-homework-eda19-r5-v1.service`, invocation `0c56cff45d2044168fb6ed9d310b0a3a`.

Validated terminal state: `SubState=exited`, `Result=success`, `ExecMainStatus=0`. Verified caps: CPU 0.5 (`500ms`), MemoryHigh 1.4 GB, MemoryMax 1.5 GB, swap 0, runtime 600 s. Launch was 2026-10-19T00:17:16.352953Z; exit timestamp was `Mon 2026-10-19 00:18:47 UTC`; elapsed wall time was 90.647047 seconds. Exact systemd commands, logs, cgroup values/events, and raw remote receipt bytes are preserved in `REMOTE_CAPTURE_RAW.json`; locally decoded metadata receipts are byte-identical files with verified transport lengths/SHA-256 in `CAPTURE_SUMMARY.json`.

The output inventory is stat-only. No arrays, model archives, coefficients, NPZ/NPY payloads, or checkpoints were read or transferred. Cgroup observations include the campaign parent and populated children, with F3 service properties when present; no F3 receipt, log, optimizer state, or checkpoint was read.

The first local capture helper had a Python syntax error and exited before starting SSH; its exact diagnostic is preserved in `CAPTURE_ATTEMPT_01_ERROR.raw`. It did not issue any remote action.

This is operational evidence only; it does not independently accept scientific results.

## Exact-unit stop

After the raw terminal envelope and decoded receipts were durably written and locally SHA/length-validated, `computer-vision-homework-eda19-r5-v1.service` was rechecked against invocation `0c56cff45d2044168fb6ed9d310b0a3a` and the exact terminal/cap predicates before the stop. The exact-unit stop receipt is in `REMOTE_STOP_RAW.json` (SHA-256 4544c7abcc13ad5b9e019335d4fc9772ee1c08ec8d328db69797c6b5f183ed7a); post-stop state is `inactive/dead`; `post_stop_terminal=True`. Parent and current F3 cgroup state/events after stop are recorded in this receipt. No other unit was stopped.

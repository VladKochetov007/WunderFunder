E419 v3 provider run failed before either QAT scientific fit. Bundle identity was `d2d8fe047d76f85b9dd5931d36b8a9daac6b23adfe197e76bb48c881328b3549` (code aggregate `36acf3c456209e195e5c8ba8d813ae436fe63adeb0a1ee1eb9b3ca6425db2796`). Kernel version 1 ran privately on two Tesla T4s; the provider marked it ERROR after about 6,818 s.

Both workers admitted on T4s and passed the CUDA preflight (55 tests, 0 failures, 2 skips), then passed their one-batch QAT production-shape probes. Each completed its matched one-epoch FP32 control (14,920 completed updates, zero AMP skips) and exported a full-valid prediction. The next control-score step failed for both arms: the packaged v3 control-screen JSON points to `c20261004_e419_fp32_control_*_t4v2`, but the produced prediction and sidecar are named `..._t4v3`. `screen_score.py` raised `FileNotFoundError` for the stale t4v2 sidecar.

The runner stopped at that failure. No parent PTQ export, same-control PTQ export, fixed-epoch QAT fit, QAT prediction, or score ran. The FP32 predictions/checkpoints are retained as partial, prespecified controls; they have not been rescored or compared here and are not a scientific QAT result.

Resource observations: recurrent-head worker (GPU index 0, UUID `GPU-af02b659-34e2-556f-a00a-be0de75ca3ef`) peaked at 8,234,680,320 bytes host RSS and 2,081,423,360 bytes GPU use. Recurrent worker (GPU index 1, UUID `GPU-87c2b8d4-bf25-3689-8df5-4175565bdfdd`) peaked at 8,192,425,984 bytes host RSS and 2,081,423,360 bytes GPU use. Both were below the admitted RAM and GPU stop limits.

Exact worker outputs, provider logs, step reports, preflight XML, artifact hashes, and both full-valid FP32 prediction arrays are preserved in `failure_20261004T1022Z/`. No retry or new identity was launched.

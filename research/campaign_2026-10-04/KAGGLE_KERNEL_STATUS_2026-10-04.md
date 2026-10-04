# Kaggle campaign kernel status — checked 2026-10-04 12:33 UTC

**3 complete, 3 error, 1 queued, 1 running.** Quota poll at12:33Z:4.96/30 GPU-hours used (25.04 remaining) and0/20 TPU-hours used. Refresh reported for2026-10-10 00:00Z.

| Kernel | Status | Outcome |
|---|---|---|
| `wnn-e413-volume-20261004-v1-runner` | COMPLETE | Both dual-T4 workers completed 4 epochs, validation predictions, and scores. Volume-history WP0.681592; current-only WP0.680630. Paired delta+0.0009618, CI[-0.0000800,+0.0020597], one exposed fold/seed. Peak6.91GiB VRAM and9.22/9.24GB host RAM; no OOM. |
| `wnn-e419-qat-dualt4-20261004-v2-runner` | ERROR | Preflight failed because portable bundle omitted sentinel_prepare dependency. No model fit. |
| `wnn-e419-qat-dualt4-20261004-v3-runner` | ERROR | Matched FP32 controls and exports ran, then the score step used stale t4v2 names. No PTQ or QAT fit. Peak2.08GiB VRAM and8.19/8.23GB host RAM; no OOM. |
| `wnn-e419-qat-dualt4-20261004-v4-runner` | RUNNING | Launched11:36Z after immutable bundle/source/screen/hash checks and CPU preflight (53 passed,2 skipped,0 failed). Dual-T4 GPU-hour cap8.5; per-device allocator11GiB and monitored stop13GiB. As of12:33Z provider logs/heartbeats, worker outputs, and memory peaks remain unavailable; no scientific result or OOM conclusion yet. Both T4s are occupied by this kernel. |
| `wnn-tpu-probe-20261004` | QUEUED | Requests `TpuV5E8` with TPU enabled. Still queued at12:33Z after about11 hours; no startup logs, device/HBM inventory, or allocation reason. TPU quota remains20/20h. |
| `wnn-t4-trainer-20261004-runner` | COMPLETE | Two-T4 CUDA training compatibility smoke passed; no campaign score experiment. |
| `wnn-t4-compat-20261004-runner` | ERROR | Bundle tarball path error; no scientific fit. |
| `wnn-t4-compat-20261004-v2-runner` | COMPLETE | Two-T4 compatibility probe passed; no campaign score experiment. |

E413 is the only scientifically completed Kaggle result to date. V4's 8.5-device-hour cap leaves the planned9-hour width budget and at least6 hours for recovery/reproduction inside the current quota; provider charge can lag, so reconcile it before adding remote work. Provider status is separate from scientific completion. The TPU probe payload only verifies framework availability, accelerator inventory, a tiny JAX GRU update, and a hidden-16 Torch/XLA model update; it is not yet a full training run. Full inventory snapshot: `checks/kaggle_kernel_inventory_20261004T1138Z.json`.

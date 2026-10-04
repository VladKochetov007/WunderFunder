# Kaggle campaign kernel status — checked 2026-10-04 11:09:25 UTC

**3 complete, 3 error, 1 queued; none running.** Latest quota poll at11:09:16Z: 26/30 GPU-hours and20/20 TPU-hours remain; quota refresh10-10 00:00Z.

| Kernel | Status | Outcome |
|---|---|---|
| `wnn-e413-volume-20261004-v1-runner` | COMPLETE | Both dual-T4 workers completed 4 epochs, validation predictions, and scores. Volume-history WP0.681592; current-only WP0.680630. Local paired delta+0.0009618, CI[-0.0000800,+0.0020597]. |
| `wnn-e419-qat-dualt4-20261004-v2-runner` | ERROR | Preflight failed because portable bundle omitted sentinel_prepare dependency. No model fit. |
| `wnn-e419-qat-dualt4-20261004-v3-runner` | ERROR | Two workers passed preflight and QAT shape probe, completed matched FP32 controls and exports. Score step used stale t4v2 names. No PTQ or QAT fit. |
| `wnn-tpu-probe-20261004` | QUEUED | No TPU worker or queue reason exposed. |
| `wnn-t4-trainer-20261004-runner` | COMPLETE | Two-T4 CUDA training compatibility smoke passed; no campaign score experiment. |
| `wnn-t4-compat-20261004-runner` | ERROR | Bundle tarball path error; no scientific fit. |
| `wnn-t4-compat-20261004-v2-runner` | COMPLETE | Two-T4 compatibility probe passed; no campaign score experiment. |

E413 is the sole Kaggle scientific result. It finished the paired four-epoch T4 fits, predictions and score runs:6.91GB VRAM and9.22/9.24GB host RAM peaks, no OOM. E419v3 also used both physical T4 workers, peaking2.08GB VRAM and8.19/8.23GB host RAM, then failed at a stale screen filename. It completed no QAT or PTQ fit. The TPU remains queued without an assigned device.

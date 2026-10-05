# E471 CUDA parity failure before training

The first guarded real-data CUDA probe failed its alpha-zero output parity assertion against E470 at the original tolerances (atol 2e-5 / rtol 2e-4). Of 2,048 output values, 459 mismatched; maximum absolute difference was 0.0001212358475. This occurred before the six optimizer updates. No full scientific fit was admitted.

The worker exited 1 after 6.6404 guarded GPU-job seconds; teardown was verified. Peak observed GPU was 308,281,344 bytes and worker RAM 1,012,953,088 bytes. Worker OOM counters were zero and parent historical OOM counters did not increase. The scientific failure and raw traceback remain in the attempt/log; the source and tolerances are unchanged.

CPU parity had passed, so the next diagnostic is the precision policy of CUDA/cuDNN versus the scripted affine scan. This is a hypothesis, not a confirmed cause. A separate diagnostic may compare default and common full-FP32 policies, record all discrepancies, and enforce the original tolerances. Do not widen tolerances or report this as an activation failure. Any subsequent fitting protocol needs a fresh identity and independent review, with its precision policy explicitly fixed for all arms.

Receipt: `research/campaign_2026-10-04/runner/attempts/e471-recurrent-real-gpu-probe-v1.json`; log has the same attempt basename under `runner/logs/`. The failed job remains quarantined. A distinct diagnostic is not a retry of this job.

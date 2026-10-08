# GPU loss-pair R2 terminal failure diagnosis

Kernel `vladkochetov07/wnn-loss-pair-gpu-20261009-v2`, version 1, reached terminal `KernelWorkerStatus.ERROR`. The root terminal collector receipt at `cloud_delivery/GPU_R2_TERMINAL_OPERATOR_RECEIPT_R1.json` was recorded at 2026-10-08T23:49:17Z, reports complete output/log collection, and has SHA-256 `9049b7957db2495caac45b439ce6072e17f287bf3c5bca9f258e94446060e003`.

The worker booted at 23:25:10Z, verified fold-0 assets and the mounted parsed-cache producer at 23:25:34Z, and passed the pre-fit resource gate at 23:26:49Z. It inventoried two Tesla T4 devices (compute capability 7.5). The cgroup sample at the gate had 0.832 GB current usage, 32.17 GB available memory, and no disk or memory guard breach. The disposable device-shape profile child then returned code 2 in 6.417 seconds. The worker stopped at 82.131 seconds, before either scientific fit.

The exact child error in `terminal_evidence_r1/outputs/out/logs/device_shape_update_gate.log` is:

```text
train_rnn.py: error: profiling/parity early exits cannot save epoch continuation state
```

R2's shared training-argument builder adds `--save-training-state` to the disposable profile command, which appends `--profile-steps 2 --profile-warmup 0`. The trainer explicitly rejects that combination. This is a startup/argument defect, not a model result or a device-capacity result. No optimizer update, scientific fit, checkpoint, or prediction was produced.

The failure occurred after real input verification. The runtime cache receipt binds manifest `7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b`, cache key `949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6`, split `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`, and membership `dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd`. Its one-pass validation read 7.785 GB and recorded 7.784 GB of physical read bytes; the mmap sample copied 54.84 MB. The fold-0 asset receipt verified all four registry-pinned files, hashing 2,247,722,202 bytes in 23.229 seconds, with read-only symlinks into the private mounted dataset.

The identified R3 repair is one argument-scope change: omit `--save-training-state` only from the short disposable device-profile argv. Keep it on both actual WMSE and Huber fit commands. R3 must exercise the actual trainer parser against that exact profile argv before admission and repeat the short real-T4 gate before either fit. R2 remains frozen as failed evidence.

Small actual worker files are preserved under `terminal_evidence_r1/outputs/`; the full provider output set remains in the root collector's original directory `cloud_delivery/gpu_loss_pair_r2_terminal/attempt_001/outputs/` and is hash-listed in the root receipt. The copied files are selected diagnostic evidence, not a second claim of complete collection.

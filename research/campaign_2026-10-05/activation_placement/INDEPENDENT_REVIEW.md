# Independent review: E474 activation placement

**Decision: GO for the sealed actual-shape probe only. This is not admission for the twelve full fits.** I rechecked the fixed repository-root binding, ran the probe's clean `--help` invocation, and verified that CPU preflight v5 binds the final source hashes. I did not launch CUDA work.

The trainer preserves the `RNNModel(42, hidden=64, layers=2, dropout=0, n_out=2)` parameter structure (61,058 parameters) and changes only parameter-free activation calls at the embedding and/or hidden head. Per seed, all six models clone one full baseline state dictionary. The training path loads the fixed 24 fit groups, excludes test groups, uses the unchanged E470 loss and `need_prediction` mask, shuffles the same fit order across arms, resets state at each B4 batch, carries and detaches state across 79 contiguous 256-row chunks, and performs 474 updates. Formula choices, AdamW settings, and common IEEE FP32 policy match the preregistration.

The final trainer records the true shuffled fit-group order and fixed-step telemetry at updates 0, 46, 236, and 473. Telemetry is computed on actual fit minibatches: element-weighted pre-clip gradient RMS and post-update parameter-delta RMS for the unchanged input, GRU, and head families. The fixed probe now binds `ROOT=HERE.parents[2]`, resets peak-memory counters, checks exact baseline output/state parity against `RNNModel`, verifies finite forward/state/loss/gradients/updated parameters for all twelve B4×256 arms, and requires that every optimizer step changes parameters. It reads fit-role data only. Its finite one-update result is a hardware/shape gate, not evidence about trained scores.

I ran the final CPU invariant suite under `scripts/limited` with one CPU and a 2 GiB memory cap; it passed. The controller-owned `preflight_cpu_v5/fit_manifest.json` contains all twelve fit-only one-update receipts, with 61,058 parameters, finite losses and changed parameters, and the expected fixed-step telemetry. The CPU preflight does not replace the actual CUDA shape probe.

Source and preflight identities verified:

- `fit_activation_placement.py`: `40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068`
- `gpu_probe.py`: `3b96ddfc05c80bb4aa3585a9957145a9b6927613e90ab23270bc80048b8c32c9`
- `tests.py`: `867c408de9a2c924b58e1c74a205c132a5f0d805ca3ebf0c68e3f9f64c1d50ce`
- `PREREG.md`: `75d9b30d6bc229f17722c6548d6a153508315404282c7a4493ffad7c3c93de9d`
- `preflight_cpu_v5/fit_manifest.json` binds the same three source hashes and the fixed cohort/dataset identities.

Keep all twelve predeclared arms and both seeds. Results concern an exposed 16-component panel, so any later contrasts are conditional, exploratory evidence; they cannot select a deployment model or establish a public forecast. The live GPU probe must still be hashed into its sealed job before execution, and a passing probe only permits the root controller to consider the full matrix.

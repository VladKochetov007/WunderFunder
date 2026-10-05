# E484 independent source review

## Disposition

**GO for the separately guarded actual-shape CUDA probe only.** This review does not authorize full fits or evaluation scoring. Root must admit the probe under its resource controls, and its actual parity, finite-update, and memory receipts must pass before any full matrix.

## Reviewed identities

- `fit_context_mixture.py`: `03d9bf45e51767499fec6f569e7af9eb46882315fe82011068a2f28f1df90bce`
- `tests.py`: `2581ec079ca5533ceb3c5f3e3d4a7973c188ecc957621d770189eae28d42d65e`
- `gpu_probe.py`: `0b75626aa776adef904a571384c85f1ead5852a9e738c8f73e635cd278c4e7a6`
- `preflight_cpu_v4/fit_manifest.json`: `b7ff0cc650f7c6b8059a140b1b55b104ad64a7add3af165965314718dd5723fd`

The final preflight manifest binds these exact three code hashes. It records 12 one-update fit-only cells, 61,058 parameters for GELU/fixed-mixture arms, 61,122 for the four learned gates, finite changed parameters, and `test_rows_loaded: false`.

## Findings

The earlier E477 cuDNN backward blocker is fixed in this version: both baseline and E477 reference GRUs use `.train()` during parity backward. Because the registered E474 model has zero dropout, this preserves the intended forward recipe while enabling cuDNN training-mode backward. The parity check now compares outputs, final state, input gradients, all shared parameter gradients, and one AdamW update after explicitly making the shared synthetic gradient norm exceed the clip threshold. It requires exact updated parameters. This closes the prior CPU-missed `eval()` backward failure and verifies the unchanged fixed-anchor optimizer path.

The implementation matches the registered E484 intervention. It retains the E474 model and replaces only the activation after the existing 64-wide readout hidden linear. All five mixture arms use the registered `g*GELU(h)+(1-g)*LeakyReLU_0.01(h)` formula; only learned arms add a 64-element gate coefficient. Context is the registered recurrent output, E474 projected embedding, or current readout hidden state. RMS normalization is per sample and time over 64 channels with epsilon `1e-6`, without centering. Gate parameters start at zero without consuming RNG, so all mixture arms share the same initial function and the GELU arm remains a distinct anchor.

The one-update trainer uses only E470 fit-role data and checks raw loaded input/target finiteness before each update. It uses the fixed seed-specific row-group order, B4×256 chunks, detached carried GRU state over each four-sequence batch, fit-only normalization, the E470 weighted objective, and the unchanged one-group E474 AdamW recipe. Parameter inventory and optimizer-group checks require every model parameter exactly once. Tests exercise nonzero gates, gate/context/parameter gradients, future-row causality, independence across batch elements, shared initialization, and baseline/fixed-mixture parity.

The GPU probe configures the worker before model/device work, captures active precision metadata within the precision context, performs the exact parity gate, and then exercises all 12 seed/arm B4×256 update cells while checking finite losses, finite parameters, actual parameter change, and the existing E474 clipping telemetry. The root controller must still enforce the physical GPU/RAM/time ceilings and verify teardown; this source review did not call CUDA.

## Verification

Under a 1-CPU, 2-GiB, no-swap systemd scope, independent checks passed:

```sh
.venv/bin/ruff check research/campaign_2026-10-05/activation_context_mixture/fit_context_mixture.py research/campaign_2026-10-05/activation_context_mixture/tests.py research/campaign_2026-10-05/activation_context_mixture/gpu_probe.py
.venv/bin/python -m py_compile research/campaign_2026-10-05/activation_context_mixture/fit_context_mixture.py research/campaign_2026-10-05/activation_context_mixture/tests.py research/campaign_2026-10-05/activation_context_mixture/gpu_probe.py
.venv/bin/python research/campaign_2026-10-05/activation_context_mixture/tests.py
```

All six synthetic tests passed. No source, GPU, full-fit, or held-out data was changed or run by this reviewer.

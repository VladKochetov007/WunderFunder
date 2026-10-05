# E481 implementation review

**Decision: GO for the sealed, bounded all-cell CUDA update probe only.** This review does not authorize or start full fits or scoring. I reviewed the preregistration, current actor, probe, CPU-preflight manifest, and synthetic tests without opening held-out data. I ran the CPU synthetic test suite; I did not run CUDA or start any fit.

## Reviewed identities

- Design: `E481_DESIGN.md`, SHA-256 `488d42609b7671d6af85eb050e910dc477706a607ade4ad16f9d6627cbfa9e93`.
- Actor: `fit_recurrent_curvature.py`, SHA-256 `3d97d2d07b116b3cf04b365c33e3d3caed2c96c35108bd28f3a6ff63b54d65cc`.
- Tests: `tests.py`, SHA-256 `ae5e277b1551d2b6c338aa8e83bb1fb7ffff0d29271d33f1f3498caa8e69ee2`.
- CUDA probe: `gpu_probe.py`, SHA-256 `e311e022232807c407d4fe56720c9e16d6474af04528064f05886487e9fd077a`.
- Fit-only CPU preflight: `preflight_cpu_v2/fit_manifest.json`, SHA-256 `83d99ffaf1b037ccf28d62dfc5d40b58a984dd7a2b49f4dea0cea9ae1f3c72b4`.
- Frozen parents: E470 `871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9`; E474 `40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068`.

## Findings

The code matches the three registered maps and changes only the reset-after GRU candidate value after `u=tanh(x_n + r*h_n)`. It retains PyTorch's r/z/n gate order and reset-after hidden candidate bias, inherited E478 two-layer H64 topology and 61,058 parameter layout, exact shared seed-specific initialization, externally fit-normalized inputs with identity model buffers, and the B4-by-256 sequence chunking. State resets at each batch, carries across all 79 chunks, and detaches at the registered truncated-BPTT boundary. The 24 fit-role rows alone are loaded for preflight/fits. Each full arm uses the preregistered 474 updates, fixed optimizer recipe and telemetry steps.

The Muon partition is guarded as an exact disjoint cover: only the four layer-specific GRU input/hidden weight matrices are assigned to Muon, with three gate blocks; all other parameters remain in AdamW. The shared E474 update helper applies one global norm clip, checks finite post-update parameters, and records the fixed telemetry. Probe and full-fit code run under the inherited full-IEEE FP32 policy, disable autocast, and preserve the existing Muon implementation.

The CPU tests cover scalar range, monotonicity, derivatives and endpoint behavior; baseline native-GRU output/state/input/parameter-gradient parity; both clipped optimizer update branches; shared initialization and ordering; chunk carry, reset and prefix causality; and finite nonzero gradients through every candidate map. The current preflight manifest binds the exact actor/test/probe hashes and records one actual finite, changed-parameter B4×256 fit-only update for all 18 cells, with `test_rows_loaded: false`. The bounded test command passed: `scripts/limited .venv/bin/python -m pytest -q research/campaign_2026-10-05/recurrent_activation_curvature/tests.py` (5 passed).

The prior blind critique's interaction-count correction remains applicable: 30 total registered contrasts comprise 18 shape pairs, 9 Muon-minus-AdamW comparisons, and 3 seed-wise paired curvature-by-optimizer interactions. The candidate comparison can distinguish sine from this matched-center cubic at this scale, but neither the sine-versus-cubic result nor a nonlinear-map gain isolates periodicity or establishes public-score improvement.

Admission is limited to a fresh probe receipt binding the exact source hashes above, with finite loss/output/state/gradients and changed parameters for every cell, full-IEEE metadata, measured peak memory, and successful teardown. Preserve any probe failure as-is. Full fits and grading require their own controller admission and receipts.

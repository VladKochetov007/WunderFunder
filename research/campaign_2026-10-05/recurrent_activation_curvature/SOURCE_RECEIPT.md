# E481 actor and bounded-preflight receipt

The E481 actor, synthetic checks, and callable all-cell CUDA probe are prepared
against `E481_DESIGN.md` and the blind `SPEC_CRITIC.md`. The actor implements
three fixed candidate formulas in one TorchScript scan: baseline `u`, sine
`u+0.1*sin(pi*u)`, and cubic `u+0.1*pi*u*(1-u*u)`. Candidate substitution only
changes the reset-after candidate value; gate equations, H64 two-layer
topology, initialization, fit order, and parameter count remain shared. All
arms have 61,058 trainable parameters. The optimizer helpers preserve AdamW
and the exact four-GRU-matrix/three-gate-block Muon partition.

The CPU preflight used `e470.load_cohort(include_test=False)`, exactly the 24
fit-role groups, and made one B4×256 update for each of 18 registered cells.
All updates had finite loss/parameters and changed parameters. The resulting
manifest explicitly reports `test_rows_loaded: false`; the task did not run a
full fit, CUDA probe, or scorer.

Synthetic pytest checks passed (5 tests, 2.40 seconds): scalar range,
monotonicity and endpoint/center derivatives for both nonlinear maps;
baseline output/state/input/parameter-gradient parity with native `nn.GRU`;
clipped AdamW and Muon optimizer-step parity at baseline; shared initialization
and seed order; chunk carry/reset and future-prefix causality; and finite
nonzero gradients through all scan shapes. Ruff format/check passed.

The probe is callable with CUDA but was not launched:

```bash
.venv/bin/python research/campaign_2026-10-05/recurrent_activation_curvature/gpu_probe.py \
  --device cuda \
  --output research/campaign_2026-10-05/recurrent_activation_curvature/gpu_probe_v1
```

Source identities:

- actor `fit_recurrent_curvature.py`: `3d97d2d07b116b3cf04b365c33e3d3caed2c96c35108bd28f3a6ff63b54d65cc`
- tests `tests.py`: `ae5e277b1551d2b6c338aa90e83bb1fb7ffff0d29271d33f1f3498caa8e69ee2`
- probe `gpu_probe.py`: `e311e022232807c407d4fe56720c9e16d6474af04528064f05886487e9fd077a`
- one-update fit-only CPU manifest: `83d99ffaf1b037ccf28d62dfc5d40b58a984dd7a2b49f4dea0cea9ae1f3c72b4`
- frozen E470 feature/training helper: `871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9`
- frozen E474 topology/optimizer helper: `40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068`
- dataset / cohort: `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782` / `068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2`

The CPU preflight command was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
ARROW_NUM_THREADS=1 POLARS_MAX_THREADS=1 WNN_THREADS=1 \
.venv/bin/python research/campaign_2026-10-05/recurrent_activation_curvature/fit_recurrent_curvature.py \
  --preflight-real --device cpu \
  --output research/campaign_2026-10-05/recurrent_activation_curvature/preflight_cpu_v2
```

This is implementation evidence only. It neither estimates comparative
performance nor authorizes full fits. E481 scoring remains for a separate
independent grader after source review and admission.

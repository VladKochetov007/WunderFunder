# E480 source and fit-only preflight receipt

Status: source-only candidate. The bounded actual-data CPU preflight passed; no CUDA probe, full fit, scorer, or provider action ran.

The actor requires both seeds explicitly (`--seed 271829 --seed 271830`) and has no mutable module-level seed default. It builds common E474 baseline `RNNModel` states by setting each explicit seed before baseline construction, then strictly clones that state to every activation/optimizer/schedule cell. The E480 test confirms seed 271829 is bitwise equal to E474's stored seed initialization and confirms the same construction is repeatable for seed 271830. Each seed uses one deterministic NumPy fit-row permutation, shared across all 12 cells for that seed.

The frozen E476 474-value LR traces are loaded from `activation_schedule/preflight_cpu_v5/lr_traces.json`; the actor regenerates and checks each trace's exact values and SHA-256 before fitting. The independent two-seed preflight receipt reports all 24 cells, exact cohort/data hashes, 61,058 parameters per cell, one finite parameter-changing update per cell, and one fixed four-group order shared within each seed. `exposed_test_groups_loaded` is false.

Bound source and preflight identities:

| Artifact | SHA-256 |
|---|---|
| Fit actor | `555a99927f94bd62b08b2f8af06b4a4ba5783d43d10e513377b954661377b4b0` |
| Tests | `511a688f40f782a4b4021d89ca0f1df6b7582155aac87a04fda1cdf627d8d1de` |
| Design | `3ad2ab9d5721b7e342b2549af81e528fab0a9fa1d2a9fe841d43d4fb978d8811` |
| CPU preflight manifest | `4c7eb346c5a85d93580d844c58c90c0cfb2ad016dcf94bae69aacd7989462aae` |
| E476 trainer reference | `5acc0efa8295338b39ffc0a0c9412484daec277967d8dc2e161affee1d1786e6` |
| E476 frozen LR traces | `046f38f67792af73450bbbb958403834d948495e4ba64b263a2a9a2ff935e4ff` |
| E474 placement source | `40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068` |
| E470 causal-feature/loss source | `871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9` |

Checks passed: `ruff format --check`, `ruff check`, `py_compile`, synthetic CPU invariants under one CPU/2 GiB, actual-data fit-only CPU preflight under one CPU/2 GiB, and clean actor `--help`. The CUDA preflight is callable through `--device cuda --preflight-only`, uses the same exact 24 fit-only B4×256 updates and records per-cell CUDA allocator peaks, but remains unrun and requires root resource admission.

The exact CPU command was:

```bash
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_SLICE=wnn-campaign.slice \
  scripts/limited .venv/bin/python \
  research/campaign_2026-10-05/activation_schedule_replication/fit_activation_schedule_replication.py \
  --device cpu --preflight-only --seed 271829 --seed 271830 \
  --output research/campaign_2026-10-05/activation_schedule_replication/preflight_cpu_v3
```

No scientific fit was started. Root-owned review must seal the exact actor, tests, design, immutable parent sources and data/cohort identities, and separately admit the actual-data CUDA probe before considering any full fits.

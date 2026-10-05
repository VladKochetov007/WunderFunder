# CPU checks — 2026-10-05

All commands below ran through `scripts/limited` with one CPU thread and a
2-GiB memory ceiling. No optimizer fit on real rows, held-out score calculation,
GPU allocation, provider launch, or shared-ledger change occurred.

- `ruff format` and `ruff check research/campaign_2026-10-05/joint_target_model`:
  clean.
- `pytest -q research/campaign_2026-10-05/joint_target_model/test_model.py`:
  6 passed, including the clipped-vs-raw nearest-atom outlier and nonfinite-step
  fail-closed tests.
- `fit.py --smoke --device cpu`: passed at batch 8192, state shape `[8192, 2,
  128]`, width-128 head; performed one real AdamW step for each of five arms
  using generated data and checked finite losses, gradients, and parameters.
- `e2e_fit.py ... --device cpu --preflight-only`: passed for fit groups
  `[0,1,2,3,6,8,10,11]`, 16,591 real fit rows, exact two-output parent callback,
  and parent checkpoint SHA `fa9f0353cf6cdd3c85fc56efc1e8a3f3dae0048fb1ccd995eeb7026afd269dc7`;
  no model updates and no evaluation access.
- `score.py ... --device cpu --preflight-only`: passed for all 468 frozen
  evaluation groups / 141 components; verified 808,863 selected rows and the
  official mask. This is provenance preflight only; no predictions were scored.

The CUDA smoke and fixed fits remain unrun and require independent review and
controller approval. CUDA entry points now call `configure_torch()` before
tensor creation, so `WNN_GPU_MEMORY_BYTES` controls the allocator; fit paths
fail closed on nonfinite loss/components, gradients, or post-step parameters.
The pooled clipped weighted-MSE value is recorded only as a training surrogate;
only the scorer's global weighted-Pearson calculation represents the official
metric.

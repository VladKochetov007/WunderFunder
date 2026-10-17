# R2 figures

- [Output policy audit, PNG](output_policies.png) and [SVG](output_policies.svg)
  show observed WP changes for three frozen arrays and conditional95% paired
  chain-bootstrap intervals. Nine causal policies are displayed; the full
  fourteen-policy matrix also includes clipped-input EMA controls. The oracle
  occupies a separate axis and is explicitly unavailable online.
- [Renderer](../theory/plot_output_audit.py) reads the immutable
  [score receipt](../prediction_audit/RESULTS.json), not model weights. The figure
  metadata binds that receipt's SHA-256. Matplotlib rendering and visual
  inspection are presentation checks, not new scientific comparisons.

Reproduce from the repository root with an isolated plotting environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-project --with matplotlib --with numpy python research/campaign_2026-10-17/research_reset_r2/theory/plot_output_audit.py research/campaign_2026-10-17/research_reset_r2/prediction_audit/RESULTS.json research/campaign_2026-10-17/research_reset_r2/plots/output_policies
```

Return to [R2](../README.md).

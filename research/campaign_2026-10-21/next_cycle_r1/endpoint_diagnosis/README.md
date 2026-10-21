# H512/TL3 endpoint diagnosis

This packet implements the frozen [endpoint diagnosis plan](../ENDPOINT_DIAGNOSIS_PLAN_R1.json) as a post hoc, exposed F0 analysis. `SOURCE_BINDING_R1.json` pins the exact score receipts, source files, target cache, prediction arrays, sidecars, and physical-disk output path. The script reuses the frozen F0 input loader and official metric, maps full-valid H512 predictions by `row_group`, then streams one evaluation sequence at a time.

Run from the repository root after the owner admits its one-core, 1.4 GB, 900-second CPU slot:

```sh
uv run python research/campaign_2026-10-21/next_cycle_r1/endpoint_diagnosis/diagnose_native_h512.py
```

The create-only JSON output reproduces the three fixed official scores, reports clipped label-magnitude-weighted moments and raw clipping/std summaries, the fixed per-target position-bin scores, the covariance/variance decomposition, H512 partial correlations conditional on TL3, and the specified differential parity residuals. It does not search coefficients or select a model. All inputs are hashed before and after scoring. The frozen plan records the exposure and claim limits: TL3 ancestry is unknown, F0 is exposed, and these descriptive diagnostics cannot establish a causal mechanism or public transfer.

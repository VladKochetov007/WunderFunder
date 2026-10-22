# Frozen F0 prediction geometry

This folder records a bounded offline comparison of five frozen F0 prediction
caches for campaign `C-ONLINE-FEATURES-20261018-R1`: the component-clean F0
parent, R2 full geometry, the fixed R2 `m0p50` inference alias, R3 moments-only,
and R3 dictionary-only. It measures where their clipped predictions and parent-relative
corrections overlap on the repeatedly exposed F0 panel. It does not make a
quality, deployment, or causal claim.

- [GEOMETRY.md](GEOMETRY.md) reports pooled and predeclared position-bin scores,
  panel coverage, weighted prediction geometry, and limits for a later
  independent MFT skeptic.
- [receipt.json](receipt.json) contains machine-readable scores, weighted
  covariances/correlations, fixed-parent projection diagnostics, bin counts,
  label-weight mass, and source hashes.
- [geometry_diagnostic.py](geometry_diagnostic.py) is the small read-only
  offline grader. It aligns only by source `row_group` IDs, uses the official
  metric helper, and checks pooled scores against the frozen independent grade.

Inputs are `cache/valid_targets.npz`, the five prediction caches and sidecars
named in the receipt, fold0 at
`research/campaign_2026-09-29/artifacts/splits/fold0.json`, and the previously
frozen grade at
`research/campaign_2026-10-18/geometry_variations_r3/geometry_r3_grade.json`.
F0 membership is 468 sequences / 141 components and remains exposed. The R2
scale alias is derived from the R2 checkpoint and is not an independent fit.
No TL3 member was added to this comparison. Its embedded description calls the
teacher sources all-data, while actual training, teacher, and initializer
ancestry remains unverified.

# Domain-stress evaluator review R1

This folder independently reviews commit `8940db0d` and the frozen `DOMAIN_PANELS_R1.json`. The three input-only panels use outer-fit medians and source-qualified complete components; their two component-frequency mixtures recompute pooled official WP through centered moments. The panel file and code hashes match, all 12 fold-by-panel outputs are available with at least 60 components in each evaluation stratum, and the three focused tests pass.

The reviewed commit's equality check on label mass/mean/M2 does not prove row identity; a distinct mask can preserve those moments and still be accepted. The current worktree patch makes the raw-moment helper private and adds `evaluate_domain_mixtures()`, which computes candidate and reference moments together from one supplied mask. I inspected this path and reran the focused tests; it closes the mismatch at the public API boundary. A grading receipt must still bind the exact wrapper hash plus row order, component IDs, source and official-mask identity. Preserve the frozen R1 panel and treat these as partial sensitivity evidence: only three of six planned families are present, and cutpoints use the explicitly restricted validation-summary source.

- [Independent review](INDEPENDENT_DOMAIN_STRESS_REVIEW_20261013.md) records method checks, the mask-alignment counterexample, and interpretation limits.
- [Machine-readable receipt](INDEPENDENT_DOMAIN_STRESS_REVIEW_20261013.json) records source and artifact hashes, checks, and disposition.

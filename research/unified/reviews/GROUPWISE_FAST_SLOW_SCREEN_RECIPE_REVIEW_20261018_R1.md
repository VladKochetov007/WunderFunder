# Groupwise fast/slow Kalman screen: static recipe review

**Disposition: pass for the bounded one-sequence profile only.** The independent reviewer inspected the recipe, implementation and imported metric/data helpers without running Python, tests, reading Parquet payloads or launching a worker. The full cross-fit fit was not admitted by this review; it remains gated on the exact profile and normal resource-controller admission.

## Findings and corrections

The first review found three concrete launch blockers: helper imports resolved the repository root one directory too high; the runner expected top-level data/sample/split fields while the recipe stored them under `selection`; and the fitter expected duplicated `model.input_widths` values that were defined under `features.arms`. These were corrected by restoring `parents[4]`, reading the nested selection fields, and deriving model widths from the single recipe definition. Ruff formatting/lint was rerun and the recipe's source hashes were refreshed. The reviewer rechecked the corrected files and confirmed that all declared hashes match.

## Static checks

- Feature rows and same-row targets align from row 99 onward; no target or scoring-mask data enters feature extraction or OOF prediction.
- Each component's labels are excluded from its assigned fold model; the grader reads held-out labels/masks after prediction artifacts are frozen and hashed.
- The feature states reset within each sequence, use the current input's pre-update innovation, and do not use a future row.
- Training clipping/weights and grading clipping/weights match their separately stated roles; the grader uses the official pooled `global_wp_clip2_v1` accumulator.
- The profile reads one complete fit-side sequence, computes the exact features and weighted-ridge shape, creates no OOF predictions and does not grade.

## Limits

This static pass is not a memory profile, execution receipt, result grade, serving integration check or quality claim. The 32 source components were already inspected as development data, so component cross-fitting does not make them a clean validation set. The existing source pin checks file size, modification time and the Parquet footer; it does not hash the full training payload.

## Provenance

Reviewed files: [experiment packet](../../campaign_2026-10-18/online_features_r1/group_fast_slow_r1/README.md), its [recipe](../../campaign_2026-10-18/online_features_r1/group_fast_slow_r1/recipe.json), and the imported feature, weighted-moment, data-reader and official metric sources listed in that recipe. The original review findings were retained in the handoff; this note records the corrected disposition.

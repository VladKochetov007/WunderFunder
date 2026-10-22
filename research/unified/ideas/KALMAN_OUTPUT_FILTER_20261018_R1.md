---
kind: experiment
status: completed_negative_for_frozen_output_filters
created_utc: 2026-10-18T04:16:53Z
updated_utc: 2026-10-18T04:33:42Z
hypothesis_ids: [H-KALMAN-OUTPUT-FILTER-20261018-R1]
experiment_ids: [E-KALMAN-OUTPUT-FILTER-20261018-R1]
evidence_paths:
  - ../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/README.md
  - ../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/recipe.json
  - ../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/result_r1.json
  - ../reviews/KALMAN_OUTPUT_FILTER_EVIDENCE_REVIEW_20261018_R1.md
---

# Causal Kalman filtering of model outputs

## Question or claim

Can causal filtering of the model's own prediction stream improve the endpoint while preserving the model's raw-input training distribution?

## Mechanism and method

The frozen CPU screen compares the unchanged parent predictions with a fixed steady-state local-level Kalman filter (`Q/R=0.25`) and a per-sequence/channel `Q/R` estimate from the first 99 model outputs. The initial 99 rows are unscored calibration/warmup; scored rows use only the prefix and current/past predictions. Both filters reset at sequence boundaries and read no labels or masks during prediction. Details and exact pins are in the [campaign packet](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/README.md).

## Evidence and interpretation

This frozen screen has completed. On 468 previously inspected F0 groups/141 components, raw parent scored 0.6812265 WP, the fixed filter 0.6755207, and the prefix-estimated filter 0.6647620. Differences from raw were −0.0057059 (95% component-bootstrap CI [−0.0065918, −0.0050088]) and −0.0164645 (CI [−0.0190768, −0.0144069]). Both lost in all four descriptive component partitions.

## Limits and validity

The comparison uses the repeatedly inspected F0 panel and a single fixed parent checkpoint. Component-bootstrap intervals condition on that endpoint and cannot establish clean CV, training-seed variation or public transfer. The filter can introduce lag; the adaptive noise estimate is a heuristic from model outputs, not observed target feedback. This result does not test learned Kalman input/hidden-state features, teacher quality, student transfer or true online learning.

## Decision and next step

The predeclared follow-up threshold was not met, so do not continue tuning these two output-filter recipes on exposed F0 or submit them. This does not close the larger retained-raw Kalman feature teacher/student direction in [draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17), nor broader online adaptation. No new B300 draft is warranted by this inference-only negative result.

## Source records

- [Recipe](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/recipe.json)
- [Generator](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/generate_predictions.py)
- [Grader](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/grade_output_filter.py)
- [Execution receipt](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/execution_receipt_r1.json)
- [Evidence review](../reviews/KALMAN_OUTPUT_FILTER_EVIDENCE_REVIEW_20261018_R1.md)

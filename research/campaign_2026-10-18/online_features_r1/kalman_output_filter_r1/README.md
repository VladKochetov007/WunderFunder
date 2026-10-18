---
kind: experiment
status: completed_negative_for_frozen_output_filters
created_utc: 2026-10-18T04:16:53Z
updated_utc: 2026-10-18T04:33:42Z
hypothesis_ids: [H-KALMAN-OUTPUT-FILTER-20261018-R1]
experiment_ids: [E-KALMAN-OUTPUT-FILTER-20261018-R1]
evidence_paths:
  - recipe.json
  - generate_predictions.py
  - grade_output_filter.py
  - preflight_r1.json
  - result_r1.json
  - execution_receipt_r1.json
  - ../../../../unified/reviews/KALMAN_OUTPUT_FILTER_EVIDENCE_REVIEW_20261018_R1.md
---

# Causal Kalman filtering of model outputs

## Question and status

Can a per-sequence causal local-level filter reduce noise in the frozen model's two output streams, without changing model weights or reading target labels at inference? The frozen screen completed. Both filters underperformed the raw parent on the exposed F0 panel; independent review confirmed the scoring and claim limits.

## Frozen method

Use the exact full-validation prediction cache for `warm_mechanism_f0_s3_c_r3`, whose checkpoint, prediction bytes, sidecar and prior fit-membership preflight are pinned in [recipe.json](recipe.json). Generate two deterministic arms:

1. A steady-state scalar local-level Kalman filter with fixed `Q/R = 0.25` for both predicted target channels.
2. The same filter with `Q/R` estimated separately per sequence/channel from the first 99 model outputs, using robust variances of first and second differences and the clipped method-of-moments equations in the recipe. The observation-variance estimate has a frozen `1e-12` floor, including for constant prefixes.

The state resets for every sequence. Rows 0–98 are an unscored calibration/warmup prefix: the adaptive gain uses those 99 predictions, and filtered prefix values are not evaluated. Every scored output from row 99 onward uses the calibrated prefix and predictions only through its current row. The filter never sees labels or scoring masks. The unchanged parent prediction is the control. No model fitting is involved.

Scoring uses the official `global_wp_clip2_v1` metric on 468 previously inspected F0 groups and 141 source-qualified components, excluding rows 0–98. The paired component bootstrap and four deterministic descriptive component partitions are conditional on fixed predictions; they do not provide clean-CV, seed, or public-transfer evidence.

## Result

| Endpoint | WP | Difference from raw parent | Paired component-bootstrap 95% CI |
|---|---:|---:|---:|
| Raw parent | 0.6812265 | — | — |
| Fixed `Q/R=0.25` | 0.6755207 | −0.0057059 | [−0.0065918, −0.0050088] |
| First-99 estimated `Q/R` | 0.6647620 | −0.0164645 | [−0.0190768, −0.0144069] |

Both filters lose in all four descriptive component partitions. The preregistered strong follow-up criterion was not met. The adaptive estimate has broad, frequently clipped gains (median Q/R 0.528; median gain 0.509; 10th–90th percentile Q/R spans 0.01–4.0), which is unstable evidence for the local-level noise model. These results narrow only these two ways of smoothing the model's output stream; they do not reject Kalman-derived inputs or hidden states, learned teacher features, student transfer, or online feedback methods.

## Source identity and pre-run correction

The F0 group, sequence and component order digests match the frozen split and the existing parent preflight. The initial recipe draft had two malformed 64-character digest pins. Its exact bytes are preserved in [recipe_r1_initial_bad_hash_pins.json](recipe_r1_initial_bad_hash_pins.json); both errors were caught using split/recipe metadata before reading prediction arrays, target arrays or masks. Static review also caught a wrong fold-helper reference and an omitted variance floor in the recipe. Both were corrected and independently rechecked before prediction generation or target/mask scoring.

## Code and next stages

- [Prediction generator](generate_predictions.py) verifies the pinned full-validation source and writes only filter predictions plus a label-blind receipt under physical scratch.
- [Grader](grade_output_filter.py) verifies prediction/output order and parent ancestry, writes [preflight_r1.json](preflight_r1.json) before target/mask numeric use, then applies the official metric and paired component bootstrap.
- [Execution receipt](execution_receipt_r1.json) records the one-group profile, full-cache identity, scope limits, sampled memory/CPU observations and unchanged parent OOM counters. The full 149,760,128-byte prediction cache and its per-group receipt remain in `/home/vlad/development/wunder/research_tmp/online_features_r1/kalman_output_filter_r1/`; exact generation wall time was not captured.
- [Independent evidence review](../../../../unified/reviews/KALMAN_OUTPUT_FILTER_EVIDENCE_REVIEW_20261018_R1.md) confirms score, folds, bootstrap and claim scope. It notes that the configured runtime cap is not a measured elapsed-time result.

This is a fixed-model serving treatment, not a large-scale training direction. It does not warrant a new B300 draft by itself. The separate retained-raw learned Kalman feature teacher/student study remains in [draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17).

## Source records

- [Frozen recipe](recipe.json)
- [Initial malformed recipe snapshot](recipe_r1_initial_bad_hash_pins.json)
- [Prediction generator](generate_predictions.py)
- [Grader](grade_output_filter.py)
- [Hypothesis ledger](../../../hypotheses.jsonl)
- [Experiment ledger](../../../experiments.jsonl)

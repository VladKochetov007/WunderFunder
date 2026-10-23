# Kalman output-filter evidence review R1

## Verdict

**PASS for internal scoring and claim consistency; no follow-up is warranted for these two fixed-filter recipes.** This is a static independent review of the frozen recipe, generator/grader source, preflight, prediction receipt and result. The reviewer did not rerun code or open parquet, target arrays or prediction arrays.

## Findings

- The independent source pass caught two pre-run defects: the grader imported the fold assignment from the wrong module, and the recipe omitted the generator's `1e-12` variance floor. Both were corrected, the fold helper seed/count were bound to the recipe, and the final source pass cleared the packet before output generation or target/mask scoring.
- The comparison matches the frozen recipe: raw parent versus fixed steady-state gain and first-99-prediction Q/R estimate, on 468 exposed F0 groups and 141 source-qualified components, scoring rows 99–19,999 under `global_wp_clip2_v1`.
- The result is internally consistent. Raw parent scores 0.6812265 WP; the fixed filter scores 0.6755207 (Δ −0.0057059; paired component-bootstrap 95% CI [−0.0065918, −0.0050088]); the prefix-moment filter scores 0.6647620 (Δ −0.0164645; CI [−0.0190768, −0.0144069]). Both treatments lose in all four descriptive component partitions.
- The adaptive filter uses only predictions from rows 0–98 to estimate its gain. Those rows are unscored; scored outputs from row 99 onward depend only on the prefix and current/past predictions. Prediction generation reads no target or mask values, and state resets per sequence.
- Preflight and result hashes bind the relevant recipe, generator, grader, official metric, component evaluation, prediction cache/receipt, parent endpoint and fit-membership receipt.

## Interpretation limits

The four partitions and component bootstrap are descriptive resampling of previously exposed F0 predictions and labels. They do not estimate fresh validation, fitting-seed, ancestry or public-transfer uncertainty, and the adaptive arm is not observed-label feedback learning. This result rejects the two specified output filters on this endpoint/panel; it does not reject Kalman-derived input features, latent-state features, teacher training, student transfer or broader online adaptation.

The recipe imposed a 900-second runtime cap, but the exact full-generation wall time was not captured in the scientific result artifacts. The separate [execution receipt](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/execution_receipt_r1.json) records the enforced cap, successful scope exit and measured memory/CPU observations without claiming an exact elapsed time.

## Source artifacts

- [Frozen recipe](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/recipe.json)
- [Preflight](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/preflight_r1.json)
- [Machine-readable result](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/result_r1.json)
- [Prediction receipt in physical scratch](../../campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/README.md)

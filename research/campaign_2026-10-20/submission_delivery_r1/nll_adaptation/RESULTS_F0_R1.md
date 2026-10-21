---
kind: result
status: inconclusive
created_utc: 2026-10-20T23:13:33.900524+00:00
updated_utc: 2026-10-20T23:13:33.900524+00:00
hypothesis_ids: [H-NLL-STRONG-GRU-20261020-R1]
experiment_ids: [E-OVERNIGHT-NLL-ADAPTATION-R1]
evidence_paths: [grades/f0_seven_endpoints_r1.json]
---

# Matched GRU continuation: actual F0 results

## Question and method

Does Gaussian NLL improve a strong GRU continuation or its contribution to a stronger saved model? Two new H256×2 fits, one per objective, share initialization, sampling/normalization identities, teacher targets and the two-epoch schedule. No new teacher fits. Strict FP32 updates start from historical mixed-precision ancestry; this is not a fresh-FP32 replication. Fixed zero-based epoch1 and seven endpoints were frozen before outcomes.

## Measured endpoints

Official `global_wp_clip2_v1`, 468 exposed valid-source groups /141 connected components.

| Endpoint | WP |
|---|---:|
| parent_cp5 | 0.6811553603 |
| wmse_cp1 | 0.6813131585 |
| nll_cp1 | 0.6812695476 |
| parent_nll_50_50 | 0.6813135999 |
| oldz4_cp1_fp32 | 0.6831643304 |
| oldz4_075_wmse_025 | 0.6835218031 |
| oldz4_075_nll_025 | 0.6835179511 |


| Paired contrast | Delta | Conditional 95% interval |
|---|---:|---|
| nll_cp1_minus_wmse_cp1 | -0.0000436109 | [-0.0006230104, +0.0004552434] |
| wmse_cp1_minus_parent_cp5 | +0.0001577981 | [-0.0000954893, +0.0004594840] |
| nll_cp1_minus_parent_cp5 | +0.0001141872 | [-0.0004979273, +0.0007036188] |
| parent_nll_50_50_minus_parent_cp5 | +0.0001582395 | [-0.0001489827, +0.0004577909] |
| oldz4_075_wmse_025_minus_oldz4_cp1_fp32 | +0.0003574728 | [-0.0003528744, +0.0010702733] |
| oldz4_075_nll_025_minus_oldz4_cp1_fp32 | +0.0003536207 | [-0.0002651170, +0.0009814120] |
| oldz4_075_nll_025_minus_oldz4_075_wmse_025 | -0.0000038521 | [-0.0001418886, +0.0001146902] |

## Interpretation and limitations

NLL does not show an advantage over matched WMSE here. The two fixed oldZ4 blends have almost identical point scores and modest positive contributions. Intervals condition on these fitted models and resample141 evaluation components; they do not measure fitting-seed variation or public transfer. No interval crossing zero closes the broader family. F0 is exposed, one seed was fitted, and this does not establish an improvement over TL3.

NLL clipped100% of updates, versus roughly7% for WMSE; readout gradients dominate the NLL squared-gradient diagnostic. This suggests a possible objective/optimization interaction, without identifying individual head rows or proving AdamW parameter-step suppression.

Independent review resolved the apparent parent discrepancy: historical0.6828233414 belongs to a different oldZ4 parent checkpoint, not this T3 CP5. Historical oldZ4 student0.6831418134 and current0.6831643304 share checkpoint/fold/target/mask identities but use FP16 and FP32 exports respectively; historical scorer-code hashes are unavailable, so precision-only attribution is not proven. Actual independent source/metric/ancestry review passes. No paired interval was frozen for oldZ4 versus T3CP5; only the seven declared contrasts have intervals.

## Decision and next step

Preserve both children and all seven endpoints. Defer the prepared F1 NLL pair for opportunity cost, rather than reject heteroscedastic modeling. Prioritize the PR25-supported cold H512 width route after an actual local cost profile. Require complete endpoint evaluation and native package checks before recommending a ZIP.

## Source records

- [Frozen scoring plan](score_plan_r1.json), [actual grade](grades/f0_seven_endpoints_r1.json), [successful R2 report](post_fit_plan_report_r2.json).
- [Post-export binding audit](prediction_post_export_r2.json), [fit terminal receipt](fit_terminal_r1.json), [packet and ancestry](README.md).
- [Conditional H512 profile](../h512_replication_profile/README.md).

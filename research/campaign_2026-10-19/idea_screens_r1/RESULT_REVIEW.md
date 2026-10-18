# Bounded screen result review

This is an independent audit of the compact EDA/ridge result and launch JSON. No sample or source-data arrays were loaded, and no fits or inferences were rerun. The selected panel contains 48 whole sequences: 32 fit sequences across 14 components and 16 evaluation sequences across 6 components. Membership was selected with target-blind shuffled components, seed 19019, greedily retaining whole eligible membership under the cost caps. The sampler labels it an exposed discovery panel with unequal component inclusion probabilities; it is not CP30. Each ridge view used 13,247 stride-40 fit rows and scored the same 35,467 masked evaluation rows.

## Binding and execution checks

- `SAMPLE.json` hashes to `0b03a4e2a6aab4eca36bdbb721d30e94b675c3425c0e0a7deca5ae562c2bbaa5`, matching both launch records and both `*_START.json` manifests.
- For EDA and ridge separately, all 11 source hashes shared by launch and start match. The EDA run is bound to the v1 EDA code. The ridge run is bound to the component-pooling-fixed v2 `ridge_probe.py` hash (`afd270…`); the launch note clarifies that the `ridge-v1` unit/output label does not denote an earlier ridge run.
- Both result-file SHA-256 values match their `*_DONE.json` receipts. Captured unit status is successful with exit status 0; CUDA was not initialized. EDA completed in 27.4 s at 501,188 KiB peak RSS, and ridge in 76.7 s at 641,728 KiB.
- The result contains exactly the nine fixed views in the screen design, including `raw`, `i0_only`, and `i1_only` controls. The ridge receipt says fit-only normalization and stride-40 fitting; the fixed ridge alpha is 1. These are diagnostic fixed screens, not evaluation-tuned model selection.

## Metric and panel audit

Every view reports 35,467 selected rows across the same six evaluation components. Summing each view's six component `pooled_raw_moments` and passing the result to canonical `wnn.gmetric.wp_from_sums` reproduces its official `global_wp_clip2_v1` score. The largest absolute difference from the reported global WP is `3.4e-16` (below the requested `1e-10`). The per-component scored-row counts are 1,624 (105), 3,124 (2951), 2,689 (3040), 1,250 (3060), 4,022 (3118), and 22,758 (995); component 995 is 64.2% of evaluation rows.

| Fixed view | Global WP | Delta vs raw |
|---|---:|---:|
| raw | 0.687250 | — |
| raw_lags_1_16_128 | 0.655190 | -0.032060 |
| raw_nonnegative_volume_proportions_log_total | 0.686367 | -0.000884 |
| raw_row_shape_location_scale | 0.667882 | -0.019368 |
| raw_rolling_beta_residual | 0.686890 | -0.000360 |
| raw_net_absinnovation_ema | 0.698439 | +0.011189 |
| raw_warmup99_context | 0.643878 | -0.043372 |
| i0_only | 0.697610 | +0.010360 |
| i1_only | 0.361145 | -0.326105 |

The component table below gives each view's change in component-pooled WP versus raw, in component order 105, 2951, 3040, 3060, 3118, 995. These component metrics are diagnostic and do not add up to the globally pooled WP delta.

| Fixed view | Component WP deltas vs raw |
|---|---|
| raw | 0, 0, 0, 0, 0, 0 |
| raw_lags_1_16_128 | -0.0078, -0.0549, -0.0377, -0.0032, -0.0770, -0.0334 |
| raw_nonnegative_volume_proportions_log_total | -0.0025, +0.0002, -0.0003, -0.0008, +0.0021, -0.0012 |
| raw_row_shape_location_scale | -0.0200, -0.1012, -0.0305, +0.0112, +0.1292, -0.0303 |
| raw_rolling_beta_residual | -0.0118, -0.0079, +0.0001, -0.0106, -0.0038, -0.0005 |
| raw_net_absinnovation_ema | -0.0449, +0.0163, +0.0448, -0.0835, -0.0973, +0.0293 |
| raw_warmup99_context | -0.0153, -0.0565, -0.0183, +0.0178, +0.0558, -0.0219 |
| i0_only | +0.0013, +0.0382, -0.0039, -0.0768, -0.0425, +0.0080 |
| i1_only | -0.1886, -0.4514, -0.4654, -0.1020, -0.2397, -0.4310 |

Leave-one-evaluation-component-out pooled deltas below are calculated from the recorded raw moments, holding the already fitted predictions fixed and recomputing WP after dropping one component. This is a scoring-concentration diagnostic, not held-out-component retraining or cross-validation. The columns summarize the six drop-one deltas as a range and show the delta when dropping the large component 995.

| Fixed view | Range across six drop-one deltas | Drop 995 |
|---|---:|---:|
| raw | 0 | 0 |
| raw_lags_1_16_128 | -0.05067 to -0.02862 | -0.05067 |
| raw_nonnegative_volume_proportions_log_total | -0.00116 to -0.00068 | -0.00092 |
| raw_row_shape_location_scale | -0.02484 to -0.00778 | -0.00778 |
| raw_rolling_beta_residual | -0.00166 to -0.00025 | -0.00166 |
| raw_net_absinnovation_ema | -0.03823 to +0.01617 | -0.03823 |
| raw_warmup99_context | -0.04832 to -0.00523 | -0.00523 |
| i0_only | +0.00849 to +0.01429 | +0.01429 |
| i1_only | -0.34555 to -0.30945 | -0.33887 |

EMA's full-panel gain is concentrated in the sense that dropping component 995 changes its pooled delta from +0.01119 to -0.03823; its individual component deltas also have mixed signs. `i0_only` remains positive in all six fixed-prediction drop-one summaries, though its individual component deltas are mixed and the evidence is limited to these six selected components. These summaries do not justify a p-value, a public leaderboard claim, or a claim that a production model uses or ignores a channel.

An existing current-F0 prediction bundle was also evaluated on the same selected rows. The captured `REPORT_REFERENCE.json` binds bundle SHA-256 `6277cca8d92f103da556929b0e88fc1c63c9eada61c5ae36d98af4a153d62d38`, reports bitwise target/order alignment for each of the 16 groups, and scores 35,467 rows: wide WP 0.807598 and deep WP 0.804993. Its captured run completed with exit status 0. This is a matched-panel context result from fixed existing FP32 predictions, not a refit or full-fold comparison. Those GRUs were fitted on much more data than the small-panel ridge, so their scores are not an equal-budget benchmark and do not support a full-F0 ranking or public inference claim.

## EDA interpretation limits

Input-only summaries show all 640,000 fit rows and 320,000 evaluation rows are fully finite. Strictly positive complete volume rows are rare in fit (771/640,000 for `i0_v`; 1,097/640,000 for `i1_v`) and absent in this evaluation panel (0/320,000 for each). The median sequence-level positive-row fraction is consequently zero for both blocks in both roles. This is a selected-panel input diagnostic, not evidence that every sequence in the full dataset has zero positive rows or that a model cannot use those channels.

The target-operator diagnostic is fit-only and tests ten fixed row-mean future-input-delta candidates at horizons 1, 8, 32, 128, and 512. In its 14-component leave-one-fit-component-out selection, target 0 selected `i0_p_mean_delta_h512` in 14/14 folds; target 1 selected `i0_dp_mean_delta_h512` in 13/14 and `i0_p_mean_delta_h32` in 1/14. Yet selected training associations are negative in all folds, and held-out magnitude-weighted associations are positive in only 3/14 folds for each target (ranges -0.368 to +0.039 for target 0 and -0.169 to +0.038 for target 1). This does not identify the target operator or establish a useful horizon. The statistic is a retrospective association to future inputs, not an online feature or an official-score prediction.

The optional `reference_panel.py` source has no obvious source-level blocker: it binds the existing prediction bundle by hash, checks selected target order/value and component alignment, and makes no inference or weight recovery. It was not run as part of this review.

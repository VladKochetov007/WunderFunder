# E474 prediction-export audit

This is an independent audit of the finished CPU-grade E474 export set. It supplements and preserves the earlier fit-source review in `INDEPENDENT_REVIEW.md`; it does not alter that review or any fit/scorer source.

I read the saved `score_result.json` and its twelve declared selected-prediction NPZ files only. I did not read the validation parquet, load checkpoints, run inference, or train models. Each export hash matched the score manifest. All twelve exports had finite raw targets and predictions, identical selected targets, component order, row-group order, and per-component counts. The outputs contain 16 component groups and 26,990 selected rows per arm. I recomputed each arm’s `global_wp_clip2_v1` score with the official `wnn.gmetric.global_wp`; all `wp`, `t0`, and `t1` values matched the stored results within `1e-10`.

| Seed | Arm | WP | t0 | t1 |
|---:|---|---:|---:|---:|
| 271828 | base | 0.444184 | 0.512219 | 0.376148 |
| 271828 | sine_embed | 0.439678 | 0.509731 | 0.369626 |
| 271828 | slope_embed | 0.438388 | 0.509599 | 0.367177 |
| 271828 | sine_head | 0.443614 | 0.509895 | 0.377334 |
| 271828 | slope_head | 0.444101 | 0.509122 | 0.379080 |
| 271828 | sine_both | 0.437504 | 0.506303 | 0.368705 |
| 271829 | base | 0.494020 | 0.491245 | 0.496796 |
| 271829 | sine_embed | 0.496496 | 0.491532 | 0.501461 |
| 271829 | slope_embed | 0.495580 | 0.490268 | 0.500892 |
| 271829 | sine_head | 0.493948 | 0.493731 | 0.494165 |
| 271829 | slope_head | 0.492958 | 0.493178 | 0.492739 |
| 271829 | sine_both | 0.495708 | 0.492995 | 0.498420 |

I independently rebuilt per-component Welford moments from the raw selected exports and replayed the scorer’s 2,000 shared component draws in frozen row order with seed `474271828`. The two representative `sine_embed − base` paired WP checks match the stored bootstrap means, standard deviations, percentile intervals, and sign probabilities:

| Seed | Point delta | Bootstrap mean | SD | 95% percentile interval | P(delta ≤ 0) |
|---:|---:|---:|---:|---:|---:|
| 271828 | -0.004506 | -0.004356 | 0.001303 | [-0.006937, -0.001789] | 0.9995 |
| 271829 | 0.002476 | 0.002455 | 0.001296 | [-0.000056, 0.004996] | 0.0290 |

The seed-271828 interaction `WP(sine_both) − WP(sine_embed) − WP(sine_head) + WP(base)` also matches when all four arms use the same resampled component multiplicities. Its point delta is `-0.001604`; the bootstrap mean is `-0.001514`, SD `0.000521`, 95% interval `[-0.002435, -0.000400]`, and `P(delta ≤ 0) = 0.996`.

The embedding contrast changes sign across the two fixed seeds, and the second interval crosses zero narrowly. These checks establish that the exported rows and reported scores/contrasts agree; they do not establish a stable activation benefit beyond this exposed 16-component panel.

Audit inputs: [score_result.json](scores/twelve_cpu_v1/score_result.json); twelve files listed in its `prediction_exports` map. The result file SHA-256 was `d3e7839761432c03e47b67e9ba9e13424dae7a70ac9dd801ce41e204af93adfe`; the scorer source SHA-256 recorded in the result was `d37634bb45788ebb8cc4d635b0f4f29d90f0b5447b7829d233efda160beeddd7`.

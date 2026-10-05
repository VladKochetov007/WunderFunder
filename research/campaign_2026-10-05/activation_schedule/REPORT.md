# E476: learning-rate shape changes the activation comparison

All12 fixed fits and independent CPU scoring finished without a new OOM. Warmup/cosine beats the constant-at-equal-mean control by +0.067328 (GELU/AdamW) and +0.072700 (sine embedding/AdamW); both conditional component intervals are positive. The corresponding Muon gains are +0.015065/+0.016691, with intervals crossing zero. These are one-seed exposed-panel results, not public gains.

The activation ranking depends on the LR protocol. Sine embedding loses to GELU under constant peak but improves under warmup/cosine. The activation-by-schedule interactions are +0.005372 (AdamW) and +0.001626 (Muon), both with positive conditional intervals. This is evidence against selecting activation independently of the optimizer schedule; independent seed replication is still needed.

Both the warmup/cosine and constant-mean traces sum to0.2607. This matches arithmetic LR area, not actual optimizer update area. All recipes have61,058 parameters, common initialization/order, 24 fit components and16 previously exposed diagnostic components. The strong public model uses different capacity/data/training. The large AdamW difference warrants reproduction, not extrapolation.

| Registered arm | Global WP | t0 | t1 |
|---|---:|---:|---:|
| base__adamw__constant_peak | 0.4441839 | 0.5122195 | 0.3761483 |
| base__adamw__warmcos_47_427 | 0.5181834 | 0.4996345 | 0.5367322 |
| base__adamw__constant_mean_warmcos | 0.4508551 | 0.5146166 | 0.3870935 |
| base__muon_adamw__constant_peak | 0.4846863 | 0.5123786 | 0.4569940 |
| base__muon_adamw__warmcos_47_427 | 0.4990650 | 0.4740198 | 0.5241102 |
| base__muon_adamw__constant_mean_warmcos | 0.4840004 | 0.4999485 | 0.4680522 |
| sine_embed__adamw__constant_peak | 0.4396783 | 0.5097308 | 0.3696257 |
| sine_embed__adamw__warmcos_47_427 | 0.5205517 | 0.4996905 | 0.5414129 |
| sine_embed__adamw__constant_mean_warmcos | 0.4478516 | 0.5087369 | 0.3869663 |
| sine_embed__muon_adamw__constant_peak | 0.4785979 | 0.5086436 | 0.4485522 |
| sine_embed__muon_adamw__warmcos_47_427 | 0.5072714 | 0.4787639 | 0.5357790 |
| sine_embed__muon_adamw__constant_mean_warmcos | 0.4905808 | 0.5025470 | 0.4786145 |

| Registered contrast | Global WP delta | Conditional95% interval |
|---|---:|---|
| sine_embed__adamw__constant_peak_minus_base__adamw__constant_peak | -0.0045056 | [-0.0069220, -0.0016905] |
| sine_embed__adamw__warmcos_47_427_minus_base__adamw__warmcos_47_427 | +0.0023683 | [+0.0004273, +0.0040903] |
| sine_embed__adamw__constant_mean_warmcos_minus_base__adamw__constant_mean_warmcos | -0.0030035 | [-0.0047585, -0.0007049] |
| sine_embed__muon_adamw__constant_peak_minus_base__muon_adamw__constant_peak | -0.0060884 | [-0.0091630, -0.0028477] |
| sine_embed__muon_adamw__warmcos_47_427_minus_base__muon_adamw__warmcos_47_427 | +0.0082065 | [+0.0055051, +0.0112746] |
| sine_embed__muon_adamw__constant_mean_warmcos_minus_base__muon_adamw__constant_mean_warmcos | +0.0065804 | [+0.0038084, +0.0099208] |
| base__muon_adamw__constant_peak_minus_base__adamw__constant_peak | +0.0405024 | [+0.0222757, +0.0570936] |
| base__muon_adamw__warmcos_47_427_minus_base__adamw__warmcos_47_427 | -0.0191184 | [-0.0336142, -0.0028796] |
| base__muon_adamw__constant_mean_warmcos_minus_base__adamw__constant_mean_warmcos | +0.0331453 | [+0.0039581, +0.0616799] |
| sine_embed__muon_adamw__constant_peak_minus_sine_embed__adamw__constant_peak | +0.0389196 | [+0.0216562, +0.0545880] |
| sine_embed__muon_adamw__warmcos_47_427_minus_sine_embed__adamw__warmcos_47_427 | -0.0132803 | [-0.0263397, +0.0025946] |
| sine_embed__muon_adamw__constant_mean_warmcos_minus_sine_embed__adamw__constant_mean_warmcos | +0.0427292 | [+0.0135639, +0.0716613] |
| base__adamw__warmcos_47_427_minus_base__adamw__constant_mean_warmcos | +0.0673283 | [+0.0275651, +0.1043927] |
| base__muon_adamw__warmcos_47_427_minus_base__muon_adamw__constant_mean_warmcos | +0.0150646 | [-0.0091102, +0.0370621] |
| sine_embed__adamw__warmcos_47_427_minus_sine_embed__adamw__constant_mean_warmcos | +0.0727001 | [+0.0315141, +0.1106531] |
| sine_embed__muon_adamw__warmcos_47_427_minus_sine_embed__muon_adamw__constant_mean_warmcos | +0.0166906 | [-0.0089257, +0.0396802] |
| base__adamw__constant_peak_minus_base__adamw__constant_mean_warmcos | -0.0066712 | [-0.0263308, +0.0096379] |
| base__muon_adamw__constant_peak_minus_base__muon_adamw__constant_mean_warmcos | +0.0006860 | [-0.0103988, +0.0137710] |
| sine_embed__adamw__constant_peak_minus_sine_embed__adamw__constant_mean_warmcos | -0.0081733 | [-0.0278186, +0.0079172] |
| sine_embed__muon_adamw__constant_peak_minus_sine_embed__muon_adamw__constant_mean_warmcos | -0.0119829 | [-0.0242554, +0.0020021] |
| sine_activation_x_warmcos_shape__adamw | +0.0053718 | [+0.0026707, +0.0070055] |
| sine_activation_x_warmcos_shape__muon_adamw | +0.0016260 | [+0.0000454, +0.0030832] |

All20 pairs and2 jointly resampled interactions are retained, with no seed pooling or multiplicity correction. The independent scorer checks actual runner/spec/runtime/artifact receipts, raw selected target masks, finite predictions before clipping, exact LR receipts, fit-only normalization and checkpoint identity. A larger-memory CPU-helper admission was queued until E478 scoring drained; it then completed127.2383s, peak0.965GB RAM. No recipe was changed.

Source review and additional independent export reproduction are recorded separately. Raw official result: eda/twelve_v1/scores.json. Figure: eda/all_schedule_arms.png. Fixed gradient/update telemetry is in the fit receipt and is descriptive, never a test-based tuner.

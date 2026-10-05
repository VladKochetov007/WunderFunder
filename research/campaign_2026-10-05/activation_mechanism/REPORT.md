# E473: curvature, local slope, scale and GELU approximation

All ten fixed one-epoch fits finished successfully. The scorer reproduced official pooled and per-target WP, verified selected targets/masks and all source/attempt identities, and retained fourteen fixed conditional component comparisons. The four unchanged plain-GELU/GELU+sine controls reproduced E470 within 1e-7.

| Recipe | Pooled WP | t0 | t1 |
|---|---:|---:|---:|
| gelu__adamw | 0.4982390 | 0.5265098 | 0.4699683 |
| gelu__muon_adamw | 0.4891222 | 0.4997636 | 0.4784808 |
| gelu_plus_sine_01__adamw | 0.4999632 | 0.5239936 | 0.4759327 |
| gelu_plus_sine_01__muon_adamw | 0.4979382 | 0.5033758 | 0.4925006 |
| gelu_plus_linear_01__adamw | 0.5009882 | 0.5253803 | 0.4765962 |
| gelu_plus_linear_01__muon_adamw | 0.4997471 | 0.5057831 | 0.4937112 |
| scaled_gelu_11__adamw | 0.5001673 | 0.5270925 | 0.4732422 |
| scaled_gelu_11__muon_adamw | 0.4947426 | 0.5030057 | 0.4864794 |
| jax_tanh_gelu_plus_sine_01__adamw | 0.4999666 | 0.5239956 | 0.4759375 |
| jax_tanh_gelu_plus_sine_01__muon_adamw | 0.4979401 | 0.5033740 | 0.4925063 |

| Fixed contrast | Delta WP | Conditional 95% interval |
|---|---:|---:|
| gelu_plus_sine_01__adamw_minus_gelu__adamw | +0.0017241 | [-0.0006990, +0.0047628] |
| gelu_plus_sine_01__muon_adamw_minus_gelu__muon_adamw | +0.0088160 | [+0.0058788, +0.0123252] |
| gelu_plus_linear_01__adamw_minus_gelu__adamw | +0.0027492 | [-0.0004101, +0.0067854] |
| gelu_plus_linear_01__muon_adamw_minus_gelu__muon_adamw | +0.0106249 | [+0.0071763, +0.0153670] |
| scaled_gelu_11__adamw_minus_gelu__adamw | +0.0019283 | [+0.0009261, +0.0032788] |
| scaled_gelu_11__muon_adamw_minus_gelu__muon_adamw | +0.0056203 | [+0.0034978, +0.0081365] |
| jax_tanh_gelu_plus_sine_01__adamw_minus_gelu__adamw | +0.0017275 | [-0.0006980, +0.0047676] |
| jax_tanh_gelu_plus_sine_01__muon_adamw_minus_gelu__muon_adamw | +0.0088179 | [+0.0058821, +0.0123182] |
| gelu_plus_linear_01__adamw_minus_gelu_plus_sine_01__adamw | +0.0010251 | [-0.0001895, +0.0026149] |
| gelu_plus_linear_01__muon_adamw_minus_gelu_plus_sine_01__muon_adamw | +0.0018089 | [+0.0007614, +0.0034684] |
| scaled_gelu_11__adamw_minus_gelu_plus_sine_01__adamw | +0.0002042 | [-0.0022276, +0.0023225] |
| scaled_gelu_11__muon_adamw_minus_gelu_plus_sine_01__muon_adamw | -0.0031956 | [-0.0049053, -0.0016786] |
| jax_tanh_gelu_plus_sine_01__adamw_minus_gelu_plus_sine_01__adamw | +0.0000034 | [+0.0000006, +0.0000072] |
| jax_tanh_gelu_plus_sine_01__muon_adamw_minus_gelu_plus_sine_01__muon_adamw | +0.0000019 | [-0.0000069, +0.0000122] |

The additive linear control beats the sine addition by +0.001025 with AdamW (interval crosses zero) and +0.001809 with Muon (conditional interval positive). Sine periodic curvature is therefore unnecessary to explain the first pilot advantage: changing the local slope remains sufficient as a candidate explanation. One seed does not establish superiority of the linear control. Retain both in the next placement comparison rather than choosing a winner from this panel.

Uniform scaling also helps versus plain GELU at this seed but does not match the derivative at zero: GELU+0.1x has derivative 0.6, while 1.1*GELU has derivative 0.55. Scaling and additive slope are distinct interventions. Approximate versus exact GELU+sine differs by +0.0000034 (AdamW) and +0.0000019 (Muon); an exploratory interval excluding zero at this magnitude is not meaningful deployment evidence.

The dataset, exposed components, one-layer GRU32, 8,235 parameters, 474 updates, nominal LR and optimizer partition remain the E470 recipe. These are conditional exploratory intervals without multiplicity correction, not independent validation, seed uncertainty, or a public expectation. No dropout, schedule, placement, strong-GRU continuation or recurrent-candidate conclusion follows from this test.

Full fit cost: 25.9463 guarded GPU-job seconds; preceding ten-update probe: 7.3145s. Peak observed GPU 0.3838GB, worker RAM 1.6403GB. Both exited zero with verified teardown, no worker OOM and no increase in parent historical OOM counters. The 12CPU/30GB hard/27GB soft/zero-swap and 5GB local observed GPU ceilings remain in force.

Machine evidence: `RESULT.json`, `GPU_PROBE_RESULT.json`, content-bound draft/sealed specs and immutable runner attempts/logs. Fit weights and selected exports are retained locally under `runs/ten_v1/` and excluded from Git as bulk artifacts.

# Three-seed activation and optimizer replication (E470 + E472)

All 48 fixed one-epoch fits completed: eight activation choices crossed with two optimizers at seeds 271828, 271829 and 271830. Official pooled and both target scores were independently recomputed from the selected-row exports. All 90 within-seed paired-component contrasts are retained in `THREE_SEED_ACTIVATION_RESULT.json`. No confidence intervals are pooled across training seeds.

This is an 8,235-parameter, one-layer GRU32 on 24 fit and 16 historically exposed diagnostic components. It is not a comparison against the complete two-layer GRU256 public incumbent. All arms use 474 optimizer updates.

| Activation / optimizer | Seed 271828 WP | Seed 271829 WP | Seed 271830 WP |
|---|---:|---:|---:|
| tanh__adamw | 0.482259 | 0.483661 | 0.548964 |
| tanh__muon_adamw | 0.505498 | 0.459306 | 0.530970 |
| gelu__adamw | 0.498239 | 0.498771 | 0.542257 |
| gelu__muon_adamw | 0.489122 | 0.469127 | 0.507702 |
| relu__adamw | 0.486599 | 0.482456 | 0.532063 |
| relu__muon_adamw | 0.481744 | 0.466605 | 0.515069 |
| leaky_relu_001__adamw | 0.479897 | 0.485396 | 0.526527 |
| leaky_relu_001__muon_adamw | 0.481479 | 0.461490 | 0.512481 |
| gelu_sine_01__adamw | 0.499963 | 0.499477 | 0.545029 |
| gelu_sine_01__muon_adamw | 0.497938 | 0.471851 | 0.515024 |
| gelu_sine_cosine_01__adamw | 0.498653 | 0.495594 | 0.545768 |
| gelu_sine_cosine_01__muon_adamw | 0.497789 | 0.466782 | 0.515289 |
| gelu_sinc_05__adamw | 0.494680 | 0.493358 | 0.541058 |
| gelu_sinc_05__muon_adamw | 0.491209 | 0.463179 | 0.509422 |
| split_gelu_leaky_relu__adamw | 0.486659 | 0.489880 | 0.533168 |
| split_gelu_leaky_relu__muon_adamw | 0.488666 | 0.465915 | 0.516452 |

## Discriminating comparisons

| Within-seed contrast | Seed 271828 delta [95% interval] | Seed 271829 | Seed 271830 |
|---|---:|---:|---:|
| gelu_sine_01__adamw_minus_gelu__adamw | +0.001724 [-0.000699, +0.004763] | +0.000706 [-0.001600, +0.003380] | +0.002771 [+0.000392, +0.005418] |
| gelu_sine_01__muon_adamw_minus_gelu__muon_adamw | +0.008816 [+0.005879, +0.012325] | +0.002723 [+0.000327, +0.005592] | +0.007322 [+0.004356, +0.010683] |
| tanh__muon_adamw_minus_tanh__adamw | +0.023239 [+0.004164, +0.043127] | -0.024355 [-0.036196, -0.012475] | -0.017994 [-0.031779, -0.004526] |
| gelu__muon_adamw_minus_gelu__adamw | -0.009117 [-0.029088, +0.007523] | -0.029644 [-0.044816, -0.014229] | -0.034555 [-0.057500, -0.013682] |

The sine addition is positive against GELU in all six seed/optimizer comparisons. That supports retaining this particular recipe for a larger matched test, not a universal activation preference or public improvement. Fit-only telemetry places most preactivations near zero, where sin(x) is approximately x. A preregistered GELU + 0.1*x control is needed before attributing gains to periodic curvature.

Muon versus AdamW is seed-sensitive here. The first-seed tanh Muon advantage reverses at the other two seeds. GELU AdamW beats the tested GELU Muon recipe at all three seeds. This is evidence about these fixed learning rates and optimizer settings, not optimally tuned optimizer families.

There are many exploratory contrasts and only three training seeds. The component intervals condition on each fitted model and do not include training or search uncertainty. The third seed improves most recipes together, further discouraging a ranking based on one run.

## Resource and provenance receipts

The two additional sixteen-arm workers exited successfully with teardown verified, no worker OOM and no increase in parent OOM counters. Together they used 75.4793 guarded GPU-job seconds. Maximum observed GPU memory was 0.5044 GB; maximum worker RAM was below 1.8 GB. Parent limits stayed at 12 CPUs, 30 decimal GB RAM, 27 GB soft and zero swap. Original E470 sources were unchanged; only the registered seed wrapper varied.

Attempts: `e472-activation-seed271829-v1` and `e472-activation-seed271830-v1`. The JSON result retains source/checkpoint-export hashes and both successful attempt records. Plot: `eda/seed_replication/all_activation_seeds.png`. No submission or public forecast was produced.

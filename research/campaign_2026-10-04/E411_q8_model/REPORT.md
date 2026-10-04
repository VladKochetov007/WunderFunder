# E411: fixed 8-bit weight degradation pilot

Both fixed presets showed small score losses on the predeclared 64-component
panel. This supports testing native byte-kernel integration. It does not yet
establish deployable runtime, full-fold noninferiority or a public score.

| Preset | Global WP | Delta from original | Paired component 95% interval | Legacy block WP delta |
|---|---:|---:|---:|---:|
| Original fused predictor | 0.6709730754 | 0 | — | 0 |
| Recurrent matrices only, q8 | 0.6709503462 | -0.0000227292 | [-0.0001014153, +0.0000471268] | -0.0000592993 |
| Recurrent and first head matrices, q8 | 0.6709538145 | -0.0000192608 | [-0.0000901548, +0.0000428880] | -0.0000457329 |

The predictor is one D256 trunk with the equally weighted, fused heads from
`c20261003_combo_s3_f0@ep3` and `c20261004_combo_head_s3_f0@ep0`. Preparation
verified every non-head tensor was identical. The two presets were fixed before
execution and both are reported; no tuning or winner selection was performed.

Quantization is symmetric signed-byte, per output row, nearest-even rounding.
The head preset uses independent row scales for hidden and projection halves.
The input projection, final output layer, normalization and every bias remain
float32. Saved codes/scales exactly reconstruct the represented weights. The
existing native engine evaluates those represented weights using **float32
arithmetic**. This is a weight-degradation test, not a measurement of the new
native byte kernel in a recurrent predictor.

The controller amended the panel before preparation or results: rank fold-0
evaluation components by SHA256 of `271828:component:ID`, choose 64, then choose
one representative per component by SHA256 of `271828:group:COMPONENT:GROUP`.
Selection reads no targets and applies no class quotas. Explicit sequence IDs
were checked against the split manifest during extraction. The panel contains
119,540 scored rows and market-class counts 8/11/11/34. All classes are present;
the worst class delta across either preset is -0.0000222830. The panel is drawn
from exposed evaluation data, represents only one sequence per selected
component, and is not the full fold or an independent holdout. Bootstrap uses
2,000 paired component replicates with seed 271828; preset standard deviations
are 0.0000377315 and 0.0000337770 respectively.

Before scoring, native and independent PyTorch GRU inference were compared on
two complete fixed sequences for every preset and the original. Maximum
absolute prediction discrepancy was 1.90735e-6; maximum absolute global-WP
discrepancy was 6.37329e-8. All explicit reset/repeat checks were bitwise equal.

Execution ran 2026-10-04 02:02:13–02:12:46 UTC. Preparation took 30.69 seconds,
peak RSS 1,456,360 KiB; inference took 602.33 seconds, peak RSS 1,214,952 KiB.
Both stages exited zero under the shared `wnn-campaign.slice` and dedicated
1-CPU/2-GiB/swap-zero scopes, pinned to logical CPU 15, with individual
1,800-second deadlines and CUDA visibility disabled. The shared parent enforced
12 CPUs and 36 GiB; its OOM counters stayed zero. Shared `high` and `max` event
counts increased during concurrent campaign activity, which cannot be attributed
to this probe. The sampled inference-scope memory event counters were all zero.

The manifest binds the full validation-data SHA through the portable registry,
the split and checkpoints, native binary, source files, selected rows, byte
codes, represented weights and extracted input arrays. Predictions, class
results, parity evidence and per-sequence progress are retained. Exact commands,
resource snapshots, exit codes and logs are in `../E411_q8_model_control/`.

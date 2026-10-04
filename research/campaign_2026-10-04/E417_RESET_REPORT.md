# E417 — current combo state reset panel

Resetting the current student's carried GRU state reduced sampled Global WP.
The observed reset250 and reset500 losses have negative component intervals;
the reset1000 and reset2000 intervals include zero. This does not establish
whether longer training credit improves predictions.

| State reset interval | Reset minus unchanged WP | Component 95% interval |
| --- | ---: | ---: |
| 250 rows | −0.04217 | [−0.07720, −0.01041] |
| 500 rows | −0.03144 | [−0.06550, −0.00348] |
| 1000 rows | −0.02142 | [−0.05239, +0.00232] |
| 2000 rows | −0.01564 | [−0.03840, +0.00126] |

The unchanged predictor scored0.60858 on498 prespecified sampled scored rows
from32 complete streams in15 connected components. The panel was selected by
fixed component hashes, without targets. The combo checkpoint and its learned
ancestors excluded all these components from fitting. The fold was historically
exposed; this is a small diagnostic panel, not new confirmation.

Each lag also has a matched0.5-state-retention control at the same boundaries.
Those controls scored0.58033,0.58867,0.59500,0.60042 respectively. Complete reset
was worse than partial attenuation by0.01392 and0.01153 at the two shortest
intervals, with negative paired component intervals. The corresponding longer
interval contrasts include zero.

Every arm preserves the first99 warmup rows and the full causal input-feature
history. Only carried recurrent state changes before rows99+k×lag. Hence this
measures disruption of the fitted predictor's state, including its possible
out-of-distribution response to reset. It does not measure the marginal value of
a particular past row or isolate useful backpropagation length. The predeclared
flat-reset outcome did not occur at250/500; no memory hypothesis is closed from
the uncertain longer lags.

The admitted GPU job finished113.67seconds, peak observed VRAM340,787,200bytes,
peak host memory1,260,822,528bytes, noOOM. All nine arms and their predictions are
retained in `feature_reset_results/f0_gpu_v1/`; exact execution is bound by
`specs/e417_reset_f0_gpu_v1.sealed.json` and its runner record. Bootstrap intervals
use2,000 draws and condition on the fitted model and fixed sampled rows.

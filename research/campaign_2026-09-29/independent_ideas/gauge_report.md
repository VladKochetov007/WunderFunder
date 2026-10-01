# I-01 frozen CPU falsifier: continuous price-code gauge rejected

The frozen [selection](gauge_selection.json) was written before any row-group price-state read. The script and machine-readable result are [gauge_screen.py](gauge_screen.py) and [gauge_result.json](gauge_result.json). Only `datasets/train.parquet` price features were read. The `cls` column in `cache/seqstats_train.parquet` stratified four train-market classes; no target values, validation examples, or scoring masks entered the statistic. Each class used 24 fit and 12 held-out train row groups, 128 fixed sampled states per group. The same held-out states and shift size were used for the coherent and random-direction control.

Predeclared pass limits: shifted nearest-code mean distance / unshifted <=1.25; coherent / equal-size random orthogonal shift <=0.75; projected quote-event energy of the proposed gauge / random direction <=0.5. All three had to pass in all four classes.

| Class | Shifted / baseline code distance | Coherent / random code distance | Gauge / random event energy | Result |
| --- | ---: | ---: | ---: | --- |
| 0 | 1.59 | 0.985 | 16.5 | Fail |
| 1 | 8.20 | 0.999 | 13.9 | Fail |
| 2 | 38.96 | 1.000 | 52.1 | Fail |
| 3 | 75.10 | 1.000 | 23.7 | Fail |

In classes 1-3, 97.7-99.9% of held-out raw price states exactly matched a sampled fit code. After either shift, none did. The top between-sequence mean direction also has much *more* quote-event energy than the orthogonal random direction, violating the intended invariant-coordinate condition. This is a direct falsification of the proposed continuous whole-sequence augmentation at the predeclared size. Do not spend GPU time on I-01 as written. A discrete valid-code substitution would be a different hypothesis requiring its own invariance proof.

Commands and outcomes (1 core, 3 GB cap, zero GPU):

```text
WNN_SLICE=wnn-campaign.slice WNN_CPU_PCT=7 WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=3G scripts/limited .venv/bin/python research/campaign_2026-09-29/independent_ideas/gauge_screen.py select
exit 0; 0.66 s

same wrapper + gauge_screen.py run
exit 1; 3.52 s; implementation KeyError in gate-name mapping, no result written

same wrapper + gauge_screen.py run after correcting gate-name mapping
exit 0; 3.47 s; result above
```

The failed invocation is preserved here because it consumed resources and had no statistical output. The selection file was not changed between the two runs.

## Implementing shared-subspace removal for I-03 without evaluation information

1. Freeze the two clean fold-matched backbones. Stream their 128-dimensional hidden states from **fit chains only**, with a deterministic event/quiet-row sample. Accumulate centered `128 x 128` covariance blocks `C33`, `C44`, and `C34` in float64. No outer evaluation states, masks, or labels enter this step. Determine centers and a ridge, for example `1e-3 * trace(C)/128`, from fit states only.
2. Form whitened cross-covariance `M = C33^{-1/2} C34 C44^{-1/2}` and compute its SVD `U diag(s) V^T`. Freeze a prespecified shared rank, say 16. The residual coordinates are `r3=(I-U16 U16^T) C33^{-1/2}(h3-m3)` and `r4=(I-V16 V16^T) C44^{-1/2}(h4-m4)`. This projection removes directions linearly predictable across the two members under the fit-state measure; it does not assert hidden coordinates have the same basis.
3. Fit `p=.5(p3+p4)+W[(A r3) ⊙ (B r4)]` with rank 8 and zero-initialized `W`. Train on fit-chain targets only. Choose regularization in an **inner** fit-chain split; freeze head epoch and coefficients before outer-fold scoring. Use the same data, update count, and regularization protocol for unary/additive and output-only controls. An inner OOF backbone-state panel is preferable if available; fitting the head on backbones' own training states can make inner selection optimistic, though the untouched outer evaluation remains a valid test of that full algorithm.
4. For streaming CPU inference, precompose each fixed whitening/residual map with its learned `A` or `B`. This collapses the calculation to one `8 x 128` affine projection per member plus 8 products and a `2 x 8` readout. The full 128-dimensional residual need not be materialized. Serialize these matrices and centers into the fixed ZIP; no online statistics, labels, mask, or sequence ID are required.

This is a mathematical implementation sketch, not a measured head gain. All covariance/statistics are trained on provided fit data only.

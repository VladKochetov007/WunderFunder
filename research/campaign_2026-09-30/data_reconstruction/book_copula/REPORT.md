# Book rank dependence and external distribution comparison

Fixed descriptive comparison of16providedTRAINcomponents (two instruments,20,000rows/group) against the first20,000snapshots in externalMDlob.csv. The provided input panel was already inspected in the additional-feature study; this is not fresh confirmation. Only price/volume fields were read. No targets, masks, VALID, model weights or predictions were used.

| Provided block | Median leading share | Median effective rank | Median signed within-side rho | Median signed cross-side rho |
|---|---:|---:|---:|---:|
| i0 p | 0.712 | 1.78 | 0.232 | 0.233 |
| i0 v | 0.161 | 12.91 | -0.002 | 0.008 |
| i1 p | 0.769 | 1.61 | 0.365 | 0.378 |
| i1 v | 0.160 | 13.40 | 0.003 | 0.003 |

External price leading shares: absolute0.999956; centering on the midpoint of the two index0quotes0.484825; fractional midpoint-relative0.480056; centering each side on its index0quote0.532014. External volume leading share0.170653 and effective rank15.41. No source/symbol/time identity was recovered.

Price-code variation has few dominant rank factors, but signed within-side correlations are much weaker than the absolute-MDprice model. This is consistent with several possibilities: moving reference coordinates, nonlinear/reflective/projected representations, book-shape factors or publication/data-window effects. It does not identify a Fourier encoder, physical price axis, ordering of competition columns or target law. The preceding nearly exact first-field conditional lookup does not itself establish monotonicity. A feature-only functional/ordering falsifier is more useful than assuming a raw price scale.

Rank correlations are invariant to **strictly monotone** individual-column transforms, with signs changing for decreasing transforms. Clipping, quantization/tie formation and row-dependent centering can change rank structure. The script docstring uses the shorter word monotone; the strict/injective condition is required for this interpretation. Overall spectra are column-permutation invariant; side summaries rely only on published bid/ask membership. Neither variance spectra nor external volume similarities establish matching joint distributions. There is one external prefix and different publication/time histories; no empirical95%source-match interval is claimed.

Numerical detail: rank calculations follow SciPy/NumPy input dtypes (providedfloat32, externalfloat64). The figures above are descriptive to three decimals; a commonfloat64independent replay should check any precise equality claim. There is no such equality or candidate-promotion claim here.

Probe exit0, outerwall3.31s (inner2.63s), peakRSS1,186,944KiB, hard2GiBprocessaddress-space and60secCPU/wall limit, threadpools1. No systemd/durable/cgroup claim under the current restricted runtime; no GPU time. No external examples or fitted numeric transforms enter competition training or inference. Immutable source/prereg/selection/prefixhash/command/results are preserved; independent reproduction pending.

# TL3 same-row diagnostic results

The native CPU-only pass completed on all1,873 official valid sequences and
3,528,273 scored rows. No model inference, fit or weight search occurred.
All four global WP results exactly equal the prior R4 replay. This is exposed
development evidence with the same unreconstructed historical consumed-cache
ancestry limitation.

| Output | Official WP | Raw-prediction weighted MSE |
| --- | ---: | ---: |
| Distilled s5 / E1 | 0.6914165040 | 0.9659722158 |
| Tlead s5 / E3 | 0.6914152791 | 0.9481380947 |
| Tlead s6 / E3 | 0.6909489195 | 0.9493038177 |
| Fixed0.34/0.33/0.33 blend | 0.6923774248 | 0.9496937503 |

WMSE uses raw predictions, clipped labels and absolute clipped-label weights;
official WP clips both predictions and labels. The WMSE and WP rankings differ:
the fixed blend has the best WP while Tlead s5 has lower WMSE. Monitor the
official score as primary rather than select a model from loss alone. No
post-hoc calibration or learned blend is fitted here.

## Same-row residual associations

Weighted Pearson between raw-prediction residuals to clipped labels:

| Pair | t0 | t1 |
| --- | ---: | ---: |
| Distilled s5 / Tlead s5 | 0.9915702304 | 0.9895420990 |
| Distilled s5 / Tlead s6 | 0.9913203039 | 0.9896395523 |
| Tlead s5 / Tlead s6 | 0.9966382186 | 0.9966406690 |

The three cached outputs share strongly associated errors on these rows.
Shared unpredictable target variation also contributes to residual correlation;
this does not establish identical predictions or rule out useful combinations.
The measured fixed blend adds0.0009609208 WP over the best cached role. A new
architecture still needs its own complete score and declared combination test.

## Position and candidate-horizon slices

The bins and the256/512 grids were declared before this pass. These are
descriptive row-population contrasts, not causal effects or fresh holdouts.

| Zero-based position | Scored rows | Fixed blend WP |
| --- | ---: | ---: |
| [99,256) | 29,132 | 0.7291932727 |
| [256,1000) | 133,079 | 0.6920265321 |
| [1000,2000) | 179,267 | 0.6641991818 |
| [2000,5000) | 520,742 | 0.7001002868 |
| [5000,10000) | 876,499 | 0.6780902820 |
| [10000,20000) | 1,789,554 | 0.6993802488 |

The position scores do not decline monotonically with sequence age. This pass
does not establish a long-memory deficiency; an architectural intervention
needs matched fitting evidence.

| Candidate grid | Near-boundary rows / WP | Interior rows / WP |
| --- | --- | --- |
| 256 | 220,855 /0.6938657318 | 3,307,418 /0.6922778106 |
| 512 | 110,961 /0.6837780443 | 3,417,312 /0.6926497938 |

The512 grid has a lower near-boundary subset score, whereas the256 grid does
not. These grids are not verified TL3 training or inference reset boundaries.
Different row populations, target distributions and source composition can
produce such differences. Inspect actual producer chunk/state metadata and
compare new Transformer exports before claiming a reset or horizon mechanism.
No grid was selected from these scores and no adaptive follow-up is admitted.

## Trace

- Frozen diagnostic commit: `ce02f2cb174f4824a526642690747a68fcf8ae1c`.
- Script SHA256: `b07f66f7b269d452bc16bc8f2aeaf9890f6033b0b0283365ef2a707504ff4122`.
- Exact [native JSON](evidence/tl3-residual-diagnostics-r1.json) SHA256:
  `603345c98505559b95198876060e9205db24ffce0d8afb734ada6ad71f269020`.
- Native owner: PID1832129 /birth158378202, CPU1,900s plus10s cleanup bound.
- Native result: runtime/transformer-optimizer-horizon-20261108-r1/grading/
  tl3-residual-diagnostics-r1.json under the existing tabdata root.
- Existing ROOT RPC receipts032/034 establish launch and exact result download.
- Five focused CPU tests and static checks passed; the
  [independent source gate](evidence/diagnostic-independent-review.md) passed.

All split/parquet, role-array, checkpoint, sidecar and provenance inputs were
rehashed after scoring. Position partitions and each grid's near/interior pair
exactly cover the official scored mask. The pass completed alongside the
already-running B300 fits; it is not retroactively claimed as a pre-fit gate.

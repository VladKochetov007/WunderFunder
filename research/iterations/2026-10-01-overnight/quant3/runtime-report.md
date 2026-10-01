# Q3 runtime result — VNNI qualifies, AVX2 fails

The fixed `d256 + plain128 seed3 + chunk500128 seed3` dynamic-int8 package passed its VNNI runtime gate. The separately forced AVX2 floating engine and AVX2 integer backend failed their first clean timing, so that fallback was stopped immediately. No target, OOF prediction, WP readout or forecasting fit was performed. The candidate is **not yet quality-qualified**, and its speed is not qualified on an unknown public CPU.

Artifact: `submissions/overnight_quant3/quant3_fixed.zip`, 6,501,798 bytes, SHA256 `48cf940bd1c7e812fe8005a90e3f11fb3199a8940c55e0adbfef00ad1de1f10f`. Original weights, frozen quantizer, one-third coefficients and all source/binary hashes stayed unchanged throughout the empirical stage. The companion `.runtime.json` binds the preparation review, root allocation and every runtime receipt. Full commands, stdout/stderr and all attempts are under `runtime/`; there were no retries or discarded measurements.

| Complete ZIP callback trace | Microseconds/row | Projected39.4M rows | Timed sibling58 busy |
|---|---:|---:|---:|
| VNNI100k run0 | 70.1973 | 46.0962min | 1.9886% |
| VNNI100k run1 | 69.1041 | 45.3784min | 0% |
| VNNI100k run2 | 69.7084 | 45.7752min | 0% |
| VNNI1M | 69.8247 | 45.8516min | .0143% |
| Forced AVX2 first100k | 190.6325 | **125.1820min — fail** | 0% |

All four primary VNNI runs satisfy the unchanged50minute all-pass rule. Measurements used one thread pinned toCPU18 in the CP311/glibc2.36 scorer environment, with sibling activity computed only from summed timed regions. Whole-window activity is also retained. Peak RSS was104,152KiB (about101.7MiB); import/model startup was .02895–.02962seconds. The entire reference/correctness/timing stage took155.5seconds under its900second/4GiB bound. CPU18 and58 were released afterward.

Every one of the1.4M timed callback outputs, including the slow AVX2 run, was byte-identical to the fixed reference generated from old frozen member predictors and independent original-order `fmaf` reduction. Fresh-ZIP correctness passed all100k real fixture rows on v3 and the prescribed2560-row samples on AVX2 and v2; scalar/AVX2/VNNI backends agreed exactly within each compiled engine. AVX2 and v3 also agreed exactly on these real checks. The v2 sample differed by at most .002253 and passed the declared numerical screen. Q3 versus same-composition FP16 passed its raw-prediction numerical screen; the largest reported per-stream/per-target relative RMSE was .007965. These numerical checks use no targets and do not establish the WP noninferiority gate.

The AVX2 timing failure is clean, large and retained; it is not explained away as noise, and no second fallback timing, kernel modification or AVX2 OOF identity audit followed. This package never substitutes a different member set or FP16 model when VNNI is absent. The public CPU/runtime gap therefore remains a material delivery limitation. The1M trace repeats the existing five20k sequences with fresh sequence IDs and resets; no claim is made about an uninterrupted1M state trajectory.

Only root may allocate the two registered quality comparisons after this report. The tested fixed-three all-h128 fallback remains available separately.

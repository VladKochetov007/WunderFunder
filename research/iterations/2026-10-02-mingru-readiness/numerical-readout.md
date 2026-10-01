# Numerical readiness, generated data only

CPU PASS; original exact-zero `exp(log_g(x))` derivative0 becomes1 on the actual Torch2.14.0+cu130 environment, with forward value0.5 unchanged. The selected positive activation branch defines derivative1 atzero. Literal FP64 recurrence and full-model vector-Jacobian checks passed. Original source preserved.

GPU PASS. B2/T2000 minGRU AMP-vsFP32 output relativeL2=0.000551, ordinary shallowGRU=0.000526; input-layer gradient relativeL2=0.07664 versus0.10701 for GRU. These are generated comparisons, not trained-quality equivalence. FP32 scan-vs-step max output discrepancy1.79e-7; full-model31-row literal VJP max discrepancy9.31e-10.

Long-retention FP32 scan/native stepping can diverge: gate=-17 over20k rows has max difference0.02205 at state about20; scan-vsFP64 max0.000458. Accumulation/rounding explains this generated case; trained retention distribution and deployment drift remain untested. No arbitrary equality veto, and no substitution of the failed direct-affine scan.

B1/T20000 full-gradient generated AdamW step completed in3.246seconds, peak952,046,592 allocated bytes,finite parameters,no scaler overflow. This does not establish B32 full-sequence feasibility and does not change chunk2000 forecast recipe. GPU scope peakowned1180MiB, released;CPU andGPU checking used no real data or forecasting metric.

Native compile attempt01 failed in our split-step adapter before numerical/benchmark execution; preserved and mechanically repaired in native-v2. Native serving decision is separate.

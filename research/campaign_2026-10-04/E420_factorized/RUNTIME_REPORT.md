# E420 clean native factor timing

Real-model factor parity was established before timing (see REAL_PARITY_REPORT.md). Five alternating trials per case, three20k streams each, complete callbacks including causal features, two scoring outputs, CPU15, oneCPU/2GiB guard. Clean v3 is primary. Timing v2 remains an overlap diagnostic: a16.79-second CPU preflight occurred within its measurement window.

| Case | Median us/row | Median kcycles/row | Native streamed weights |
|---|---:|---:|---:|
| factor_f32 |112.871603|458.198425|4,674,112 bytes|
| effective_dense_f32 |184.747202|763.034143|8,606,272 bytes|
| original W384 q16x |99.063995|396.069816|4,337,856 bytes|

Factor code reduces cycles39.95% versus its exact effective-dense FP32 reference; cycles are15.69% greater than the original q16x implementation. Original q16x predicts the unapproximated parent, so that comparison includes different weights/arithmetic. No deployment speedup over q16x was demonstrated. Import-only dense matrices are freed; factor storage is a subset of total weights, not an additional amount. Scratch16,716 bytes, recurrent state3,072 bytes.

No fit or full scientific score comparison has occurred. The rank128 SVD model is a zero-update approximation. Parameter/state geometry motivates training factorized models and matching budget controls, but provides no score guarantee. No public runtime or score claim.

Raw evidence: ../E420_factorized_runtime_v3/result.json, timing_trials.jsonl, sealed draft/code/input identities, resource-guard attempt e420-factorized-runtime-v3.

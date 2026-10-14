# R4 sealed profile specifications

[Preflight](preflight.spec.json) is byte-identical to R3. The [FP32](qat_deploy_f0_s4_100pct_fp32_profile_r1.spec.json), [uniformW4](qat_deploy_f0_s4_100pct_uniform_w4_profile_r1.spec.json) and [mixedW4/W8](qat_deploy_f0_s4_100pct_mixed_w4w8_profile_r1.spec.json) candidates each retain100 actual updates, pinned source/cache/ancestry and8GiB reservation. Only the three copied cache paths leave the deferred-symlink list and receive full runner identity checks. The [packet README](../README.md) explains the actual-producer check and remaining execution limits.

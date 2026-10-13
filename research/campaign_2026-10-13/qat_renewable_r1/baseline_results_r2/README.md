# Native CP5 full-fold baseline results R2

Completed 2026-10-13T17:05:39.035957+00:00. [Frozen manifest](MANIFEST_R2.json) binds the clean-T3 H256 CP5 F0 seed3 parent, native library, source files, archives and full evaluation membership. Every case covers 468 groups, 141 components and 808,863 selected rows after complete 20,000-row callback trajectories. Full-sequence reset/replay passes.

| Deployment | t0 | t1 | Official WP | Callback seconds |
|---|---:|---:|---:|---:|
| [fp32](FP32_R2.json) | 0.674375728 | 0.687932577 | 0.681154152 | 1572.85 |
| [native_q8](NATIVE_Q8_R2.json) | 0.674365731 | 0.687905068 | 0.681135399 | 198.48 |
| [packed_q8](PACKED_Q8_R2.json) | 0.674365731 | 0.687905068 | 0.681135399 | 239.61 |
| [recurrent_w4](RECURRENT_W4_R2.json) | 0.673006189 | 0.687571025 | 0.680288607 | 228.01 |
| [input_w4_recurrent_w8](INPUT_W4_RECURRENT_W8_R2.json) | 0.672537953 | 0.687576478 | 0.680057216 | 290.73 |

Native Q8 and exact packed-Q8 predictions have the same complete-file SHA-256, `a303b73885515090926169c7df3fa45b11e943fbfb6c1a7af3ff15e670f777be`. Exact transport therefore preserves this full-fold endpoint. FP32 is only 0.000018753 above native Q8; the larger measured degradation comes from W4 in this parent. Mixed precision does not automatically improve quality: its unchanged-parent point score is below uniform W4 here.

Packed Q8, uniform W4 and mixed archives are 1,086,824, 678,883 and 882,966 bytes. All three expand to 1,071,552 resident weight bytes and retain 2,048 bytes of FP32 recurrent state. This is transport compression without INT4 execution or resident-memory savings. Callback times were collected during other work on the local CPU; they are not a controlled speed comparison or an EPYC serving guarantee.

The [independent baseline bootstrap](../../../unified/reviews/QAT_DEPLOYMENT_COMPONENT_BOOTSTRAP_METHOD.md) records native Q8 minus uniform W4 +0.000846792 with an interval spanning zero. Full-fold results are exposed-F0 unchanged-parent diagnostics, not QAT effects, independent fitting-seed evidence or a public forecast. The matched trained comparisons are separately graded in the [campaign](../README.md). Predictions and component statistics remain at `artifacts/qat_renewable_r1/deployment_frozen_r2/<case>/` on physical local storage; each result records their hashes. H-QAT-W4-20261013; E-QAT-DEPLOYMENT-20261013-R1.

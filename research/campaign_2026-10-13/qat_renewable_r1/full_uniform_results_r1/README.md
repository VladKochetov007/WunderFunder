---
kind: result
status: complete
created_utc: 2026-10-13T18:31:00Z
hypothesis_ids: [H-QAT-W4-20261013]
experiment_ids: [E-QAT-WEIGHT-100PCT-F0S3-20261013-R1]
---

# Full-data uniform-W4 QAT endpoint

One epoch from clean-T3 H256 CP5 F0 seed3 completed 746 batches and 14,920 updates. Native W4 evaluation covers all 468 exposed F0 groups, 141 components and 808,863 official scored rows, with exact reset/replay.

WP is **0.681362789**, versus **0.680741171** for the matched floating continuation exported to W4: point delta **+0.000621618**. Target scores are 0.674999560 and 0.687726018. Independent paired uncertainty and fit/export audit are pending; this is one fitting seed and an exposed fold, without a public-score forecast or submission promotion.

- [Result](recurrent_w4.json) retains scores, callback timings, state replay and prediction hash. Concurrent local callback timings do not establish a CPU speedup.
- [Deployment manifest](MANIFEST_R1.json) binds exact selected checkpoint, archives, native library and evaluation membership; [inventory](RESULT_INDEX_R1.json) identifies original prediction storage.
- [Fit completion](../FULL_UNIFORM_FIT_COMPLETION_R1.json) binds history, checkpoint hash and measured resource peaks; [native execution](../NATIVE_100PCT_UNIFORM_QAT_R1.json) retains its command and admission vector.
- [Matched full-data PTQ controls](../full_float_results_r1/README.md) and [allocation](../ALLOCATION_100PCT_QAT_R1.json) define the comparison. The mixed-precision arm remains in progress.

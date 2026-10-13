---
kind: result
status: complete
hypothesis_ids: [H-QAT-W4-20261013]
experiment_ids: [E-QAT-WEIGHT-100PCT-F0S3-20261013-R1]
---

# Full-data mixed W4/W8 QAT endpoint

The fixed raw epoch-0 model completed 14,920 updates from clean-T3 H256 CP5 F0 seed3. Native evaluation replayed all468 exposed F0 groups,141 components and808,863 official scored rows, with exact reset/replay.

WP is **0.681420269**, versus **0.680660270** for matched mixed PTQ: point delta **+0.000759999**. Target scores are0.675029550 and0.687810989. The [independent grade](../../../unified/reviews/QAT_100PCT_MIXED_W4W8_MATCHED_GRADE_R1.md) and fit/export audit pass; the approximate paired 95% component interval is [−0.000013193, +0.001655967]. This is one fitting seed and an exposed fold; no public-score forecast or submission promotion follows.

- [Result](input_w4_recurrent_w8.json) retains full scores, callback timings, replay and prediction hash. Local concurrent timing is not a CPU speedup claim.
- [Deployment manifest](MANIFEST_R1.json) and [inventory](RESULT_INDEX_R1.json) pin exact checkpoint, quantized archives, native library and physical prediction storage.
- [Fit completion](../FULL_MIXED_FIT_COMPLETION_R1.json) and [native execution](../NATIVE_100PCT_MIXED_QAT_R1.json) record source/resource identities and exact command.
- [Matched full-data controls](../full_float_results_r1/README.md) and [uniform-W4 endpoint](../full_uniform_results_r1/README.md) complete the initial weight-only scale ladder.

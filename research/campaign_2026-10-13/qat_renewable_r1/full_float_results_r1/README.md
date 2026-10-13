# Full-data floating continuation: native control endpoints

The fixed raw epoch-0 checkpoint followed one full-data continuation from clean-T3 H256 CP5 F0 seed3, with 746 complete-sequence batches and14,920 updates. Each native endpoint replays all468 exposed F0 evaluation groups,141 components and808,863 official scored rows. These are controls for the ongoing full-data weight-only QAT cohort, not independently confirmed submission winners.

| Format | WP | Target0 | Target1 |
|---|---:|---:|---:|
| Native Q8 |0.681605037|0.675158015|0.688052058|
| Uniform W4 PTQ |0.680741171|0.674331983|0.687150359|
| Mixed W4/W8 PTQ |0.680660270|0.674184445|0.687136095|

- [Native Q8 result](native_q8.json), [uniform W4 result](recurrent_w4.json), and [mixed result](input_w4_recurrent_w8.json) preserve complete scores, reset/replay, callbacks, source manifest and prediction hashes. Local concurrent callback timings do not establish final CPU speedups.
- [Frozen deployment manifest](MANIFEST_R1.json) binds exact parent checkpoint, all archive bytes, split/membership, native library and source. [Result inventory](RESULT_INDEX_R1.json) records the original artifact locations; large predictions remain on physical approved storage.
- [Explicit import/export verification](../EXPORT_IMPORT_IDENTITY_VERIFICATION_R2.json) reproduced the complete manifest and all archives byte for byte with frozen Python module paths.
- [Execution sequence](../NATIVE_100PCT_FLOAT_SEQUENCE_R1.json) and [terminal receipt](../NATIVE_100PCT_FLOAT_SEQUENCE_RECEIPT_R1.json) bind all three successful workers.

H-QAT-W4-20261013 / E-QAT-WEIGHT-100PCT-F0S3-20261013-R1. The teacher/initializer ancestry remains the full fitting membership. Bootstrap uncertainty, QAT comparisons and fitting-seed variation are separate later evidence; no public forecast is implied.

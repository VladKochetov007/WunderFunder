# Conditional first-cell score admission V4

V4 carries the selected-GPU ownership correction forward: it reads cgroup metadata for every GPU owner, reads command lines only for same-UID owners, and rejects a campaign cgroup regardless of UID. It also binds the corrected exporter/scorer V2 pair: the flat-support `EXPORT_BINDING_V1.json` byte/SHA, all four V2 scorer modules, V2 dependency manifest, and independent `PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR` review.

The V2 binding differs from V1 only in its two flat support-relative paths. The scorer core differs only in the expected binding length and digest; contract, prediction, and CLI files remain byte-identical to V1. V4's conditional template still has unbound root-owned export acceptance and exact-stop pins, so the renderer remains closed until root supplies those receipts. This packet authorizes no B query, staging, admission, scoring, or quality result.

`test_score_admission_v4_binding.py` checks the local binding/review/module/plan joins and confirms the V4 body differs from V3 only in the new V2 source pins. It does not run the renderer, import the scorer or WNN source, inspect `/proc`, query a GPU, or access model/data/cache/prediction payloads.

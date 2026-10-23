# Local checks — scorer V3

The new metadata-only fixture passed three tests. It verifies the exporter V3 binding bytes/SHA, confirms the four changed root/AllThree receipt fields and equality of every other binding field, proves the V2-to-V3 scorer AST delta is limited to two binding constants plus the fixed binding filename/evidence-root literal, and checks dependency-manifest pins.

The fixture did not import or execute the scorer, exporter, WNN, or source tree. It read only small local binding JSON files and source/manifest text. No B/network actions, model, target, cache, dataset, prediction, or checkpoint payload reads occurred. V1/V2 scorer suites were not rerun. The subsequent independent pair review is [EXPORT_SOURCE_REVIEW_V3.json](../first_cell_evaluation_export_v3/EXPORT_SOURCE_REVIEW_V3.json), 7,284 bytes / SHA-256 `ee53ce06a6119152866e8f39801e7575e59fb42bc09515d102db4c8df5d9c37a`, status `PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR`; it did not execute tests or access B or payloads.

The source-only pair gate is complete; any future execution still requires root-owned fresh admission and actual export/stop gates.

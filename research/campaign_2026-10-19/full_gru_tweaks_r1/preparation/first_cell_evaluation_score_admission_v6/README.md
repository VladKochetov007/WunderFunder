# Conditional first-cell score admission V6

V6 is a local-only conditional template for a later single CPU score admission. It binds the reviewed exporter/scorer V3 pair, exporter V2's exact unit, and the `evaluation_v2` evidence root while preserving the V5 score unit and score directory. The score worker and result contract remain the frozen V3 scorer contract.

The renderer checks the exact V3 producer/consumer review bytes (7,284 bytes, SHA-256 `ee53ce06a6119152866e8f39801e7575e59fb42bc09515d102db4c8df5d9c37a`) and validates its actual `reviewed_files` array: the eleven reviewed paths, byte counts, hashes, the one nested review status, and the complete producer/consumer join flags. It rejects duplicate or missing file rows and altered pins. The focused fixture also proves the V6 body is an AST-exact adaptation of the V5 body after normalizing only the exporter/scorer V3 pins, evidence-root basename, and exporter unit.

The export acceptance and exact-stop receipt paths, byte counts, and hashes remain null in `SCORE_ADMISSION_BINDING_PLAN_V6.json`. The packet has no rendered or bound remote body and makes no B query, staging, launch, prediction or target read, score, or quality claim. Any later admission requires a separate root-owned binding of the completed export's acceptance and exact-stop receipts and a fresh independent review.

`test_score_admission_v6_review_metadata.py` reads only small local metadata and source files. It does not call the renderer's `render()` or `main()` functions, load the scorer, read arrays, or contact a remote host.

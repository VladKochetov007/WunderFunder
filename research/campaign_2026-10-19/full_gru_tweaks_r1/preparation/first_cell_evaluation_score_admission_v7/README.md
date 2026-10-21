# Conditional first-cell score admission V7

V7 carries forward the frozen conditional score packet and adds one source-integrity gate: it enumerates the canonical source tree and requires the exact 167-file manifest closure before hashing any source file. The inventory rejects unlisted files (including extra `.py`, `.so`, and `.pyc` entries), symlinks, non-regular entries, and unlisted directories. The existing per-file stable no-follow hashes run only after this check.

The V7 body differs from V6 only in `inventory_exact_source_files` and `check_source_tree`; [`V7_INVERSE_PROOF.json`](V7_INVERSE_PROOF.json) records the AST comparison with those two functions removed. The focused fixture exercises a tiny exact source tree, extra files, file and directory symlinks, the inventory-before-hash ordering, the V6 gap note, and the unchanged body outside the inventory functions. It does not load the scorer or read model, dataset, cache, checkpoint, prediction, or target payloads.

[`V6_SOURCE_INVENTORY_GAP_NOTE_V1.json`](V6_SOURCE_INVENTORY_GAP_NOTE_V1.json) preserves the V6 limitation and its exact body pin. The V6 admission packet itself remains unchanged.

The prior actual export-v2 unit failed during checkpoint verification after model loading and before prediction outputs. Its failed records remain authoritative history. The future export acceptance and exact-stop are unbound in [`SCORE_ADMISSION_BINDING_PLAN_V7.json`](SCORE_ADMISSION_BINDING_PLAN_V7.json), and V7 retains the inherited export-v2 identity only as a historical conditional anchor. The V7 renderer was not called; no score body was rendered or admitted, no unit was queried, and no score or quality result exists.

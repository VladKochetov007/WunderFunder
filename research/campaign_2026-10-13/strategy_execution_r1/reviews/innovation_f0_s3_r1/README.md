# S09 independent recipe review — R1

This folder independently audits the frozen F0 seed-3 causal-innovation/moments-control recipe in `innovation_f0_s3_r1/`. The verdict is **FAIL R1; no profile or fit launch**: the actual homogeneous cache metadata reaches the candidate validator and is rejected on its one-family schema before target indexes or cache arrays are opened. The failure is operational and gives no scientific result.

- [Independent review](INDEPENDENT_RECIPE_REVIEW_R1_20261013.md) explains what passed, the exact consumer blocker, the target scalar and intended row mapping, and the limits of the check.
- [Machine-readable receipt](review_receipt.json) binds candidate/packet identities, first rejection and bounded resource evidence.
- [Compact actual-cache metadata and validator result](CACHE_METADATA_R1.json) records the sidecar hash, declared array metadata and an `np.load` spy showing no array reads before rejection. Payload hashes/shapes remain metadata claims, not independently verified facts.
- [Source/membership preflight](VERIFY_R1_RESULT.json), [parent ancestry review](ANCESTRY_R1_RESULT.json), and [command review](COMMAND_REVIEW_R1_RESULT.json) preserve the independent checks and exact staged command hashes. Their bounded runners are [`verify_s09_r1.py`](verify_s09_r1.py), [`verify_ancestry_r1.py`](verify_ancestry_r1.py), and [`verify_commands_r1.py`](verify_commands_r1.py).
- [Reproducible metadata/consumer checker](inspect_cache_metadata_r1.py) loads only the JSON sidecar and hash-bound split manifest, reconstructs the frozen split membership object, and calls the candidate validator with array loading forbidden.
- `verify_s09_r1.py`, `verify_ancestry_r1.py`, and `verify_commands_r1.py` are the read-only scripts behind the corresponding receipts.

The fit prior and pair matching pass the reviewed checks, but no training, GPU profile, prediction, grading, or quality update is supported. A successor needs actual-producer compatibility and row/order/scale checks before it can be considered runnable.

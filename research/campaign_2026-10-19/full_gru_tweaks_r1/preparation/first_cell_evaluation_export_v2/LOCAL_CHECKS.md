# V2 local checks

The V2 correction is isolated to two binding path strings and the exporter's exact binding byte-count/SHA-256 constants. The inverse proof verifies byte-for-byte reconstruction of the frozen V1 binding and exporter after reversing those changes; the 27,569-byte, 167-file source manifest is identical to V1.

Checks run with the literal maintenance virtual-environment executables:

- `./.venv/bin/ruff format` on the V2 exporter, focused fixture, and inverse-proof script.
- `./.venv/bin/ruff check` on the same three files: passed.
- `./.venv/bin/python -B prove_v2_inverse_bytes.py`: passed; wrote `EXPORT_V2_INVERSE_PROOF.json` (2,553 bytes, SHA-256 `4b5e1d8fe72bc2f26601573ade2a33146221928328e7d9dd17fb3f7461209acc`).
- `./.venv/bin/python -B -m unittest test_export_v2_binding_metadata -v`: three new metadata-only fixtures passed. They compare only the V1/V2 source and binding bytes plus the captured small preflight metadata JSON; they do not import the exporter or read model, cache, dataset, or prediction payloads.

The captured metadata source was independently pinned as 1,509 bytes, SHA-256 `40a55d9e1c2d251627e6208775c0736699e3f1aa0378cdbc2ba7b126acd126c2`. It records a failed V2 preflight before evidence-directory creation, an absent new unit, no mutation, and the exact three flat support-file stats. This local source packet has not been staged or executed remotely. No prediction export, scoring, or quality evaluation was run, and no unchanged fixture suite was replayed.

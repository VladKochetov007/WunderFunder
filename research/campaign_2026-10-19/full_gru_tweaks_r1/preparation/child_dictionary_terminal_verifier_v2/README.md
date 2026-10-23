# Dictionary terminal verifier body V2

This version isolates one worker-launch compatibility fix from the frozen V1 packet. The V1 body tries to read `Path(__file__)` with a 250,000-byte cap, although the root sends the submission through stdin and the exact body is larger than that cap. V2 derives from the pinned V1 rendered body and makes only two changes: it removes that self-read, and replaces the worker-reported body hash/stat field with a root-owned source-review and raw-transport authority record. The worker does not read or claim a digest for its own body.

`build_dictionary_terminal_body_v2.py` checks the complete V1 source byte count and SHA-256 before making the two anchored replacements. It writes the new `REMOTE_SUBMISSION.py` create-only and has a check mode that compares the complete output bytes against a fresh deterministic derivation. The V1 files remain unchanged. The new body retains the same six-file checker closure, native proof, predecessor/source/cache/parent gates, payload limits, and CPU resource limits as V1.

The two focused AST regressions verify that the V2 body has no worker `__file__` self-read or self-cap and that its admission record contains root source-review/raw-transport authority without a worker body-hash claim. Ruff and these local checks pass; see [LOCAL_CHECKS_V2.json](LOCAL_CHECKS_V2.json) and [V2_BINDING.json](V2_BINDING.json).

This is a local repair candidate only. Independent review of the exact V2 bytes and a root-owned raw-transport binding remain required before any support admission. No B action, SSH, unit launch, fit, evaluation, unchanged suite, or model/data/cache/dictionary payload read occurred for this version.

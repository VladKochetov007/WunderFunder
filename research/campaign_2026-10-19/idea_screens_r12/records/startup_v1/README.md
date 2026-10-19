# R12 transport preparation

The successful local assembly uses the pinned original R11 v1 assembler and inline template with recorded R11-to-R12 substitutions. `ASSEMBLY_SPEC.json` binds the fixed R3 sample, 32 source pins, 26 runtime modules, six canonical pins, the reviewed R7 allowlist, and the fixed resource envelope. `REPLAY.json` records exact source hashes, helper pins, substitutions, receipt identity, and generated script hashes.

Generated scripts remain in the private local runtime directory named in `REPLAY.json`; no large script copies are stored here. `LOCAL_IMPORT_SMOKE.json` records the cold import check with reader sentinels. The earlier adapter based on R11 attempt 2 failed locally before assembly and remains preserved in `replay_transport.py` and `root_inline_r12.py`.

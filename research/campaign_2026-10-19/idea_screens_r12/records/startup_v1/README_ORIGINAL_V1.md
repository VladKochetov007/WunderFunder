# R12 startup transport record

The first local adapter based on the R11 attempt-2 assembler failed before producing an assembly receipt. Its adapter and inline template remain at `replay_transport.py` and `root_inline_r12.py`. `ASSEMBLY_SPEC.json` is the current validated spec and was used by the successful replay; it is not a failed-only artifact.

The successful replay uses the pinned original R11 v1 assembler and inline template with occurrence-counted R11-to-R12 substitutions. `REPLAY.json` records the input pins, helper pins, substitutions, receipt, and generated script hashes. `LOCAL_IMPORT_SMOKE.json` records the bounded 26-module import and zero-read sentinel check. Generated scripts live only in the private runtime directory recorded in `REPLAY.json`; this file contains no script copies. The local assembly performed no remote actions.

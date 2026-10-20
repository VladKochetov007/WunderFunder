# Cold-import proof history

`PARENT_FIT_TERMINAL_COLD_IMPORT_V1.json` is retained as a preliminary record only (current SHA-256 `fb066cca6a68b0f9c48ee6a142addd4cde471f008de74506f708a2381d7b77e6`). Its 25-module proof did not call the S5 sibling's lazy `_canonical_tensor_identity` helper, so it omitted `wnn.run_identity`. The temporary V1 runner was not retained; the V1 receipt records its script digest but not a recoverable script path. Treat V1 as superseded and incomplete evidence.

Use [V2](PARENT_FIT_TERMINAL_COLD_IMPORT_V2.json) instead. Its [runner](parent_fit_terminal_cold_import_proof_v2.py) calls the sibling launch parser and tensor identity helper on synthetic inputs, and records all 26 source-manifest-matched WNN module paths and hashes. V2 is a local import/read-sentinel proof only; it does not inspect parent model/data payloads or establish terminal acceptance.

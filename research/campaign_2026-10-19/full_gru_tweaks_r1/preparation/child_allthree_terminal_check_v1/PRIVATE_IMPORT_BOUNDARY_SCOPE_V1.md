# AllThree private import-boundary scope

`private_import_boundary_proof_allthree_terminal_check_v1.py` runs a narrow local CPU import check for the exact seven-code closure in `dependency_manifest.json`. It uses the literal repository interpreter path `.venv/bin/python`, copies only the seven pinned Python files to a fresh temporary directory, checks source and temporary-copy hashes before import, and verifies all seven post-import hashes and module origins.

The child runs with `-E -B -I`, verifies `sys.dont_write_bytecode is True` and `sys.pycache_prefix is None`, and checks that the fresh private directory has no `__pycache__` or `.pyc` files before import. The Python `open` audit hook is installed before importing local Torch. It counts and rejects opens in the enumerated worker/B, model/state, dictionary, cache, and data/split paths. After Torch baseline setup, the child records a separate `torch.cuda.is_initialized()` observation, guards the enumerated CUDA action and device-transfer APIs, rejects any of the seven private checker module names already present in `sys.modules`, imports the private checker, and calls only `assert_helper_closure()`. It checks bytecode controls and CUDA initialization again and confirms no `wnn` source modules loaded. The initialization-observation counter is separate from CUDA action attempts.

The resulting `PRIVATE_IMPORT_BOUNDARY_PROOF_V1.json` is limited to private-code origin/integrity and this focused Python open/CUDA boundary. It does not call `verify_terminal`, a CLI or `main`, training or evaluation, source representation code, or model/state acceptance. It does not prove exhaustive OS-level file access controls, worker behavior, or the worker Torch runtime; local Torch 2.14.1 remains distinct from the required worker runtime `2.14.0+cu130`.

The recorded result is schema `fg19.child-allthree-terminal-import-boundary-proof.v1`, status `PASS_NARROW_ALLTHREE_PRIVATE_IMPORT_BOUNDARY_V1`: seven files matched before and after import and their origins point into the fresh temporary private directory; no private module was preloaded; bytecode writing was disabled, the cache prefix was null, and no private `.pyc` existed before or after import; protected open attempts were 0; enumerated CUDA action attempts were 0; the separate `is_initialized` observer returned false before and after (2 observations); and zero WNN modules loaded. The harness is 23,177 bytes with SHA-256 `6f4318aaa3eea9c88acc693b3d7a822a6b0e9ae61450202f6dedf79ef5bceda8`. The final receipt is 12,712 bytes with SHA-256 `1456ca8e051aa44ca39c836e1d6a531deecd0cac87307e1be2aae982aba7ba93`.

Two earlier successful observations are retained as non-final evidence. `PRIVATE_IMPORT_BOUNDARY_PROOF_V1_INITIAL_NONFINAL.json` (12,428 bytes, SHA-256 `14fee1f2998651124c965c7f9b742fd2f9ba20bb147917e239331011bbbb7c30`) did not record the explicit `-E -B` and pre-import bytecode-absence checks. `PRIVATE_IMPORT_BOUNDARY_PROOF_V1_BYTECODE_GUARD_NONFINAL.json` (12,672 bytes, SHA-256 `6a19af04419d27efeb85240bd17ab930c55b572ef40f9eafa916d5c12ea7e9b1`) included those bytecode checks but did not reject preloaded private module names. The root preparation must bind only the final receipt above.

Run the harness from the repository root with the literal path:

```sh
.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_check_v1/private_import_boundary_proof_allthree_terminal_check_v1.py
```

The harness refuses to overwrite an existing proof receipt. The root preparation record must bind the final harness and proof receipt bytes and hashes after the run.

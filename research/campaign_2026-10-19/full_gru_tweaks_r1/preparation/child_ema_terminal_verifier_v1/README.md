# EMA terminal verifier admission packet

This folder contains a frozen local one-shot CPU admission body for the completed EMA fit. It does not stage files, contact B, run the verifier, accept tensor integrity, stop the native unit, or add a fit.

The body is bound to EMA run `fg19_child_ema_f0_s1001_fit_v1`, native unit `computer-vision-homework-fg19-child-ema-fit-v1.service`, invocation `da5baf82d22b4f02890e184193b8caf7`, and the exact original root launch acceptance. It requires the native capture's strict integer `MainPID=0`, `ExecMainCode=1`, and `ExecMainStatus=0`, then asks the unchanged EMA-specific checker to perform the full restricted-load integrity check. It does not require the native unit to be stopped before verification.

The private closure contains exactly five code files: the frozen EMA checker and its four reviewed helper dependencies. The body creates only `B/research_tmp/fg19_child_ema_verify_v1`, containing those code files, the source manifest, root launch acceptance, typed terminal evidence, and native raw capture. Original EMA admission/launch, parent/dictionary/profile/CONTROL receipts, profile files, and cache acceptance/stop are verified read-only at their existing paths.

Fresh source167 content, split, cache metadata and array stats, registered data aliases/stats, ancestor checkpoint and dictionary stats, empty population and ownership, unused one-time headroom reservation, project scratch free space and device, support-unit absence, and create-only output absence are gated before the first write. The one support unit is capped at 50% CPU, 1.4 GB MemoryHigh, 1.5 GB MemoryMax, zero swap, 600 seconds, hidden CUDA, and one BLAS thread. The runner uses the canonical project Python with `-E -B` and `WNN_PROJECT_ROOT=B/data`.

The direct-human exact dictionary-path decoder exception is already recorded in the linked authority receipt and is required before mutation. The body preserves `weights_only=True`, all existing model/state/metadata/aggregate caps, and the exact one-field Path restriction. A submit timeout or unknown result is durably retained and must never be retried.

The local binding receipts pin the original EMA root acceptance, terminal capture, frozen checker closure, eight existing ancestry receipts, three profile metadata files, and cache receipts. Local work read only small metadata and code; model, training-state, cache-array, dictionary, and dataset payloads were not read or hashed. No checker, model/source/profile suite, or remote operation ran here.

The focused fixture `test_child_ema_terminal_verifier_v1.py` passes five tests against actual stored metadata and negative body-binding cases; this does not execute the terminal checker or load model/state payloads.

`REMOTE_SUBMISSION.py` SHA-256: `03879ea709e48a93a7c11fcfd204d2dabb5c2039a48dde872177860bb157b35f` (503514 bytes). See `ROOT_BINDING.json`, `DEPENDENCY_MANIFEST.json`, and `LOCAL_CHECKS.json` for exact pins and scope.

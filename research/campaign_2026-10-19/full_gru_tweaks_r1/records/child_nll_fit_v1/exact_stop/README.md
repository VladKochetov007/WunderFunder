# NLL exact stop

This record documents the one exact stop performed after the durable NLL terminal acceptance. `TERMINAL_ACCEPTANCE.json` was durably written with SHA-256 `d94457b11ce22a368cdc93df02e473c9a82cde6ddebd26cc6957bafc143750c5` before the stop at 2026-10-21 01:54:50.161896 UTC. The stop command returned 0 and targeted only:

- Native fit: `computer-vision-homework-fg19-child-nll-fit-v1.service`, invocation `c833c613fb92426981ab255a5faf4452`.
- Successful bounded verifier: `computer-vision-homework-fg19-child-nll-verify-v1.service`, invocation `91948e17c5e74cefa32ab2a546a1b46c`.

Both were observed before stop as loaded/active/exited with `MainPID=0`, `ExecMainCode=1`, and `ExecMainStatus=0`. Afterward both were not found/inactive/dead with `MainPID=0` and empty invocation IDs. The captured before/after campaign population maps were empty. Failed, foreign, and other units were excluded and left untouched. No model, state, dictionary, cache-array, dataset, source, or trainer payload was read or hashed for the stop record.

The canonical stop JSON and raw response are byte-identical (3,140 bytes, SHA-256 `c63be79b27d131acb0a38d2ad641fa7d72344cc344c26ab94ca77faf3b4a74f1`). The root binding records the acceptance-before-stop ordering and the exact native/verifier unit and invocation IDs.

- [EXACT_STOP.json](EXACT_STOP.json)
- [STDOUT.raw.json](STDOUT.raw.json)
- [TRANSPORT.json](TRANSPORT.json)
- [ROOT_STOP_BINDING.json](ROOT_STOP_BINDING.json)
- [Accepted terminal result](../TERMINAL_ACCEPTANCE.json)
- [Terminal capture observation](../capture/OBSERVATION.json)
- [Verifier capture README](../capture/README.md)
- [Human-scoped Path decoder authority](../../../preparation/child_control_terminal_verifier_v3/HUMAN_PATH_COMPAT_EXCEPTION_AUTHORITY_V1.json)

This is process lifecycle evidence accompanying the accepted integrity/provenance result. It does not report predictive quality.

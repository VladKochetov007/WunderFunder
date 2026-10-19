# R12 records navigation

The current local transport assembly is recorded in [`startup_v1/REPLAY.json`](startup_v1/REPLAY.json) and [`startup_v1/ASSEMBLY_RECEIPT.json`](startup_v1/ASSEMBLY_RECEIPT.json). The replay binds the R12 spec, 32 source pins, 26-module runtime map, six canonical pins, and the private generated script hashes. It pins the reusable original R11 v1 assembler and inline template plus the three reviewed R10 helpers. Generated scripts are kept in the private runtime directory named by the replay and are not copied into this repository.

[`startup_v1/LOCAL_IMPORT_SMOKE.json`](startup_v1/LOCAL_IMPORT_SMOKE.json) records the cold 26-module import and zero-read sentinel check. [`startup_v1/README.md`](startup_v1/README.md) gives the local preparation summary. `startup_v1/ASSEMBLY_SPEC.json` is the validated spec used by the successful assembly.

[`startup_v1/PREPARED_README.md`](startup_v1/PREPARED_README.md) is the launch-time byte copy of the packet README (SHA-256 `777ccb735928751bbe668affe26d18ada9500bc15ff176bc86f3f4a52b93dcc8`); its relative links intentionally resolve from the R12 packet root.

An earlier local-only adapter based on the R11 attempt-2 assembler failed before assembly. Its artifacts are preserved at [`startup_v1/replay_transport.py`](startup_v1/replay_transport.py) and [`startup_v1/root_inline_r12.py`](startup_v1/root_inline_r12.py); neither is the successful transport source.

The successful launch transport record and raw output are [`LAUNCH_TRANSPORT.json`](LAUNCH_TRANSPORT.json), [`launch.stdout`](launch.stdout), and [`launch.stderr`](launch.stderr). Exact terminal evidence, including raw capture and stop output, is in [`terminal_capture/`](terminal_capture/). [`RESOURCE_RECONCILIATION.json`](RESOURCE_RECONCILIATION.json) separates unit properties from helper timings and stage memory observations; [`RESOURCE_METADATA_REVIEW.json`](RESOURCE_METADATA_REVIEW.json) records the scope checks. The unit exited successfully and was exact-stopped after raw capture validation passed. The model archive was stat-inventoried and its producer SHA retained without transferring its payload.

The top-level [resource reconciliation](../RESOURCE_RECONCILIATION.json) and [resource review](../RESOURCE_METADATA_REVIEW.json) are byte-identical navigation aliases of the records in this folder.

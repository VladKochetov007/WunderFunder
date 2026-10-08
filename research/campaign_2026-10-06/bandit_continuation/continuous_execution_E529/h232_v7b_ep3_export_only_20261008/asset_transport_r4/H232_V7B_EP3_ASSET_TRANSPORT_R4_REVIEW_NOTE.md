# H232 V7B EP3 asset transport R4

The R3 `SaveKernel` rejection had no response body, so its cause remains unconfirmed. Its embedded source measured 11,875,142 bytes. Kaggle's [kernel source-size discussion](https://www.kaggle.com/discussions/questions-and-answers/103948) gives a plausible size-limit explanation. This R4 candidate replaces only that large wrapper with a 9,227-byte bootstrap; it does not change the reviewed payload, checkpoint, worker, or scientific recipe.

The bootstrap searches recursively beneath `/kaggle/input` for exactly one mounted payload under the pinned dataset slug. It checks the mount manifest's private flag, dataset ID, filename, byte count, and SHA-256, then verifies the ZIP digest, exact 17-member set, safe regular-file paths, and every member digest before importing and calling the unchanged `export_worker.main`. It records a transport receipt in `/kaggle/working` before launching the worker. The kernel metadata pins private dataset version 1 and keeps internet disabled.

The offline builder takes an explicit local payload path, checks its full size and SHA-256 against the R3 build receipt, then stages the asset with the existing `scripts/kaggle_assets.py` helper. It does not require the rejected 11.9-MB R3 kernel source to be present or re-embedded.

The private dataset `vladkochetov07/h232-v7b-ep3-payload-b9e14e8509df` reports `ready`. A read-only download of its asset and manifest matched the local hashes exactly: payload 8,321,018 bytes, SHA-256 `b9e14e8509dff6abc18f4e3aaffe4af031d78883e91fb98f28fb4a845ed1490e`; manifest 358 bytes, SHA-256 `d777e41460190e1355f8d5e48a41ee78c2a15fd538bc3c4768c505371ffb0578`. The asset manifest identifies version 1 and marks the dataset private. No second dataset version was created.

The frozen payload still has 17 members, with checkpoint SHA-256 `aaa731da1692ee5083c80296fd665b0a60815c9dc18430457f702cf779c2b829` and source archive SHA-256 `f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4`. The embedded root source review is unchanged. No training, profiling, scoring, or prediction evaluation was run locally.

Five focused offline tests passed, including a fake nested Kaggle mount, byte tampering, member verification, unsafe ZIP rejection, and the version-pinned small-source packet. Ruff format/check and `git diff --check` passed under the 1-CPU, 1.5-GB, 120-second local cap. Full details and hashes are in `H232_V7B_EP3_ASSET_TRANSPORT_R4_VERIFICATION_RECEIPT.json`.

Candidate identity is `vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r4-asset`, title `H232 GRU384 V7B EP3 Export 20261008 R4 Asset`. Its inventory was `Not found` before any push. This packet is pending independent root review and fresh kernel quota admission. No kernel push was attempted or authorized here; the prior R2/R3 packets and rejection receipts remain preserved.

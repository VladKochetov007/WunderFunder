# D02 identified R3 control transport

D02 R3 keeps the frozen R2 Muon-versus-loss recipe and changes its control
transport from the incomplete R4 fit-only packet to the root-reviewed R5
training contract and exports. R2 remains preserved. The exact R5 kernel is
`vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1`, version 1, and the
training-contract SHA-256 is
`5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7`.

The D02 runtime finds exactly one mounted `out/training_contract.json` with that
digest, then checks the five shared prefit, normalization, sample-order,
schedule and profile-input artifacts by their contract-bound sizes and hashes.
The loader also requires the audited 350/28/468 membership and two-epoch,
220-update AdamW OneCycle settings. It never refits or rewrites R5 controls.
The source mount adds the R5 kernel to the parsed-cache kernel and existing
fold-0 asset sources.

The packet records the reviewed R5 checkpoint, prediction and shared-role
identities, plus the fixed exposed fold-0 grade. WMSE scored 0.606001923,
Huber delta 1 scored 0.601239135, and their predeclared equal blend scored
0.605480201 across 468 exposed sequences and 141 components. This is a
development screen; it does not forecast public transfer or reject Huber.

`root_admission.json` keeps `push_allowed` and `conditional_fit_allowed` false.
This is a source and transport candidate only. No dataset upload, kernel push,
device profile or fit is authorized by this packet.

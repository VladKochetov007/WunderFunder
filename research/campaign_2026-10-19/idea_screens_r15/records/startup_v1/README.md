# R15 startup transport record

This packet binds the frozen R15 V2 manifest, independent implementation review, test receipt, and exact transport scripts. The source README launch-time bytes are preserved in `PREPARED_README.md`; they are not edited by this packet.

`ASSEMBLY_SPEC.json` records the fixed R15 unit/source/output identity, exact 25-module runtime map, six canonical WNN pins, exact three-file R7 source allowlist, R10 resource envelope, and the four transport extras. `LOCAL_IMPORT_SMOKE.json` records a private B-shaped import check: all 25 actual owned-prefix modules matched the manifest, Torch imported with CUDA hidden and uninitialized, and NumPy/Parquet/Torch payload-reader sentinels were untouched.

`ASSEMBLY_RECEIPT.json` records the byte-exact 29-file bundle and generated script hashes. `REPLAY.json` binds the private generated scripts to the pinned R13 assembler and Torch-capable inline template with occurrence-counted R15 identity/root/sentinel patches. Generated transport scripts are kept under the private runtime directory; no remote action was performed in this task.

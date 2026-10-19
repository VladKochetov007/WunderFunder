# R11 startup assembly

This packet adapts the reviewed R10 resource/staging/launch body for the frozen R11 assignment. It does not connect to B or launch work. The assembler verifies the R10 helper byte pins and R11 preparation manifest, applies occurrence-counted text patches, compiles the generated local scripts, and writes an assembly receipt.

assemble_transport_v1.py expects the final R11 PREPARATION_MANIFEST.json, IMPLEMENTATION_REVIEW.json, TEST_RECEIPT.json, and README.md. Its 24-module runtime map must exactly match ASSEMBLY_SPEC.json and every referenced source must be pinned by the manifest. The three canonical WNN pins are semantic checks only and are never staged. Bundle files are bounded at 16 MB each and 64 MB total. R7 transport is restricted to the four listed model-math/panel primitives. All inputs and outputs are create-only.

The initial local assembly is preserved as V1 and superseded because its review metadata was corrected. The final local assembly uses the 24-module wording in ASSEMBLY_SPEC_V2.json and is recorded by ASSEMBLY_RECEIPT_V2.json plus ASSEMBLY_V2_REPLAY.json. Its generated private scripts are:

- /Users/nb/Documents/wunder/runtime-20261018/eda19-r11-v2/r11_remote.py — one-time ownership, resource, source-staging, and launch body.
- /Users/nb/Documents/wunder/runtime-20261018/eda19-r11-v2/r11_capture.py — identity-substituted R10 V2 success-only raw metadata capture (systemctl show --all retained).
- /Users/nb/Documents/wunder/runtime-20261018/eda19-r11-v2/r11_stop.py — exact successful-exited-unit stop helper.

root_inline_v1.py checks the canonical source pins, verifies each runtime source path/hash before imports, installs cold-import sentinels for Torch, NumPy, and PyArrow readers, imports the exact manifest closure, then replaces itself with the fresh R11 CLI process. The campaign owner must independently review the generated assembly receipt and perform the actual admission, capture-byte verification, and exact stop. The capture/stop helpers intentionally retain R10's successful-terminal preconditions; an unsuccessful unit is preserved for owner diagnosis rather than automatically retried.

Validation performed for this packet is the 24-module cold-import/read-sentinel smoke, local syntax, pinned-source patch compilation, generated-script byte verification, and Ruff. No remote command or R11 job was run while preparing it.

# R16 startup packet

This packet binds the frozen R16 NumPy-only source bundle and the reviewed R10-derived staging, admission, capture, and stop operations. No remote action was performed by this local preparation.

- [Prepared README snapshot](PREPARED_README.md) preserves the source-prepared README bytes.
- [Assembly spec](ASSEMBLY_SPEC.json) binds the R16 V1 unit/source/output, accepted R3 sample, exact runtime map, resource caps, and helper pins.
- [B-shaped local import smoke](LOCAL_IMPORT_SMOKE.json) passed on the 22-file transport fixture: all nine declared R10/R16 modules resolved to fixture files, NumPy load/memmap sentinels were untouched, Torch/PyArrow stayed absent, and fresh `python -m idea_screens_r16.run_probe --help` returned 0. No sample arrays or models were opened.
- [Assembly receipt](ASSEMBLY_RECEIPT.json) contains exact transport file sizes and SHA-256 values plus generated helper hashes. [Replay record](REPLAY.json) links the pinned reusable operations and private generated scripts.
- [Attempt 01 spec](ASSEMBLY_SPEC_ATTEMPT_01.json) is preserved as a local-only rejected draft: it omitted the R3 metadata-fixture bundle root. The corrected final spec includes that root without adding R3 to the runtime import map.

Generated scripts and the B-shaped fixture remain under `/Users/nb/Documents/wunder/runtime-20261018`. The remote body is the sealed 22-file payload; no arrays, models, or checkpoint payloads are transported. Root owns SSH staging, admission, launch, capture, and stop.

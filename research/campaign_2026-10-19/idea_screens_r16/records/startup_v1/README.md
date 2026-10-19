# R16 startup packet

This packet binds the frozen R16 NumPy-only source bundle and the reviewed R10-derived stage/admission/capture/stop operations. No remote action was performed here.

- [Prepared README snapshot](PREPARED_README.md) preserves the source-prepared bytes.
- [Assembly spec](ASSEMBLY_SPEC.json) binds the R16 V1 unit/source/output, accepted R3 sample, exact runtime map, caps, and helper pins.
- [B-shaped local import smoke](LOCAL_IMPORT_SMOKE.json) passed on the 22-file transport fixture: all nine declared R10/R16 modules resolved to fixture files, NumPy load/memmap sentinels were untouched, Torch/PyArrow stayed absent, and fresh `python -m idea_screens_r16.run_probe --help` returned 0. No sample arrays or models were opened.
- [Assembly receipt](ASSEMBLY_RECEIPT.json) records exact transport sizes/hashes and generated helper hashes. [Replay record](REPLAY.json) binds the final corrected scripts and the pinned base operations.

The first local-only spec omitted the R3 metadata-fixture root and is preserved as [attempt 01](ASSEMBLY_SPEC_ATTEMPT_01.json); the assembler rejected it before constructing payloads. A first generated body is also preserved in [attempt 01 receipt](ASSEMBLY_RECEIPT_ATTEMPT_01.json) but was never launched. The final corrected body places only staged support on `PYTHONPATH`, consistent with the zero-canonical-pin runtime closure.

Generated scripts and the B-shaped fixture remain under `/Users/nb/Documents/wunder/runtime-20261018`. No arrays, models, or checkpoints were transported. Root owns SSH staging/admission/launch/capture/stop.

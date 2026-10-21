# V4 local checks

The V4 stdin body is 246,596 bytes, SHA-256 abe3251d9d65200b8e0c745db287e76f77e2dcb2a6524f3de7cd2f31c0cfa9ec. The renderer emitted ADMISSION_SOURCE_BINDING_V4.json; ADMISSION_V4_INVERSE_PROOF.json is 4,488 bytes, SHA-256 5bd746a4e53eb946dd3568b5a5513d2b6ed2abbe99aa2310d434d2228c399fb1, with status PASS_EXACT_V3_BODY_INVERSE_OUTSIDE_CHECK_FIT_RECEIPTS.

Checks used the literal maintenance virtual-environment tools:

- Ruff format check passed for the V4 template, renderer, rendered body, inverse proof, and focused fixture.
- Ruff check passed for the renderer, rendered body, inverse proof, and focused fixture. The unrendered template intentionally omits globals imported by the pinned stdlib helper source; the complete rendered body is linted.
- The new test_admission_v4_actual_receipts.py focused fixture passed 3/3 tests. It runs the extracted receipt-check function with exact local receipt metadata and fake checkpoint-stat responses; it does not inspect checkpoint contents or import project/model code.
- prove_admission_v4_inverse.py verified that restoring only the frozen V3 check_fit_receipts function makes the V4 body byte-identical to V3, and verified the five fixed payload joins.

No historical fixture or model/source suite was replayed. No B query, staging, admission, SSH, or execution occurred in this local V4 packet.

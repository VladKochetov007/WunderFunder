# E488 renderer v5 independent review

**Disposition: GO for canonical report generation from the already-graded E488 JSON receipts.** This review covers only the exact v5 bytes below; v4 remains preserved.

- `render_report_v5.py`: `fa00ae17e4b533387b1812989466cbeb9b744c325e8c6f35fb0aea21eb2e1ba7`
- `test_render_report_v5.py`: `f6150865e399cd7d0f950901ba3d651767c0585303f1d1958f7a15ffb5664ae3`

The v5 diff fixes the report’s NaN arithmetic label by reading the raw reproduction receipt’s validated `max_abs_error` field. It binds both the raw arithmetic receipt and the separate root provenance wrapper to the same score/export hashes and exact 24/27/54 inventory. It labels the duration interaction `E3−E1`, not epoch 4. The confidence plot keeps percentile bounds intact and draws each point independently, including when the interval does not contain it. Rendering reads saved JSON and hashes the selected-export file; it never loads the NPZ payload, Parquet rows, or checkpoints.

The synthetic end-to-end fixture builds score, proof, fit/grade attempt, raw reproduction, and root-wrapper JSON under a temporary root, then runs `render` against only those files. It asserts the displayed error is finite, checks the root receipt bindings, and verifies output creation. The interval-outside-point test passes as well. All 6 synthetic tests, Ruff, and `py_compile` passed under `scripts/limited` with 1 CPU and 2 GiB. This review did not inspect actual scores/exports or generate the canonical report.

# R15 preparation logs — accepted V3 source

V3 is the accepted source version used for the single bounded run. The [cold import smoke](cold_import_smoke.json) verified the 25-module runtime closure and read sentinels without loading sample arrays, model weights, or predictions. The final synthetic checks and Ruff outputs are preserved in this directory; [TEST_RECEIPT.json](../../TEST_RECEIPT.json) summarizes them, and [PREPARATION_MANIFEST.json](../../PREPARATION_MANIFEST.json) pins the source closure.

The actual R15 endpoint and capture provenance are documented in [RESULTS.md](../../RESULTS.md). Earlier preserved source versions are indexed in [SOURCE_HISTORY.md](../SOURCE_HISTORY.md).

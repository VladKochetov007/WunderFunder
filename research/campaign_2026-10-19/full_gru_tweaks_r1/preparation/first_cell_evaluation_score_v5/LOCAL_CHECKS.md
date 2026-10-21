# Local checks — scorer V5

The new focused fixture passed three tests. It calls the actual `_read_export_bundle` function with the exact local V4 binding and canonical 167-row source manifest, plus synthetic completion metadata. All eight compact API pins pass; mutations to bytes, SHA-256, source path, and row keys fail at the normalized API-row gate. An inverse source-delta assertion confirms that the contract change is limited to that comparison block and that core, prediction I/O, and CLI modules remain byte-identical to V4.

Ruff `check` and `format --check` passed for the four scorer modules and the new fixture. The fixture imports only this packet's scorer contract/core and standard-library test helpers; it does not import WNN or load project/model modules. No dataset, cache, checkpoint, prediction, or target payload was read; no B action, exporter run, actual score, or metric claim occurred.

The exporter V4 and its binding remain pinned at 66,790 bytes / SHA-256 `87b6d3626bdafe6353865d95b1d8116bd187cb4da46888815658db6fc192738e` and 15,796 bytes / SHA-256 `4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd`. Their 27,569-byte source manifest is pinned at SHA-256 `a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621`. Independent producer/consumer source review is pending.

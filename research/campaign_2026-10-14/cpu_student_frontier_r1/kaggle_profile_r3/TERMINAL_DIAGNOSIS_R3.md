# R3 terminal diagnosis

R3 ended ERROR before a candidate worker started. The isolated input preflight passed, and the cache materializer copied all three files as ordinary files under the registered cache root: 1,910,723,660 bytes with exact expected hashes. This repairs the original R2 containment failure.

The unchanged runner then rejected the first candidate input with `Deferred mounted input size mismatch`. Expected and observed size were both 1,685,600,128 bytes. The actual failing predicate was `path.is_symlink()`: candidate specs still listed the three copied files in `WNN_DEFERRED_IDENTITY_PATHS`, whose contract requires mounted symlinks. No optimizer update, profile or quality result followed. This is a producer/consumer handoff error, not a quantization result.

[Terminal collection](TERMINAL_R3.json) and [collected evidence](collected/README.md) preserve the session log, successful preflight/copy receipts and failing candidate identity receipt. The worker-side session took about506 seconds; portal RUNNING duration included additional provider time and was not a fitting duration. Latest GPU quota was75,200.311 seconds used of108,000, with0 reserved;32,799.689 seconds remained. TPU remainedQUEUED.

The [R4 successor](../kaggle_profile_r4/README.md) removes exactly those three materialized paths from candidate deferred lists. The runner consequently applies its normal full hash/size validation to them. Preflight remains unchanged, all other mounted paths retain the existing deferred treatment, and the strict scientific consumer remains unchanged. The original R3 source, specifications and receipts stay frozen.

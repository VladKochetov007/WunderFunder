# E481 complete-fit root checks before scoring

The full matrix finished:18 fits,2052.4498627 GPU-worker seconds, exit0, failureNone and teardown_verifiedTrue. Parent OOM-kill counters did not increase. Observed peak GPU398458880 bytes; worker RAM1678028800 bytes. FIT_RECEIPT.json is a byte-exact copy of the completed runtime manifest, SHA256 `9ddb8e20506a996187bba1235ce9c0251bb724939643d543351d04ccf781812a`.

Root checked every18 checkpoint byte hash against the receipt, each474-update/61058-parameter declaration, fit_onlyTrue/test_rows_loadedFalse, and all72 sampled telemetry records. Each chunk_start equals `(update_index % 79)*256`; losses/state maxima are finite and sampled state maxima are≤1.000001. This sampled check does not prove the full-trajectory invariant or gradient stability.

Independent complete-receipt/model validation passed on the unmodified v2 scorer before the scoring spec is sealed; its separate GRADER_V2_INDEPENDENT_REVIEW.md binds the actual finished attempt, parent/source/preflight hashes, strict checkpoint state, identity buffers and complete18-arm role/order/telemetry proof. No test data was loaded by these root receipt checks. Scoring uses the unchanged independent scorer and an immutable new output directory.

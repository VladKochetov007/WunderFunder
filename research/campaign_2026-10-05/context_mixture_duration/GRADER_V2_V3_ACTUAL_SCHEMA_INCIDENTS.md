# E488 grading contract failures, versions 2–4

These verify-only executions failed before test row-group decoding. No checkpoint predictions or official scores were produced. All original sources, reviews and actual traceback logs are retained.

Version 2 assumed src/wnn/evaluation.py belonged to the sealed training code. It is a grader-only dependency: the actual training import chain does not use it. The same draft expected design and admission documents under inputs, although the immutable fit spec bound them under code. Version 3 checks each complete, exact role inventory and separately pins the evaluator dependency; it does not change the training seal.

The actual version 3 then rejected head_update_rms as nonnumeric. The actor records this as a family dictionary, not a scalar. The source review and partial fixtures had not exercised the entire actual fit-only evidence schema. A separately versioned correction must validate the whole fit-only manifest and every telemetry field before an actual retry. The fixed training outputs remain unchanged; no scientific run is repeated.

Actual logs: ROOT_VERIFY_ONLY_V2.log and ROOT_VERIFY_ONLY_V3.log. This is a grader implementation failure, not a negative mixture result. Test evaluation remains gated on a complete successful provenance verification.

Version4 then added a condition that every actual-shape probe cell must have an active clipping factor. The registered design requires active clipping for the two native-reference parity checks, while the twelve real-data cells require finite changed updates with the registered clipping operator. A seed271841 GELU update correctly had factor1. Version5 removes the invented all-cell activation requirement, validates every factor against min(1,1/(norm+1e-6)), and retains active clipping in both native-reference parity checks. Its integrated actual fit/probe/attempt fixture and root CLI verify-only passed with the test loader blocked. The final CPU grade uses a separate immutable spec binding all24 checkpoint artifacts and the complete reviewed source closure. Original v4 log remains ROOT_VERIFY_ONLY_V4.log.

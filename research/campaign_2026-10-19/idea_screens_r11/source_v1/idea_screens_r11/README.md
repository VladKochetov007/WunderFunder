# R11 preparation

R11 is a fixed-step comparison of uniform row sampling with one initial-gradient importance sampler on the accepted R3 panel. The complete protocol, estimator, limits, and interpretation boundaries are in [DESIGN.md](DESIGN.md); the pre-implementation design bytes are preserved in [DESIGN_INITIAL.md](DESIGN_INITIAL.md). The pre-code review and selection rationale are in [DESIGN_REVIEW.json](DESIGN_REVIEW.json) and [SELECTION_HISTORY.json](SELECTION_HISTORY.json).

The frozen runtime is a six-module R11 package entry/closure (`__init__`, `importance_math`, `importance_training`, `probe_execution`, `result_payload`, `run_probe`) plus `test_probe.py`. Its transitive runtime imports are pinned in [PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json), including the R1/R2/R3/R7 helpers and canonical WNN files. The exact prior manifest and test receipt bytes are retained as `PREPARATION_MANIFEST_INITIAL.json` and `TEST_RECEIPT_INITIAL.json`.

Synthetic CPU checks and a source-only 24-module import smoke passed. This preparation did not read the real R3 arrays, load a checkpoint, train or evaluate a model, or run a remote job. Root/operator owns the B-shaped import/read-sentinel gate, actual bounded execution, capture, and result interpretation. The bounded job must run only after the frozen source and launch packet pass independent review.

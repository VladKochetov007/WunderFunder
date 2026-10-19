# R11 preparation

R11 is a fixed-step comparison of uniform row sampling with one initial-gradient importance sampler on the accepted R3 panel. The complete protocol, estimator, limits, and interpretation boundaries are in [DESIGN.md](DESIGN.md); the pre-implementation design bytes are preserved in [DESIGN_INITIAL.md](DESIGN_INITIAL.md). The pre-code review and selection rationale are in [DESIGN_REVIEW.json](DESIGN_REVIEW.json) and [SELECTION_HISTORY.json](SELECTION_HISTORY.json).

The frozen runtime is a six-module R11 package entry/closure (`__init__`, `importance_math`, `importance_training`, `probe_execution`, `result_payload`, `run_probe`) plus `test_probe.py`. Its transitive runtime imports are pinned in [PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json), including the R1/R2/R3/R7 helpers and canonical WNN files. The exact prior manifest and test receipt bytes are retained as `PREPARATION_MANIFEST_INITIAL.json` and `TEST_RECEIPT_INITIAL.json`.

The first V1 launch stopped at the import-inventory gate: `python -m` executes the CLI as `__main__`, so the named `idea_screens_r11.run_probe` key is absent from `sys.modules`. It stopped before output initialization, array reads, or model work. V1 bytes and its failure record are preserved. V2 accepts the alias only when both `__spec__.name` and `__file__` match the pinned CLI module, and requires the exact `computer-vision-homework-eda19-r11-v2.service` cgroup.

The synthetic CPU checks and a source-only 24-module import smoke passed. The V2 checks also exercise the real package `python -m ... --help` entry and a read-sentinel inventory test. This preparation did not read the real R3 arrays, load a checkpoint, train or evaluate a model, or run a remote job. Root/operator owns the V2 B-shaped import/read-sentinel gate, actual bounded execution, capture, and result interpretation.

## Completed result

The V2 bounded run completed the fixed pair. The importance arm scored 0.45491058 pooled WP versus 0.48819247 for uniform (delta -0.03328189), with both target differences negative. See [RESULTS.md](RESULTS.md) for the score, component/drop-one summaries, timing scope, and limits. This closes this exact fixed-initial-gradient variant on the exposed R3 panel; no retuning or second model was run.

- [Independent compact audit](RESULT_AUDIT.json) reconstructs fixed endpoint/component/drop-one scores without model or array replay.
- [Execution records](records/README.md) preserve raw bytes, the pre-data V1 failure and corrected V2 capture/stop. [Preserved V1 bundle](source_v1/README.md) inventories immutable source and metadata.
- [Resource reconciliation](RESOURCE_RECONCILIATION.json) and [metadata review](RESOURCE_METADATA_REVIEW.json) are byte-identical root aliases of records/attempt_v2 originals. Their preserved V1 metadata separates the earlier message-label clarification from raw values; no compute was repeated.
- [Continuation](CONTINUATION.md) carries the current CPU-only exploration boundary and prevents replay of completed work.

The original scalar audit is preserved as [RESULT_AUDIT_V1.json](RESULT_AUDIT_V1.json); the current audit narrows one memory-scope sentence without changing any numerical result. Systemd MemoryPeak remains available even when terminal cgroup files are absent.

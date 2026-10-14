# Independent student widths and native transport R1

This implementation adds independent encoder, recurrent, and readout widths plus an optional second readout layer to the student model, trainer factory, NumPy reference, native callback, and weight exporter. The legacy equal-width one-hidden-layer model retains its state-dict names and WNNWTS01 format. Synthetic tests and storage probes verify the code path; no model was fitted or ranked.

- [Implementation report](REPORT.md): behavior, supported native configurations, checks, and limits.
- [Test receipt](TESTS_R1.json): focused model/native checks and broader compatibility suite.
- [Style check receipt](STYLE_CHECK_R1.json): corrected new import/format issues and records pre-existing Ruff findings outside changed lines.
- [Build provenance](BUILD_R1.json): compiler/runtime versions, native build artifact, and command outcomes.
- [Preserved Rust test failure](CARGO_TEST_FAILURE_R1.json): the first full unit run exposed a wide-head dispatch regression; records the exact command, pre-fix source hash, and correction.
- [Source manifest](SOURCE_MANIFEST_R1.json): SHA-256 inventory for the implementation and tests.
- [Storage probe](STORAGE_PROBE_R1.json): exact matrix-count and FP32/packed-Q8 byte accounting for deterministic E12/R8/M6/L2/head2 weights.
- [WNNWTS02 consumer receipt](NATIVE_TRANSPORT_R1.json): complete 20,000-row synthetic raw-feature callback consumed from the exported binary in FP32 and portable Q8. The observed Q8 deviation is numerical, not bitwise parity.
- [Synthetic WNNWTS02 fixture](SYNTHETIC_WIDTHS_WNNWTS02.wbin) and [20k raw-feature fixture](SYNTHETIC_RAW_20K.f32): deterministic, hash-bound transport inputs; these contain no competition rows or trained weights.

The test sequence covers legacy state-dict/output compatibility, model construction and optimization, format selection, native callback/chunk/causality/reset behavior, dynamic Q8 storage, and compatibility modules. Native serving currently accepts only the existing 211-input, linear-encoder, GRU, GELU, two-output student path; other experimental preprocessors/cells remain outside this exporter. The benchmark example reports one run on synthetic inputs and is not a trained-model speed claim.

Root integration retains the 8,960,000-byte synthetic raw fixture and 16,468-byte weight fixture at their recorded physical paths, excluded from main Git. The original actor commit `e02b358430ac385fbbe513e3b355b4d5c87a5886` preserves its submitted state; hashes and generation provenance remain in the receipts above. No fitting checkpoint or private dataset is copied into version control.

[Root integration receipt](ROOT_INTEGRATION_R1.json) records preserved ledger history, physical fixture storage, and a guard preventing width overrides from using the frozen equal-width comparison contract. [Final merged tests](ROOT_INTEGRATION_TESTS_R3.xml) pass34 checks, including default parameter/output compatibility, source-qualified ordering and optimizer grouping; [preceding33-check run](ROOT_INTEGRATION_TESTS_R2.xml) is retained. The initial test invocation named a nonexistent test file and ran no tests. No fit, score or target-host runtime claim.

The root [initial no-test receipt](ROOT_INTEGRATION_TESTS_R1.xml) preserves an invocation error: a requested test path did not exist, so pytest exited4 before running tests. The indexed R2/R3 runs used the actual paths and passed; this failure is not a model or implementation result.

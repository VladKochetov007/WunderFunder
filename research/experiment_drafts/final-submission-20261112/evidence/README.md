# Actual engineering receipts

- `final_cpu_report_r1.json` is the extracted feature-only baseline report, including FP32/Q8 drift and runtime projections.
- `NATIVE_CPU_STATUS_117.json` retains the original process/report collection with source hash, one-core owner and exit-zero status.
- `PRECISION_COMPONENT_DIAG_R2.json` confirms actual AVX512 Q8 VNNI dispatch and compares W18-only with the full callback.
- `MEMBER_CORE_R2.json` records each isolated recurrent core's time; stripped tiedlead cores exclude the nonlinear encoder.
- `KERNEL_PANEL_R1_NATIVE.json` records all four existing Q8 mode timings and bitwise output comparisons, including negative results.
- `GELU_BUILD_STATUS_132.json` records the actual successful offline build and unit test after correcting the Python shared-library loader environment; earlier attempts remain in the task archive.
- `CACHE_ORDER_BUILD_STATUS_135.json` records the combined native build and three passing residual tests.
- `OPTIMIZED_QUICK_TIMINGS_R1.json` records the first isolated quick timings for GELU-only and combined cache-order changes.
- `GELU_CHECK_FAILURE_RPC_138.json` preserves the first checker's parser-field error before inference. It is not a failed model parity result.
- `CACHE_ORDER_PARITY_R2.json` records the repaired checker, actual bitwise member/callback parity and1.2705x paired speedup on two full sequences.

No report in this folder scores all-labeled final weights for scientific selection.

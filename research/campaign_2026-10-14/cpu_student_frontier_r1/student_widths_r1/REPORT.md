---
kind: experiment
status: complete
created_utc: 2026-10-14T01:40:21Z
updated_utc: 2026-10-14T01:40:21Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-STUDENT-WIDTHS-CODE-20261014-R1]
evidence_paths: [TESTS_R1.json, BUILD_R1.json, STORAGE_PROBE_R1.json, NATIVE_TRANSPORT_R1.json]
---

# Independent student widths and native export: implementation evidence

## Question or claim

Can the student implementation train and serve independently allocated encoder, recurrent, and readout widths, with either one or two hidden readout layers, while preserving the legacy default model and archive interface? This delivery implements the code path and validates synthetic behavior. It does not fit weights or establish a quality improvement.

## Mechanism and method

`RNNModel` now takes optional `encoder_width`, `recurrent_width`, `readout_width`, and `head_layers` arguments. The projected feature stream has width E, the GRU state has width R, and the readout maps the concatenated `[state; projection]` of width R+E to M; an optional second GELU readout maps M to M before the existing output layer. The historical `hidden` argument remains the fallback for all three widths and the recurrent-state alias. Existing parameter names—including `head.0` and `head.2`—remain in place, with only the new matrix registered as `head.extra`.

The trainer factory, Muon head scope, manual activation-QAT path, NumPy reference, checkpoint reconstruction helpers, and native callback accept these shapes. The exporter retains WNNWTS01 for equal E=R=M with one hidden head; other supported shapes use WNNWTS02 with explicit dimensions and the optional extra head tensors. The native GRU state stays FP32. Independent readout dimensions use the generic native path unless their width is an integer multiple of the recurrent width, which retains the established merged-wide-head kernels. A synthetic WNNWTS02 file was consumed by the Rust callback benchmark over one complete 20,000-row raw-feature sequence.

The model-matrix count is `211E + 3R(E+R) + 6(L-1)R² + (R+E)M + 2M`, plus `M²` for the second head. Biases are counted separately. The storage probe checks that formula against model tensors and reports both exported FP32 bytes and the current dynamic-Q8 VNNI packed allocation.

## Evidence and interpretation

The final focused suite passed **34 tests**, including equal E/R with M>R on portable Q8; the broader compatibility set passed **107 tests**, skipped **7** cases requiring separately built native quantizer variants or trained checkpoint fixtures, and passed **7 subtests**. That broader set preceded the final M-only dispatch repair; the affected behavior was covered by the focused width tests and the final native unit run. The repaired native Rust unit suite passed **59 tests**, with **1 ignored**; `cargo check --tests --examples` and `cargo build --features extension-module` succeeded. The initial Rust unit failure and its source hash are preserved in [failure receipt](CARGO_TEST_FAILURE_R1.json). See [test receipt](TESTS_R1.json) and [build provenance](BUILD_R1.json).

The scoped Ruff check passes after fixing the new probe's import order; its receipt preserves six unrelated findings on unchanged legacy lines. All three new Python files pass Ruff formatting. The workspace-wide Rust formatting check reports extensive baseline differences across the crate, so no broad formatting-only rewrite was made.

The deterministic E12/R8/M6/L2/head2 storage probe counted **3,564 matrix weights**, matching the tensor count. FP32 matrix weights occupy **14,256 bytes** and all exported tensors **16,432 bytes**. The current native dynamic-Q8 representation allocates **7,168 bytes** for packed matrices and **36,096 bytes** for all streamed weights, including shared-backbone/head payloads; recurrent FP32 state is **64 bytes** and scratch is **1,732 bytes**. These values describe a tiny synthetic shape and implementation-specific padding, not archive size for a trained student.

The native example successfully loaded the 16,468-byte WNNWTS02 fixture and processed all 20,000 raw rows. Its one-repetition measurement was **107.57 μs/row FP32** and **257.75 μs/row portable Q8**. On scored rows, Q8 maximum absolute error was **0.00110** for output 0 and **0.00051** for output 1, with RMS **0.000108** over scored output elements. Q8 was slower in this synthetic run, so this is transport and numeric evidence only. Python/native tests separately cover full/chunk callback, causality and reset replay; the synthetic comparison does not claim exact FP32/Q8 output parity.

Equal-width legacy defaults are bitwise checked against the old expected state dict and outputs. Native-compatible Python callback checks cover E≠R and an optional extra head. The implementation source and checked outputs are hash-indexed in the [source manifest](SOURCE_MANIFEST_R1.json), [storage receipt](STORAGE_PROBE_R1.json), and [native transport receipt](NATIVE_TRANSPORT_R1.json).

## Limits and validity

This is code-only work using deterministic synthetic weights and inputs. There was no fitting, teacher target, grade, or architecture selection. Native export currently rejects unsupported feature maps, non-GRU cells, non-GELU readouts, non-211 input shapes, and privileged/research preprocessing. No candidate package was built for competition submission. The full 20k callback result uses one random-weight member and one synthetic raw sequence on an Intel i9-11900H; its one-repetition timings are not a stable performance estimate or AMD competition-host prediction.

## Decision and next step

Keep the width/depth path available for the root-owned student ladder. The next scientific comparison must fit matched candidates with identical teacher/norm ancestry, source-qualified sequences, optimization budget and evaluation membership, then measure trained export size and complete callback behavior. This code change alone does not alter the ladder ranking.

## Source records

The active shape ladder and scope are in the [CPU student frontier](../README.md); the implementation commits to the existing training and native-serving APIs. The reusable matrix formula is documented in the frontier's [theory and shape accounting](../THEORY_AND_DISCRIMINATING_TESTS_R1.md) and [shape table](../SHAPE_ACCOUNTING_R1.csv). The append-only experiment ledger records this as E-STUDENT-WIDTHS-CODE-20261014-R1 with H-STUDENT-LADDER-20261014 ancestry; it adds implementation evidence only, with no fit or quality claim.

---
kind: result
status: complete
created_utc: 2026-10-14T02:15:26Z
updated_utc: 2026-10-14T02:20:41Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-MINGRU-NATIVE-CALLBACK-20261014-R1]
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/min_gru_native_r1/BUILD_R1.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/min_gru_native_r1/TESTS_R1.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/min_gru_native_r1/RUNTIME_PROBE_R1.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/min_gru_native_r1/SOURCE_MANIFEST_R1.json
---

# Native minGRU callback prototype

## Question or claim

Can the separate native callback implement the root-owned zero-state minGRU, preserve its normalized GELU encoder and skip readout, and match a complete causal 112-raw-feature sequence closely enough to validate serving transport? This is implementation evidence for one small deterministic fixture, not a model-quality or competition-speed claim.

## Mechanism and method

The model loader consumes `mean`, `std`, `inp.*`, `cells.N.input_projection.*`, `head.0.*`, and `head.2.*` tensors from the Python `MinGRUModel` committed at `750701ce96807c2fc6921c7e12512893f8ed33c1`. It infers encoder, recurrent and readout widths, and requires 211 engineered inputs and two outputs.

`MinGruPredictor` accepts the competition's 112 raw values, calls the existing causal `StreamFeatures` builder for the 99 event values, and steps the minGRU on `[raw; event]`. `MinGruStepper` exposes the same model for direct 211-column inputs. Both start at actual all-zero FP32 hidden state; reset clears recurrent state, and the raw callback reset also clears feature-builder statistics.

Each recurrent layer projects only its current input to candidate-then-gate rows. It uses `z = sigmoid(gate)`, `g(c) = c + 0.5` for nonnegative candidates and `sigmoid(c)` otherwise, followed by `h = (1-z) * old_h + z * g(c)`. The encoder and hidden readout both retain GELU. The optional Q8 path reuses existing `DynQ8` packing and leaves the final two-output layer FP32.

The fixture uses seed `9027` for E=12/R=8/M=6/L=2 weights and seed `10062026` for 20,000 synthetic raw rows. The state-dict SHA-256 is `5a4b35a9b4f80ede01d8ec437e88c98bbc3fa8342409feae2b6e226f793437f8`; the raw-row sample SHA-256 is `db877443cda09d8a1385dda477b48346fcc642b74e6866c9366ba20cecc6a86d`. The reference runs on CPU with one thread. The native callback and direct engineered-input stepper use the same deterministic weights and feature sequence.

## Evidence and interpretation

The full native Rust library suite passed 62 tests with one existing ignored test. The focused Python reference/native suite passed 15 tests, covering zero state, direct row stepping, raw 112-to-211 features, full/chunk/reset behavior, Q8 modes and shape rejection. For default causal-feature coverage, the Python fixture expands the same raw rows with `StreamingEventFeatures`, while `MinGruPredictor` uses the existing Rust `StreamFeatures`; the 20k native callback and direct stepper outputs have zero maximum difference on that fixture. This is end-to-end output evidence for the default builder path, not a separately exported per-row event-array proof. Ruff passes on the added Python test and probe files.

Across the 20,000-row probe, exact-activation FP32 native output differs from the Python model by at most `7.45e-8` per prediction and `1.19e-7` per final hidden value; the first 99-row maximum prediction difference is `4.47e-8`. Native full-versus-split-at-row-99 and reset replay are bitwise identical. The direct 211-input native stepper and raw callback predictions have zero maximum difference on this fixture. Fast FP32 differs from exact FP32 by at most `8.20e-8` in predictions and has identical final hidden values.

Fast Q8-VNNI differs from exact FP32 by at most `1.78e-3` in predictions and `1.47e-3` in hidden state. For this tiny shape, Q8 is larger: matrix payload is 18,224 bytes versus 11,936 bytes in FP32, and `weight_bytes` including biases and normalization are 20,120 versus 13,832. Full callback storage including scratch, recurrent state and feature-builder state is 22,924 versus 16,384 bytes.

On one core of an 11th Gen Intel Core i9-11900H, single-pass 20,000-row times were 0.925 s for exact FP32, 0.906 s for fast FP32 and 0.518 s for fast Q8-VNNI. The probe process peak RSS was 652 MB under a 2 GB memory cap. The Python reference's vectorized affine scan took 0.027 s. These are single measurements of random weights at E=12/R=8/M=6/L=2, with different execution structures; they do not establish H256L2 serving capacity or a target-host speed ranking.

## Limits and validity

Weights and inputs are deterministic synthetic fixtures; no fit, actual competition sequence, grade, score or trained export was used. The Python reference is CPU PyTorch 2.14.0 on Python 3.13.9. The extension is a Cargo debug build for local verification, not a release wheel or submission artifact.

The native FP32 and Python predictions are numerically close but not bitwise identical. This reflects different floating-point reduction order and must remain an explicit tolerance claim. Q8 changes predictions and increases storage for the tiny fixture. No quality or resource conclusion transfers from this shape to the planned student ladder.

The repository-wide `cargo fmt --all -- --check` found formatting differences in untouched native sources. The focused diagnostic also includes older, unchanged regions of `native/src/python.rs`; those regions were preserved. The new minGRU binding hunks shown in the preserved output were adjusted to rustfmt's formatting. Exact failed-command hashes and diagnoses are indexed in this folder.

## Decision and next step

The isolated prototype, build provenance, parity tests and synthetic complete-sequence receipt are ready for root integration. Keep the shipped GRU callback unchanged. A separate bounded, random-weight H256L2-GRU versus H384L2-minGRU cost profile can measure a meaningful serving-shape difference; this fixture does not select minGRU for fitting or release.

## Source records

The hypothesis and experiment are `H-STUDENT-LADDER-20261014` and `E-MINGRU-NATIVE-CALLBACK-20261014-R1` in the append-only research ledgers. The Python model and its zero-state reference tests are linked from [the minGRU reference note](../MINGRU_REFERENCE_R1.md). Source hashes and ancestry are in [the source manifest](SOURCE_MANIFEST_R1.json); build, checks, runtime probe and preserved command failures are listed in the [folder README](README.md).

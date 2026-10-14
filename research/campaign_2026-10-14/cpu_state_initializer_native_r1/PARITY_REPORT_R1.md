---
kind: experiment
status: complete
created_utc: 2026-10-14
updated_utc: 2026-10-14
hypothesis_ids: [H-STATE-INIT-20261014]
experiment_ids: [E-STATE-INIT-IMPLEMENTATION-20261014-R1]
evidence_paths: [NATIVE_PARITY_R1.xml, BUILD_R1.json, BUILD_FAILURE_R1.json]
---

# Native causal state initialization R1

## Claim and implementation

The existing callback now accepts an optional FP32 initializer export per ensemble member through `EnsemblePredictor(state_initializers=...)`. The export helper in `src/wnn/state_initialization.py` transports the unquantized constant or MLP tensors. The Rust callback normalizes the already-preprocessed input with the member's original backbone `mean` and `std`, applies the initializer immediately before that member's recurrent step, and leaves its input projection, GRU, head, coefficients and preprocessing unchanged.

All four wrapper modes are supported: `zero`, bounded `constant`, bounded `first_row`, and bounded `prefix`. Prefix collects normalized rows 0–98 in float32 Welford order, then adds its bounded residual once before row 99. Sequence change and explicit reset clear hidden state, position and all prefix moments together. The Q8 VNNI and fixed-code recurrent paths refresh their activation-code cache from the injected FP32 state; the stored native recurrent state remains FP32. Nonzero initialization is rejected with the shared-backbone optimization, whose secondary head does not own a recurrent state. Omitting the new constructor argument preserves the existing callback.

## Measurements

The deterministic synthetic fixture uses a two-layer, width-8 GRU, 211 preprocessed features, 137 raw rows, model seed 197 and input seed 5921. Exact-activation native callback predictions for each of the four modes differ from the Python wrapper by at most **2.98e-8 absolute**; RMS prediction error ranges from **1.30e-8 to 1.34e-8**. Final hidden-state error is at most **5.96e-8**. Zero configuration and omitted configuration are bitwise identical, including final hidden state.

For every mode, replay after a new callback sequence ID is bitwise identical in predictions and final hidden state. Python full/chunk outputs also agree at the existing wrapper tolerance across cuts 1, 37, 98, 99, 100 and end-of-sequence. In the native prefix test, changing raw rows starting at 100 leaves callback predictions for rows 0–99 bitwise unchanged.

With `q8vnni_portable`, `q8x_portable`, and `q16x_portable`, constant, first-row and prefix injection all change the expected part of the trajectory and produce bitwise-identical reset replays. Maximum output deltas versus the zero-state control are 0.03143/0.03133/0.03142 for constant, 0.01587/0.01587/0.01591 for first-row, and 0.01596/0.01585/0.01590 for prefix, respectively. These are initializer effects, not Python-parity errors. On this same small fixture, native fast activations differ from native exact activations by at most **2.98e-8 absolute** (RMS **1.56e-8**). The small activation delta is specific to these random, small weights and does not bound a trained model.

The [JUnit receipt](NATIVE_PARITY_R1.xml) records **25 passing tests**: ten pre-existing wrapper tests and fifteen native export/callback checks. The bounded release extension build, source hashes and binary hash are in [BUILD_R1.json](BUILD_R1.json). The focused Rust reset unit test also passed (`state_init::tests::prefix_stats_reset_with_the_sequence_state`). Ruff format/check passed for the changed Python files. All compile and test commands used one CPU, a 1.5 GB memory maximum and no swap.

The abi3 extension was also imported with NumPy 2.2.6 under Python 3.11.16. Since that profile environment has no Torch, [make_py311_fixture.py](make_py311_fixture.py) exports deterministic wrapper weights, four initializer configurations, raw rows and Python 3.13 reference predictions to [PY311_CALLBACK_FIXTURE_R1.npz](PY311_CALLBACK_FIXTURE_R1.npz); [verify_py311_fixture.py](verify_py311_fixture.py) loads that NumPy-only fixture and exercises each native callback under 3.11. The [receipt](PY311_ABI3_NATIVE_RECEIPT_R1.json) records at most **2.98e-8** prediction error, at most **5.96e-8** final hidden-state error, bitwise reset replay, and bitwise zero-versus-omitted equivalence. This verifies serving import/transport on the profile interpreter without installing Torch.

The first bounded compiler attempt and its exact source hashes are preserved in [BUILD_FAILURE_R1.json](BUILD_FAILURE_R1.json). It failed on a missing PyO3 return lifetime and a retained mutable governor borrow; the next build was made only after those compiler errors were diagnosed and corrected.

## Limits and next owner

These are implementation and transport checks on synthetic H8×2 weights and one short sequence, not initializer fitting, actual-data stability, official quality, runtime ranking, final-package compatibility or submission evidence. Quantized paths are checked for native state-code transport and exact reset replay, not numerical equivalence to the FP32 Python wrapper. The Python 3.11 receipt tests the FP32 callback against exported references; quantized transport presets were tested under Python 3.13. The root owner controls initializer training, matched zero-state comparison and any production package integration.

---
kind: review
status: complete
created_utc: 2026-10-14T01:13:44Z
updated_utc: 2026-10-14T01:13:44Z
hypothesis_ids: [H-STATE-INIT-20261014]
experiment_ids: [E-STATE-INIT-IMPLEMENTATION-20261014-R1]
evidence_paths:
  - research/campaign_2026-10-14/cpu_state_initializer_native_r1/ACTUAL_NATIVE_CALLBACK_PARITY_20K_R1.json
  - research/campaign_2026-10-14/cpu_state_initializer_native_r1/BUILD_R1.json
  - research/campaign_2026-10-14/cpu_state_initializer_native_r1/NATIVE_ACTUAL_SHAPE_20K_R1.json
---

# Independent review of actual-sequence native initializer parity R1

## Question or claim

Does the sealed native state-initializer callback preserve the reviewed row-99 causal boundary, normalization, reset behavior and FP32 reference predictions on a complete real 20,000-row sequence, without changing the frozen recurrent/readout weights?

## Method

I inspected the sealed runner, Python wrapper, Rust `StateInitializer`, callback reset and `FastRnn` hidden-state mutation paths, plus the JSON receipts. I did not rerun the callback. The actual-sequence receipt identifies source `train`, row group 37, `seq_ix=7,598,721`, the full raw feature hash, and the unchanged clean-F0 H256×2 seed-3 epoch-5 checkpoint hash. The runner hash is `8d27fcdf17ac01a65d16f0c242f562501e155d120eb261632e9c36389fd54ed7`; the checkpoint hash is `e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`.

## Findings

The source and execution receipt support the intended causal timing. Prefix mode normalizes the existing 211-feature rows with the frozen backbone buffers, accumulates Welford mean/population variance and last-row context over positions 0–98, and injects the bounded state residual immediately before position 99 is consumed. Native callback sequence changes reset feature history, online normalization, recurrent state and initializer counters/moments. `FastRnn` validates and writes the FP32 state, then refreshes dynamic/fixed activation-code caches before the recurrent step. Zero initialization is bitwise identical to omission in the callback receipt.

The comparison preserves the full 23-output Python checkpoint reference and adapts only the native callback's two-output interface by passing the checkpoint's existing first two head rows, matching the reference diagnostic's `[..., :2]` comparison. No checkpoint tensors are edited. The receipt's native-library hash matches `BUILD_R1.json`; the Rust callback/state and Python wrapper source hashes also match that build receipt.

On this sequence, all four modes remain finite. Exact-activation native versus Python deltas are at most `1.91e-6` for predictions and `2.75e-6` for final hidden state, inside the recorded `rtol=3e-4, atol=3e-5` check. Reset replay is bitwise identical for predictions and hidden state. The exact-versus-fast prediction delta is at most `4.18e-7`. Python full-sequence versus chunked calls cut around positions 0, 1, 37, 98, 99 and 100 differ by at most `1.15e-5` in predictions, also within the receipt tolerance.

## Limits and validity

This establishes callback numerics and state transport for one exposed fitting sequence, one clean-F0 checkpoint and deterministic synthetic initializer parameters. It does not measure a learned initializer, score change, population robustness, production callback latency, quantized parity, or hardware ranking. The companion decay receipt is one finite perturbation on the same sequence and likewise does not establish initializer quality or a Jacobian bound. The first attempt's output-shape mismatch is preserved; the successful repair keeps the checkpoint intact and applies the documented two-output projection.

## Decision

Pass for the narrowly stated actual-sequence parity, row-99 causality, reset and transport checks. No implementation blocker was found in this path. Any initializer fit or quality interpretation requires separate matched training and grading evidence.

## Source records

- [Native initializer campaign record and receipts](../../campaign_2026-10-14/cpu_state_initializer_native_r1/README.md)
- [Actual callback comparison](../../campaign_2026-10-14/cpu_state_initializer_native_r1/ACTUAL_NATIVE_CALLBACK_PARITY_20K_R1.json)
- [Native source/build provenance](../../campaign_2026-10-14/cpu_state_initializer_native_r1/BUILD_R1.json)
- [Actual-sequence state-decay diagnostic](../../campaign_2026-10-14/cpu_state_initializer_native_r1/NATIVE_ACTUAL_SHAPE_20K_R1.json)
- [Synthetic native implementation review](STATE_INITIALIZER_NATIVE_IMPLEMENTATION_REVIEW_20261014_R1.md)

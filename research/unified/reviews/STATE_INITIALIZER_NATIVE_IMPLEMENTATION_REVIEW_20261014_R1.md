---
kind: review
status: complete
created_utc: 2026-10-14T01:01:40Z
updated_utc: 2026-10-14T01:01:40Z
hypothesis_ids: [H-STATE-INIT-20261014]
experiment_ids: [E-STATE-INIT-IMPLEMENTATION-20261014-R1]
evidence_paths:
  - research/campaign_2026-10-14/cpu_state_initializer_native_r1/README.md
  - research/campaign_2026-10-14/cpu_state_initializer_native_r1/VALIDATION_R1.json
---

# Independent review of native causal GRU initializers

## Question or claim

Does the native `EnsemblePredictor` transport the Python initializer modes with the declared normalization, reset and prefix timing, including refresh of quantized hidden-state codes, while retaining an exact zero-state baseline when initialization is omitted?

## Mechanism and method

I inspected sealed source commit `d3686dfd8d9dd5e1860129d3aa1fa892f83dca74` against its Python wrapper, native callback tests, build receipt, deterministic Python 3.11 ABI3 fixture and validation receipt. I checked that only the additive initializer wrapper/export, native callback and GRU state-cache code changed; no checked xteam package or original native binary was modified.

The prefix path normalizes the current 211-feature input with the backbone's stored mean/std, updates Welford moments for zero-based rows 0–98, then adds its bounded residual immediately before row 99. The callback reset clears streaming features, online normalization, recurrent state and initializer position/moments on sequence changes. Replacing or adding FP32 hidden state refreshes the dynamic-int8 or fixed-code activation cache before the next recurrent update.

## Evidence and interpretation

The source hashes in `BUILD_R1.json` match the sealed files and the built 1,156,608-byte extension hash. The reported focused suite has 25 passing Python/native parity and export tests; the Rust suite has 59 passed, one ignored. Exact-activation synthetic H8×2 predictions differ from the Python wrapper by at most `2.98e-8`, and final hidden state by at most `5.96e-8`. The first-99-row causality check preserves predictions through row 99 when later raw rows change. New-sequence replay is bitwise identical for all four modes. Zero-initializer configuration and omitted configuration produce bitwise-identical FP32 predictions and final hidden state.

Portable Q8VNNI, Q8x and Q16x tests exercise constant, first-row and prefix injection, verify a trajectory change at or after the intended rows, and verify exact reset replay. These tests establish state-code transport and refresh on the exercised portable variants, not quantized parity with Python or optimized-target performance. The Python 3.11 fixture confirms the abi3 extension import and callback transport without installing packages; Torch is absent in that fixture environment.

## Limits and validity

All parity and state-transport measurements use deterministic synthetic H8×2 data. No initializer was trained, and there is no actual-data quality result, competition callback result, production runtime measurement, or AMD measurement. Root's separate actual H256×2 complete-sequence check was pending at review time. One hidden-state copy allocation occurs at an injection event to refresh the native activation cache; it is not a per-row allocation and was not benchmarked.

## Decision and next step

No implementation blocker was found for the reviewed native callback/export path. The code and receipts support continuing to root-owned actual-data validation; training admission and any quality interpretation remain separate decisions. Preserve the synthetic and ABI3 scope labels when reporting subsequent outcomes.

## Source records

- [Native initializer implementation and receipts](../../campaign_2026-10-14/cpu_state_initializer_native_r1/README.md)
- [Build/source/binary hashes](../../campaign_2026-10-14/cpu_state_initializer_native_r1/BUILD_R1.json)
- [Native and Python validation receipt](../../campaign_2026-10-14/cpu_state_initializer_native_r1/VALIDATION_R1.json)
- [Python 3.11 ABI3 callback receipt](../../campaign_2026-10-14/cpu_state_initializer_native_r1/PY311_ABI3_NATIVE_RECEIPT_R1.json)
- [Python state initializer and exporter](../../../src/wnn/state_initialization.py)
- [Native callback and state initializer](../../../native/src/state_init.rs)

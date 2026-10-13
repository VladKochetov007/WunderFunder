---
kind: review
status: complete
created_utc: 2026-10-13T11:56:00Z
updated_utc: 2026-10-13T11:56:00Z
hypothesis_ids: [H-TEAM-NATIVE-20261013]
experiment_ids: [E-TEAM-ARCHIVE-REPLAY-20261013-R1]
evidence_paths: [INVENTORY.json, profiles/]
---

# Independent packaged-runtime audit

## Question or claim

Read-only native role `team_score_critic`, gpt-6-luna/max, inspected the actual imported ELF libraries and wrappers within a bounded ten-minute assignment. Root records its public findings here. No fit, model inference, build or source edit was performed by the reviewer.

## Mechanism and method

Used archive/library hashes from [inventory](INVENTORY.json), bounded symbols/disassembly, available native source, and a no-model query of the exact C++ extension. Source descriptions and machine-code observations are distinguished: the inventory does not bind an exact native source commit to the supplied Rust ELF, and the imported pair ZIP has no C++ source.

## Evidence and interpretation

- The pair C++ v3 ELF has GCC13.3 metadata, scalar/AVX2 symbols and `gemv_block128[_prequant]`. Disassembly confirms ZMM `vpdpbusd` in those block kernels. `resolve_backend` includes CPU checks; its VNNI branch returns2. The exact extension returned `quant_backend()==2` on this Intel i9-11900H, whose flags expose AVX512F/BW/VNNI. This is local dispatch evidence only.
- Xteam and E Rust v3 ELFs have identical SHA and build ID prefix `f4b2c387`; wrappers are also identical. Both binaries contain `DynQ8::gemv_vnni`, `vpdpbusd` and AVX2 fallback symbols. Both precision files request `q8vnni`. Available Rust source documents feature checks and exact integer arithmetic, but source-to-binary build provenance is not independently reconstructed here.
- Both wrappers choose their `v3` binary using x86-64-v3 checks, not AVX-512 checks. They can load it and still fall back to AVX2. Existing profile receipts omit selected `.precision` / `quant_backend`; local observations cannot establish competition-container dispatch.
- The pair wrapper uses `half_weights=False`, `dynamic_int8=True`, `quant_backend='auto'`, converts each112-value row to contiguous FP32, resets in Python, and calls one native ensemble `.step`. A `run_sequence` symbol exists but is not the shipped callback.
- The own wrapper binds native `.predict`, which handles DataPoint fields, reset, streamed features and ensemble stepping. Less Python glue is a plausible advantage to measure, not a demonstrated runtime gain.
- The precise C++ quantizer rounding/layout was not established from the archive. Rust's documented formula must not be attributed to the teammate engine. The measured original/conversion prediction differences make this distinction material.

## Limits and validity

No EPYC benchmark or host CPUID observation exists in this audit. The short Intel profiles are unpaired; portal endpoints have different weights. Neither supports an engine winner. No private reasoning transcript is retained.

## Decision and next step

Compare identical pair weights and fixed0.25 coefficients, first one member and then all four, with repeated interleaved full callback trials and actual backend/CPU logging. The single-to-four comparison can expose cache/scheduling effects. Distinguish native-step time, complete callback time and whole-run overhead. Complete official quality checks are required for changed quantization.

## Source records

[Runtime reflection](EPYC_RUNTIME_REFLECTION_R1.md), [archive inventory](INVENTORY.json), [short profiles](profiles/README.md), [native arithmetic report](../../../native/INT8_VNNI_REPORT.md), [Rust VNNI source](../../../native/src/vnni.rs).

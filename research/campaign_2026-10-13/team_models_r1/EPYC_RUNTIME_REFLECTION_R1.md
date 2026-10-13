---
kind: review
status: complete
created_utc: 2026-10-13T11:50:00Z
updated_utc: 2026-10-13T11:50:00Z
hypothesis_ids: [H-TEAM-NATIVE-20261013]
experiment_ids: [E-TEAM-ARCHIVE-REPLAY-20261013-R1]
evidence_paths: [INVENTORY.json, public_results.json, profiles/, FULL_REPLAY_ADMISSION_R1.json]
---

# Runtime interpretation for the competition EPYC 9224

## Question or claim

Which team inference implementation should we use on the reported AMD EPYC 9224 competition host? The documented allocation is **one vCPU, Python3.11, 16 GB RAM, 60 minutes, and a 20 MB ZIP**. CPU identity comes from the project's earlier host observation; we obtained no new host probe. A vCPU does not establish an exclusive physical core, clock, SMT-neighbour load or exposed instruction set.

Retain the working Q8 VNNI path as the baseline and compare the teammate C++ engine on identical weights. There is no established Rust-versus-C++ winner on EPYC. `xteam_h4` already contains our byte-identical `solution.py` and all three Rust libraries: adopting our engine is not an unperformed optimization for that archive.

## Mechanism and method

AMD documents Zen4's AVX-512 VNNI support, 256-bit internal execution paths and 1 MB L2 per core. A 512-bit instruction does not imply twice the throughput of well-scheduled 256-bit VNNI. This also does **not** establish an AVX2 advantage: our AVX2 fallback widens bytes and uses `vpmaddwd`; a fair 256-bit VNNI comparison requires EVEX AVX512VL/BW/VNNI. Keep runtime CPUID dispatch; an archive directory named `v3` alone does not identify the inner kernel.

Our engine quantizes recurrent and first-head matrices to Q8. Dynamic signed activation codes receive a +128 offset for `vpdpbusd`, with row-sum correction and int32 accumulation. Input projection, floating-point gates/state and head epilogues remain separate costs. Q8 reduces weight traffic as well as arithmetic. FP16 **storage** reduces ZIP bytes without changing loaded Q8 compute.

Optimize complete causal callbacks: preprocessing, conversion, recurrent layers, nonlinear heads, output blending and reset. GPU batches and dense-GEMM benchmarks do not measure this path. Shared features, preallocated scratch, packed weights and one-thread execution are suitable defaults; multithreaded BLAS can oversubscribe one vCPU.

## Evidence and interpretation

[Independent binary inspection](INDEPENDENT_RUNTIME_REVIEW_R1.md) confirms VNNI instructions and AVX2 alternatives in **both** runtime families. The exact C++ extension selects VNNI locally; its original profile did not record dispatch, and the competition container remains unobserved. Available source descriptions of Rust arithmetic are not proof of exact source-to-ELF build identity. The C++ quantizer formula is not established by its archive.

User-transcribed completed portal durations:

| Endpoint | Composition | Duration | Interpretation |
|---|---|---:|---|
| Older distilled pair | Two H256 students | 24:57 | Historical compact endpoint completed |
| H390 s3/s4 pair | Two H256 members, own Q8 | 24:31 | Compact endpoint completed |
| C | Three members, own Q8 | 36:36 | Three-model endpoint completed |
| pair_ftt2 | Four distinct H256×2 trunks, C++ dynamic int8 | 44:42 | Named endpoint completed |
| E | Four members, own Q8; wider merged heads | 49:16 | Named endpoint completed |
| xteam_h4 | Four H256×2 trunks, own Q8; merged Z4 head | 50:16 | Completed using our runtime |
| H388 teacher trio | Different, larger models | TIMEOUT at 61:56 | Right-censored failure, not completed runtime |

Model count broadly tracks cost, but weights, head dimensions, shapes and host conditions differ. Pair's shorter time cannot isolate a C++ advantage. The pending E upload is excluded. Portal names are associated with local archives without portal-verified hashes.

The local Intel i9-11900H short profile used the **same pair_ftt2 weights and coefficients** through both engines: approximately119.32 µs/row for original C++ and106.73 for the Rust wrapper over two complete sequences. This unpaired, unpinned screen is not an11% EPYC speedup claim. Predictions are not bitwise identical: two-sequence WP changes from0.72556209 to0.72557066. The difference establishes a numerical treatment requiring complete checks, not a quality gain.

Admitted complete replays evaluate all1,873 supplied-valid sequences with the official metric. Concurrent worker timings remain throughput diagnostics, not controlled kernel benchmarks. Final weights have fitted or unknown overlap with this validation source, so these scores cannot select generalizing blend coefficients.

## Limits and validity

- Hardware capabilities do not prove what a container exposes. Successful Q8 packages establish compatibility of their dispatched path; a host dispatch receipt would identify it.
- Intel clocks, cache, execution and scheduling differ. One guessed scale factor cannot certify a60-minute completion.
- Distinct recurrent tensors cannot share hidden-state computation exactly, even with identical architectures. Exact reuse needs identical normalization, projection and recurrent weights; preserve separate nonlinear heads.
- Deduplicating xteam plus pair leaves six trunks and about22.28 MB of compressed weights before libraries. Pair plus E leaves eight and30.04 MB. A naive combined ZIP exceeds size limits and lacks measured runtime margin.
- No new EPYC rental or target-device benchmark occurred. Portal durations are actual-host observations with the confounding above.

## Decision and next step

1. Preserve original callbacks and grade the same-weight Rust conversion. Matching output shape does not establish quantizer parity.
2. Run repeated paired **A→B→B→A** one-core callback timing on identical complete sequences, warmup/reset, weights and coefficients. Record wall/CPU time, affinity, loaded ELF and selected ISA under controlled load. This establishes local performance; target-device confirmation is needed for a host-speed claim.
3. Prioritize exact duplicate-work removal where tensors match and distillation of the complete team ensemble into one or two students. Select endpoints with source-qualified CV and native quantized quality.
4. Treat genuine256-bit VNNI, register blocking and cache layout as bounded target-specific experiments. Existing Q4 rounding lost quality without demonstrating INT4 speed. Lower-bit arithmetic needs both quality and callback measurements.

## Source records

- [Inventory](INVENTORY.json), [portal transcript](public_results.json), [profiles](profiles/README.md), [contract](CONTRACT.json), [independent review](INDEPENDENT_REVIEW_R1.md).
- [Competition environment](../../../docs/submission_guide.md#evaluation-environment-and-limits), [VNNI arithmetic](../../../native/src/vnni.rs), [prior CPU synthesis](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/zen4_cpu_research_20261012_r1/RESEARCH_SYNTHESIS_R1.md).
- [AMD architecture white paper](https://www.amd.com/content/dam/amd/en/documents/products/epyc/4th-gen-amd-epyc-processor-architecture-whitepaper.pdf), May2024, pp.6/9, read October13,2026: cache/vector hardware facts, not our benchmark.
- [EPYC9224 specifications](https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9224.html), read October13,2026: identity/frequency/cache; maximum boost is not an observed container clock.

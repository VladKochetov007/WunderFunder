---
kind: review
status: complete
created_utc: 2026-10-14T00:39:12Z
updated_utc: 2026-10-14T00:39:12Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-PACKED-W4-MATRIX-20261014-R1]
evidence_paths:
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/packed_w4_screen_r1/README.md
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/packed_w4_screen_r1/PROVENANCE_R3.json
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/packed_w4_screen_r1/MATRIX_TIMINGS_R3.csv
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/packed_w4_screen_r1/PACKED_VNNI_ASSEMBLY_R3.txt
---

# Packed-W4 matrix screen review R1

## Question or claim

Does the frozen R3 microbenchmark compare the same represented `[-7, 7]` W4 matrices across expanded-byte VNNI, scalar nibble decode, scalar bit-plane lookup, and vectorized in-kernel unpack, and what does its timing support?

## Mechanism and method

I read the benchmark source, the native `DynQ8` layout/API, the provenance packet, R1/R2 failure and timing receipts, the final CSV, and the assembly excerpt. The current benchmark SHA-256 matches the source pin in `PROVENANCE_R3.json`; the native source, Cargo files, binary, CSV, and assembly hashes also match their pins. The disassembly is the `bench_packed_w4::packed_vnni` function from that binary and includes packed-byte widening, nibble unpack/sign adjustment, and four `vpdpbusd` instructions per 16-row/four-column tile. The worker asserts AVX-512 VNNI support without fallback.

The arithmetic is consistent across the four paths. Two's-complement nibble decoding maps the represented codes to signed values; the bit-plane lookup weights its four patterns as `b0 + 2b1 + 4b2 - 8b3`; and the vectorized path reads the native 16-by-4 tiled layout, sign-extends unpacked values, accumulates unsigned offset activations with signed weights, then subtracts `128 * sum(weights)` per row. Every tested width and row count satisfies its four-column and 64-row block assumptions. The expanded, scalar nibble, bit-plane lookup, and vectorized paths overwrite the output and are checked against the same integer row-dot reference after each 40-call timing sample.

## Evidence and interpretation

The preserved R1 receipt correctly diagnoses the harness error: `DynQ8.gemv(..., bias=None, ...)` accumulates into the existing output, so 40 benchmark calls produced a result 40 times the reference. R3 instead supplies an explicit zero bias, which gives overwrite semantics and matches the reference API path. R1 has no valid timing rows; R2 records the scalar screen; R3 is the vectorized successor.

R3 median timings put the vectorized unpack path at 1.30–2.34 times the expanded VNNI time across widths 192, 256, 384, 448, 512 and 640. The calculation uses the reported nine-sample medians; raw dispersion is not included. This supports a slowdown for this specific unpack kernel on this Intel i9-11900H host. It does not reject packed W4 generally. The scalar lookup is a correctness prototype, not an optimized T-MAC result.

## Limits and validity

The benchmark uses structured synthetic codes and one deterministic activation vector per width, unit row scales and zero bias. The static arithmetic and layouts are consistent, but output equality for that one vector is not an elementwise round-trip proof for every arbitrary matrix/input. Scales, nonzero biases, activation quantization, preprocessing, nonlinearities, trained matrices, and full GRU/callback execution are outside the measurement. The vectorized path also excludes offline packed-layout preparation; the LUT total includes table construction. The CSV's packed columns count weight payload bytes only, while expanded storage includes native scales and offset sums; neither column is an RSS or deployment-footprint measurement. No AMD, callback-quality, trained-model, competition-host, or optimized-T-MAC claim follows.

## Decision and next step

Accept the exactness and provenance checks for this bounded matrix screen and report its local timing ratios as point estimates. Retain expanded Q8 as the currently demonstrated CPU baseline. A packed successor needs an optimized kernel, real represented model matrices, and a complete GRU/callback measurement before it can support a deployment decision. The broader larger-W4 versus smaller-W8 quality/runtime question remains open.

## Source records

- [Frozen benchmark source](../../../native/examples/bench_packed_w4.rs)
- [Campaign result and interpretation](../../campaign_2026-10-14/cpu_student_frontier_r1/packed_w4_screen_r1/README.md)
- [Provenance and source/binary hashes](../../campaign_2026-10-14/cpu_student_frontier_r1/packed_w4_screen_r1/PROVENANCE_R3.json)
- [R3 timing table](../../campaign_2026-10-14/cpu_student_frontier_r1/packed_w4_screen_r1/MATRIX_TIMINGS_R3.csv)
- [R1 accumulation failure receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/packed_w4_screen_r1/FAILURE_R1.json)

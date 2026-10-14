---
kind: result
status: complete
created_utc: 2026-10-14T00:24:53Z
updated_utc: 2026-10-14T00:24:53Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-PACKED-W4-MATRIX-20261014-R1]
evidence_paths: [MATRIX_TIMINGS_R3.csv, PROVENANCE_R3.json]
---

# Exact packed-W4 matrix screen on the local Intel CPU

## Question or claim

Can genuinely packed W4 beat expanded-byte VNNI on the declared recurrent widths? This first vectorized unpack implementation is exact on the synthetic matrices but slower than expanded VNNI at every tested width. It provides no extra serving capacity yet. The scalar lookup prototype is also exact and substantially slower; it is not an optimized T-MAC replication.

## Mechanism and method

[`native/examples/bench_packed_w4.rs`](../../../../native/examples/bench_packed_w4.rs) constructs identical symmetric W4 codes in[-7,7] and offset-u8 activations for four executions: expanded DynQ8 VNNI; scalar nibble decode; packed bit-plane lookup with16-entry int16 activation-sum tables; and vectorized nibble decoding directly before512-bit VNNI accumulation. No represented weight is requantized between methods. An independent row-major integer dot verifies every output exactly after each sample. Unit scales and zero biases make the float outputs exactly representable; general scaled epilogues and quantizer boundaries are outside this test.

The vectorized method preserves the production64-row/four-column blocking, stores each64-byte weight line in32 packed bytes, expands/sign-extends in registers, and subtracts the same unsigned-activation offset correction. The [assembly excerpt](PACKED_VNNI_ASSEMBLY_R3.txt) contains packed32-byte loads widened by `vpmovzxbw`, shifts, and actual `vpdpbusd` instructions. Packed bit planes also occupy half a byte per weight; their activation tables cost8bytes per input coordinate.

[Provenance](PROVENANCE_R3.json) binds source, compiler, binary, assembly and CSV. The local Intel i9-11900H run is pinned to logicalCPU2, with nine alternating-order repeats of40 calls and an asserted VNNI implementation. Build cap is1CPU/1.5GB; execution is1CPU/0.5GB, no swap,120s limit. These are matrix timings including LUT construction where applicable, without activation quantization, nonlinearities, preprocessing or Python callback costs.

## Evidence and interpretation

[Final matrix timings](MATRIX_TIMINGS_R3.csv), microseconds per3H×H matrix:

| H | Expanded VNNI | In-kernel vectorized W4 decode | Scalar LUT, including table build |
|---:|---:|---:|---:|
|192|0.919|1.759|35.246|
|256|1.349|2.911|61.461|
|384|2.747|6.415|137.304|
|448|3.755|8.301|183.452|
|512|4.872|10.708|236.217|
|640|13.024|16.878|370.097|

The vectorized decode penalty narrows atH640, but still costs about1.30× the expanded kernel there. Extra decode instructions outweigh the weight-storage reduction for this implementation and host. The lookup table itself costs0.13–0.45µs; scalar lookup/accumulation dominates its total. This distinguishes a poor scalar lookup implementation from a claim that table construction inherently prevents a fast SIMD lookup method.

Packed byte columns report weight payloads only; expanded storage includes native scales/offset sums. Neither is an RSS measurement. The benchmark retains multiple representations to compare them, so its process memory cannot be presented as a packed deployment footprint.

## Limits and validity

Structured synthetic codes, unit scales, one CPU and microbenchmarks do not establish trained-model quality or competition-host capacity. All dimensions are multiples of64; irregular padding is untested here. Production RNN kernels and checked A/B/C/D packages are unchanged. No complete packed-W4 GRU/callback or BF16 path is claimed.

## Decision and next step

Retain expanded Q8 as the demonstrated CPU baseline. Do not use this packed prototype to justify a larger served student. A packed successor needs register/scheduling or SIMD lookup improvements, actual represented model matrices, and complete GRU/callback measurement before promotion. The broader larger-W4 versus smaller-W8 quality/runtime question remains open.

## Source records

[Failure receipt](FAILURE_R1.json) and [failed source snapshot](BENCH_FAILED_R1.rs) preserve a benchmark API error: `bias=None` accumulates into existing output, so40 repetitions yielded40× the reference. [R1 output](MATRIX_TIMINGS_R1.csv) has no valid timing rows. The corrected explicit-zero-bias [R2 scalar screen](MATRIX_TIMINGS_R2.csv) precedes the vectorized R3 successor. This is a harness failure, not a scientific negative. [Theory and T-MAC source](../THEORY_AND_DISCRIMINATING_TESTS_R1.md) provide the separately labeled mechanism basis. [Independent review](../../../unified/reviews/PACKED_W4_MATRIX_SCREEN_REVIEW_20261014_R1.md) passes the narrow represented-arithmetic and provenance claims. It records structured-input, unmeasured-spread and matrix-only limitations; it does not promote a serving path.

# Performance profile, 2026-09-15

Host: i9-11900H laptop (AVX-512), RTX 3060 Laptop 6 GB. Jobs run under
`scripts/limited` (80% CPU quota, 30 GB memory cap, no swap). CPU averaged
~3.1 GHz during the runs (thermally shared with the GPU), so absolute numbers
are pessimistic; ratios are the evidence.

## Training (scripts/profile_train.py, batch 32, 20k rows, GRU 256x2, chunk 2000)

| stage | ms/batch |
|---|---:|
| parquet decode, 1 core, 32 seqs | 653 |
| pin + host->device | 48 |
| event features on GPU | 19 |
| fwd+bwd+opt, 10 TBPTT chunks | 836 |

Training is GPU-bound: one decode core nearly keeps up, 4 workers give 5x
headroom. Native CPU code would not speed training. Observed end-to-end 22 seq/s
with the 80% GPU duty cycle (one epoch of 10,607 seqs ~8 min).

## Per-row inference, one core (scripts/bench_infer.py, native/examples)

GRU 256x2 with 211 inputs, ~1.0 M FLOPs/row. Budget if the test set matches
validation (37.46 M rows in 60 min): ~96 us/row for everything.

| runtime | us/row |
|---|---:|
| numpy event features (stream) | 18 |
| numpy GRU | 190 |
| torch eager | 243 |
| onnxruntime | 119 |
| Rust (PyO3, lanes dot, target-cpu=native) | 112 |

Kernel ablations (Rust, 768x256 gemv): row-dot 16 lanes 13.6 GFLOPS, column
SAXPY 11.9, x86-64-v3 11.6, prefer-512-bit 13.9, `matrixmultiply` sgemm 1.6.
Every dense variant sits at 12-15 GFLOPS: cost is FLOP-bound on this host.
Scalar sigmoid/tanh/GELU cost ~11 us for 768 activations.

Wins that remain:
- Sparse inputs: skipping zero input columns runs 3x faster at 1/3 density
  (5.6 vs 16.6 us); event deltas and flags are mostly zero.
- Whole pipeline (features + model) in one native call removes ~18 us of numpy.
- Model design: keep <= ~0.6 M FLOPs/row for margin on an unknown scorer vCPU.

## Perf pass 2 (E-022): layout, allocations, approximations

`native/src/rnn_fast.rs`: one 64-byte-aligned arena (bytemuck blocks, no
unsafe) in read order, state+scratch in a second aligned arena, caller-owned
output (no per-row allocation), normalization folded into a column-major input
projection so zero inputs are skipped, fused branchless gate loop.
`native/src/approx.rs`: exp as 2^n e^r (exact power of two from bits, degree-7
core), sigmoid/tanh from it, erf by Abramowitz-Stegun 7.1.26; max errors vs
f64 libm: sigmoid < 2e-6, tanh < 4e-6, GELU < 5e-6.

| hidden x 2 | reference | FastRnn exact | FastRnn approx |
|---|---:|---:|---:|
| 256 | 150.5 us | 144.8 (1.04x) | 131.1 (1.15x) |
| 192 | 93.3 | 82.1 (1.14x) | 74.8 (1.25x) |
| 128 | 44.9 | 35.1 (1.28x) | 30.8 (1.46x) |

Gains concentrate where non-matmul work is a large share (small hidden). The
first attempt at exp by 2^8 squarings amplified f32 rounding to 5.7e-6 and was
replaced. `x86-64-v3` portable builds are up to 13% slower than native.

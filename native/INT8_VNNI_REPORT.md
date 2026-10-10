# Dynamic int8 (VNNI) inference path, `Precision::Q8Vnni`

Independent Rust implementation. The idea (int8 weights with per-row scales, per-vector dynamic int8
activations, int32 accumulation, VNNI dot products) comes from the E525 report. No teammate source
was read for the code, and none was copied; weights are our own checkpoints.

## What was built

- `native/src/vnni.rs`: packed int8 matrix `DynQ8` and three kernels with runtime dispatch.
  - AVX-512 VNNI (`_mm512_dpbusd_epi32`, `std::arch` intrinsics under `#[target_feature]`, stable
    Rust 1.94, runtime `is_x86_feature_detected!`; no inline asm). Activations carry a +128 offset
    (unsigned), and a precomputed `128 * rowsum(w)` is subtracted.
  - AVX2 (`vpmaddwd` on widened bytes; exact, deliberately not `vpmaddubsw`, which saturates).
  - Portable scalar integer loops (the test oracle).
  - Layout: 64-row super-blocks, 4-column groups, one 64-byte line per 16 rows x 4 columns, so a
    super-block is a contiguous stream and one line is one `vpdpbusd` operand.
  - The integer sums are exact in int32 and the fp32 epilogue is one shared loop (convert, multiply,
    multiply, add; never fused), so **all three kernels return identical bits**.
  - Dynamic activation quantization (`quantize`, `quantize_avx2`): symmetric per vector,
    `q = round_ties_even(x * 127/max|x|)`, scale `max|x|/127`, NaN to -127, all-zero or non-finite
    vector gives scale 0.
- `native/src/rnn_fast.rs`: `Precision::Q8Vnni` (best available; becomes `Q8VnniAvx2` or
  `Q8VnniPortable` where the CPU lacks it, reported by `precision()`), plus explicit `Q8VnniAvx2` and
  `Q8VnniPortable`. GRU `weight_ih`/`weight_hh` for every layer and the split head.0 (hidden and
  projection blocks) use the int8 path. Input projection stays int16 (existing sparse kernel), GELU/GRU
  update in fp32 (existing AVX2 twins), head.2 and state in fp32. Hidden state is re-quantized right
  after each update and persists across steps; `reset()` clears codes and scales. GRU and LSTM, any
  hidden width, merged heads.
- `native/src/python.rs`: names `q8vnni`, `q8vnni_avx2`, `q8vnni_portable`.
- `native/examples/bench_int8.rs` (timing, output error, `--kernel` matvec microbench),
  `scripts/export_native_weights.py` (checkpoint to flat file), `scripts/int8_vnni_quality.py`
  (official-metric check).

Not changed: packaging, existing precisions, any existing submission. Weights are quantized from the
float checkpoint at load (PTQ, no calibration, no QAT); `to_rows`/`from_rows` exist for preserved codes.

## Tests (`cargo test --release`: 53 pass)

- Kernel vs an independent integer oracle from exported codes: bitwise equal for all supported ISAs
  across shapes 1x1 .. 1152x384, bias and accumulate forms; extreme +-127 inputs; quantize paths
  bitwise equal incl. tails/NaN/inf; import validation.
- Model level (GRU, LSTM, 3 layers, odd widths 20/33/24, 128/256): `Q8Vnni`, `Q8VnniAvx2`,
  `Q8VnniPortable` outputs bitwise identical over 300 steps, and `reset()` replay bitwise equal to the
  first pass. (Synthetic random weights are chaotic: error vs fp32 reaches 2.0 at width 256 for q8x too,
  so that test only asserts bit-identity; real-weight error below.)

## Measurements (real checkpoints, i9-11900H, one pinned core, load average about 4-6)

Rows: 40,000 real fold-1 rows (two 20k sequences), per-row work = sanitize + streaming features +
K members `step_then`, state reset per sequence, no Python. Median of 5 alternating-order trials,
weights: bd26_h230_f1_s3_ep3 (ep2), lr2e4, alpha05, muon (GRU256x2) and ab2_E_mix_h384 (width 384).
The machine was shared with a training job and desktop apps, so absolute values are inflated against
E525 (i16 one member 64.8 us there, 115.6 here); compare ratios within a column set, not with E525.

| us/row | i16 | q16x | q8x | q8vnni_avx2 | **q8vnni** |
|---|---:|---:|---:|---:|---:|
| 256 x1 | 115.6 | 75.2 | 85.3 | 64.1 | **41.8** |
| 256 x2 | 188.5 | 139.0 | 153.7 | 134.8 | **94.5** |
| 256 x3 | 293.0 | 210.9 | 223.1 | 216.2 | **148.3** |
| 256 x4 | 375.9 | 275.1 | 279.3 | 272.8 | **179.1** |
| 384 x1 | 191.0 | 144.2 | 145.7 | 148.1 | **92.5** |

Scalar portable path: about 490 us/row for one member (first run; a correctness oracle, not a target).

Ratio q8vnni / i16: 0.36, 0.50, 0.51, 0.48, 0.48; vs q8x 0.49-0.64. The C++ reference's 19.3 us (one
member, quiet machine) is not reproduced on this loaded machine; our ratio to i16 (0.36) is near its
0.30. Matvec microbenchmark (768x256): 3.1-3.7 us when L2-resident (60 GB/s), 6-7 us when streamed
from L3 (30 GB/s). Phase split (rdtsc, temporary, one member): ih matvecs 29%, hh matvecs 32%, head 10%,
int16 input projection 16%, GRU update 9%, quantization about 2%. Streaming from L3 is why two
members cost 2.3x one: a member's int8 weights are 0.9 MB, and two exceed the 1.25 MB L2.

Software prefetch hints (T0, 4 KB ahead) measured no better than the hardware prefetcher, so were removed.

Budget (91 us/row end to end): under this load 2 members at 94.5 us just exceed it, 3 do not fit.
Scaling the quiet-machine i16 baseline from E525 (64.8 / 136.2 us for 1 / 2 members) by the measured
ratio gives roughly 23 / 68 us for 1 / 2 members, about 3 members within 91 us before Python callback and
harness cost, which this benchmark does not include. That is an extrapolation and needs a quiet-machine rerun.

## Quality

Output error against fp32 on the real fold-1 rows, 40,000 rows, scored region (max abs, two outputs):
q8vnni 0.0162 / 0.0174 (rms 0.0029) at width 256; 0.0059 / 0.0079 (rms 0.0009) at width 384;
q8x 0.0161 / 0.0110; i16 6e-5. (Reference C++ int8 reported 0.072.) Over the 60-sequence check
below the largest output difference was 0.0256.

Official metric `global_wp_clip2_v1` (`wnn.gmetric.global_wp`), first 60 fold-1 evaluation sequences
(valid.parquet groups from the frozen fold1 manifest; the run was fit with fold 1 held out), 112,495
scored rows, single member bd26_h230_f1_s3_ep3 through the production `Predictor` path:

| precision | WP | WP - f32 | paired sequence bootstrap 95% |
|---|---:|---:|---|
| f32 | 0.624051 | 0 | |
| i16 | 0.624052 | +2.9e-7 | [-1.3e-7, +6.4e-7] |
| q8x | 0.624131 | +8.0e-5 | [-6.3e-5, +2.8e-4] |
| **q8vnni** | 0.624093 | **+4.2e-5** | [-1.1e-4, +2.2e-4] |

The int8 effect is indistinguishable from zero on this sample (exposed fold, diagnostic only; not a
public-score forecast). Raw data: `native/int8_vnni_quality_60.json`.

## Reproduce

```sh
WNN_RUNS_DIR=artifacts/runs .venv/bin/python scripts/export_native_weights.py bd26_h230_f1_s3_ep3@ep2 a.wbin   # also others, ab2_E_mix_h384
cd native && cargo build --release --example bench_int8
taskset -c 2 target/release/examples/bench_int8 --rows rows.f32 --reps 5 --members a.wbin b.wbin c.wbin d.wbin --wide h384.wbin --precisions i16,q16x,q8x,q8vnni_avx2,q8vnni
taskset -c 2 target/release/examples/bench_int8 --kernel
cargo build --release --lib --features extension-module   # copy libwnn_native.so to DIR/wnn_native.abi3.so
PYTHONPATH=src python scripts/int8_vnni_quality.py --lib DIR --weights ep3.npz --sequences 60
```

## What remains before a package can use it

1. Quiet-machine timing (no competing load) on the scorer-class CPU; the scoring CPU's ISA is unknown.
   Keep i16/q16x as the conservative choice and confirm AVX2 (about 1.5x slower) is acceptable if VNNI is absent.
2. Packaging: `scripts/package_submission.py`, the ABI3 build and the solution callback do not know
   `q8vnni` yet (only the Python name parser was added); build the wheel for CPython 3.11 and run the
   delivered-ZIP checks (offline, CPU-only, reset, determinism, size, runtime).
3. Ensemble governor/budget code in `python.rs` must be re-calibrated for int8 per-member cost.
4. Quality per member set: score the exact ensemble to be shipped; add QAT or per-channel activation
   calibration if a member degrades; width-384 and merged-head members need their own check.
5. Optional speed: reuse of activation quantization across matrices, fusing projection into int8,
   cross-member cache blocking (L2 is the limit beyond one member), AVX-VNNI (256-bit) path for CPUs
   without AVX-512.

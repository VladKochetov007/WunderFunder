//! Cache-conscious twin of `rnn::RnnModel`.
//!
//! - All weights live in one 64-byte-aligned arena, laid out in the order a
//!   step reads them; every segment starts on a 64-byte boundary.
//! - Hidden state and scratch share a second small aligned arena.
//! - Input normalization is folded into the projection (W' = W/std,
//!   b' = b - W' mean), so zero raw inputs are skipped column by column.
//! - Gate and activation loops are branchless for vectorization, using
//!   `approx` functions unless `Activations::Exact` is requested.
//! - `step` writes into caller-provided output; nothing allocates per row.
//! - The recurrent cell is a GRU or, when `weight_ih` has 4h rows, a PyTorch `nn.LSTM`
//!   (gates i, f, g, o; cell state reset with h).
//! - `Precision::F16` / `I16` store the input projection, GRU and head.0 matrices as 16-bit
//!   panels (`half`), halving the bytes streamed per row; accumulation, biases, head.2 and state
//!   stay fp32.
//! - `Precision::Q8Vnni` runs the GRU and head.0 matvecs as dynamic int8 (`vnni`): byte weights,
//!   per-vector byte activations, exact int32 sums, AVX-512 VNNI / AVX2 / scalar with identical bits.
//! - `Precision::Q16x` keeps the I16 weights and runs the GRU and head.0 matvecs as integer dot
//!   products against int16 activation codes (`half::gemv_q16x`).

use std::ops::Range;

use bytemuck::{Pod, Zeroable};

use crate::approx::{gelu_fast, sigmoid_fast, tanh_fast};
use crate::half::{Ahead, Codes, Encoding, HalfArena, TILE_COLS};
use crate::kernels::{gelu, sigmoid};
use crate::rnn::{RnnWeights, ShapeError};

const LANES: usize = 16;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Block([f32; LANES]);

struct Arena {
    blocks: Vec<Block>,
    len: usize,
}

impl Arena {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            len: 0,
        }
    }

    /// Reserve `n` values starting on a block (64-byte) boundary.
    fn push(&mut self, values: &[f32]) -> Range<usize> {
        let start = self.blocks.len() * LANES;
        let blocks = values.len().div_ceil(LANES).max(1);
        self.blocks
            .resize(self.blocks.len() + blocks, Block([0.0; LANES]));
        let range = start..start + values.len();
        self.as_mut()[range.clone()].copy_from_slice(values);
        self.len = self.blocks.len() * LANES;
        range
    }

    fn zeros(&mut self, n: usize) -> Range<usize> {
        self.push(&vec![0.0; n])
    }

    fn as_ref(&self) -> &[f32] {
        bytemuck::cast_slice(&self.blocks)
    }

    fn as_mut(&mut self) -> &mut [f32] {
        bytemuck::cast_slice_mut(&mut self.blocks)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activations {
    Exact,
    Fast,
}

/// Storage of the input projection, GRU (`weight_ih`, `weight_hh`) and head.0 matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F32,
    /// IEEE half, widened by F16C inside the matvec. Needs AVX2 + FMA + F16C at run time.
    F16,
    /// int16 with an fp32 scale per output row (per input column for the projection); same
    /// kernels and CPU requirement as F16, ~6x smaller weight error on trained GRUs.
    I16,
    /// I16 weights; the GRU/LSTM and head.0 matvecs take int16 activation codes (hidden states
    /// at the fixed scale 1/16383, the projection output at its own max |z| / 16383) and sum
    /// exactly in int32 (`half::gemv_q16x`). head.0 is split into its hidden and projection
    /// column blocks, each with its own row scales. Every fp32 operation is plain IEEE in a fixed
    /// order, so the AVX2 kernels and the portable fallback return the same bits.
    Q16x,
    /// `Q16x` arithmetic without the AVX2 kernels; what `Q16x` becomes on CPUs without
    /// AVX2/FMA/F16C, and selectable to check that both paths agree.
    Q16xPortable,
    /// Byte recurrent weights, float32 projection/head/state; transient int16 activations.
    Q8r,
    Q8rPortable,
    /// Byte recurrent and split head.0 weights; float32 projection, final head and state.
    Q8x,
    Q8xPortable,
    /// Imported FP32 U/V recurrence, float32 state, fixed non-fused reductions.
    FactorF32,
    FactorF32Portable,
    /// Dynamic int8 (`vnni`): byte recurrent and split head.0 weights with per-row scales, byte
    /// activations quantized per vector, int32 sums, int16 input projection, float32 final
    /// head and state. Runs on AVX-512 VNNI; becomes `Q8VnniAvx2` or `Q8VnniPortable` on CPUs
    /// without it. All three return the same bits.
    Q8Vnni,
    Q8VnniAvx2,
    Q8VnniPortable,
    /// Dynamic-int8 matrix products using EVEX 256-bit AVX-512VL/VNNI instructions.
    Q8Vnni256,
    /// Dynamic-int8 matrix products using 16-row register blocks and four accumulators.
    Q8VnniBlocked,
    /// Dynamic-int8 GRU input/recurrent projections share one VNNI matrix traversal.
    Q8VnniFused,
}

impl Precision {
    /// Kernel instruction set of a dynamic-int8 precision.
    fn dyn8(self) -> Option<crate::vnni::Isa> {
        match self {
            Precision::Q8Vnni => Some(crate::vnni::Isa::Vnni512),
            Precision::Q8VnniAvx2 => Some(crate::vnni::Isa::Avx2),
            Precision::Q8VnniPortable => Some(crate::vnni::Isa::Portable),
            Precision::Q8Vnni256 => Some(crate::vnni::Isa::Vnni256),
            Precision::Q8VnniBlocked => Some(crate::vnni::Isa::Vnni512Blocked),
            Precision::Q8VnniFused => Some(crate::vnni::Isa::Vnni512),
            _ => None,
        }
    }

    fn encoding(self) -> Option<Encoding> {
        match self {
            Precision::F32
            | Precision::Q8r
            | Precision::Q8rPortable
            | Precision::Q8x
            | Precision::Q8xPortable
            | Precision::FactorF32
            | Precision::FactorF32Portable => None,
            Precision::F16 => Some(Encoding::F16),
            Precision::I16
            | Precision::Q16x
            | Precision::Q16xPortable
            | Precision::Q8Vnni
            | Precision::Q8VnniAvx2
            | Precision::Q8VnniPortable
            | Precision::Q8Vnni256
            | Precision::Q8VnniBlocked
            | Precision::Q8VnniFused => Some(Encoding::I16),
        }
    }

    fn q8(self) -> bool {
        matches!(
            self,
            Self::Q8r | Self::Q8rPortable | Self::Q8x | Self::Q8xPortable
        )
    }

    fn q8_head(self) -> bool {
        matches!(self, Self::Q8x | Self::Q8xPortable)
    }

    fn factorized(self) -> bool {
        matches!(self, Self::FactorF32 | Self::FactorF32Portable)
    }
}

/// Recurrent cell, inferred from the weights: `weight_ih_l*` has 3h rows (GRU) or 4h (LSTM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Gru,
    Lstm,
}

impl Cell {
    pub fn gates(self) -> usize {
        match self {
            Cell::Gru => 3,
            Cell::Lstm => 4,
        }
    }
}

fn infer_cell_and_width(w: &RnnWeights) -> Result<(Cell, usize), ShapeError> {
    let layer = w
        .layers
        .first()
        .ok_or_else(|| ShapeError("at least one recurrent layer required".into()))?;
    for (gates, cell) in [(3usize, Cell::Gru), (4usize, Cell::Lstm)] {
        if layer.b_hh.len() % gates != 0 {
            continue;
        }
        let recurrent = layer.b_hh.len() / gates;
        if layer.b_ih.len() == gates * recurrent
            && layer.w_hh.len() == gates * recurrent * recurrent
        {
            return Ok((cell, recurrent));
        }
    }
    Err(ShapeError("cannot infer a square GRU/LSTM recurrent matrix".into()))
}

/// Shape checks for a `gates`-gate stack with independent encoder, recurrent and readout widths.
fn check_shapes(w: &RnnWeights, gates: usize) -> Result<(), ShapeError> {
    let n_in = w.mean.len();
    let encoder = w.b_in.len();
    let n_out = w.b2.len();
    let head = w.b1.len();
    let expect = |name: String, got: usize, want: usize| {
        if got == want {
            Ok(())
        } else {
            Err(ShapeError(format!(
                "{name}: expected {want} values, got {got}"
            )))
        }
    };
    expect("std".into(), w.std.len(), n_in)?;
    if w.layers.is_empty() {
        return Err(ShapeError("at least one recurrent layer required".into()));
    }
    let recurrent = w.layers[0].b_hh.len() / gates;
    expect("inp.weight".into(), w.w_in.len(), encoder * n_in)?;
    for (k, l) in w.layers.iter().enumerate() {
        let input = if k == 0 { encoder } else { recurrent };
        expect(
            format!("weight_ih_l{k}"),
            l.w_ih.len(),
            gates * recurrent * input,
        )?;
        expect(
            format!("weight_hh_l{k}"),
            l.w_hh.len(),
            gates * recurrent * recurrent,
        )?;
        expect(format!("bias_ih_l{k}"), l.b_ih.len(), gates * recurrent)?;
        expect(format!("bias_hh_l{k}"), l.b_hh.len(), gates * recurrent)?;
    }
    if head == 0 || n_out == 0 {
        return Err(ShapeError("empty head".into()));
    }
    expect("head.0.weight".into(), w.w1.len(), head * (recurrent + encoder))?;
    if w.w_mid.is_empty() != w.b_mid.is_empty() {
        return Err(ShapeError("head.extra weight and bias must be present together".into()));
    }
    if !w.w_mid.is_empty() {
        expect("head.extra.weight".into(), w.w_mid.len(), head * head)?;
        expect("head.extra.bias".into(), w.b_mid.len(), head)?;
    }
    expect("head.2.weight".into(), w.w2.len(), n_out * head)
}

fn activate_slice(act: Activations, values: &mut [f32]) {
    if act == Activations::Fast {
        values.iter_mut().for_each(|value| *value = gelu_fast(*value));
    } else {
        values.iter_mut().for_each(|value| *value = gelu(*value));
    }
}

const MODE_F32: u8 = 0;
const MODE_F16: u8 = 1;
const MODE_I16: u8 = 2;

/// Dequantization scale of hidden-state codes (`half::quantize` with bound 1).
const HIDDEN_SCALE: f32 = 1.0 / crate::half::QX;

/// A matrix: `w` indexes `weights` (F32) or `half`; `s` its I16 scales in `weights` (else empty).
#[derive(Clone)]
struct Mat {
    w: Range<usize>,
    s: Range<usize>,
    q8: Option<usize>,
}

struct LayerIdx {
    w_ih: Mat,
    b_ih: Range<usize>,
    w_hh: Mat,
    b_hh: Range<usize>,
}

pub struct FastRnn {
    n_in: usize,
    encoder_width: usize,
    hidden: usize,
    readout_width: usize,
    general_dims: bool,
    n_out: usize,
    input_mean: Vec<f32>,
    input_std: Vec<f32>,
    act: Activations,
    cell: Cell,
    precision: Precision,
    weights: Arena,
    /// 16-bit matrices when `precision` is F16 or I16.
    half: HalfArena,
    q8: Vec<(String, crate::quant8::PackedQ8)>,
    /// Dynamic-int8 matrices; `Mat::q8` indexes this vector when `precision.dyn8()` is set.
    dq: Vec<crate::vnni::DynQ8>,
    /// Names parallel to `dq`; exact import/export retains the Python checkpoint identity.
    dq_names: Vec<String>,
    /// Dynamic-int8 activation bytes (offset codes), `padded_cols(hidden)` per block:
    /// [h per layer | z]; the hidden blocks persist across steps. `dscales` holds each block's
    /// dequantization scale.
    dcodes: Vec<u8>,
    dscales: Vec<f32>,
    factors: Vec<(String, crate::factorized::FloatFactors)>,
    factor_scratch: Vec<f32>,
    import_only_dense_bytes: usize,
    state: Arena,
    w_in_cols: Mat, // column-major hidden x n_in, folded normalization
    b_in: Range<usize>,
    layers: Vec<LayerIdx>,
    w1: Mat,
    /// Q16x: the projection (z) column block of head.0; `w1` then holds the hidden block.
    w1z: Mat,
    b1: Range<usize>,
    w_mid: Mat,
    b_mid: Range<usize>,
    w2: Range<usize>,
    b2: Range<usize>,
    h: Range<usize>,
    z: Range<usize>,
    gi: Range<usize>,
    gh: Range<usize>,
    cat: Range<usize>,
    a: Range<usize>,
    a_mid: Range<usize>,
    /// LSTM cell state, one hidden vector per layer (empty for a GRU).
    c: Range<usize>,
    /// Nonzero input columns of the current row (half projection scratch).
    nz: Vec<u32>,
    /// Q16x activation codes, `hidden` padded to whole chunks per block: [h per layer | z]. The
    /// hidden blocks persist across steps (each is requantized right after its update).
    codes: Vec<Codes>,
    /// Dimension-aware dynamic-int8 scratch used only when encoder/recurrent/readout widths differ
    /// or the optional second hidden readout layer is present.
    general_q_z: Vec<u8>,
    general_q_hidden: Vec<u8>,
    general_q_recurrent: Vec<u8>,
    general_q_cat: Vec<u8>,
    general_q_mid: Vec<u8>,
    general_hidden_scales: Vec<f32>,
    general_input: Vec<f32>,
}

/// AVX2 dot product for the tail rows of `gemv_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= a.len() {
            s0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                s0,
            );
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                s1,
            );
            s2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 16)),
                _mm256_loadu_ps(b.as_ptr().add(i + 16)),
                s2,
            );
            s3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 24)),
                _mm256_loadu_ps(b.as_ptr().add(i + 24)),
                s3,
            );
            i += 32;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        let mut result: f32 = lanes.iter().sum();
        while i < a.len() {
            result += a[i] * b[i];
            i += 1;
        }
        result
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());
    unsafe {
        let mut s0 = _mm_setzero_ps();
        let mut s1 = _mm_setzero_ps();
        let mut s2 = _mm_setzero_ps();
        let mut s3 = _mm_setzero_ps();
        let mut i = 0;
        while i + 16 <= a.len() {
            s0 = _mm_add_ps(
                s0,
                _mm_mul_ps(
                    _mm_loadu_ps(a.as_ptr().add(i)),
                    _mm_loadu_ps(b.as_ptr().add(i)),
                ),
            );
            s1 = _mm_add_ps(
                s1,
                _mm_mul_ps(
                    _mm_loadu_ps(a.as_ptr().add(i + 4)),
                    _mm_loadu_ps(b.as_ptr().add(i + 4)),
                ),
            );
            s2 = _mm_add_ps(
                s2,
                _mm_mul_ps(
                    _mm_loadu_ps(a.as_ptr().add(i + 8)),
                    _mm_loadu_ps(b.as_ptr().add(i + 8)),
                ),
            );
            s3 = _mm_add_ps(
                s3,
                _mm_mul_ps(
                    _mm_loadu_ps(a.as_ptr().add(i + 12)),
                    _mm_loadu_ps(b.as_ptr().add(i + 12)),
                ),
            );
            i += 16;
        }
        let sum = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
        let mut lanes = [0.0f32; 4];
        _mm_storeu_ps(lanes.as_mut_ptr(), sum);
        let mut result: f32 = lanes.iter().sum();
        while i < a.len() {
            result += a[i] * b[i];
            i += 1;
        }
        result
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for k in 0..LANES {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        s += x * y;
    }
    s
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
#[inline(always)]
fn gemv(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    for ((o, row), bias) in out.iter_mut().zip(w.chunks_exact(x.len())).zip(b) {
        *o = dot(row, x) + bias;
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
fn gemv(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    // SAFETY: this build targets AVX2 + FMA.
    unsafe { gemv_avx2(w, b, x, out) }
}

/// Four output rows per pass: each x chunk is loaded once and feeds eight independent FMA
/// chains, so the kernel is throughput- rather than latency-bound. Remaining rows use `dot`.
/// The 16-bit step uses it for head.2 in every build, so those outputs agree across builds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn gemv_avx2(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = x.len();
    let rows = out.len();
    assert!(w.len() >= rows * n && b.len() >= rows);
    let main = n / 16 * 16;
    let blocks = rows / 4;
    unsafe {
        let xp = x.as_ptr();
        for blk in 0..blocks {
            let r0 = blk * 4;
            let base = w.as_ptr().add(r0 * n);
            let mut acc = [_mm256_setzero_ps(); 8];
            let mut i = 0;
            while i < main {
                let x0 = _mm256_loadu_ps(xp.add(i));
                let x1 = _mm256_loadu_ps(xp.add(i + 8));
                for r in 0..4 {
                    let wr = base.add(r * n + i);
                    acc[2 * r] = _mm256_fmadd_ps(_mm256_loadu_ps(wr), x0, acc[2 * r]);
                    acc[2 * r + 1] =
                        _mm256_fmadd_ps(_mm256_loadu_ps(wr.add(8)), x1, acc[2 * r + 1]);
                }
                i += 16;
            }
            let s0 = _mm256_add_ps(acc[0], acc[1]);
            let s1 = _mm256_add_ps(acc[2], acc[3]);
            let s2 = _mm256_add_ps(acc[4], acc[5]);
            let s3 = _mm256_add_ps(acc[6], acc[7]);
            // hadd twice leaves [row0..row3 low half | row0..row3 high half].
            let t = _mm256_hadd_ps(_mm256_hadd_ps(s0, s1), _mm256_hadd_ps(s2, s3));
            let sum = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
            let mut lanes = [0.0f32; 4];
            _mm_storeu_ps(lanes.as_mut_ptr(), sum);
            for (r, lane) in lanes.iter().enumerate() {
                let mut v = *lane;
                for j in main..n {
                    v += *base.add(r * n + j) * x[j];
                }
                out[r0 + r] = v + b[r0 + r];
            }
        }
    }
    for r in blocks * 4..rows {
        out[r] = unsafe { dot_avx2(&w[r * n..(r + 1) * n], x) } + b[r];
    }
}

/// PyTorch GRU update from the two gate preactivation blocks [r | u | n].
#[inline(always)]
fn gru_update(h: &mut [f32], gi: &[f32], gh: &[f32], fast: bool) {
    let n = h.len();
    let (gi_r, gi_rest) = gi.split_at(n);
    let (gi_u, gi_c) = gi_rest.split_at(n);
    let (gh_r, gh_rest) = gh.split_at(n);
    let (gh_u, gh_c) = gh_rest.split_at(n);
    let gates = gi_r
        .iter()
        .zip(gh_r)
        .zip(gi_u.iter().zip(gh_u))
        .zip(gi_c.iter().zip(gh_c));
    if fast {
        for (hj, (((ir, hr), (iu, hu)), (ic, hc))) in h.iter_mut().zip(gates) {
            let r = sigmoid_fast(ir + hr);
            let u = sigmoid_fast(iu + hu);
            let c = tanh_fast(ic + r * hc);
            *hj = (1.0 - u) * c + u * *hj;
        }
    } else {
        for (hj, (((ir, hr), (iu, hu)), (ic, hc))) in h.iter_mut().zip(gates) {
            let r = sigmoid(ir + hr);
            let u = sigmoid(iu + hu);
            let c = (ic + r * hc).tanh();
            *hj = (1.0 - u) * c + u * *hj;
        }
    }
}

/// `gru_update` with fast activations on eight lanes at a time (`approx::avx2`), the same bits,
/// prefetching `ahead` as it goes.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn gru_update_avx2(h: &mut [f32], gi: &[f32], gh: &[f32], ahead: &mut Ahead) {
    use crate::approx::avx2::{sigmoid8, tanh8};
    use std::arch::x86_64::*;
    let n = h.len();
    assert!(gi.len() >= 3 * n && gh.len() >= 3 * n);
    let main = n / 8 * 8;
    unsafe {
        let (i, g, hp) = (gi.as_ptr(), gh.as_ptr(), h.as_mut_ptr());
        let ld = |p: *const f32, k: usize| _mm256_loadu_ps(p.add(k));
        for j in (0..main).step_by(8) {
            ahead.issue(2 * crate::half::AHEAD_LINES);
            let r = sigmoid8(_mm256_add_ps(ld(i, j), ld(g, j)));
            let u = sigmoid8(_mm256_add_ps(ld(i, n + j), ld(g, n + j)));
            let c = tanh8(_mm256_add_ps(
                ld(i, 2 * n + j),
                _mm256_mul_ps(r, ld(g, 2 * n + j)),
            ));
            let keep = _mm256_mul_ps(u, ld(hp, j));
            let y = _mm256_add_ps(
                _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), u), c),
                keep,
            );
            _mm256_storeu_ps(hp.add(j), y);
        }
    }
    for j in main..n {
        let r = sigmoid_fast(gi[j] + gh[j]);
        let u = sigmoid_fast(gi[n + j] + gh[n + j]);
        let c = tanh_fast(gi[2 * n + j] + r * gh[2 * n + j]);
        h[j] = (1.0 - u) * c + u * h[j];
    }
}

/// `gru_update_avx2` on sixteen lanes (`approx::avx512`); the same bits.
///
/// # Safety
/// The CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn gru_update_avx512(h: &mut [f32], gi: &[f32], gh: &[f32], ahead: &mut Ahead) {
    use crate::approx::avx512::{sigmoid16, tanh16};
    use std::arch::x86_64::*;
    let n = h.len();
    assert!(gi.len() >= 3 * n && gh.len() >= 3 * n);
    let main = n / 16 * 16;
    unsafe {
        let (i, g, hp) = (gi.as_ptr(), gh.as_ptr(), h.as_mut_ptr());
        let ld = |p: *const f32, k: usize| _mm512_loadu_ps(p.add(k));
        for j in (0..main).step_by(16) {
            ahead.issue(4 * crate::half::AHEAD_LINES);
            let r = sigmoid16(_mm512_add_ps(ld(i, j), ld(g, j)));
            let u = sigmoid16(_mm512_add_ps(ld(i, n + j), ld(g, n + j)));
            let c = tanh16(_mm512_add_ps(
                ld(i, 2 * n + j),
                _mm512_mul_ps(r, ld(g, 2 * n + j)),
            ));
            let keep = _mm512_mul_ps(u, ld(hp, j));
            let y = _mm512_add_ps(
                _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps(1.0), u), c),
                keep,
            );
            _mm512_storeu_ps(hp.add(j), y);
        }
    }
    for j in main..n {
        let r = sigmoid_fast(gi[j] + gh[j]);
        let u = sigmoid_fast(gi[n + j] + gh[n + j]);
        let c = tanh_fast(gi[2 * n + j] + r * gh[2 * n + j]);
        h[j] = (1.0 - u) * c + u * h[j];
    }
}

/// out = W x + b with eight fp32 lanes per row (multiply, then add) reduced in a fixed order.
/// Plain IEEE operations that LLVM vectorizes without reassociating, so every build and CPU
/// returns the same bits (Q16x head.2).
#[inline(always)]
fn gemv_lanes(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    let n = x.len();
    for ((o, row), bias) in out.iter_mut().zip(w.chunks_exact(n)).zip(b) {
        let mut acc = [0.0f32; 8];
        let (wc, wt) = row.split_at(n / 8 * 8);
        let (xc, xt) = x.split_at(n / 8 * 8);
        for (w8, x8) in wc.chunks_exact(8).zip(xc.chunks_exact(8)) {
            for k in 0..8 {
                acc[k] += w8[k] * x8[k];
            }
        }
        let mut s =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (wv, xv) in wt.iter().zip(xt) {
            s += wv * xv;
        }
        *o = s + bias;
    }
}

/// PyTorch LSTM cell update from the two gate preactivation blocks [i | f | g | o]:
/// c = f c + i g, h = o tanh(c).
#[inline(always)]
fn lstm_update(h: &mut [f32], c: &mut [f32], gi: &[f32], gh: &[f32], fast: bool) {
    let n = h.len();
    let (gi_i, rest) = gi.split_at(n);
    let (gi_f, rest) = rest.split_at(n);
    let (gi_g, gi_o) = rest.split_at(n);
    let (gh_i, rest) = gh.split_at(n);
    let (gh_f, rest) = rest.split_at(n);
    let (gh_g, gh_o) = rest.split_at(n);
    let gates = gi_i
        .iter()
        .zip(gh_i)
        .zip(gi_f.iter().zip(gh_f))
        .zip(gi_g.iter().zip(gh_g).zip(gi_o.iter().zip(gh_o)));
    let cells = h.iter_mut().zip(c.iter_mut());
    if fast {
        for ((hj, cj), (((ii, hi), (fi, fh)), ((gi, gh), (oi, oh)))) in cells.zip(gates) {
            let i = sigmoid_fast(ii + hi);
            let f = sigmoid_fast(fi + fh);
            let g = tanh_fast(gi + gh);
            let o = sigmoid_fast(oi + oh);
            *cj = f * *cj + i * g;
            *hj = o * tanh_fast(*cj);
        }
    } else {
        for ((hj, cj), (((ii, hi), (fi, fh)), ((gi, gh), (oi, oh)))) in cells.zip(gates) {
            let i = sigmoid(ii + hi);
            let f = sigmoid(fi + fh);
            let g = (gi + gh).tanh();
            let o = sigmoid(oi + oh);
            *cj = f * *cj + i * g;
            *hj = o * cj.tanh();
        }
    }
}

impl FastRnn {
    pub fn new(w: RnnWeights, act: Activations) -> Result<Self, ShapeError> {
        Self::with_precision(w, act, Precision::F32)
    }

    /// Like `new`; 16-bit precisions fall back to F32 when the CPU lacks AVX2/FMA/F16C (and F16
    /// also when a matrix entry exceeds the half range). `precision()` reports what is used.
    pub fn with_precision(
        w: RnnWeights,
        act: Activations,
        precision: Precision,
    ) -> Result<Self, ShapeError> {
        Self::build(w, act, precision, Vec::new())
    }

    /// Import exact factors; dense recurrent input copies are discarded after construction.
    pub fn with_factors(
        w: RnnWeights,
        act: Activations,
        precision: Precision,
        factors: Vec<(String, crate::factorized::FloatFactors)>,
    ) -> Result<Self, ShapeError> {
        if !precision.factorized() {
            return Err(ShapeError("factors require factor_f32 precision".into()));
        }
        Self::build(w, act, precision, factors)
    }

    fn build(
        w: RnnWeights,
        act: Activations,
        precision: Precision,
        mut supplied: Vec<(String, crate::factorized::FloatFactors)>,
    ) -> Result<Self, ShapeError> {
        let n_in = w.mean.len();
        let input_mean = w.mean.clone();
        let input_std = w.std.clone();
        let encoder_width = w.b_in.len();
        let (cell, hidden) = infer_cell_and_width(&w)?;
        let readout_width = w.b1.len();
        let n_out = w.b2.len();
        let gates = cell.gates();
        check_shapes(&w, gates)?;
        // Existing merged-head ensembles use M=kR and keep their established optimized kernels.
        let legacy_wide_head = hidden > 0 && readout_width > hidden && readout_width % hidden == 0;
        let independent_readout = readout_width != hidden && !legacy_wide_head;
        let general_dims = encoder_width != hidden || independent_readout || !w.w_mid.is_empty();
        if general_dims && precision != Precision::F32 && precision.dyn8().is_none() {
            return Err(ShapeError(
                "independent widths currently support f32 and dynamic q8vnni precision".into(),
            ));
        }
        if general_dims && precision == Precision::Q8VnniFused {
            return Err(ShapeError(
                "q8vnni_fused is not implemented for independent-width models".into(),
            ));
        }
        let required_policy = match precision {
            Precision::Q8Vnni256 => Some(crate::vnni::Isa::Vnni256),
            Precision::Q8VnniBlocked => Some(crate::vnni::Isa::Vnni512Blocked),
            Precision::Q8VnniFused => Some(crate::vnni::Isa::Vnni512),
            _ => None,
        };
        if required_policy.is_some_and(|isa| !crate::vnni::supported(isa)) {
            return Err(ShapeError(format!(
                "requested CPU policy {precision:?} is unsupported; refusing fallback"
            )));
        }
        if precision == Precision::Q8VnniFused && cell != Cell::Gru {
            return Err(ShapeError("fused recurrent policy requires a GRU".into()));
        }
        let head = w.b1.len();
        let mut factors = Vec::new();
        if precision.factorized() {
            if cell != Cell::Gru || supplied.len() != 2 * w.layers.len() {
                return Err(ShapeError(
                    "factor_f32 requires all GRU input/hidden matrices".into(),
                ));
            }
            for k in 0..w.layers.len() {
                for kind in ["ih", "hh"] {
                    let name = format!("gru.weight_{kind}_l{k}");
                    let index = supplied
                        .iter()
                        .position(|(key, _)| key == &name)
                        .ok_or_else(|| ShapeError(format!("missing factor matrix: {name}")))?;
                    let item = supplied.swap_remove(index);
                    if item.1.rows() != 3 * hidden || item.1.cols() != hidden {
                        return Err(ShapeError(format!("factor matrix shape mismatch: {name}")));
                    }
                    factors.push(item);
                }
            }
            let rank = factors[0].1.rank();
            if factors.iter().any(|(_, matrix)| matrix.rank() != rank) {
                return Err(ShapeError(
                    "factor ranks must match the exported common rank".into(),
                ));
            }
        } else if !supplied.is_empty() {
            return Err(ShapeError("unexpected factor matrices".into()));
        }
        let factor_scratch = vec![0.0; factors.first().map_or(0, |(_, matrix)| matrix.rank())];
        let import_only_dense_bytes = if precision.factorized() {
            w.layers
                .iter()
                .map(|layer| (layer.w_ih.len() + layer.w_hh.len()) * 4)
                .sum()
        } else {
            0
        };

        // Fold normalization: W'[r][c] = W[r][c] / std[c]; b'[r] = b[r] - sum_c W'[r][c] mean[c].
        let mut cols = vec![0.0f32; encoder_width * n_in];
        let mut b_in = w.b_in.clone();
        for r in 0..encoder_width {
            for c in 0..n_in {
                let v = w.w_in[r * n_in + c] / w.std[c];
                cols[c * encoder_width + r] = v;
                b_in[r] -= v * w.mean[c];
            }
        }
        let in_range = |m: &[f32]| m.iter().all(|v| v.abs() <= crate::half::MAX);
        let precision = match precision {
            Precision::F32 | Precision::FactorF32 | Precision::FactorF32Portable => precision,
            Precision::Q8r if !crate::half::available() => Precision::Q8rPortable,
            Precision::Q8x if !crate::half::available() => Precision::Q8xPortable,
            Precision::Q8r | Precision::Q8rPortable | Precision::Q8x | Precision::Q8xPortable => {
                precision
            }
            Precision::Q8Vnni if !crate::vnni::supported(crate::vnni::Isa::Vnni512) => {
                if crate::vnni::supported(crate::vnni::Isa::Avx2) {
                    Precision::Q8VnniAvx2
                } else {
                    Precision::Q8VnniPortable
                }
            }
            Precision::Q8VnniAvx2 if !crate::vnni::supported(crate::vnni::Isa::Avx2) => {
                Precision::Q8VnniPortable
            }
            Precision::Q8Vnni
            | Precision::Q8VnniAvx2
            | Precision::Q8VnniPortable
            | Precision::Q8Vnni256
            | Precision::Q8VnniBlocked
            | Precision::Q8VnniFused => precision,
            Precision::Q16x if !crate::half::available() => Precision::Q16xPortable,
            Precision::Q16x | Precision::Q16xPortable => precision,
            _ if !crate::half::available() => Precision::F32,
            Precision::F16
                if !(in_range(&cols)
                    && w.layers
                        .iter()
                        .all(|l| in_range(&l.w_ih) && in_range(&l.w_hh))
                    && in_range(&w.w1)) =>
            {
                Precision::F32
            }
            p => p,
        };
        let mut weights = Arena::new();
        let mut half = HalfArena::new();
        let mut q8 = Vec::new();
        let mut dq = Vec::new();
        let mut dq_names = Vec::new();
        // The 16-bit projection writes whole COL_TILE row tiles; z and b_in are padded to match.
        let z_len = if precision.encoding().is_none() {
            encoder_width
        } else {
            encoder_width.div_ceil(crate::half::COL_TILE) * crate::half::COL_TILE
        };
        let w_in_cols = match precision.encoding() {
            None => Mat {
                w: weights.push(&cols),
                s: 0..0,
                q8: None,
            },
            Some(enc) => {
                let scales = match enc {
                    Encoding::F16 => Vec::new(),
                    Encoding::I16 => crate::half::column_scales(&cols, encoder_width, n_in),
                };
                let packed = half.push_columns_as(enc, &cols, &scales, encoder_width, n_in);
                // Fold the mean with the decoded weights the kernel uses, so W'(x - mean) stays
                // exact in the rounding of W' instead of picking up (W'16 - W') mean.
                for c in 0..n_in {
                    let scale = scales.get(c).copied().unwrap_or(1.0);
                    for v in &mut cols[c * encoder_width..(c + 1) * encoder_width] {
                        *v = enc.decode(*v, scale);
                    }
                }
                b_in.copy_from_slice(&w.b_in);
                for r in 0..encoder_width {
                    for c in 0..n_in {
                        b_in[r] -= cols[c * encoder_width + r] * w.mean[c];
                    }
                }
                b_in.resize(z_len, 0.0);
                Mat {
                    w: packed,
                    q8: None,
                    s: if scales.is_empty() {
                        0..0
                    } else {
                        weights.push(&scales)
                    },
                }
            }
        };
        let b_in = weights.push(&b_in);
        // Matrices in step order: `weights` for F32, `half` (+ scales in `weights`) otherwise.
        let mut matrix = |weights: &mut Arena,
                          name: String,
                          m: &[f32],
                          rows: usize,
                          cols: usize,
                          byte: bool|
         -> Result<Mat, ShapeError> {
            if precision.factorized() && name.starts_with("gru.") {
                return Ok(Mat {
                    w: 0..0,
                    s: 0..0,
                    q8: None,
                });
            }
            if precision.dyn8().is_some() {
                let index = dq.len();
                dq.push(
                    crate::vnni::DynQ8::pack(m, rows, cols).map_err(|e| ShapeError(e.into()))?,
                );
                dq_names.push(name.clone());
                return Ok(Mat {
                    w: 0..0,
                    s: 0..0,
                    q8: Some(index),
                });
            }
            if byte {
                let index = q8.len();
                let packed = crate::quant8::PackedQ8::pack(m, rows, cols)
                    .map_err(|e| ShapeError(e.into()))?;
                q8.push((name, packed));
                return Ok(Mat {
                    w: 0..0,
                    s: 0..0,
                    q8: Some(index),
                });
            }
            Ok(match precision.encoding() {
                None => Mat {
                    w: weights.push(m),
                    s: 0..0,
                    q8: None,
                },
                Some(Encoding::F16) => Mat {
                    w: half.push_panel(m, rows, cols),
                    s: 0..0,
                    q8: None,
                },
                Some(Encoding::I16) => {
                    let scales = crate::half::row_scales(m, rows, cols);
                    Mat {
                        w: half.push_panel_as(Encoding::I16, m, &scales, rows, cols),
                        s: weights.push(&scales),
                        q8: None,
                    }
                }
            })
        };
        let layers = w
            .layers
            .iter()
            .enumerate()
            .map(|(k, l)| {
                let input_width = if k == 0 { encoder_width } else { hidden };
                Ok(LayerIdx {
                    w_ih: matrix(
                        &mut weights,
                        format!("gru.weight_ih_l{k}.0"),
                        &l.w_ih,
                        gates * hidden,
                        input_width,
                        precision.q8(),
                    )?,
                    b_ih: weights.push(&l.b_ih),
                    w_hh: matrix(
                        &mut weights,
                        format!("gru.weight_hh_l{k}.0"),
                        &l.w_hh,
                        gates * hidden,
                        hidden,
                        precision.q8(),
                    )?,
                    b_hh: weights.push(&l.b_hh),
                })
            })
            .collect::<Result<Vec<_>, ShapeError>>()?;
        let q16x = matches!(precision, Precision::Q16x | Precision::Q16xPortable);
        let empty_mat = Mat {
            w: 0..0,
            s: 0..0,
            q8: None,
        };
        let (w1, w1z) = if general_dims {
            (
                matrix(
                    &mut weights,
                    "head.0.weight".into(),
                    &w.w1,
                    head,
                    hidden + encoder_width,
                    precision.q8_head(),
                )?,
                empty_mat.clone(),
            )
        } else if q16x || precision.q8_head() || precision.dyn8().is_some() {
            let block = |k: usize| -> Vec<f32> {
                w.w1.chunks_exact(2 * hidden)
                    .flat_map(|row| &row[k * hidden..(k + 1) * hidden])
                    .copied()
                    .collect()
            };
            let w1 = matrix(
                &mut weights,
                "head.0.weight.0".into(),
                &block(0),
                head,
                hidden,
                precision.q8_head(),
            )?;
            let w1z = matrix(
                    &mut weights,
                    "head.0.weight.1".into(),
                    &block(1),
                    head,
                    hidden,
                    precision.q8_head(),
                )?;
            (w1, w1z)
        } else {
            let w1 = matrix(
                &mut weights,
                "head.0.weight".into(),
                &w.w1,
                head,
                2 * hidden,
                false,
            )?;
            (
                w1,
                empty_mat.clone(),
            )
        };
        let b1 = weights.push(&w.b1);
        let w_mid = if w.w_mid.is_empty() {
            empty_mat
        } else {
            matrix(
                &mut weights,
                "head.extra.weight".into(),
                &w.w_mid,
                head,
                head,
                precision.q8_head(),
            )?
        };
        let b_mid = if w.b_mid.is_empty() {
            let end = weights.len;
            end..end
        } else {
            weights.push(&w.b_mid)
        };
        let w2 = weights.push(&w.w2);
        let b2 = weights.push(&w.b2);

        let mut state = Arena::new();
        let h = state.zeros(layers.len() * hidden);
        let z = {
            let padded = state.zeros(z_len);
            padded.start..padded.start + encoder_width
        };
        let gi = state.zeros(gates * hidden);
        let gh = state.zeros(gates * hidden);
        let cat = state.zeros(hidden + encoder_width);
        let a = state.zeros(head);
        let a_mid = if w.w_mid.is_empty() {
            let end = state.len;
            end..end
        } else {
            state.zeros(head)
        };
        let c = match cell {
            Cell::Gru => {
                let end = state.len;
                end..end
            }
            Cell::Lstm => state.zeros(layers.len() * hidden),
        };
        let code_blocks = if !general_dims && (q16x || precision.q8()) {
            layers.len() + 1
        } else {
            0
        };
        let codes = vec![Codes([0; TILE_COLS]); code_blocks * hidden.div_ceil(TILE_COLS)];
        let dyn_blocks = if precision.dyn8().is_some() && !general_dims { layers.len() + 1 } else { 0 };
        let dcodes = vec![crate::vnni::OFFSET; dyn_blocks * crate::vnni::padded_cols(hidden)];
        let dscales = vec![0.0; dyn_blocks];
        let general_q_z = if general_dims && precision.dyn8().is_some() {
            vec![crate::vnni::OFFSET; crate::vnni::padded_cols(encoder_width)]
        } else { Vec::new() };
        let general_q_hidden = if general_dims && precision.dyn8().is_some() {
            vec![crate::vnni::OFFSET; layers.len() * crate::vnni::padded_cols(hidden)]
        } else { Vec::new() };
        let general_q_recurrent = if general_dims && precision.dyn8().is_some() {
            vec![crate::vnni::OFFSET; crate::vnni::padded_cols(hidden)]
        } else { Vec::new() };
        let general_q_cat = if general_dims && precision.dyn8().is_some() {
            vec![crate::vnni::OFFSET; crate::vnni::padded_cols(hidden + encoder_width)]
        } else { Vec::new() };
        let general_q_mid = if general_dims && precision.dyn8().is_some() {
            vec![crate::vnni::OFFSET; crate::vnni::padded_cols(head)]
        } else { Vec::new() };
        let general_hidden_scales = if general_dims && precision.dyn8().is_some() {
            vec![0.0; layers.len()]
        } else { Vec::new() };
        let general_input = if general_dims {
            vec![0.0; encoder_width.max(hidden)]
        } else { Vec::new() };
        debug_assert!(weights.len >= weights.as_ref().len());
        Ok(Self {
            n_in,
            encoder_width,
            hidden,
            readout_width,
            general_dims,
            n_out,
            input_mean,
            input_std,
            act,
            cell,
            precision,
            weights,
            half,
            q8,
            dq,
            dq_names,
            dcodes,
            dscales,
            factors,
            factor_scratch,
            import_only_dense_bytes,
            state,
            w_in_cols,
            b_in,
            layers,
            w1,
            w1z,
            b1,
            w_mid,
            b_mid,
            w2,
            b2,
            h,
            z,
            gi,
            gh,
            cat,
            a,
            a_mid,
            c,
            nz: Vec::with_capacity(n_in),
            codes,
            general_q_z,
            general_q_hidden,
            general_q_recurrent,
            general_q_cat,
            general_q_mid,
            general_hidden_scales,
            general_input,
        })
    }

    pub fn n_in(&self) -> usize {
        self.n_in
    }

    pub fn n_out(&self) -> usize {
        self.n_out
    }

    pub fn encoder_width(&self) -> usize {
        self.encoder_width
    }

    /// Input normalization used by the Python backbone, before its input projection.
    pub fn normalize_input(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.n_in, "input width");
        assert_eq!(output.len(), self.n_in, "normalized input width");
        for ((value, source), (mean, std)) in output
            .iter_mut()
            .zip(input)
            .zip(self.input_mean.iter().zip(&self.input_std))
        {
            *value = (*source - *mean) / *std;
        }
    }

    /// Reproduce the tiedlead encoder's `(x - mean) / std * std + mean` reconstruction.
    pub fn restore_normalized_input(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.n_in, "input width");
        assert_eq!(output.len(), self.n_in, "restored input width");
        for ((restored, source), (mean, std)) in output
            .iter_mut()
            .zip(input)
            .zip(self.input_mean.iter().zip(&self.input_std))
        {
            *restored = ((*source - *mean) / *std) * *std + *mean;
        }
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Shape of the persistent hidden state, ordered as `(layers, hidden)`.
    pub fn hidden_state_shape(&self) -> (usize, usize) {
        (self.layers.len(), self.hidden)
    }

    /// Read-only flattened hidden vectors, one contiguous `hidden`-wide row per layer.
    pub fn hidden_state(&self) -> &[f32] {
        &self.state.as_ref()[self.h.clone()]
    }

    /// Replace the FP32 GRU state and refresh any activation-code cache used by this precision.
    pub fn set_hidden_state(&mut self, values: &[f32]) -> Result<(), ShapeError> {
        if self.cell != Cell::Gru {
            return Err(ShapeError("state initialization requires a GRU".into()));
        }
        if values.len() != self.layers.len() * self.hidden {
            return Err(ShapeError(format!(
                "hidden-state size mismatch: expected {}, got {}",
                self.layers.len() * self.hidden,
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ShapeError("hidden state must be finite".into()));
        }
        self.state.as_mut()[self.h.clone()].copy_from_slice(values);
        self.refresh_hidden_codes();
        Ok(())
    }

    /// Add a bounded FP32 residual to every GRU layer and refresh quantized activation caches.
    pub fn add_hidden_state(&mut self, values: &[f32]) -> Result<(), ShapeError> {
        if self.cell != Cell::Gru {
            return Err(ShapeError("state initialization requires a GRU".into()));
        }
        if values.len() != self.layers.len() * self.hidden {
            return Err(ShapeError(format!(
                "hidden-state size mismatch: expected {}, got {}",
                self.layers.len() * self.hidden,
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ShapeError("hidden-state residual must be finite".into()));
        }
        for (state, residual) in self.state.as_mut()[self.h.clone()].iter_mut().zip(values) {
            *state += *residual;
        }
        if self.hidden_state().iter().any(|value| !value.is_finite()) {
            return Err(ShapeError(
                "initialized hidden state must remain finite".into(),
            ));
        }
        self.refresh_hidden_codes();
        Ok(())
    }

    fn refresh_hidden_codes(&mut self) {
        if self.general_dims {
            // The dimension-aware Q8 path quantizes persistent FP32 states at use sites.
            return;
        }
        let layers = self.layers.len();
        let hidden = self.hidden;
        let state = self.hidden_state().to_vec();
        if self.precision.dyn8().is_some() {
            let padded = crate::vnni::padded_cols(hidden);
            for layer in 0..layers {
                let start = layer * padded;
                self.dscales[layer] = crate::vnni::quantize(
                    &state[layer * hidden..(layer + 1) * hidden],
                    &mut self.dcodes[start..start + padded],
                );
            }
        } else if self.precision == Precision::Q16x
            || self.precision == Precision::Q16xPortable
            || self.precision.q8()
        {
            let padded = hidden.div_ceil(TILE_COLS) * TILE_COLS;
            let codes: &mut [i16] = bytemuck::cast_slice_mut(&mut self.codes);
            for layer in 0..layers {
                let start = layer * padded;
                crate::half::quantize(
                    &state[layer * hidden..(layer + 1) * hidden],
                    Some(1.0),
                    &mut codes[start..start + padded],
                );
            }
        }
    }

    /// Whether the step runs on the dynamic-int8 kernels (`Q8Vnni` and its fallbacks).
    pub fn is_dynamic_int8(&self) -> bool {
        self.precision.dyn8().is_some()
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Exact byte tensors, named using the portable quantization checkpoint schema.
    pub fn q8_matrices(&self) -> &[(String, crate::quant8::PackedQ8)] {
        &self.q8
    }

    /// Exact row-wise byte matrices used by `q8vnni*`, paired with checkpoint tensor names.
    pub fn dyn8_matrices(&self) -> impl Iterator<Item = (&str, &crate::vnni::DynQ8)> {
        self.dq_names.iter().map(String::as_str).zip(self.dq.iter())
    }

    pub fn factor_matrices(&self) -> &[(String, crate::factorized::FloatFactors)] {
        &self.factors
    }

    /// Actual persistent recurrent state, excluding scratch and alignment padding.
    pub fn recurrent_state_bytes(&self) -> usize {
        (self.h.len() + self.c.len()) * 4
    }

    /// State-arena padding, projection/gates/head temporaries and factor projection scratch.
    pub fn scratch_bytes(&self) -> usize {
        self.state.len * 4 - self.recurrent_state_bytes()
            + self.factor_scratch.len() * 4
            + self.codes.len() * size_of::<Codes>()
            + self.dcodes.len()
            + self.dscales.len() * 4
            + self.nz.capacity() * size_of::<u32>()
            + (self.general_q_z.capacity()
                + self.general_q_hidden.capacity()
                + self.general_q_recurrent.capacity()
                + self.general_q_cat.capacity()
                + self.general_q_mid.capacity())
            * size_of::<u8>()
            + (self.general_hidden_scales.capacity() + self.general_input.capacity())
                * size_of::<f32>()
    }

    /// Dense recurrent copies accepted by the existing Python weights API, not retained.
    pub fn import_only_dense_bytes(&self) -> usize {
        self.import_only_dense_bytes
    }

    /// Replace constructor quantization with preserved codes/scales from an export.
    pub fn load_q8_matrices(
        &mut self,
        matrices: Vec<(String, crate::quant8::PackedQ8)>,
    ) -> Result<(), ShapeError> {
        if matrices.len() != self.q8.len() || self.q8.is_empty() {
            return Err(ShapeError(
                "q8 tensor count disagrees with precision preset".into(),
            ));
        }
        for ((expected, old), (name, matrix)) in self.q8.iter().zip(&matrices) {
            if name != expected || old.rows() != matrix.rows() || old.cols() != matrix.cols() {
                return Err(ShapeError(format!(
                    "q8 tensor identity or shape mismatch: {name}"
                )));
            }
        }
        self.q8 = matrices;
        Ok(())
    }

    /// Replace constructor quantization with exact dynamic-int8 codes/scales.
    pub fn load_dyn8_matrices(
        &mut self,
        matrices: Vec<(String, crate::vnni::DynQ8)>,
    ) -> Result<(), ShapeError> {
        if !self.is_dynamic_int8() || matrices.len() != self.dq.len() || self.dq.is_empty() {
            return Err(ShapeError(
                "dynamic q8 tensor count disagrees with precision preset".into(),
            ));
        }
        for ((expected, old), (name, matrix)) in self
            .dq_names
            .iter()
            .zip(&self.dq)
            .zip(&matrices)
        {
            if name != expected || old.rows() != matrix.rows() || old.cols() != matrix.cols() {
                return Err(ShapeError(format!(
                    "dynamic q8 tensor identity or shape mismatch: {name}"
                )));
            }
        }
        self.dq = matrices.into_iter().map(|(_, matrix)| matrix).collect();
        Ok(())
    }

    /// Weight bytes a step streams (all arenas, padding included).
    pub fn weight_bytes(&self) -> usize {
        self.weights.len * 4
            + self.half.bytes()
            + self
                .q8
                .iter()
                .map(|(_, matrix)| matrix.storage_bytes())
                .sum::<usize>()
            + self.dq.iter().map(|m| m.storage_bytes()).sum::<usize>()
            + self
                .factors
                .iter()
                .map(|(_, matrix)| matrix.storage_bytes())
                .sum::<usize>()
    }

    /// Logical per-step bytes read from the shared input projection and recurrent weights.
    ///
    /// This excludes arena alignment padding and is defined for dynamic-int8 precision only.
    pub fn shared_backbone_step_payload_bytes(&self) -> usize {
        if self.precision.dyn8().is_none() {
            return 0;
        }
        let recurrent_matrices = self
            .layers
            .iter()
            .map(|layer| {
                self.dq[layer.w_ih.q8.expect("dynamic input matrix")].storage_bytes()
                    + self.dq[layer.w_hh.q8.expect("dynamic recurrent matrix")].storage_bytes()
                    + (layer.b_ih.len() + layer.b_hh.len()) * size_of::<f32>()
            })
            .sum::<usize>();
        self.half.bytes()
            + (self.w_in_cols.s.len() + self.b_in.len()) * size_of::<f32>()
            + recurrent_matrices
    }

    /// Logical per-step bytes read from this model's independent head weights.
    ///
    /// This excludes arena alignment padding and is defined for dynamic-int8 precision only.
    pub fn independent_head_step_payload_bytes(&self) -> usize {
        if self.precision.dyn8().is_none() {
            return 0;
        }
        let mut bytes = self.dq[self.w1.q8.expect("dynamic head matrix")].storage_bytes()
            + (self.b1.len() + self.w2.len() + self.b2.len()) * size_of::<f32>();
        if self.general_dims {
            if let Some(index) = self.w_mid.q8 {
                bytes += self.dq[index].storage_bytes();
                bytes += self.b_mid.len() * size_of::<f32>();
            }
        } else {
            bytes += self.dq[self.w1z.q8.expect("dynamic projection-head matrix")].storage_bytes();
        }
        bytes
    }

    pub fn cell(&self) -> Cell {
        self.cell
    }

    pub fn reset(&mut self) {
        let (h, c) = (self.h.clone(), self.c.clone());
        self.state.as_mut()[h].fill(0.0);
        self.state.as_mut()[c].fill(0.0);
        self.codes.fill(Codes([0; TILE_COLS]));
        self.dcodes.fill(crate::vnni::OFFSET);
        self.dscales.fill(0.0);
        self.general_q_z.fill(crate::vnni::OFFSET);
        self.general_q_hidden.fill(crate::vnni::OFFSET);
        self.general_q_recurrent.fill(crate::vnni::OFFSET);
        self.general_q_cat.fill(crate::vnni::OFFSET);
        self.general_q_mid.fill(crate::vnni::OFFSET);
        self.general_hidden_scales.fill(0.0);
        self.general_input.fill(0.0);
    }

    pub fn step(&mut self, x: &[f32], out: &mut [f32]) {
        let after = self.first_weights();
        self.step_then(x, out, after)
    }

    /// Run an FP32 model after adding a per-row residual to its encoder preactivation.
    ///
    /// This supports the causal tiedlead encoder while leaving ordinary model stepping
    /// and every existing predictor unchanged.
    pub fn step_with_encoder_residual(
        &mut self,
        x: &[f32],
        residual: &[f32],
        out: &mut [f32],
    ) -> Result<(), ShapeError> {
        if self.precision != Precision::F32 {
            return Err(ShapeError(
                "encoder residual requires the final predictor's fp32 precision".into(),
            ));
        }
        if x.len() != self.n_in {
            return Err(ShapeError(format!(
                "input width: expected {}, got {}",
                self.n_in,
                x.len()
            )));
        }
        if residual.len() != self.encoder_width {
            return Err(ShapeError(format!(
                "encoder residual width: expected {}, got {}",
                self.encoder_width,
                residual.len()
            )));
        }
        if out.len() != self.n_out {
            return Err(ShapeError(format!(
                "output width: expected {}, got {}",
                self.n_out,
                out.len()
            )));
        }
        if self.general_dims {
            self.step_general_f32_with_residual(x, residual, out);
        } else {
            // Equal-width models use the same optimized GEMV kernels as ordinary F32 stepping.
            // The residual is added at the same point as in the general path: after input
            // projection and before the encoder activation.
            self.step_impl_with_encoder_residual::<MODE_F32>(x, residual, out);
        }
        Ok(())
    }

    /// The weights a step reads first (the input projection): what `step_then` of the member
    /// stepped just before this one prefetches.
    pub fn first_weights(&self) -> Ahead {
        match self.precision {
            Precision::F32
            | Precision::Q8r
            | Precision::Q8rPortable
            | Precision::Q8x
            | Precision::Q8xPortable
            | Precision::FactorF32
            | Precision::FactorF32Portable => Ahead::NONE,
            _ => Ahead::new(&self.half.as_ref()[self.w_in_cols.w.clone()]),
        }
    }

    /// `step`, prefetching `after` (the weights the caller reads next) during the last
    /// activation pass. Only Q16x prefetches; outputs never depend on it.
    pub fn step_then(&mut self, x: &[f32], out: &mut [f32], after: Ahead) {
        assert_eq!(x.len(), self.n_in, "input width");
        assert_eq!(out.len(), self.n_out, "output width");
        if self.general_dims {
            if self.precision == Precision::F32 {
                self.step_general_f32(x, out);
            } else {
                self.step_general_dyn8(x, out);
            }
            return;
        }
        match self.precision {
            Precision::F32 => self.step_impl::<MODE_F32>(x, out),
            Precision::FactorF32 | Precision::FactorF32Portable => self.step_factorized(x, out),
            // SAFETY: 16-bit precisions are only selected when `half::available()`.
            #[cfg(target_arch = "x86_64")]
            Precision::F16 => unsafe { self.step_f16(x, out) },
            #[cfg(target_arch = "x86_64")]
            Precision::I16 => unsafe { self.step_i16(x, out) },
            // SAFETY: Q16x is only selected when `half::available()` (else Q16xPortable).
            #[cfg(target_arch = "x86_64")]
            Precision::Q16x => unsafe { self.step_q16x(x, out, after) },
            Precision::Q16xPortable => self.step_q_impl::<false>(x, out, after),
            // SAFETY: selected only when `vnni::supported` reported the instruction set.
            #[cfg(target_arch = "x86_64")]
            Precision::Q8Vnni => unsafe { self.step_dyn_vnni(x, out, after) },
            #[cfg(target_arch = "x86_64")]
            Precision::Q8Vnni256 => unsafe { self.step_dyn_vnni256(x, out, after) },
            #[cfg(target_arch = "x86_64")]
            Precision::Q8VnniBlocked => unsafe { self.step_dyn_vnni_blocked(x, out, after) },
            #[cfg(target_arch = "x86_64")]
            Precision::Q8VnniFused => unsafe { self.step_dyn_vnni_fused(x, out, after) },
            #[cfg(target_arch = "x86_64")]
            Precision::Q8VnniAvx2 => unsafe { self.step_dyn_avx2(x, out, after) },
            Precision::Q8VnniPortable => {
                self.step_dyn_impl::<false, false, false>(
                    x,
                    out,
                    after,
                    crate::vnni::Isa::Portable,
                )
            }
            #[cfg(target_arch = "x86_64")]
            Precision::Q8r | Precision::Q8x => unsafe { self.step_q16x(x, out, after) },
            Precision::Q8rPortable | Precision::Q8xPortable => {
                self.step_q_impl::<false>(x, out, after)
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => unreachable!(),
        }
    }

    /// Run one dynamic-int8 backbone and evaluate a second model's independent head on it.
    ///
    /// The caller preflights byte-identical input/projection/recurrent tensors, matching
    /// precision and dimensions, and two-output heads. The second model is used only as head
    /// storage; its recurrent state is deliberately not advanced.
    pub(crate) fn step_with_shared_backbone(
        &mut self,
        second_head: &mut Self,
        x: &[f32],
        first_out: &mut [f32],
        second_out: &mut [f32],
    ) {
        debug_assert!(self.precision.dyn8().is_some());
        debug_assert!(!self.general_dims);
        debug_assert_eq!(self.precision, second_head.precision);
        debug_assert_eq!(self.hidden, second_head.hidden);
        debug_assert_eq!(self.layers.len(), second_head.layers.len());
        debug_assert_eq!(self.n_out, second_head.n_out);
        self.step(x, first_out);
        let after = self.first_weights();
        second_head.step_dyn_head_from(self, second_out, after);
    }

    fn step_dyn_head_from(&mut self, backbone: &FastRnn, out: &mut [f32], mut after: Ahead) {
        use crate::vnni::padded_cols;

        debug_assert!(self.precision.dyn8().is_some());
        debug_assert_eq!(self.precision, backbone.precision);
        debug_assert_eq!(self.hidden, backbone.hidden);
        debug_assert_eq!(self.layers.len(), backbone.layers.len());
        assert_eq!(out.len(), self.n_out, "output width");

        let isa = self
            .precision
            .dyn8()
            .expect("shared head requires dynamic int8");
        let n_layers = self.layers.len();
        let kp = padded_cols(self.hidden);
        let (hq, zq) = backbone.dcodes.split_at(n_layers * kp);
        let (hs, zs) = backbone.dscales.split_at(n_layers);
        let last_hidden = &hq[(n_layers - 1) * kp..n_layers * kp];
        let hidden_scale = hs[n_layers - 1];
        let z_scale = zs[0];
        let z_codes = &zq[..kp];

        let wts = self.weights.as_ref();
        let a = &mut self.state.as_mut()[self.a.clone()];
        self.dq[self.w1.q8.expect("dynamic head weight")].gemv(
            isa,
            last_hidden,
            hidden_scale,
            Some(&wts[self.b1.clone()]),
            a,
        );
        self.dq[self.w1z.q8.expect("dynamic projection-head weight")]
            .gemv(isa, z_codes, z_scale, None, a);

        let fast = self.act == Activations::Fast;
        #[cfg(target_arch = "x86_64")]
        if fast
            && matches!(
                self.precision,
                Precision::Q8Vnni
                    | Precision::Q8VnniBlocked
                    | Precision::Q8VnniFused
            )
        {
            // SAFETY: Q8Vnni is selected only after the AVX-512 VNNI capability check.
            unsafe { crate::approx::avx512::gelu_slice(a, &mut after) };
        } else if fast
            && matches!(
                self.precision,
                Precision::Q8VnniAvx2 | Precision::Q8Vnni256
            )
        {
            // SAFETY: Q8VnniAvx2 is selected only after the AVX2 capability check.
            unsafe { crate::approx::avx2::gelu_slice(a, &mut after) };
        } else if fast {
            let _ = after;
            a.iter_mut().for_each(|value| *value = gelu_fast(*value));
        } else {
            let _ = after;
            a.iter_mut().for_each(|value| *value = gelu(*value));
        }
        gemv_lanes(&wts[self.w2.clone()], &wts[self.b2.clone()], a, out);
    }

    /// Portable matrix path for independent E/R/M widths and an optional second head layer.
    /// Equal-width one-head models stay on their existing specialized path.
    fn step_general_f32(&mut self, x: &[f32], out: &mut [f32]) {
        self.step_general_f32_with_residual(x, &[], out);
    }

    fn step_general_f32_with_residual(&mut self, x: &[f32], residual: &[f32], out: &mut [f32]) {
        let weights = self.weights.as_ref();
        let state = self.state.as_mut();
        let e = self.encoder_width;
        let r = self.hidden;
        let m = self.readout_width;
        // Equal-width models normally use the specialized path and leave this scratch empty.
        // The tiedlead residual path deliberately enters the general kernel for those models.
        if self.general_input.len() < e.max(r) {
            self.general_input.resize(e.max(r), 0.0);
        }
        {
            let z = &mut state[self.z.clone()];
            z.copy_from_slice(&weights[self.b_in.clone()][..e]);
            for (column, &value) in weights[self.w_in_cols.w.clone()].chunks_exact(e).zip(x) {
                if value != 0.0 {
                    for (output, &weight) in z.iter_mut().zip(column) {
                        *output += weight * value;
                    }
                }
            }
            for (value, addition) in z.iter_mut().zip(residual) {
                *value += addition;
            }
            activate_slice(self.act, z);
        }

        for (index, layer) in self.layers.iter().enumerate() {
            let input_width = if index == 0 { e } else { r };
            let input_range = if index == 0 {
                self.z.clone()
            } else {
                self.h.start + (index - 1) * r..self.h.start + index * r
            };
            self.general_input[..input_width]
                .copy_from_slice(&state[input_range]);
            let h_start = self.h.start + index * r;
            let (low, high) = state.split_at_mut(self.gi.start);
            let (gi, rest) = high.split_at_mut(self.gi.len());
            let gh_offset = self.gh.start - self.gi.end;
            let (rest, cells) = rest.split_at_mut(self.c.start - self.gi.end);
            let gh = &mut rest[gh_offset..gh_offset + self.cell.gates() * r];
            let h = &mut low[h_start..h_start + r];
            gemv_lanes(
                &weights[layer.w_ih.w.clone()],
                &weights[layer.b_ih.clone()],
                &self.general_input[..input_width],
                gi,
            );
            gemv_lanes(
                &weights[layer.w_hh.w.clone()],
                &weights[layer.b_hh.clone()],
                h,
                gh,
            );
            match self.cell {
                Cell::Gru => gru_update(h, gi, gh, self.act == Activations::Fast),
                Cell::Lstm => lstm_update(
                    h,
                    &mut cells[index * r..(index + 1) * r],
                    gi,
                    gh,
                    self.act == Activations::Fast,
                ),
            }
        }

        let last = self.h.start + (self.layers.len() - 1) * r;
        let (low, rest) = state.split_at_mut(self.cat.start);
        let (cat, rest) = rest.split_at_mut(self.cat.len());
        cat[..r].copy_from_slice(&low[last..last + r]);
        cat[r..].copy_from_slice(&low[self.z.clone()]);
        let a_offset = self.a.start - self.cat.end;
        let (_padding, after_a) = rest.split_at_mut(a_offset);
        let (a, after_a) = after_a.split_at_mut(m);
        gemv_lanes(
            &weights[self.w1.w.clone()],
            &weights[self.b1.clone()],
            cat,
            a,
        );
        activate_slice(self.act, a);
        if !self.w_mid.w.is_empty() {
            let mid_offset = self.a_mid.start - self.a.end;
            let (_padding, after_mid) = after_a.split_at_mut(mid_offset);
            let mid = &mut after_mid[..m];
            gemv_lanes(
                &weights[self.w_mid.w.clone()],
                &weights[self.b_mid.clone()],
                a,
                mid,
            );
            activate_slice(self.act, mid);
            gemv_lanes(&weights[self.w2.clone()], &weights[self.b2.clone()], mid, out);
        } else {
            gemv_lanes(&weights[self.w2.clone()], &weights[self.b2.clone()], a, out);
        }
    }

    fn step_general_dyn8(&mut self, x: &[f32], out: &mut [f32]) {
        use crate::vnni::{padded_cols, quantize};

        let isa = self.precision.dyn8().expect("general Q8 path");
        let fast = self.act == Activations::Fast;
        let weights = self.weights.as_ref();
        let state = self.state.as_mut();
        let e = self.encoder_width;
        let r = self.hidden;
        let m = self.readout_width;
        {
            let b = &weights[self.b_in.clone()];
            let padded = &mut state[self.z.start..self.z.start + b.len()];
            crate::half::project_q16_portable(
                &self.half.as_ref()[self.w_in_cols.w.clone()],
                &weights[self.w_in_cols.s.clone()],
                b,
                x,
                padded,
            );
            let z = &mut state[self.z.clone()];
            activate_slice(self.act, z);
        }
        let z_scale = quantize(&state[self.z.clone()], &mut self.general_q_z);
        let kr = padded_cols(r);

        for (index, layer) in self.layers.iter().enumerate() {
            let input_width = if index == 0 { e } else { r };
            let input_codes: &[u8] = if index == 0 {
                &self.general_q_z
            } else {
                let start = (index - 1) * kr;
                &self.general_q_hidden[start..start + kr]
            };
            let input_scale = if index == 0 {
                z_scale
            } else {
                self.general_hidden_scales[index - 1]
            };
            debug_assert_eq!(input_codes.len(), padded_cols(input_width));

            let h_start = self.h.start + index * r;
            let (low, high) = state.split_at_mut(self.gi.start);
            let (gi, rest) = high.split_at_mut(self.gi.len());
            let gh_offset = self.gh.start - self.gi.end;
            let (rest, cells) = rest.split_at_mut(self.c.start - self.gi.end);
            let gh = &mut rest[gh_offset..gh_offset + self.cell.gates() * r];
            let h = &mut low[h_start..h_start + r];
            let recurrent_scale = quantize(h, &mut self.general_q_recurrent);
            self.dq[layer.w_ih.q8.expect("dynamic input matrix")].gemv(
                isa,
                input_codes,
                input_scale,
                Some(&weights[layer.b_ih.clone()]),
                gi,
            );
            self.dq[layer.w_hh.q8.expect("dynamic recurrent matrix")].gemv(
                isa,
                &self.general_q_recurrent,
                recurrent_scale,
                Some(&weights[layer.b_hh.clone()]),
                gh,
            );
            match self.cell {
                Cell::Gru => gru_update(h, gi, gh, fast),
                Cell::Lstm => lstm_update(
                    h,
                    &mut cells[index * r..(index + 1) * r],
                    gi,
                    gh,
                    fast,
                ),
            }
            let start = index * kr;
            self.general_hidden_scales[index] = quantize(
                h,
                &mut self.general_q_hidden[start..start + kr],
            );
        }

        let last = self.h.start + (self.layers.len() - 1) * r;
        let (low, rest) = state.split_at_mut(self.cat.start);
        let (cat, rest) = rest.split_at_mut(self.cat.len());
        cat[..r].copy_from_slice(&low[last..last + r]);
        cat[r..].copy_from_slice(&low[self.z.clone()]);
        let cat_scale = quantize(cat, &mut self.general_q_cat);
        let a_offset = self.a.start - self.cat.end;
        let (_padding, after_a) = rest.split_at_mut(a_offset);
        let (a, rest) = after_a.split_at_mut(m);
        self.dq[self.w1.q8.expect("dynamic head matrix")].gemv(
            isa,
            &self.general_q_cat,
            cat_scale,
            Some(&weights[self.b1.clone()]),
            a,
        );
        activate_slice(self.act, a);
        if self.w_mid.q8.is_some() {
            let mid_scale = quantize(a, &mut self.general_q_mid);
            let mid_offset = self.a_mid.start - self.a.end;
            let (_padding, after_mid) = rest.split_at_mut(mid_offset);
            let mid = &mut after_mid[..m];
            self.dq[self.w_mid.q8.expect("dynamic second head matrix")].gemv(
                isa,
                &self.general_q_mid,
                mid_scale,
                Some(&weights[self.b_mid.clone()]),
                mid,
            );
            activate_slice(self.act, mid);
            gemv_lanes(&weights[self.w2.clone()], &weights[self.b2.clone()], mid, out);
        } else {
            gemv_lanes(&weights[self.w2.clone()], &weights[self.b2.clone()], a, out);
        }
    }

    fn step_factorized(&mut self, x: &[f32], out: &mut [f32]) {
        let n = self.hidden;
        let fast = self.act == Activations::Fast;
        let portable = self.precision == Precision::FactorF32Portable;
        let weights = self.weights.as_ref();
        let state = self.state.as_mut();
        {
            let z = &mut state[self.z.clone()];
            z.copy_from_slice(&weights[self.b_in.clone()]);
            for (column, &value) in weights[self.w_in_cols.w.clone()].chunks_exact(n).zip(x) {
                if value != 0.0 {
                    for (output, weight) in z.iter_mut().zip(column) {
                        *output += weight * value;
                    }
                }
            }
            z.iter_mut().for_each(|value| {
                *value = if fast {
                    gelu_fast(*value)
                } else {
                    gelu(*value)
                }
            });
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let (low, gates) = state.split_at_mut(self.gi.start);
            let (gi, rest) = gates.split_at_mut(3 * n);
            let offset = self.gh.start - self.gi.end;
            let gh = &mut rest[offset..offset + 3 * n];
            let input = if index == 0 {
                &low[self.z.clone()]
            } else {
                &low[self.h.start + (index - 1) * n..self.h.start + index * n]
            };
            self.factors[2 * index].1.gemv(
                input,
                &mut self.factor_scratch,
                &weights[layer.b_ih.clone()],
                gi,
                portable,
            );
            let hidden = &mut low[self.h.start + index * n..self.h.start + (index + 1) * n];
            self.factors[2 * index + 1].1.gemv(
                hidden,
                &mut self.factor_scratch,
                &weights[layer.b_hh.clone()],
                gh,
                portable,
            );
            gru_update(hidden, gi, gh, fast);
        }
        let (low, head) = state.split_at_mut(self.cat.start);
        let (cat, rest) = head.split_at_mut(2 * n);
        let last = self.h.start + (self.layers.len() - 1) * n;
        cat[..n].copy_from_slice(&low[last..last + n]);
        cat[n..].copy_from_slice(&low[self.z.clone()]);
        let activated = &mut rest[self.a.start - self.cat.end..self.a.end - self.cat.end];
        crate::factorized::matvec(
            &weights[self.w1.w.clone()],
            cat,
            Some(&weights[self.b1.clone()]),
            activated,
            portable,
        );
        activated.iter_mut().for_each(|value| {
            *value = if fast {
                gelu_fast(*value)
            } else {
                gelu(*value)
            }
        });
        crate::factorized::matvec(
            &weights[self.w2.clone()],
            activated,
            Some(&weights[self.b2.clone()]),
            out,
            portable,
        );
    }

    /// The whole step compiled with AVX2/FMA/F16C, so the 16-bit kernels and the activation
    /// loops get those instructions even in the x86-64-v2 build.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn step_f16(&mut self, x: &[f32], out: &mut [f32]) {
        self.step_impl::<MODE_F16>(x, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn step_i16(&mut self, x: &[f32], out: &mut [f32]) {
        self.step_impl::<MODE_I16>(x, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vnni")]
    unsafe fn step_dyn_vnni(&mut self, x: &[f32], out: &mut [f32], after: Ahead) {
        self.step_dyn_impl::<true, true, false>(x, out, after, crate::vnni::Isa::Vnni512)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512vnni")]
    unsafe fn step_dyn_vnni256(&mut self, x: &[f32], out: &mut [f32], after: Ahead) {
        self.step_dyn_impl::<true, false, false>(x, out, after, crate::vnni::Isa::Vnni256)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vnni")]
    unsafe fn step_dyn_vnni_blocked(&mut self, x: &[f32], out: &mut [f32], after: Ahead) {
        self.step_dyn_impl::<true, true, false>(
            x,
            out,
            after,
            crate::vnni::Isa::Vnni512Blocked,
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vnni")]
    unsafe fn step_dyn_vnni_fused(&mut self, x: &[f32], out: &mut [f32], after: Ahead) {
        self.step_dyn_impl::<true, true, true>(x, out, after, crate::vnni::Isa::Vnni512)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn step_dyn_avx2(&mut self, x: &[f32], out: &mut [f32], after: Ahead) {
        self.step_dyn_impl::<true, false, false>(x, out, after, crate::vnni::Isa::Avx2)
    }

    /// The dynamic-int8 step. `SIMD` selects the AVX2 helpers (projection, activations,
    /// quantization), `ZMM` (only from the AVX-512 wrapper) their sixteen-lane twins; `isa` the
    /// matvec kernel. Every combination returns the same bits.
    #[inline(always)]
    fn step_dyn_impl<const SIMD: bool, const ZMM: bool, const FUSED: bool>(
        &mut self,
        x: &[f32],
        out: &mut [f32],
        mut after: Ahead,
        isa: crate::vnni::Isa,
    ) {
        use crate::vnni::{padded_cols, quantize};
        let n = self.hidden;
        let fast = self.act == Activations::Fast;
        let wts = self.weights.as_ref();
        let half = self.half.as_ref();
        let st = self.state.as_mut();
        let kp = padded_cols(n);
        let n_layers = self.layers.len();
        let mut clock = crate::vnni::prof::tick();
        let quant = |v: &[f32], q: &mut [u8]| -> f32 {
            // SAFETY: SIMD is instantiated only from AVX2-enabled wrappers.
            #[cfg(target_arch = "x86_64")]
            if SIMD {
                return unsafe { crate::vnni::quantize_avx2(v, q) };
            }
            quantize(v, q)
        };
        let gelu_all = |v: &mut [f32], ahead: &mut Ahead| {
            // SAFETY: as for `quant`; ZMM only from the AVX-512 wrapper.
            #[cfg(target_arch = "x86_64")]
            if ZMM && fast {
                return unsafe { crate::approx::avx512::gelu_slice(v, ahead) };
            }
            #[cfg(target_arch = "x86_64")]
            if SIMD && fast {
                return unsafe { crate::approx::avx2::gelu_slice(v, ahead) };
            }
            let _ = ahead;
            if fast {
                v.iter_mut().for_each(|v| *v = gelu_fast(*v));
            } else {
                v.iter_mut().for_each(|v| *v = gelu(*v));
            }
        };
        let matrix_ahead = |m: &Mat| match m.q8 {
            Some(index) => Ahead::bytes(self.dq[index].bytes()),
            None => Ahead::NONE,
        };

        {
            let b = &wts[self.b_in.clone()];
            let padded = &mut st[self.z.start..self.z.start + b.len()];
            let cols = &half[self.w_in_cols.w.clone()];
            let scales = &wts[self.w_in_cols.s.clone()];
            if SIMD && cfg!(target_arch = "x86_64") {
                // SAFETY: SIMD is instantiated only from AVX2 + FMA + F16C wrappers, ZMM only
                // from the AVX-512 one.
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if ZMM {
                        crate::vnni::project_zmm(cols, scales, b, x, &mut self.nz, padded)
                    } else {
                        crate::half::project_as::<true>(cols, scales, b, x, &mut self.nz, padded)
                    }
                }
            } else {
                crate::half::project_q16_portable(cols, scales, b, x, padded);
            }
            gelu_all(&mut st[self.z.clone()], &mut matrix_ahead(&self.layers[0].w_ih));
        }
        crate::vnni::prof::lap(0, &mut clock);
        let (hq, zq) = self.dcodes.split_at_mut(n_layers * kp);
        let (hs, zs) = self.dscales.split_at_mut(n_layers);
        zs[0] = quant(&st[self.z.clone()], zq);
        let zs = zs[0];
        crate::vnni::prof::lap(1, &mut clock);

        for (k, li) in self.layers.iter().enumerate() {
            let g = self.cell.gates() * n;
            let (lo, hi) = st.split_at_mut(self.gi.start);
            let (gi, rest) = hi.split_at_mut(g);
            let gh_off = self.gh.start - self.gi.end;
            let (rest, cells) = rest.split_at_mut(self.c.start - self.gi.end);
            let gh = &mut rest[gh_off..gh_off + g];
            let h = &mut lo[self.h.start + k * n..self.h.start + (k + 1) * n];
            let (done, cur) = hq.split_at_mut(k * kp);
            let (q_in, s_in) = if k == 0 {
                (&*zq, zs)
            } else {
                (&done[(k - 1) * kp..], hs[k - 1])
            };
            if FUSED {
                debug_assert_eq!(self.cell, Cell::Gru);
                self.dq[li.w_ih.q8.unwrap()].gemv_pair_vnni(
                    &self.dq[li.w_hh.q8.unwrap()],
                    q_in,
                    s_in,
                    Some(&wts[li.b_ih.clone()]),
                    gi,
                    &cur[..kp],
                    hs[k],
                    Some(&wts[li.b_hh.clone()]),
                    gh,
                );
                crate::vnni::prof::lap(2 + 4 * k, &mut clock);
                crate::vnni::prof::lap(3 + 4 * k, &mut clock);
            } else {
                self.dq[li.w_ih.q8.unwrap()].gemv(
                    isa,
                    q_in,
                    s_in,
                    Some(&wts[li.b_ih.clone()]),
                    gi,
                );
                crate::vnni::prof::lap(2 + 4 * k, &mut clock);
                self.dq[li.w_hh.q8.unwrap()].gemv(
                    isa,
                    &cur[..kp],
                    hs[k],
                    Some(&wts[li.b_hh.clone()]),
                    gh,
                );
                crate::vnni::prof::lap(3 + 4 * k, &mut clock);
            }
            let next = self.layers.get(k + 1).map_or(&self.w1, |l| &l.w_ih);
            match self.cell {
                // SAFETY: as for `quant`; ZMM only from the AVX-512 wrapper.
                #[cfg(target_arch = "x86_64")]
                Cell::Gru if ZMM && fast => unsafe {
                    gru_update_avx512(h, gi, gh, &mut matrix_ahead(next))
                },
                #[cfg(target_arch = "x86_64")]
                Cell::Gru if SIMD && fast => unsafe {
                    gru_update_avx2(h, gi, gh, &mut matrix_ahead(next))
                },
                Cell::Gru => gru_update(h, gi, gh, fast),
                Cell::Lstm => lstm_update(h, &mut cells[k * n..(k + 1) * n], gi, gh, fast),
            }
            crate::vnni::prof::lap(4 + 4 * k, &mut clock);
            hs[k] = quant(h, &mut cur[..kp]);
            crate::vnni::prof::lap(5 + 4 * k, &mut clock);
        }

        let a = &mut st[self.a.clone()];
        let last = &hq[(n_layers - 1) * kp..n_layers * kp];
        self.dq[self.w1.q8.unwrap()].gemv(isa, last, hs[n_layers - 1], Some(&wts[self.b1.clone()]), a);
        self.dq[self.w1z.q8.unwrap()].gemv(isa, zq, zs, None, a);
        crate::vnni::prof::lap(2 + 4 * n_layers, &mut clock);
        gelu_all(a, &mut after);
        crate::vnni::prof::lap(3 + 4 * n_layers, &mut clock);
        gemv_lanes(&wts[self.w2.clone()], &wts[self.b2.clone()], a, out);
        crate::vnni::prof::lap(4 + 4 * n_layers, &mut clock);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn step_q16x(&mut self, x: &[f32], out: &mut [f32], after: Ahead) {
        self.step_q_impl::<true>(x, out, after)
    }

    /// The Q16x step; `SIMD` selects the AVX2 kernels (only from `step_q16x`) or the portable
    /// ones, which return the same bits.
    #[inline(always)]
    fn step_q_impl<const SIMD: bool>(&mut self, x: &[f32], out: &mut [f32], mut after: Ahead) {
        let n = self.hidden;
        let fast = self.act == Activations::Fast;
        let wts = self.weights.as_ref();
        let half = self.half.as_ref();
        let st = self.state.as_mut();
        let codes: &mut [i16] = bytemuck::cast_slice_mut(&mut self.codes);
        let kc = n.div_ceil(TILE_COLS) * TILE_COLS;
        let matvec = |m: &Mat, xs: f32, q: &[i16], b: Option<&[f32]>, out: &mut [f32]| {
            if let Some(index) = m.q8 {
                let packed = &self.q8[index].1;
                if SIMD {
                    packed.gemv(q, xs, b, out);
                } else {
                    packed.gemv_portable(q, xs, b, out);
                }
                return;
            }
            let (panel, scales) = (&half[m.w.clone()], &wts[m.s.clone()]);
            // SAFETY: SIMD is instantiated only from `step_q16x` (AVX2 + FMA + F16C).
            #[cfg(target_arch = "x86_64")]
            if SIMD {
                return unsafe { crate::half::gemv_q16x(panel, scales, xs, q, n, b, out) };
            }
            crate::half::gemv_q16x_portable(panel, scales, xs, q, n, b, out)
        };
        let matrix_ahead = |m: &Mat| match m.q8 {
            Some(index) => Ahead::bytes(self.q8[index].1.bytes()),
            None if self.precision.q8() => Ahead::NONE,
            None => Ahead::new(&half[m.w.clone()]),
        };
        // Elementwise passes: the AVX2 twins (`approx::avx2`, `quantize_avx2`) return the scalar
        // functions' bits, and prefetch the next matrix while they compute.
        let gelu_all = |v: &mut [f32], ahead: &mut Ahead| {
            // SAFETY: as for `matvec`.
            #[cfg(target_arch = "x86_64")]
            if SIMD && fast {
                return unsafe { crate::approx::avx2::gelu_slice(v, ahead) };
            }
            let _ = ahead;
            if fast {
                v.iter_mut().for_each(|v| *v = gelu_fast(*v));
            } else {
                v.iter_mut().for_each(|v| *v = gelu(*v));
            }
        };
        let quantize = |x: &[f32], bound: Option<f32>, q: &mut [i16]| -> f32 {
            // SAFETY: as for `matvec`.
            #[cfg(target_arch = "x86_64")]
            if SIMD {
                return unsafe { crate::half::quantize_avx2(x, bound, q) };
            }
            crate::half::quantize(x, bound, q)
        };

        {
            if self.precision.q8() {
                let z = &mut st[self.z.clone()];
                z.copy_from_slice(&wts[self.b_in.clone()]);
                for (col, &xj) in wts[self.w_in_cols.w.clone()].chunks_exact(n).zip(x) {
                    if xj != 0.0 {
                        for (value, weight) in z.iter_mut().zip(col) {
                            *value += weight * xj;
                        }
                    }
                }
            } else {
                let b = &wts[self.b_in.clone()];
                let padded = &mut st[self.z.start..self.z.start + b.len()];
                let cols = &half[self.w_in_cols.w.clone()];
                let scales = &wts[self.w_in_cols.s.clone()];
                let simd = SIMD && cfg!(target_arch = "x86_64");
                if simd {
                    // SAFETY: as for `matvec`.
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        crate::half::project_as::<true>(cols, scales, b, x, &mut self.nz, padded)
                    }
                } else {
                    crate::half::project_q16_portable(cols, scales, b, x, padded);
                }
            }
            gelu_all(
                &mut st[self.z.clone()],
                &mut matrix_ahead(&self.layers[0].w_ih),
            );
        }
        let n_layers = self.layers.len();
        let (hq, zq) = codes.split_at_mut(n_layers * kc);
        let zs = quantize(&st[self.z.clone()], None, zq);

        for (k, li) in self.layers.iter().enumerate() {
            // State layout is [h (all layers) | z | gi | gh | cat | a | c (LSTM)].
            let g = self.cell.gates() * n;
            let (lo, hi) = st.split_at_mut(self.gi.start);
            let (gi, rest) = hi.split_at_mut(g);
            let gh_off = self.gh.start - self.gi.end;
            let (rest, cells) = rest.split_at_mut(self.c.start - self.gi.end);
            let gh = &mut rest[gh_off..gh_off + g];
            let h = &mut lo[self.h.start + k * n..self.h.start + (k + 1) * n];
            // Layer input codes: z for the first layer, the previous layer's fresh h otherwise.
            let (done, cur) = hq.split_at_mut(k * kc);
            let (q_in, s_in) = if k == 0 {
                (&*zq, zs)
            } else {
                (&done[(k - 1) * kc..], HIDDEN_SCALE)
            };
            let cur = &mut cur[..kc];
            matvec(&li.w_ih, s_in, q_in, Some(&wts[li.b_ih.clone()]), gi);
            matvec(&li.w_hh, HIDDEN_SCALE, cur, Some(&wts[li.b_hh.clone()]), gh);
            let next = self.layers.get(k + 1).map_or(&self.w1, |l| &l.w_ih);
            match self.cell {
                // SAFETY: as for `matvec`.
                #[cfg(target_arch = "x86_64")]
                Cell::Gru if SIMD && fast => unsafe {
                    gru_update_avx2(h, gi, gh, &mut matrix_ahead(next))
                },
                Cell::Gru => gru_update(h, gi, gh, fast),
                Cell::Lstm => lstm_update(h, &mut cells[k * n..(k + 1) * n], gi, gh, fast),
            }
            quantize(h, Some(1.0), cur);
        }

        if self.precision.q8() && !self.precision.q8_head() {
            let last = self.h.start + (n_layers - 1) * n;
            let (lo, hi) = st.split_at_mut(self.cat.start);
            let (cat, rest) = hi.split_at_mut(2 * n);
            cat[..n].copy_from_slice(&lo[last..last + n]);
            cat[n..].copy_from_slice(&lo[self.z.clone()]);
            let a = &mut rest[self.a.start - self.cat.end..self.a.end - self.cat.end];
            gemv_lanes(&wts[self.w1.w.clone()], &wts[self.b1.clone()], cat, a);
            gelu_all(a, &mut after);
            gemv_lanes(&wts[self.w2.clone()], &wts[self.b2.clone()], a, out);
            return;
        }
        let a = &mut st[self.a.clone()];
        let last = &hq[(n_layers - 1) * kc..];
        matvec(&self.w1, HIDDEN_SCALE, last, Some(&wts[self.b1.clone()]), a);
        matvec(&self.w1z, zs, zq, None, a);
        gelu_all(a, &mut after);
        gemv_lanes(&wts[self.w2.clone()], &wts[self.b2.clone()], a, out);
    }

    #[inline(always)]
    fn step_impl<const MODE: u8>(&mut self, x: &[f32], out: &mut [f32]) {
        self.step_impl_with_encoder_residual::<MODE>(x, &[], out);
    }

    #[inline(always)]
    fn step_impl_with_encoder_residual<const MODE: u8>(
        &mut self,
        x: &[f32],
        residual: &[f32],
        out: &mut [f32],
    ) {
        let n = self.hidden;
        let fast = self.act == Activations::Fast;
        let wts = self.weights.as_ref();
        let half = self.half.as_ref();
        let st = self.state.as_mut();
        // SAFETY (both unsafe arms): 16-bit modes are instantiated only from `step_f16`/`step_i16`.
        let matvec = |m: &Mat, b: &[f32], x: &[f32], out: &mut [f32]| match MODE {
            #[cfg(target_arch = "x86_64")]
            MODE_F16 => unsafe { crate::half::gemv(&half[m.w.clone()], b, x, out) },
            #[cfg(target_arch = "x86_64")]
            MODE_I16 => unsafe {
                crate::half::gemv_q16(&half[m.w.clone()], &wts[m.s.clone()], b, x, out)
            },
            _ => gemv(&wts[m.w.clone()], b, x, out),
        };

        // Sparse input projection (column saxpy, zero inputs skipped).
        {
            if MODE != MODE_F32 {
                let b = &wts[self.b_in.clone()];
                let padded = &mut st[self.z.start..self.z.start + b.len()];
                let cols = &half[self.w_in_cols.w.clone()];
                let scales = &wts[self.w_in_cols.s.clone()];
                // SAFETY: 16-bit modes are instantiated only from `step_f16`/`step_i16`.
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if MODE == MODE_I16 {
                        crate::half::project_as::<true>(cols, scales, b, x, &mut self.nz, padded)
                    } else {
                        crate::half::project_as::<false>(cols, scales, b, x, &mut self.nz, padded)
                    }
                }
            } else {
                let z = &mut st[self.z.clone()];
                z.copy_from_slice(&wts[self.b_in.clone()]);
                let cols = &wts[self.w_in_cols.w.clone()];
                for (col, &xj) in cols.chunks_exact(n).zip(x) {
                    if xj != 0.0 {
                        for (o, w) in z.iter_mut().zip(col) {
                            *o += w * xj;
                        }
                    }
                }
            }
            let z = &mut st[self.z.clone()];
            if !residual.is_empty() {
                debug_assert_eq!(MODE, MODE_F32);
                debug_assert_eq!(residual.len(), z.len());
                for (value, &addition) in z.iter_mut().zip(residual) {
                    *value += addition;
                }
            }
            if fast {
                z.iter_mut().for_each(|v| *v = gelu_fast(*v));
            } else {
                z.iter_mut().for_each(|v| *v = gelu(*v));
            }
        }

        for (k, li) in self.layers.iter().enumerate() {
            let h_start = self.h.start + k * n;
            // Layer input: z for the first layer, the previous layer's h otherwise.
            let inp_start = if k == 0 { self.z.start } else { h_start - n };
            // State layout is [h (all layers) | z | gi | gh | cat | a | c (LSTM)].
            let g = self.cell.gates() * n;
            let (lo, hi) = st.split_at_mut(self.gi.start);
            let (gi, rest) = hi.split_at_mut(g);
            let gh_off = self.gh.start - self.gi.end;
            let (rest, cells) = rest.split_at_mut(self.c.start - self.gi.end);
            let gh = &mut rest[gh_off..gh_off + g];
            let (inp, h): (&[f32], &mut [f32]) = if inp_start < h_start {
                let (a, b) = lo.split_at_mut(h_start);
                (&a[inp_start..inp_start + n], &mut b[..n])
            } else {
                let (a, b) = lo.split_at_mut(inp_start);
                (&b[..n], &mut a[h_start..h_start + n])
            };
            matvec(&li.w_ih, &wts[li.b_ih.clone()], inp, gi);
            matvec(&li.w_hh, &wts[li.b_hh.clone()], h, gh);
            if self.cell == Cell::Lstm {
                lstm_update(h, &mut cells[k * n..(k + 1) * n], gi, gh, fast);
                continue;
            }
            gru_update(h, gi, gh, fast);
        }

        let last = self.h.start + (self.layers.len() - 1) * n;
        let (lo, hi) = st.split_at_mut(self.cat.start);
        let (cat, rest) = hi.split_at_mut(2 * n);
        cat[..n].copy_from_slice(&lo[last..last + n]);
        cat[n..].copy_from_slice(&lo[self.z.clone()]);
        let a = &mut rest[self.a.start - self.cat.end..self.a.end - self.cat.end];
        matvec(&self.w1, &wts[self.b1.clone()], cat, a);
        if fast {
            a.iter_mut().for_each(|v| *v = gelu_fast(*v));
        } else {
            a.iter_mut().for_each(|v| *v = gelu(*v));
        }
        let (w2, b2) = (&wts[self.w2.clone()], &wts[self.b2.clone()]);
        match MODE {
            // SAFETY: 16-bit modes run only on AVX2 + FMA + F16C CPUs.
            #[cfg(target_arch = "x86_64")]
            MODE_F16 | MODE_I16 => unsafe { gemv_avx2(w2, b2, a, out) },
            _ => gemv(w2, b2, a, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rnn::{GruLayer, RnnModel};

    #[test]
    fn q8_preserves_own_representation_and_matches_scalar_reset() {
        for (n_in, hidden, layers) in [(37, 20, 2), (40, 33, 3), (211, 128, 2)] {
            for (precision, portable) in [
                (Precision::Q8r, Precision::Q8rPortable),
                (Precision::Q8x, Precision::Q8xPortable),
            ] {
                let make = || {
                    let mut value = weights(n_in, hidden, layers, 17);
                    // Bound recurrent gain as width grows. Unscaled random recurrent
                    // matrices are a determinism stress test, not a uniform error bound.
                    let scale = 1.0 / (hidden as f32).sqrt();
                    for layer in &mut value.layers {
                        layer.w_ih.iter_mut().chain(&mut layer.w_hh).for_each(|w| *w *= scale);
                    }
                    value.w1.iter_mut().chain(&mut value.w2).for_each(|w| *w *= scale);
                    value
                };
                let mut represented = make();
                for layer in &mut represented.layers {
                    for values in [&mut layer.w_ih, &mut layer.w_hh] {
                        *values = crate::quant8::PackedQ8::pack(values, 3 * hidden, hidden).unwrap().dequantize();
                    }
                }
                if precision.q8_head() {
                    for half in 0..2 {
                        let block: Vec<f32> = represented.w1.chunks_exact(2 * hidden)
                            .flat_map(|row| &row[half * hidden..(half + 1) * hidden]).copied().collect();
                        let decoded = crate::quant8::PackedQ8::pack(&block, hidden, hidden).unwrap().dequantize();
                        for (row, values) in represented.w1.chunks_exact_mut(2 * hidden).zip(decoded.chunks_exact(hidden)) {
                            row[half * hidden..(half + 1) * hidden].copy_from_slice(values);
                        }
                    }
                }
                let mut q = FastRnn::with_precision(make(), Activations::Fast, precision).unwrap();
                let mut scalar = FastRnn::with_precision(make(), Activations::Fast, portable).unwrap();
                let mut reference = FastRnn::new(represented, Activations::Fast).unwrap();
                assert!(q.weight_bytes() < reference.weight_bytes());
                let preserved = q.q8_matrices().iter().map(|(name, matrix)| {
                    let (codes, scales) = matrix.to_rows();
                    (name.clone(), crate::quant8::PackedQ8::from_rows(&codes, &scales, matrix.rows(), matrix.cols()).unwrap())
                }).collect();
                scalar.load_q8_matrices(preserved).unwrap();
                let (mut a, mut b, mut c) = ([0.0; 2], [0.0; 2], [0.0; 2]);
                let mut outputs = Vec::new();
                let mut worst = 0.0f32;
                for input in inputs(n_in, 300) {
                    q.step(&input, &mut a);
                    scalar.step(&input, &mut b);
                    reference.step(&input, &mut c);
                    assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
                    worst = worst.max((a[0] - c[0]).abs()).max((a[1] - c[1]).abs());
                    outputs.push(a);
                }
                assert!(worst < 1e-3, "{precision:?} H{hidden} own-reference error: {worst}");
                q.reset();
                for (input, expected) in inputs(n_in, 300).iter().zip(outputs) {
                    q.step(input, &mut a);
                    assert_eq!(a.map(f32::to_bits), expected.map(f32::to_bits));
                }
            }
        }
    }

    #[test]
    fn q8_unscaled_recurrent_stress_is_finite_and_bitwise_portable() {
        for (precision, portable) in [(Precision::Q8r, Precision::Q8rPortable), (Precision::Q8x, Precision::Q8xPortable)] {
            let mut simd = FastRnn::with_precision(weights(211, 128, 2, 17), Activations::Fast, precision).unwrap();
            let mut scalar = FastRnn::with_precision(weights(211, 128, 2, 17), Activations::Fast, portable).unwrap();
            let (mut a, mut b) = ([0.0; 2], [0.0; 2]);
            for input in inputs(211, 300) {
                simd.step(&input, &mut a);
                scalar.step(&input, &mut b);
                assert!(a.iter().all(|value| value.is_finite()));
                assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
            }
        }
    }

    fn weights(n_in: usize, hidden: usize, layers: usize, seed: u64) -> RnnWeights {
        let mut s = seed | 1;
        let mut v = |len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s % 2001) as f32 / 1000.0 - 1.0) * scale
                })
                .collect()
        };
        RnnWeights {
            mean: v(n_in, 1.0),
            std: v(n_in, 0.5).iter().map(|x| 1.0 + x.abs()).collect(),
            w_in: v(hidden * n_in, 0.4),
            b_in: v(hidden, 0.2),
            layers: (0..layers)
                .map(|_| GruLayer {
                    w_ih: v(3 * hidden * hidden, 0.4),
                    w_hh: v(3 * hidden * hidden, 0.4),
                    b_ih: v(3 * hidden, 0.2),
                    b_hh: v(3 * hidden, 0.2),
                })
                .collect(),
            w1: v(2 * hidden * hidden, 0.3),
            b1: v(hidden, 0.1),
            w_mid: Vec::new(),
            b_mid: Vec::new(),
            w2: v(2 * hidden, 0.3),
            b2: v(2, 0.1),
        }
    }

    fn inputs(n_in: usize, rows: usize) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|r| {
                (0..n_in)
                    .map(|c| {
                        if (r + c) % 3 == 0 {
                            0.0
                        } else {
                            ((r * 31 + c * 17) % 200) as f32 / 50.0 - 2.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn run_both(act: Activations, n_in: usize, hidden: usize, layers: usize) -> f32 {
        let mut exact = RnnModel::new(weights(n_in, hidden, layers, 7)).unwrap();
        let mut fast = FastRnn::new(weights(n_in, hidden, layers, 7), act).unwrap();
        let mut out = vec![0.0; 2];
        let mut worst = 0.0f32;
        for x in inputs(n_in, 300) {
            let want = exact.step(&x).to_vec();
            fast.step(&x, &mut out);
            for (a, b) in want.iter().zip(&out) {
                worst = worst.max((a - b).abs());
            }
        }
        worst
    }

    #[test]
    fn exact_activations_match_reference_model() {
        for (n_in, hidden, layers) in [(5, 3, 1), (37, 20, 2), (40, 33, 3)] {
            let err = run_both(Activations::Exact, n_in, hidden, layers);
            assert!(
                err < 2e-5,
                "n_in {n_in} hidden {hidden} layers {layers}: {err}"
            );
        }
    }

    #[test]
    fn fast_activations_stay_close_over_300_steps() {
        let err = run_both(Activations::Fast, 40, 33, 2);
        assert!(err < 1e-4, "{err}");
    }

    #[test]
    fn arena_segments_are_64_byte_aligned() {
        let m = FastRnn::new(weights(37, 20, 2, 3), Activations::Fast).unwrap();
        let base = m.weights.as_ref().as_ptr() as usize;
        assert_eq!(base % 64, 0);
        for r in [&m.w_in_cols.w, &m.b_in, &m.layers[1].w_hh.w, &m.w2] {
            assert_eq!((base + r.start * 4) % 64, 0);
        }
        assert_eq!(m.state.as_ref().as_ptr() as usize % 64, 0);
    }

    #[test]
    fn gemv_matches_naive_for_ragged_shapes() {
        let mut s = 99u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2001) as f32 / 1000.0 - 1.0
        };
        for rows in [1, 2, 3, 4, 5, 7, 8, 13, 384] {
            for n in [1, 7, 8, 15, 16, 17, 33, 128, 211, 256] {
                let w: Vec<f32> = (0..rows * n).map(|_| rnd()).collect();
                let b: Vec<f32> = (0..rows).map(|_| rnd()).collect();
                let x: Vec<f32> = (0..n).map(|_| rnd()).collect();
                let mut out = vec![0.0; rows];
                gemv(&w, &b, &x, &mut out);
                for r in 0..rows {
                    let want: f64 = b[r] as f64
                        + (0..n)
                            .map(|j| w[r * n + j] as f64 * x[j] as f64)
                            .sum::<f64>();
                    assert!(
                        (out[r] as f64 - want).abs() < 1e-4,
                        "rows {rows} n {n} r {r}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_step_breakdown() {
        use std::time::Instant;
        let (n_in, hidden) = (211, 128);
        let mut m = FastRnn::new(weights(n_in, hidden, 2, 11), Activations::Fast).unwrap();
        let xs = inputs(n_in, 2000);
        let mut out = vec![0.0; 2];
        let reps = 10;
        let t = Instant::now();
        for _ in 0..reps {
            for x in &xs {
                m.step(x, &mut out);
            }
        }
        let step_us = t.elapsed().as_secs_f64() * 1e6 / (reps * xs.len()) as f64;
        let w = vec![0.01f32; 3 * hidden * hidden];
        let b = vec![0.0f32; 3 * hidden];
        let h = vec![0.5f32; hidden];
        let mut g = vec![0.0f32; 3 * hidden];
        let t = Instant::now();
        for _ in 0..reps * xs.len() {
            for _ in 0..4 {
                gemv(&w, &b, std::hint::black_box(&h), &mut g);
                std::hint::black_box(&mut g);
            }
        }
        let gemv_us = t.elapsed().as_secs_f64() * 1e6 / (reps * xs.len()) as f64;
        let w1 = vec![0.01f32; 2 * hidden * hidden];
        let cat = vec![0.5f32; 2 * hidden];
        let mut a = vec![0.0f32; hidden];
        let t = Instant::now();
        for _ in 0..reps * xs.len() {
            gemv(&w1, &b[..hidden], std::hint::black_box(&cat), &mut a);
            std::hint::black_box(&mut a);
        }
        let head_us = t.elapsed().as_secs_f64() * 1e6 / (reps * xs.len()) as f64;
        let t = Instant::now();
        let mut z = vec![0.0f32; hidden];
        let cols = vec![0.01f32; n_in * hidden];
        for _ in 0..reps {
            for x in &xs {
                for (col, &xj) in cols.chunks_exact(hidden).zip(x) {
                    if xj != 0.0 {
                        for (o, w) in z.iter_mut().zip(col) {
                            *o += w * xj;
                        }
                    }
                }
                std::hint::black_box(&mut z);
            }
        }
        let saxpy_us = t.elapsed().as_secs_f64() * 1e6 / (reps * xs.len()) as f64;
        eprintln!("input saxpy {saxpy_us:.2}us");
        eprintln!("step {step_us:.2}us  4x gru gemv {gemv_us:.2}us  head gemv {head_us:.2}us");
    }

    /// Unit std, so the folded input projection W/std is W itself and rounds identically.
    fn unit_std(mut w: RnnWeights) -> RnnWeights {
        w.std.iter_mut().for_each(|v| *v = 1.0);
        w
    }

    /// The weights the half path multiplies by (for unit-std inputs).
    fn rounded(mut w: RnnWeights) -> RnnWeights {
        let round = |m: &mut Vec<f32>| m.iter_mut().for_each(|v| *v = crate::half::round_trip(*v));
        round(&mut w.w_in);
        for l in &mut w.layers {
            round(&mut l.w_ih);
            round(&mut l.w_hh);
        }
        round(&mut w.w1);
        w
    }

    #[test]
    fn f16_step_is_f32_step_on_half_rounded_weights() {
        if !crate::half::available() {
            return;
        }
        for (n_in, hidden, layers, act) in [
            (37, 32, 1, Activations::Fast),
            (211, 128, 2, Activations::Fast),
            (211, 64, 3, Activations::Exact),
        ] {
            let w = || unit_std(weights(n_in, hidden, layers, 7));
            let mut f16 = FastRnn::with_precision(w(), act, Precision::F16).unwrap();
            assert_eq!(f16.precision(), Precision::F16);
            let mut f32 = FastRnn::new(rounded(w()), act).unwrap();
            let (mut a, mut b) = (vec![0.0; 2], vec![0.0; 2]);
            // Bitwise only against the AVX2 fp32 kernel; the SSE build sums in another order.
            let avx2 = cfg!(all(target_feature = "avx2", target_feature = "fma"));
            for x in inputs(n_in, 300) {
                f16.step(&x, &mut a);
                f32.step(&x, &mut b);
                for k in 0..2 {
                    if avx2 {
                        assert_eq!(a[k].to_bits(), b[k].to_bits(), "hidden {hidden}");
                    } else {
                        assert!((a[k] - b[k]).abs() < 1e-5, "hidden {hidden}");
                    }
                }
            }
        }
    }

    #[test]
    fn f16_ragged_hidden_tracks_reference_and_is_repeatable() {
        if !crate::half::available() {
            return;
        }
        for (n_in, hidden, layers) in [(5, 3, 1), (37, 20, 2), (40, 33, 3)] {
            let w = || unit_std(weights(n_in, hidden, layers, 7));
            let mut exact = RnnModel::new(rounded(w())).unwrap();
            let mut f16 = FastRnn::with_precision(w(), Activations::Exact, Precision::F16).unwrap();
            let mut out = vec![0.0; 2];
            let mut first = Vec::new();
            for x in inputs(n_in, 300) {
                let want = exact.step(&x).to_vec();
                f16.step(&x, &mut out);
                first.extend_from_slice(&out);
                for (a, b) in want.iter().zip(&out) {
                    assert!((a - b).abs() < 2e-5, "hidden {hidden}: {a} vs {b}");
                }
            }
            f16.reset();
            let mut again = Vec::new();
            for x in inputs(n_in, 300) {
                f16.step(&x, &mut out);
                again.extend_from_slice(&out);
            }
            assert!(
                first
                    .iter()
                    .zip(&again)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    #[test]
    fn f16_falls_back_to_f32_outside_half_range() {
        let mut w = weights(12, 16, 1, 5);
        w.layers[0].w_hh[3] = 1e5;
        let m = FastRnn::with_precision(w, Activations::Fast, Precision::F16).unwrap();
        assert_eq!(m.precision(), Precision::F32);
    }

    #[test]
    fn i16_tracks_f32_on_dequantized_weights_and_repeats_bitwise() {
        if !crate::half::available() {
            return;
        }
        let dequant = |m: &mut Vec<f32>, rows: usize, cols: usize| {
            let sc = crate::half::row_scales(m, rows, cols);
            for r in 0..rows {
                for v in &mut m[r * cols..(r + 1) * cols] {
                    *v = Encoding::I16.decode(*v, sc[r]);
                }
            }
        };
        for (n_in, hidden, layers) in [(37, 20, 2), (211, 128, 2)] {
            let w = || unit_std(weights(n_in, hidden, layers, 9));
            let mut q = w();
            // Input projection scales are per input column, i.e. per row of the transpose.
            let mut t: Vec<f32> = (0..n_in * hidden)
                .map(|i| q.w_in[(i % hidden) * n_in + i / hidden])
                .collect();
            dequant(&mut t, n_in, hidden);
            q.w_in = (0..hidden * n_in)
                .map(|i| t[(i % n_in) * hidden + i / n_in])
                .collect();
            for l in &mut q.layers {
                dequant(&mut l.w_ih, 3 * hidden, hidden);
                dequant(&mut l.w_hh, 3 * hidden, hidden);
            }
            dequant(&mut q.w1, hidden, 2 * hidden);
            let mut i16 = FastRnn::with_precision(w(), Activations::Fast, Precision::I16).unwrap();
            assert_eq!(i16.precision(), Precision::I16);
            let mut f32 = FastRnn::new(q, Activations::Fast).unwrap();
            let mut orig = FastRnn::new(w(), Activations::Fast).unwrap();
            let (mut a, mut b, mut c) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
            let mut first = Vec::new();
            let (mut worst_deq, mut worst_orig) = (0.0f32, 0.0f32);
            for x in inputs(n_in, 300) {
                i16.step(&x, &mut a);
                f32.step(&x, &mut b);
                orig.step(&x, &mut c);
                first.extend_from_slice(&a);
                for k in 0..2 {
                    worst_deq = worst_deq.max((a[k] - b[k]).abs());
                    worst_orig = worst_orig.max((a[k] - c[k]).abs());
                }
            }
            assert!(
                worst_deq < 5e-5,
                "hidden {hidden}: vs dequantized {worst_deq}"
            );
            assert!(worst_orig < 2e-3, "hidden {hidden}: vs fp32 {worst_orig}");
            i16.reset();
            for (t, x) in inputs(n_in, 300).iter().enumerate() {
                i16.step(x, &mut a);
                assert_eq!(a[0].to_bits(), first[2 * t].to_bits());
                assert_eq!(a[1].to_bits(), first[2 * t + 1].to_bits());
            }
        }
    }

    #[test]
    fn q16x_tracks_f32_and_portable_path_is_bitwise_identical() {
        for (n_in, hidden, layers, lstm) in [(37, 20, 2, false), (211, 128, 2, false), (40, 33, 3, false), (37, 24, 2, true)] {
            let w = || {
                if lstm {
                    lstm_weights(n_in, hidden, layers, 9)
                } else {
                    weights(n_in, hidden, layers, 9)
                }
            };
            let mut q = FastRnn::with_precision(w(), Activations::Fast, Precision::Q16x).unwrap();
            let mut port =
                FastRnn::with_precision(w(), Activations::Fast, Precision::Q16xPortable).unwrap();
            assert_eq!(port.precision(), Precision::Q16xPortable);
            if crate::half::available() {
                assert_eq!(q.precision(), Precision::Q16x);
            }
            let mut i16 = FastRnn::with_precision(w(), Activations::Fast, Precision::I16).unwrap();
            let mut f32 = FastRnn::new(w(), Activations::Fast).unwrap();
            let (mut a, mut b, mut c, mut d) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
            let mut first = Vec::new();
            let (mut worst_q, mut worst_i16) = (0.0f32, 0.0f32);
            for x in inputs(n_in, 300) {
                q.step(&x, &mut a);
                port.step(&x, &mut b);
                i16.step(&x, &mut c);
                f32.step(&x, &mut d);
                first.extend_from_slice(&a);
                assert_eq!(a[0].to_bits(), b[0].to_bits(), "{hidden} lstm {lstm}");
                assert_eq!(a[1].to_bits(), b[1].to_bits(), "{hidden} lstm {lstm}");
                for k in 0..2 {
                    worst_q = worst_q.max((a[k] - d[k]).abs());
                    worst_i16 = worst_i16.max((c[k] - d[k]).abs());
                }
            }
            eprintln!("{n_in}/{hidden}/{layers} lstm {lstm}: q16x vs f32 {worst_q:.2e}, i16 vs f32 {worst_i16:.2e}");
            assert!(worst_q < 3e-3, "hidden {hidden}: q16x vs fp32 {worst_q}");
            q.reset();
            for (t, x) in inputs(n_in, 300).iter().enumerate() {
                q.step(x, &mut a);
                assert_eq!(a[0].to_bits(), first[2 * t].to_bits());
                assert_eq!(a[1].to_bits(), first[2 * t + 1].to_bits());
            }
        }
    }

    #[test]
    fn dynamic_int8_isas_agree_bitwise_and_track_f32() {
        use crate::vnni::{Isa, supported};
        for (n_in, hidden, layers, lstm) in [(37, 20, 2, false), (211, 128, 2, false), (40, 33, 3, false), (211, 256, 2, false), (37, 24, 2, true)] {
            let w = || {
                if lstm {
                    lstm_weights(n_in, hidden, layers, 9)
                } else {
                    weights(n_in, hidden, layers, 9)
                }
            };
            let mut precisions = vec![
                Precision::Q8Vnni,
                Precision::Q8VnniAvx2,
                Precision::Q8VnniPortable,
            ];
            if supported(Isa::Vnni256) {
                precisions.push(Precision::Q8Vnni256);
            }
            if supported(Isa::Vnni512Blocked) {
                precisions.push(Precision::Q8VnniBlocked);
            }
            if supported(Isa::Vnni512) && !lstm {
                precisions.push(Precision::Q8VnniFused);
            }
            let mut models: Vec<FastRnn> = precisions
                .iter()
                .copied()
                .map(|p| FastRnn::with_precision(w(), Activations::Fast, p).unwrap())
                .collect();
            if supported(Isa::Vnni512) {
                assert_eq!(models[0].precision(), Precision::Q8Vnni);
            }
            assert_eq!(models[2].precision(), Precision::Q8VnniPortable);
            for (model, precision) in models.iter().zip(&precisions) {
                if matches!(
                    precision,
                    Precision::Q8Vnni256
                        | Precision::Q8VnniBlocked
                        | Precision::Q8VnniFused
                ) {
                    assert_eq!(model.precision(), *precision, "strict policy selection");
                }
            }
            let mut f32 = FastRnn::new(w(), Activations::Fast).unwrap();
            let mut q8x = FastRnn::with_precision(w(), Activations::Fast, Precision::Q8x).unwrap();
            let mut q8x_out = vec![0.0f32; 2];
            let mut q8x_worst = 0.0f32;
            let mut outs = vec![vec![0.0f32; 2]; models.len()];
            let mut d = vec![0.0f32; 2];
            let (mut first, mut worst) = (Vec::new(), 0.0f32);
            for x in inputs(n_in, 300) {
                for (m, o) in models.iter_mut().zip(&mut outs) {
                    m.step(&x, o);
                }
                f32.step(&x, &mut d);
                q8x.step(&x, &mut q8x_out);
                first.extend_from_slice(&outs[0]);
                for o in &outs[1..] {
                    assert_eq!(o[0].to_bits(), outs[0][0].to_bits(), "{hidden} lstm {lstm}");
                    assert_eq!(o[1].to_bits(), outs[0][1].to_bits(), "{hidden} lstm {lstm}");
                }
                for k in 0..2 {
                    worst = worst.max((outs[0][k] - d[k]).abs());
                    q8x_worst = q8x_worst.max((q8x_out[k] - d[k]).abs());
                }
            }
            eprintln!("{n_in}/{hidden}/{layers} lstm {lstm}: dynamic int8 vs f32 max {worst:.2e} (q8x {q8x_worst:.2e})");
            // A reset restores the zero state exactly: the replay is bitwise the first pass.
            for m in &mut models {
                m.reset();
            }
            for (t, x) in inputs(n_in, 300).iter().enumerate() {
                for (m, o) in models.iter_mut().zip(&mut outs) {
                    m.step(x, o);
                    assert_eq!(o[0].to_bits(), first[2 * t].to_bits());
                    assert_eq!(o[1].to_bits(), first[2 * t + 1].to_bits());
                }
            }
        }
    }

    #[test]
    fn imported_dynamic_q8_codes_round_trip_and_match_every_isa() {
        let weights = weights(37, 20, 2, 41);
        let source = FastRnn::with_precision(
            weights_like(&weights),
            Activations::Fast,
            Precision::Q8VnniPortable,
        )
        .unwrap();
        let exported: Vec<_> = source
            .dyn8_matrices()
            .map(|(name, matrix)| {
                let (codes, scales) = matrix.to_rows();
                (name.to_owned(), matrix.rows(), matrix.cols(), codes, scales)
            })
            .collect();
        assert_eq!(exported.len(), 6);
        assert!(exported.iter().any(|(name, ..)| name == "gru.weight_ih_l0.0"));
        assert!(exported.iter().any(|(name, ..)| name == "gru.weight_hh_l1.0"));
        assert!(exported.iter().any(|(name, ..)| name == "head.0.weight.0"));
        assert!(exported.iter().any(|(name, ..)| name == "head.0.weight.1"));

        let sequence = inputs(37, 96);
        let mut reference = FastRnn::with_precision(
            weights_like(&weights),
            Activations::Fast,
            Precision::Q8VnniPortable,
        )
        .unwrap();
        assert_eq!(reference.hidden_state_shape(), (2, 20));
        assert!(reference.hidden_state().iter().all(|value| *value == 0.0));
        let mut output = [0.0f32; 2];
        let mut expected = Vec::with_capacity(sequence.len());
        let mut expected_hidden = Vec::with_capacity(sequence.len());
        for input in &sequence {
            reference.step(input, &mut output);
            expected.push(output.map(f32::to_bits));
            expected_hidden.push(
                reference
                    .hidden_state()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
        }

        for precision in [
            Precision::Q8VnniPortable,
            Precision::Q8VnniAvx2,
            Precision::Q8Vnni,
        ] {
            let mut imported = FastRnn::with_precision(
                weights_like(&weights),
                Activations::Fast,
                precision,
            )
            .unwrap();
            let payload = exported
                .iter()
                .map(|(name, rows, cols, codes, scales)| {
                    (
                        name.clone(),
                        crate::vnni::DynQ8::from_rows(codes, scales, *rows, *cols).unwrap(),
                    )
                })
                .collect();
            imported.load_dyn8_matrices(payload).unwrap();
            let round_trip: Vec<_> = imported
                .dyn8_matrices()
                .map(|(name, matrix)| {
                    let (codes, scales) = matrix.to_rows();
                    (
                        name.to_owned(),
                        matrix.rows(),
                        matrix.cols(),
                        codes,
                        scales.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    )
                })
                .collect();
            let exact: Vec<_> = exported
                .iter()
                .map(|(name, rows, cols, codes, scales)| {
                    (
                        name.clone(),
                        *rows,
                        *cols,
                        codes.clone(),
                        scales.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    )
                })
                .collect();
            assert_eq!(round_trip, exact, "{precision:?}");

            for (step, (input, expected)) in sequence.iter().zip(&expected).enumerate() {
                imported.step(input, &mut output);
                assert_eq!(output.map(f32::to_bits), *expected, "{precision:?}");
                assert_eq!(
                    imported
                        .hidden_state()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_hidden[step],
                    "{precision:?} hidden state",
                );
            }
            imported.reset();
            assert!(imported.hidden_state().iter().all(|value| *value == 0.0));
            imported.step(&sequence[0], &mut output);
            assert_eq!(output.map(f32::to_bits), expected[0], "{precision:?} reset");
            assert_eq!(
                imported
                    .hidden_state()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_hidden[0],
                "{precision:?} reset hidden state",
            );
        }

        let mut target = FastRnn::with_precision(
            weights_like(&weights),
            Activations::Fast,
            Precision::Q8VnniPortable,
        )
        .unwrap();
        let payload = || {
            exported
                .iter()
                .map(|(name, rows, cols, codes, scales)| {
                    (
                        name.clone(),
                        crate::vnni::DynQ8::from_rows(codes, scales, *rows, *cols).unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let mut wrong_name = payload();
        wrong_name[0].0.push_str(".unexpected");
        assert!(target.load_dyn8_matrices(wrong_name).is_err());
        assert!(target.load_dyn8_matrices(Vec::new()).is_err());
        let mut wrong_shape = payload();
        let name = wrong_shape[0].0.clone();
        let matrix = &wrong_shape[0].1;
        let (codes, _scales) = matrix.to_rows();
        let (rows, cols) = (matrix.rows() * 2, matrix.cols() / 2);
        let scales = vec![0.25; rows];
        wrong_shape[0] = (
            name,
            crate::vnni::DynQ8::from_rows(&codes, &scales, rows, cols).unwrap(),
        );
        assert!(target.load_dyn8_matrices(wrong_shape).is_err());
    }

    #[test]
    fn q16x_wide_head_is_the_mean_of_its_heads() {
        let (n_in, hidden) = (37, 32);
        let heads: Vec<RnnWeights> = (0..2).map(|k| weights(n_in, hidden, 2, 50 + k)).collect();
        let mut merged = weights(n_in, hidden, 2, 50);
        merged.w1 = heads.iter().flat_map(|h| h.w1.clone()).collect();
        merged.b1 = heads.iter().flat_map(|h| h.b1.clone()).collect();
        merged.w2 = (0..2)
            .flat_map(|o| {
                heads
                    .iter()
                    .flat_map(move |h| h.w2[o * hidden..(o + 1) * hidden].iter().map(|v| v / 2.0))
            })
            .collect();
        merged.b2 = (0..2)
            .map(|o| heads.iter().map(|h| h.b2[o]).sum::<f32>() / 2.0)
            .collect();
        let mut wide = FastRnn::with_precision(merged, Activations::Exact, Precision::Q16x).unwrap();
        let mut single: Vec<FastRnn> = (0..2)
            .map(|k| {
                let mut w = weights(n_in, hidden, 2, 50);
                w.w1 = heads[k].w1.clone();
                w.b1 = heads[k].b1.clone();
                w.w2 = heads[k].w2.clone();
                w.b2 = heads[k].b2.clone();
                FastRnn::with_precision(w, Activations::Exact, Precision::F32).unwrap()
            })
            .collect();
        let (mut a, mut b) = (vec![0.0; 2], vec![0.0; 2]);
        for x in inputs(n_in, 100) {
            wide.step(&x, &mut a);
            let mut mean = [0.0f32; 2];
            for m in &mut single {
                m.step(&x, &mut b);
                mean[0] += b[0] / 2.0;
                mean[1] += b[1] / 2.0;
            }
            assert!((a[0] - mean[0]).abs() < 5e-3 && (a[1] - mean[1]).abs() < 5e-3);
        }
    }

    #[test]
    fn f16_halves_matrix_bytes() {
        if !crate::half::available() {
            return;
        }
        let a = FastRnn::new(weights(211, 128, 2, 1), Activations::Fast).unwrap();
        let b = FastRnn::with_precision(weights(211, 128, 2, 1), Activations::Fast, Precision::F16)
            .unwrap();
        // 14 h^2 matrix entries and the 211 x h input projection move from 4 to 2 bytes.
        assert_eq!(
            a.weight_bytes() - b.weight_bytes(),
            (14 * 128 + 211) * 128 * 2
        );
    }

    fn lstm_weights(n_in: usize, hidden: usize, layers: usize, seed: u64) -> RnnWeights {
        let mut w = weights(n_in, hidden, layers, seed);
        let gru = weights(n_in, hidden, layers, seed + 1);
        for (l, extra) in w.layers.iter_mut().zip(gru.layers) {
            l.w_ih.extend_from_slice(&extra.w_ih[..hidden * hidden]);
            l.w_hh.extend_from_slice(&extra.w_hh[..hidden * hidden]);
            l.b_ih.extend_from_slice(&extra.b_ih[..hidden]);
            l.b_hh.extend_from_slice(&extra.b_hh[..hidden]);
        }
        w
    }

    /// f64 PyTorch-order LSTM model, straight from the definitions.
    fn lstm_reference(w: &RnnWeights, xs: &[Vec<f32>]) -> Vec<[f64; 2]> {
        let (n_in, n) = (w.mean.len(), w.b_in.len());
        let f = |v: f32| f64::from(v);
        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
        let gelu = |v: f64| 0.5 * v * (1.0 + libm::erf(v / std::f64::consts::SQRT_2));
        let mv = |m: &[f32], b: &[f32], x: &[f64]| -> Vec<f64> {
            b.iter()
                .enumerate()
                .map(|(r, bias)| {
                    f(*bias)
                        + x.iter()
                            .enumerate()
                            .map(|(j, v)| f(m[r * x.len() + j]) * v)
                            .sum::<f64>()
                })
                .collect()
        };
        let mut h = vec![vec![0.0f64; n]; w.layers.len()];
        let mut c = h.clone();
        xs.iter()
            .map(|x| {
                let xn: Vec<f64> = (0..n_in)
                    .map(|j| (f(x[j]) - f(w.mean[j])) / f(w.std[j]))
                    .collect();
                let z: Vec<f64> = mv(&w.w_in, &w.b_in, &xn).into_iter().map(gelu).collect();
                let mut inp = z.clone();
                for (k, l) in w.layers.iter().enumerate() {
                    let gi = mv(&l.w_ih, &l.b_ih, &inp);
                    let gh = mv(&l.w_hh, &l.b_hh, &h[k]);
                    for j in 0..n {
                        let g = |q: usize| gi[q * n + j] + gh[q * n + j];
                        c[k][j] = sig(g(1)) * c[k][j] + sig(g(0)) * g(2).tanh();
                        h[k][j] = sig(g(3)) * c[k][j].tanh();
                    }
                    inp = h[k].clone();
                }
                let cat: Vec<f64> = inp.iter().chain(&z).copied().collect();
                let a: Vec<f64> = mv(&w.w1, &w.b1, &cat).into_iter().map(gelu).collect();
                let y = mv(&w.w2, &w.b2, &a);
                [y[0], y[1]]
            })
            .collect()
    }

    #[test]
    fn lstm_matches_f64_reference_in_every_precision() {
        for (n_in, hidden, layers) in [(9, 5, 1), (37, 20, 2), (211, 64, 2)] {
            let xs = inputs(n_in, 200);
            let want = lstm_reference(&lstm_weights(n_in, hidden, layers, 3), &xs);
            for (act, precision, tol) in [
                (Activations::Exact, Precision::F32, 2e-5),
                (Activations::Fast, Precision::F32, 1e-4),
                (Activations::Exact, Precision::F16, 2e-2),
                (Activations::Exact, Precision::I16, 5e-3),
            ] {
                let mut m =
                    FastRnn::with_precision(lstm_weights(n_in, hidden, layers, 3), act, precision)
                        .unwrap();
                assert_eq!(m.cell(), Cell::Lstm);
                let mut out = vec![0.0; 2];
                let mut worst = 0.0f64;
                for (x, y) in xs.iter().zip(&want) {
                    m.step(x, &mut out);
                    worst = worst
                        .max((f64::from(out[0]) - y[0]).abs())
                        .max((f64::from(out[1]) - y[1]).abs());
                }
                assert!(
                    worst < tol,
                    "{n_in}/{hidden}/{layers} {act:?} {precision:?}: {worst}"
                );
                m.reset();
                m.step(&xs[0], &mut out);
                assert!(
                    (f64::from(out[0]) - want[0][0]).abs() < tol,
                    "reset clears h and c"
                );
            }
        }
        let mut bad = lstm_weights(9, 5, 2, 3);
        bad.layers[1].w_hh.pop();
        assert!(FastRnn::new(bad, Activations::Fast).is_err());
        let mut bad = weights(9, 5, 2, 3);
        bad.layers[0].b_hh.pop();
        assert!(FastRnn::new(bad, Activations::Fast).is_err());
    }

    /// k heads stacked into one head of width k*hidden (W1/b1 rows stacked, W2 columns / k) give
    /// the mean of the k heads' outputs.
    #[test]
    fn merged_wide_head_is_the_mean_of_its_heads() {
        let (n_in, hidden) = (37, 32);
        let heads: Vec<RnnWeights> = (0..3).map(|k| weights(n_in, hidden, 2, 40 + k)).collect();
        let mut merged = weights(n_in, hidden, 2, 40);
        merged.w1 = heads.iter().flat_map(|h| h.w1.clone()).collect();
        merged.b1 = heads.iter().flat_map(|h| h.b1.clone()).collect();
        merged.w2 = (0..2)
            .flat_map(|o| {
                heads
                    .iter()
                    .flat_map(move |h| h.w2[o * hidden..(o + 1) * hidden].iter().map(|v| v / 3.0))
            })
            .collect();
        merged.b2 = (0..2)
            .map(|o| heads.iter().map(|h| h.b2[o]).sum::<f32>() / 3.0)
            .collect();
        let trunk = |k: usize| {
            let mut w = weights(n_in, hidden, 2, 40);
            w.w1 = heads[k].w1.clone();
            w.b1 = heads[k].b1.clone();
            w.w2 = heads[k].w2.clone();
            w.b2 = heads[k].b2.clone();
            w
        };
        for precision in [Precision::F32, Precision::I16] {
            let mut wide =
                FastRnn::with_precision(weights_like(&merged), Activations::Exact, precision)
                    .unwrap();
            let mut single: Vec<FastRnn> = (0..3)
                .map(|k| {
                    FastRnn::with_precision(trunk(k), Activations::Exact, Precision::F32).unwrap()
                })
                .collect();
            let (mut a, mut b) = (vec![0.0; 2], vec![0.0; 2]);
            let tol = if precision == Precision::F32 {
                2e-5
            } else {
                5e-3
            };
            for x in inputs(n_in, 100) {
                wide.step(&x, &mut a);
                let mut mean = [0.0f32; 2];
                for m in &mut single {
                    m.step(&x, &mut b);
                    mean[0] += b[0] / 3.0;
                    mean[1] += b[1] / 3.0;
                }
                assert!(
                    (a[0] - mean[0]).abs() < tol && (a[1] - mean[1]).abs() < tol,
                    "{precision:?}"
                );
            }
        }
    }

    fn weights_like(w: &RnnWeights) -> RnnWeights {
        RnnWeights {
            mean: w.mean.clone(),
            std: w.std.clone(),
            w_in: w.w_in.clone(),
            b_in: w.b_in.clone(),
            layers: w
                .layers
                .iter()
                .map(|l| GruLayer {
                    w_ih: l.w_ih.clone(),
                    w_hh: l.w_hh.clone(),
                    b_ih: l.b_ih.clone(),
                    b_hh: l.b_hh.clone(),
                })
                .collect(),
            w1: w.w1.clone(),
            b1: w.b1.clone(),
            w_mid: w.w_mid.clone(),
            b_mid: w.b_mid.clone(),
            w2: w.w2.clone(),
            b2: w.b2.clone(),
        }
    }

    #[test]
    fn reset_restores_initial_output() {
        let mut m = FastRnn::new(weights(12, 8, 2, 5), Activations::Fast).unwrap();
        let x = inputs(12, 3);
        let mut first = vec![0.0; 2];
        let mut out = vec![0.0; 2];
        m.step(&x[0], &mut first);
        m.step(&x[1], &mut out);
        m.step(&x[2], &mut out);
        m.reset();
        m.step(&x[0], &mut out);
        assert_eq!(out, first);
    }

    #[test]
    fn encoder_residual_uses_equal_width_fast_path_and_matches_general_reference() {
        let residual = [0.07, -0.04, 0.03, -0.02, 0.05, -0.06, 0.01, -0.08];
        let mut residual_model = FastRnn::new(weights(12, 8, 2, 37), Activations::Exact).unwrap();
        let mut ordinary_model = FastRnn::new(weights(12, 8, 2, 37), Activations::Exact).unwrap();
        let mut general_reference =
            FastRnn::new(weights(12, 8, 2, 37), Activations::Exact).unwrap();
        assert!(residual_model.general_input.is_empty());

        let mut actual = [0.0; 2];
        let mut ordinary = [0.0; 2];
        let mut reference = [0.0; 2];
        let mut saved = Vec::new();
        let mut largest_difference = 0.0f32;
        let mut largest_residual_effect = 0.0f32;
        for input in inputs(12, 80) {
            residual_model
                .step_with_encoder_residual(&input, &residual, &mut actual)
                .unwrap();
            ordinary_model.step(&input, &mut ordinary);
            general_reference.step_general_f32_with_residual(&input, &residual, &mut reference);
            assert!(actual.iter().chain(&reference).all(|value| value.is_finite()));
            largest_difference = largest_difference
                .max((actual[0] - reference[0]).abs())
                .max((actual[1] - reference[1]).abs());
            largest_residual_effect = largest_residual_effect
                .max((actual[0] - ordinary[0]).abs())
                .max((actual[1] - ordinary[1]).abs());
            assert!(
                (actual[0] - reference[0]).abs() < 2e-4
                    && (actual[1] - reference[1]).abs() < 2e-4,
                "optimized and general residual paths diverged: {actual:?} vs {reference:?}"
            );
            saved.push(actual.map(f32::to_bits));
        }
        assert!(largest_difference < 2e-4, "path difference: {largest_difference}");
        assert!(largest_residual_effect > 1e-6, "residual was not applied");
        assert!(residual_model.general_input.is_empty());
        assert!(ordinary_model.general_input.is_empty());
        assert_eq!(general_reference.general_input.len(), 8);

        residual_model.reset();
        for (input, expected) in inputs(12, 80).iter().zip(saved) {
            residual_model
                .step_with_encoder_residual(input, &residual, &mut actual)
                .unwrap();
            assert_eq!(actual.map(f32::to_bits), expected);
        }
    }

    #[test]
    fn q8vnni_shared_backbone_matches_two_independent_heads_across_resets() {
        let mut first = weights(211, 32, 2, 91);
        let mut second = weights(211, 32, 2, 92);
        second.mean = first.mean.clone();
        second.std = first.std.clone();
        second.w_in = first.w_in.clone();
        second.b_in = first.b_in.clone();
        second.layers = first
            .layers
            .iter()
            .map(|layer| GruLayer {
                w_ih: layer.w_ih.clone(),
                w_hh: layer.w_hh.clone(),
                b_ih: layer.b_ih.clone(),
                b_hh: layer.b_hh.clone(),
            })
            .collect();
        for model in [&mut first, &mut second] {
            for layer in &mut model.layers {
                layer
                    .w_ih
                    .iter_mut()
                    .chain(&mut layer.w_hh)
                    .for_each(|value| *value *= 0.05);
            }
            model
                .w1
                .iter_mut()
                .chain(&mut model.w2)
                .for_each(|value| *value *= 0.05);
        }

        let precision = Precision::Q8VnniPortable;
        let mut baseline_first =
            FastRnn::with_precision(weights_like(&first), Activations::Fast, precision).unwrap();
        let mut baseline_second =
            FastRnn::with_precision(weights_like(&second), Activations::Fast, precision).unwrap();
        let mut shared_backbone =
            FastRnn::with_precision(weights_like(&first), Activations::Fast, precision).unwrap();
        let mut shared_second_head =
            FastRnn::with_precision(weights_like(&second), Activations::Fast, precision).unwrap();
        let input = inputs(211, 256);
        let (mut expected_first, mut expected_second, mut actual_first, mut actual_second) =
            ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);

        for _ in 0..2 {
            baseline_first.reset();
            baseline_second.reset();
            shared_backbone.reset();
            shared_second_head.reset();
            for row in &input {
                baseline_first.step(row, &mut expected_first);
                baseline_second.step(row, &mut expected_second);
                shared_backbone.step_with_shared_backbone(
                    &mut shared_second_head,
                    row,
                    &mut actual_first,
                    &mut actual_second,
                );
                assert_eq!(
                    expected_first.map(f32::to_bits),
                    actual_first.map(f32::to_bits)
                );
                assert_eq!(
                    expected_second.map(f32::to_bits),
                    actual_second.map(f32::to_bits)
                );
            }
        }
    }
}

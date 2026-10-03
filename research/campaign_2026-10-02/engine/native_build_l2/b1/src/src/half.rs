//! 16-bit weight storage for the streaming GRU.
//!
//! Matrices are packed into row panels: rows are taken four at a time and, within a group,
//! columns sixteen at a time, so one 128-byte tile holds a 4 x 16 block and a matvec reads the
//! whole matrix as a single forward stream. Ragged edges are zero-padded. Two encodings share the
//! layout and kernels:
//! - `F16`: IEEE binary16, widened with F16C. The kernel reduces in the same order as the fp32
//!   AVX2 kernel in `rnn_fast`, so for `cols % 16 == 0` and `rows % 4 == 0` it returns bit for bit
//!   what that kernel returns for the widened weights.
//! - `I16`: int16 with one fp32 scale per output row (max |w| / 32767), widened by sign extension
//!   and int->float conversion (same port cost as F16C); the scale multiplies the row's dot
//!   product. On trained GRUs its error is ~6x below F16 (absolute step max|row| / 65534 vs a
//!   relative 2^-11 per weight).
//!
//! Accumulation is fp32 throughout. Packing is portable; the kernels need AVX2, FMA and F16C at run
//! time (`available`).

use std::ops::Range;

use bytemuck::{Pod, Zeroable};

/// Rows per panel.
pub const PANEL_ROWS: usize = 4;
/// Columns per tile.
pub const TILE_COLS: usize = 16;
/// Halves per 4 x 16 tile (128 bytes).
pub const TILE: usize = PANEL_ROWS * TILE_COLS;

/// Round-to-nearest-even fp32 -> binary16 bits, with subnormals, infinities and NaN.
pub fn to_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal: shift the full significand (implicit bit included) into 10 bits.
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut out = m >> shift;
        if rest > half || (rest == half && out & 1 == 1) {
            out += 1;
        }
        return sign | out as u16;
    }
    let mut out = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && out & 1 == 1) {
        out += 1; // may carry into the exponent, which is the correct rounding to the next binade
    }
    sign | out as u16
}

/// Exact binary16 bits -> fp32.
pub fn from_bits(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: value = man * 2^-24, exactly representable in fp32.
            let v = man as f32 * f32::from_bits(0x3380_0000);
            return if sign != 0 { -v } else { v };
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// fp32 -> binary16 -> fp32, the value the half kernel actually multiplies by.
pub fn round_trip(x: f32) -> f32 {
    from_bits(to_bits(x))
}

/// Largest finite binary16 magnitude.
pub const MAX: f32 = 65504.0;

/// True when the running CPU can execute `gemv`.
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Halves a `rows x cols` panel occupies.
pub fn panel_len(rows: usize, cols: usize) -> usize {
    rows.div_ceil(PANEL_ROWS) * cols.div_ceil(TILE_COLS) * TILE
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    F16,
    I16,
}

impl Encoding {
    /// Scale for a group of weights (I16 only): max |v| / 32767, 1 for an all-zero group.
    fn scale(self, values: impl Iterator<Item = f32>) -> f32 {
        let max = values.fold(0.0f32, |m, v| m.max(v.abs()));
        if max > 0.0 { max / 32767.0 } else { 1.0 }
    }

    fn encode(self, v: f32, scale: f32) -> u16 {
        match self {
            Encoding::F16 => to_bits(v),
            Encoding::I16 => (v / scale).round().clamp(-32767.0, 32767.0) as i16 as u16,
        }
    }

    /// The value a kernel multiplies by for `v` in a group with `scale`.
    pub fn decode(self, v: f32, scale: f32) -> f32 {
        match self {
            Encoding::F16 => round_trip(v),
            Encoding::I16 => f32::from(self.encode(v, scale) as i16) * scale,
        }
    }
}

/// I16 scale of each row of a row-major `rows x cols` matrix, padded with 1.0 to whole panels.
pub fn row_scales(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut s: Vec<f32> = (0..rows)
        .map(|r| Encoding::I16.scale(w[r * cols..(r + 1) * cols].iter().copied()))
        .collect();
    s.resize(rows.div_ceil(PANEL_ROWS) * PANEL_ROWS, 1.0);
    s
}

/// I16 scale of each column of a column-major (`w[c * rows + r]`) matrix.
pub fn column_scales(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|c| Encoding::I16.scale(w[c * rows..(c + 1) * rows].iter().copied()))
        .collect()
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Line([u16; 32]);

/// 64-byte-aligned binary16 storage; every segment starts on a cache line.
pub struct HalfArena {
    lines: Vec<Line>,
}

impl Default for HalfArena {
    fn default() -> Self {
        Self::new()
    }
}

impl HalfArena {
    pub fn new() -> Self {
        Self { lines: Vec::new() }
    }

    /// Packs a row-major `rows x cols` fp32 matrix as a zero-padded F16 panel.
    pub fn push_panel(&mut self, w: &[f32], rows: usize, cols: usize) -> Range<usize> {
        self.push_panel_as(Encoding::F16, w, &[], rows, cols)
    }

    /// Packs a panel in `enc`; `scales` are `row_scales(w, ..)` for I16 and ignored for F16.
    pub fn push_panel_as(
        &mut self,
        enc: Encoding,
        w: &[f32],
        scales: &[f32],
        rows: usize,
        cols: usize,
    ) -> Range<usize> {
        assert_eq!(w.len(), rows * cols, "panel shape");
        assert!(enc == Encoding::F16 || scales.len() >= rows, "row scales");
        let len = panel_len(rows, cols);
        let start = self.lines.len() * 32;
        self.lines
            .resize(self.lines.len() + len.div_ceil(32).max(1), Line([0; 32]));
        let dst = &mut self.as_mut()[start..start + len];
        let chunks = cols.div_ceil(TILE_COLS);
        for r in 0..rows {
            let (panel, lane) = (r / PANEL_ROWS, r % PANEL_ROWS);
            for c in 0..cols {
                let (chunk, k) = (c / TILE_COLS, c % TILE_COLS);
                let scale = scales.get(r).copied().unwrap_or(1.0);
                dst[(panel * chunks + chunk) * TILE + lane * TILE_COLS + k] =
                    enc.encode(w[r * cols + c], scale);
            }
        }
        start..start + len
    }

    /// Packs a column-major `rows x cols` fp32 matrix (`w[c * rows + r]`) for `project`: rows
    /// padded to `COL_TILE`, stored tile by tile, each tile holding every column's slice.
    pub fn push_columns(&mut self, w: &[f32], rows: usize, cols: usize) -> Range<usize> {
        self.push_columns_as(Encoding::F16, w, &[], rows, cols)
    }

    /// `push_columns` in `enc`; `scales` are `column_scales(w, ..)` for I16.
    pub fn push_columns_as(
        &mut self,
        enc: Encoding,
        w: &[f32],
        scales: &[f32],
        rows: usize,
        cols: usize,
    ) -> Range<usize> {
        assert_eq!(w.len(), rows * cols, "column shape");
        assert!(
            enc == Encoding::F16 || scales.len() >= cols,
            "column scales"
        );
        let tiles = rows.div_ceil(COL_TILE);
        let len = tiles * cols * COL_TILE;
        let start = self.lines.len() * 32;
        self.lines
            .resize(self.lines.len() + len.div_ceil(32).max(1), Line([0; 32]));
        let dst = &mut self.as_mut()[start..start + len];
        for c in 0..cols {
            for r in 0..rows {
                let scale = scales.get(c).copied().unwrap_or(1.0);
                dst[((r / COL_TILE) * cols + c) * COL_TILE + r % COL_TILE] =
                    enc.encode(w[c * rows + r], scale);
            }
        }
        start..start + len
    }

    pub fn as_ref(&self) -> &[u16] {
        bytemuck::cast_slice(&self.lines)
    }

    fn as_mut(&mut self) -> &mut [u16] {
        bytemuck::cast_slice_mut(&mut self.lines)
    }

    pub fn bytes(&self) -> usize {
        self.lines.len() * 64
    }
}

/// Software prefetch distance in halves (0 disables); one tile is 64 halves.
pub const PREFETCH: usize = 0;

/// Output rows per register tile of `project` (eight 8-lane accumulators).
pub const COL_TILE: usize = 64;

/// Sparse column projection: `out[..tiles * COL_TILE] = b + sum_j col_j * x[j]` over the nonzero
/// `x[j]` in column order, for columns packed by `HalfArena::push_columns`. Each term is a
/// multiply then an add (no FMA), exactly the fp32 column saxpy in `rnn_fast` on the widened
/// weights. `nz` is scratch for the nonzero column indices (capacity `x.len()`).
///
/// # Safety
/// The CPU must support AVX2, FMA and F16C (`available()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub unsafe fn project(cols: &[u16], b: &[f32], x: &[f32], nz: &mut Vec<u32>, out: &mut [f32]) {
    unsafe { project_as::<false>(cols, &[], b, x, nz, out) }
}

/// `project` for I16 columns (`Q`), each nonzero input pre-multiplied by its column scale.
///
/// # Safety
/// The CPU must support AVX2, FMA and F16C (`available()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub unsafe fn project_as<const Q: bool>(
    cols: &[u16],
    scales: &[f32],
    b: &[f32],
    x: &[f32],
    nz: &mut Vec<u32>,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let n_cols = x.len();
    let tiles = out.len() / COL_TILE;
    assert!(out.len().is_multiple_of(COL_TILE) && b.len() >= out.len());
    assert!(cols.len() >= tiles * n_cols * COL_TILE);
    assert!(!Q || scales.len() >= n_cols);
    nz.clear();
    nz.extend((0..n_cols as u32).filter(|&j| x[j as usize] != 0.0));
    unsafe {
        for t in 0..tiles {
            let bt = b.as_ptr().add(t * COL_TILE);
            let mut acc: [__m256; 8] = std::array::from_fn(|i| _mm256_loadu_ps(bt.add(8 * i)));
            let tile = cols.as_ptr().add(t * n_cols * COL_TILE);
            for &j in nz.iter() {
                let mut xj = *x.get_unchecked(j as usize);
                if Q {
                    xj *= *scales.get_unchecked(j as usize);
                }
                let xj = _mm256_set1_ps(xj);
                let col = tile.add(j as usize * COL_TILE);
                for (i, a) in acc.iter_mut().enumerate() {
                    let w = widen::<Q>(col.add(8 * i));
                    *a = _mm256_add_ps(*a, _mm256_mul_ps(w, xj));
                }
            }
            let o = out.as_mut_ptr().add(t * COL_TILE);
            for (i, a) in acc.iter().enumerate() {
                _mm256_storeu_ps(o.add(8 * i), *a);
            }
        }
    }
}

/// out = W x + b for a panel from `HalfArena::push_panel` with `out.len()` rows and `x.len()`
/// columns.
///
/// # Safety
/// The CPU must support AVX2, FMA and F16C (`available()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub unsafe fn gemv(panel: &[u16], b: &[f32], x: &[f32], out: &mut [f32]) {
    unsafe { gemv_pf::<PREFETCH>(panel, b, x, out) }
}

/// out = diag(scales) Q x + b for an I16 panel (`push_panel_as(Encoding::I16, ..)`).
///
/// # Safety
/// The CPU must support AVX2, FMA and F16C (`available()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub unsafe fn gemv_q16(panel: &[u16], scales: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    unsafe { gemv_impl::<PREFETCH, true>(panel, scales, b, x, out) }
}

/// F16 `gemv` with an explicit prefetch distance, for tuning.
///
/// # Safety
/// The CPU must support AVX2, FMA and F16C (`available()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub unsafe fn gemv_pf<const PF: usize>(panel: &[u16], b: &[f32], x: &[f32], out: &mut [f32]) {
    unsafe { gemv_impl::<PF, false>(panel, &[], b, x, out) }
}

/// Eight 16-bit weights as fp32 lanes: binary16 (F16C) or scaled int16 (`Q`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn widen<const Q: bool>(p: *const u16) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm_loadu_si128(p as *const __m128i);
        if Q {
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v))
        } else {
            _mm256_cvtph_ps(v)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn gemv_impl<const PF: usize, const Q: bool>(
    panel: &[u16],
    scales: &[f32],
    b: &[f32],
    x: &[f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let rows = out.len();
    let cols = x.len();
    let full = cols / TILE_COLS;
    let chunks = cols.div_ceil(TILE_COLS);
    let panels = rows.div_ceil(PANEL_ROWS);
    assert!(panel.len() >= panels * chunks * TILE && b.len() >= rows);
    assert!(!Q || scales.len() >= panels * PANEL_ROWS);
    let mut tail = [0.0f32; TILE_COLS];
    tail[..cols - full * TILE_COLS].copy_from_slice(&x[full * TILE_COLS..]);
    unsafe {
        let mut p = panel.as_ptr();
        for blk in 0..panels {
            let mut acc = [_mm256_setzero_ps(); 8];
            for chunk in 0..chunks {
                let xc = if chunk < full {
                    x.as_ptr().add(chunk * TILE_COLS)
                } else {
                    tail.as_ptr()
                };
                let x0 = _mm256_loadu_ps(xc);
                let x1 = _mm256_loadu_ps(xc.add(8));
                if PF > 0 {
                    _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(PF) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(PF + 32) as *const i8);
                }
                for r in 0..PANEL_ROWS {
                    let t = p.add(r * TILE_COLS);
                    let w0 = widen::<Q>(t);
                    let w1 = widen::<Q>(t.add(8));
                    acc[2 * r] = _mm256_fmadd_ps(w0, x0, acc[2 * r]);
                    acc[2 * r + 1] = _mm256_fmadd_ps(w1, x1, acc[2 * r + 1]);
                }
                p = p.add(TILE);
            }
            let s0 = _mm256_add_ps(acc[0], acc[1]);
            let s1 = _mm256_add_ps(acc[2], acc[3]);
            let s2 = _mm256_add_ps(acc[4], acc[5]);
            let s3 = _mm256_add_ps(acc[6], acc[7]);
            // hadd twice leaves [row0..row3 low half | row0..row3 high half].
            let t = _mm256_hadd_ps(_mm256_hadd_ps(s0, s1), _mm256_hadd_ps(s2, s3));
            let mut sum = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
            let r0 = blk * PANEL_ROWS;
            if Q {
                sum = _mm_mul_ps(sum, _mm_loadu_ps(scales.as_ptr().add(r0)));
            }
            if r0 + PANEL_ROWS <= rows {
                let y = _mm_add_ps(sum, _mm_loadu_ps(b.as_ptr().add(r0)));
                _mm_storeu_ps(out.as_mut_ptr().add(r0), y);
            } else {
                let mut lanes = [0.0f32; 4];
                _mm_storeu_ps(lanes.as_mut_ptr(), sum);
                for r in r0..rows {
                    out[r] = lanes[r - r0] + b[r];
                }
            }
        }
    }
}

/// Software prefetch distance of `gemv_q16x` in bytes (0 disables). The GRU256 matrices stream
/// from L3 (or DRAM) every row; 4 KiB ahead gained ~2-5% on the i9-11900H.
pub const Q_PREFETCH: usize = 4096;

/// Largest |activation code| of `gemv_q16x`; weights use the I16 codes (|w| <= 32767).
pub const QX: f32 = 16383.0;

/// One `TILE_COLS` chunk of activation codes, 32-byte aligned for the integer kernels.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(32))]
pub struct Codes(pub [i16; TILE_COLS]);

/// Activation codes for `gemv_q16x`: `q[j] = round_half_even(clamp(x[j] / scale, ±QX))` for
/// `j < x.len()` (padding past it is left alone); returns `scale`. `bound` is a fixed max |x|
/// (1 for GRU/LSTM hidden states, which are tanh-bounded); `None` uses the vector's max |x|.
/// The clamp keeps every code within ±QX, which the overflow bound of `gemv_q16x` relies on, and
/// maps NaN to -QX in every build (`max` returns its non-NaN operand, like `maxps`). Plain IEEE
/// operations only, so all builds produce the same codes.
#[inline(always)]
pub fn quantize(x: &[f32], bound: Option<f32>, q: &mut [i16]) -> f32 {
    let m = bound.unwrap_or_else(|| x.iter().fold(0.0f32, |m, v| m.max(v.abs())));
    let q = &mut q[..x.len()];
    if !(m > 0.0 && m.is_finite()) {
        q.fill(0);
        return 0.0;
    }
    let inv = QX / m;
    for (o, v) in q.iter_mut().zip(x) {
        *o = (v * inv).max(-QX).min(QX).round_ties_even() as i16;
    }
    m / QX
}

/// Software prefetch cursor over the weights a step reads next. Compute-bound steps
/// (activations) pull a few lines per vector toward L2, so the memory system keeps streaming
/// while they run instead of idling until the next matvec misses. Prefetches never fault and
/// never change results.
#[derive(Clone, Copy)]
pub struct Ahead {
    cur: *const u8,
    end: *const u8,
}

/// Lines `Ahead::issue` requests per eight-lane vector of activation work.
pub const AHEAD_LINES: usize = 2;

impl Ahead {
    pub const NONE: Ahead = Ahead {
        cur: std::ptr::null(),
        end: std::ptr::null(),
    };

    pub fn new(s: &[u16]) -> Self {
        let cur = s.as_ptr() as *const u8;
        Self {
            cur,
            end: cur.wrapping_add(2 * s.len()),
        }
    }

    #[inline(always)]
    pub fn issue(&mut self, lines: usize) {
        #[cfg(target_arch = "x86_64")]
        for _ in 0..lines {
            if self.cur >= self.end {
                return;
            }
            // SAFETY: prefetch is a hint; any address is allowed.
            unsafe {
                std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(self.cur as *const i8)
            };
            self.cur = self.cur.wrapping_add(64);
        }
    }
}

/// `quantize` on eight lanes at a time, the same codes and scale bit for bit.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn quantize_avx2(x: &[f32], bound: Option<f32>, q: &mut [i16]) -> f32 {
    use std::arch::x86_64::*;
    let main = x.len() / 16 * 16;
    let m = match bound {
        Some(m) => m,
        None => unsafe {
            // max(|v|, acc) keeps acc when |v| is NaN, like f32::max; max is exact, so the
            // lane order does not matter.
            let mut acc = _mm256_setzero_ps();
            let abs = _mm256_set1_ps(-0.0);
            for c in x[..x.len() / 8 * 8].chunks_exact(8) {
                acc = _mm256_max_ps(_mm256_andnot_ps(abs, _mm256_loadu_ps(c.as_ptr())), acc);
            }
            let mut lanes = [0.0f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
            let m = lanes.iter().fold(0.0f32, |m, v| m.max(*v));
            x[x.len() / 8 * 8..].iter().fold(m, |m, v| m.max(v.abs()))
        },
    };
    let q = &mut q[..x.len()];
    if !(m > 0.0 && m.is_finite()) {
        q.fill(0);
        return 0.0;
    }
    let inv = QX / m;
    unsafe {
        let (vi, lo, hi) = (_mm256_set1_ps(inv), _mm256_set1_ps(-QX), _mm256_set1_ps(QX));
        let code = |p: *const f32| {
            // max(v, -QX) returns -QX for NaN like f32::max; codes stay within +-QX.
            let v = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(p), vi), lo), hi);
            _mm256_cvtps_epi32(_mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v))
        };
        for i in (0..main).step_by(16) {
            let packed = _mm256_packs_epi32(code(x.as_ptr().add(i)), code(x.as_ptr().add(i + 8)));
            let ordered = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
            _mm256_storeu_si256(q.as_mut_ptr().add(i) as *mut __m256i, ordered);
        }
    }
    for (o, v) in q[main..].iter_mut().zip(&x[main..]) {
        *o = (v * inv).max(-QX).min(QX).round_ties_even() as i16;
    }
    m / QX
}

/// Integer `gemv` for an I16 panel: `out[r] = dot_r * (scales[r] * xs) + base[r]`, where `dot_r`
/// is the exact integer dot product of row r's codes with the activation codes `q` (from
/// `quantize`, zero-padded to whole chunks) and `base` is `b`, or `out` itself when `b` is
/// `None` (accumulating a second column block). `cols` is the panel's column count.
///
/// Overflow bound: |w| <= 32767 and |q| <= QX = 16383, so one `madd_epi16` lane
/// (w0 q0 + w1 q1) is at most 2 * 32767 * 16383 = 1_073_643_522 in magnitude, and the int32
/// lane that sums two chunks' madds before it is flushed to fp32 is at most 2_147_287_044
/// < 2^31 - 1 = 2_147_483_647. Every integer is therefore exact; each flush rounds to fp32
/// (round to nearest even, as `i32 as f32`) and adds into eight fp32 lanes per row, reduced as
/// ((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7)). `gemv_q16x_portable` performs exactly these
/// operations, so both return the same bits.
///
/// # Safety
/// The CPU must support AVX2, FMA and F16C (`available()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub unsafe fn gemv_q16x(
    panel: &[u16],
    scales: &[f32],
    xs: f32,
    q: &[i16],
    cols: usize,
    b: Option<&[f32]>,
    out: &mut [f32],
) {
    unsafe { gemv_q16x_hint::<false, Q_PREFETCH>(panel, scales, xs, q, cols, b, out) }
}

/// `gemv_q16x` with its software prefetch as a parameter: `PF` bytes ahead (0 disables), with
/// the non-temporal hint when `NTA` (the line goes to L1 without displacing L2-resident weights).
/// Prefetches never change the result.
///
/// # Safety
/// The CPU must support AVX2, FMA and F16C (`available()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub unsafe fn gemv_q16x_hint<const NTA: bool, const PF: usize>(
    panel: &[u16],
    scales: &[f32],
    xs: f32,
    q: &[i16],
    cols: usize,
    b: Option<&[f32]>,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let rows = out.len();
    let chunks = cols.div_ceil(TILE_COLS);
    let panels = rows.div_ceil(PANEL_ROWS);
    assert!(panel.len() >= panels * chunks * TILE && q.len() >= chunks * TILE_COLS);
    assert!(scales.len() >= panels * PANEL_ROWS && b.is_none_or(|b| b.len() >= rows));
    unsafe {
        let mut p = panel.as_ptr() as *const __m256i;
        let qp = q.as_ptr() as *const __m256i;
        let base = b.map_or(out.as_ptr(), |b| b.as_ptr());
        let xs4 = _mm_set1_ps(xs);
        for blk in 0..panels {
            let mut f = [_mm256_setzero_ps(); PANEL_ROWS];
            let mut c = 0;
            while c + 2 <= chunks {
                if PF > 0 {
                    // Runs past the panel's end into the next matrix, which the arenas store in
                    // step order; prefetches never fault.
                    let pf = (p as *const i8).wrapping_add(PF);
                    for line in 0..4 {
                        if NTA {
                            _mm_prefetch::<_MM_HINT_NTA>(pf.wrapping_add(64 * line));
                        } else {
                            _mm_prefetch::<_MM_HINT_T0>(pf.wrapping_add(64 * line));
                        }
                    }
                }
                let x0 = _mm256_loadu_si256(qp.add(c));
                let x1 = _mm256_loadu_si256(qp.add(c + 1));
                for (r, fr) in f.iter_mut().enumerate() {
                    let i0 = _mm256_madd_epi16(_mm256_loadu_si256(p.add(r)), x0);
                    let i1 = _mm256_madd_epi16(_mm256_loadu_si256(p.add(PANEL_ROWS + r)), x1);
                    *fr = _mm256_add_ps(*fr, _mm256_cvtepi32_ps(_mm256_add_epi32(i0, i1)));
                }
                p = p.add(2 * PANEL_ROWS);
                c += 2;
            }
            if c < chunks {
                let x0 = _mm256_loadu_si256(qp.add(c));
                for (r, fr) in f.iter_mut().enumerate() {
                    let i0 = _mm256_madd_epi16(_mm256_loadu_si256(p.add(r)), x0);
                    *fr = _mm256_add_ps(*fr, _mm256_cvtepi32_ps(i0));
                }
                p = p.add(PANEL_ROWS);
            }
            // hadd twice leaves [row0..row3 low half | row0..row3 high half].
            let t = _mm256_hadd_ps(_mm256_hadd_ps(f[0], f[1]), _mm256_hadd_ps(f[2], f[3]));
            let sum = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
            let r0 = blk * PANEL_ROWS;
            let sc = _mm_mul_ps(_mm_loadu_ps(scales.as_ptr().add(r0)), xs4);
            let dot = _mm_mul_ps(sum, sc);
            if r0 + PANEL_ROWS <= rows {
                let y = _mm_add_ps(dot, _mm_loadu_ps(base.add(r0)));
                _mm_storeu_ps(out.as_mut_ptr().add(r0), y);
            } else {
                let mut lanes = [0.0f32; 4];
                _mm_storeu_ps(lanes.as_mut_ptr(), dot);
                for r in r0..rows {
                    out[r] = lanes[r - r0] + *base.add(r);
                }
            }
        }
    }
}

/// `gemv_q16x` without SIMD requirements: the same integers, roundings and fp32 additions in the
/// same order, hence the same bits (CPUs without AVX2, and the oracle of the bitwise tests).
pub fn gemv_q16x_portable(
    panel: &[u16],
    scales: &[f32],
    xs: f32,
    q: &[i16],
    cols: usize,
    b: Option<&[f32]>,
    out: &mut [f32],
) {
    let rows = out.len();
    let chunks = cols.div_ceil(TILE_COLS);
    let panels = rows.div_ceil(PANEL_ROWS);
    assert!(panel.len() >= panels * chunks * TILE && q.len() >= chunks * TILE_COLS);
    assert!(scales.len() >= panels * PANEL_ROWS && b.is_none_or(|b| b.len() >= rows));
    // One madd_epi16 lane: codes 2l and 2l+1 of row r in chunk c.
    let madd = |blk: usize, c: usize, r: usize, l: usize| -> i32 {
        let t = (blk * chunks + c) * TILE + r * TILE_COLS + 2 * l;
        let x = c * TILE_COLS + 2 * l;
        i32::from(panel[t] as i16) * i32::from(q[x])
            + i32::from(panel[t + 1] as i16) * i32::from(q[x + 1])
    };
    for blk in 0..panels {
        let mut f = [[0.0f32; 8]; PANEL_ROWS];
        let mut c = 0;
        while c < chunks {
            let pair = c + 2 <= chunks;
            for (r, fr) in f.iter_mut().enumerate() {
                for (l, lane) in fr.iter_mut().enumerate() {
                    let mut i = madd(blk, c, r, l);
                    if pair {
                        i += madd(blk, c + 1, r, l);
                    }
                    *lane += i as f32;
                }
            }
            c += if pair { 2 } else { 1 };
        }
        let r0 = blk * PANEL_ROWS;
        for (r, l) in f.iter().enumerate().take(rows.saturating_sub(r0)) {
            let sum = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
            let base = b.map_or(out[r0 + r], |b| b[r0 + r]);
            out[r0 + r] = sum * (scales[r0 + r] * xs) + base;
        }
    }
}

/// `project_as::<true>` without SIMD requirements, bit for bit: per output row, the nonzero
/// inputs in column order, each term `code * (x[j] * scale[j])` added after rounding.
pub fn project_q16_portable(cols: &[u16], scales: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    let n_cols = x.len();
    let tiles = out.len() / COL_TILE;
    assert!(out.len().is_multiple_of(COL_TILE) && b.len() >= out.len());
    assert!(cols.len() >= tiles * n_cols * COL_TILE && scales.len() >= n_cols);
    out.copy_from_slice(&b[..out.len()]);
    for (j, &xj) in x.iter().enumerate() {
        if xj == 0.0 {
            continue;
        }
        let xj = xj * scales[j];
        for (t, acc) in out.chunks_exact_mut(COL_TILE).enumerate() {
            let col = &cols[(t * n_cols + j) * COL_TILE..][..COL_TILE];
            for (a, w) in acc.iter_mut().zip(col) {
                *a += f32::from(*w as i16) * xj;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_matches_f16c_on_every_half_and_rounds_to_nearest_even() {
        for h in 0..=u16::MAX {
            let x = from_bits(h);
            if x.is_nan() {
                assert!(from_bits(to_bits(x)).is_nan());
                continue;
            }
            assert_eq!(to_bits(x), h, "round trip of {h:#06x}");
        }
        // Halfway cases between 1.0 and its successor 1 + 2^-10.
        assert_eq!(to_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(to_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(to_bits(65520.0), 0x7c00);
        assert_eq!(to_bits(65519.0), 0x7bff);
        assert_eq!(to_bits(2f32.powi(-25)), 0);
        assert_eq!(to_bits(2f32.powi(-25) * 1.5), 1);
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("f16c") {
            let mut s = 12345u32;
            for _ in 0..200_000 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let x = f32::from_bits(s);
                if x.is_nan() {
                    continue;
                }
                let hw = unsafe { f16c_bits(x) };
                assert_eq!(to_bits(x), hw, "{x:e}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "f16c")]
    unsafe fn f16c_bits(x: f32) -> u16 {
        use std::arch::x86_64::*;
        let v = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_set1_ps(x));
        _mm_extract_epi16::<0>(v) as u16
    }

    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % 2001) as f32 / 1000.0 - 1.0
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gemv_matches_widened_oracle_on_ragged_shapes() {
        if !available() {
            return;
        }
        let mut s = 7u64;
        for rows in [1, 2, 3, 4, 5, 7, 8, 13, 384] {
            for cols in [1, 7, 8, 15, 16, 17, 33, 128, 211, 256] {
                let w: Vec<f32> = (0..rows * cols).map(|_| rnd(&mut s)).collect();
                let b: Vec<f32> = (0..rows).map(|_| rnd(&mut s)).collect();
                let x: Vec<f32> = (0..cols).map(|_| rnd(&mut s)).collect();
                let mut arena = HalfArena::new();
                let _pad = arena.push_panel(&[0.5; 3], 1, 3);
                let range = arena.push_panel(&w, rows, cols);
                let mut out = vec![0.0; rows];
                unsafe { gemv(&arena.as_ref()[range], &b, &x, &mut out) };
                for r in 0..rows {
                    let want: f64 = f64::from(b[r])
                        + (0..cols)
                            .map(|j| f64::from(round_trip(w[r * cols + j])) * f64::from(x[j]))
                            .sum::<f64>();
                    assert!(
                        (f64::from(out[r]) - want).abs() < 1e-4,
                        "rows {rows} cols {cols} r {r}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn project_matches_column_saxpy_bitwise() {
        if !available() {
            return;
        }
        let mut s = 11u64;
        for (rows, cols) in [(1, 1), (20, 5), (64, 211), (128, 211), (200, 37)] {
            let w: Vec<f32> = (0..rows * cols).map(|_| rnd(&mut s)).collect();
            let b: Vec<f32> = (0..rows).map(|_| rnd(&mut s)).collect();
            let x: Vec<f32> = (0..cols)
                .map(|j| if j % 3 == 1 { 0.0 } else { rnd(&mut s) })
                .collect();
            let mut arena = HalfArena::new();
            let range = arena.push_columns(&w, rows, cols);
            let padded = rows.div_ceil(COL_TILE) * COL_TILE;
            let mut bp = b.clone();
            bp.resize(padded, 0.0);
            let mut out = vec![f32::NAN; padded];
            let mut nz = Vec::with_capacity(cols);
            unsafe { project(&arena.as_ref()[range], &bp, &x, &mut nz, &mut out) };
            let mut want = b.clone();
            for c in 0..cols {
                if x[c] != 0.0 {
                    for r in 0..rows {
                        want[r] += round_trip(w[c * rows + r]) * x[c];
                    }
                }
            }
            for r in 0..rows {
                assert_eq!(
                    out[r].to_bits(),
                    want[r].to_bits(),
                    "rows {rows} cols {cols} r {r}"
                );
            }
            assert!(out[rows..].iter().all(|v| *v == 0.0));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q16_kernels_match_dequantized_oracle() {
        if !available() {
            return;
        }
        let mut s = 21u64;
        for rows in [1, 3, 4, 13, 384] {
            for cols in [1, 15, 16, 33, 128, 211] {
                let w: Vec<f32> = (0..rows * cols).map(|_| rnd(&mut s) * 0.7).collect();
                let b: Vec<f32> = (0..rows).map(|_| rnd(&mut s)).collect();
                let x: Vec<f32> = (0..cols).map(|_| rnd(&mut s)).collect();
                let scales = row_scales(&w, rows, cols);
                let mut arena = HalfArena::new();
                let range = arena.push_panel_as(Encoding::I16, &w, &scales, rows, cols);
                let mut out = vec![0.0; rows];
                unsafe { gemv_q16(&arena.as_ref()[range], &scales, &b, &x, &mut out) };
                for r in 0..rows {
                    let dot: f64 = (0..cols)
                        .map(|j| {
                            f64::from(Encoding::I16.decode(w[r * cols + j], scales[r]))
                                * f64::from(x[j])
                        })
                        .sum();
                    let want = f64::from(b[r]) + dot;
                    assert!(
                        (f64::from(out[r]) - want).abs() < 1e-4,
                        "rows {rows} cols {cols}"
                    );
                    // Quantization error itself stays within half a step per weight.
                    let exact: f64 = (0..cols)
                        .map(|j| f64::from(w[r * cols + j]) * f64::from(x[j]))
                        .sum();
                    let bound = f64::from(scales[r])
                        * 0.5
                        * x.iter().map(|v| f64::from(v.abs())).sum::<f64>();
                    assert!((dot - exact).abs() <= bound + 1e-6);
                }
                // Column-major projection of the transposed matrix.
                let wt: Vec<f32> = (0..cols * rows)
                    .map(|i| w[(i % rows) * cols + i / rows])
                    .collect();
                let cscale = column_scales(&wt, rows, cols);
                let mut arena = HalfArena::new();
                let range = arena.push_columns_as(Encoding::I16, &wt, &cscale, rows, cols);
                let padded = rows.div_ceil(COL_TILE) * COL_TILE;
                let mut bp = b.clone();
                bp.resize(padded, 0.0);
                let mut out = vec![0.0; padded];
                let mut nz = Vec::new();
                unsafe {
                    project_as::<true>(&arena.as_ref()[range], &cscale, &bp, &x, &mut nz, &mut out)
                };
                for r in 0..rows {
                    let want: f64 = f64::from(b[r])
                        + (0..cols)
                            .map(|c| {
                                f64::from(Encoding::I16.decode(wt[c * rows + r], cscale[c]))
                                    * f64::from(x[c])
                            })
                            .sum::<f64>();
                    assert!((f64::from(out[r]) - want).abs() < 1e-4);
                }
            }
        }
    }

    /// Codes for `quantize` tests and kernels: x padded with zero codes to whole chunks.
    fn codes(x: &[f32], bound: Option<f32>) -> (Vec<i16>, f32) {
        let mut q = vec![0i16; x.len().div_ceil(TILE_COLS) * TILE_COLS];
        let s = quantize(x, bound, &mut q);
        (q, s)
    }

    #[test]
    fn quantize_rounds_half_even_within_bounds() {
        let (q, s) = codes(&[1.0, -1.0, 0.5, 2.0, -3.0, f32::NAN, 0.0], Some(1.0));
        assert_eq!(s, 1.0 / QX);
        // 0.5 * 16383 = 8191.5 rounds to the even 8192; out-of-bound and NaN inputs clamp.
        assert_eq!(&q[..7], &[16383, -16383, 8192, 16383, -16383, -16383, 0]);
        let (q, s) = codes(&[0.25, -2.0, 1.0], None);
        assert_eq!(s, 2.0 / QX);
        assert_eq!(&q[..3], &[2048, -16383, 8192]);
        assert!(q[3..].iter().all(|v| *v == 0));
        let (q, s) = codes(&[0.0; 5], None);
        assert_eq!(s, 0.0);
        assert!(q.iter().all(|v| *v == 0));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn quantize_avx2_matches_scalar_bitwise() {
        if !available() {
            return;
        }
        let mut s = 5u64;
        for len in [1usize, 7, 8, 15, 16, 17, 31, 256, 263, 512] {
            for scale in [1.0f32, 3.0, 1e-3, 0.0] {
                let mut x: Vec<f32> = (0..len).map(|_| rnd(&mut s) * scale).collect();
                if len > 3 {
                    x[1] = 0.5 / QX; // halfway codes round to even
                    x[2] = -2.5 / QX;
                }
                for bound in [None, Some(1.0), Some(0.25)] {
                    let mut want = vec![7i16; len + 9];
                    let mut got = vec![7i16; len + 9];
                    let ws = quantize(&x, bound, &mut want);
                    let gs = unsafe { quantize_avx2(&x, bound, &mut got) };
                    assert_eq!(ws.to_bits(), gs.to_bits(), "len {len} scale {scale} {bound:?}");
                    assert_eq!(want, got, "len {len} scale {scale} {bound:?}");
                }
            }
        }
        let x = [f32::NAN, 2.0, -0.7, f32::INFINITY, 0.3, -0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        for bound in [None, Some(1.0)] {
            let (mut want, mut got) = (vec![0i16; 17], vec![0i16; 17]);
            let ws = quantize(&x, bound, &mut want);
            let gs = unsafe { quantize_avx2(&x, bound, &mut got) };
            assert_eq!((ws.to_bits(), &want), (gs.to_bits(), &got), "{bound:?}");
        }
    }

    proptest::proptest! {
        /// Integer kernel vs the exact dequantized product (f64), and AVX2 vs portable bitwise.
        #[test]
        fn q16x_gemv_matches_dequantized_oracle_and_portable_bits(
            rows in 1usize..70,
            cols in 1usize..300,
            seed in proptest::prelude::any::<u64>(),
            bounded in proptest::prelude::any::<bool>(),
            accumulate in proptest::prelude::any::<bool>(),
        ) {
            let mut s = seed | 1;
            let w: Vec<f32> = (0..rows * cols).map(|_| rnd(&mut s) * 0.9).collect();
            let b: Vec<f32> = (0..rows).map(|_| rnd(&mut s)).collect();
            let x: Vec<f32> = (0..cols)
                .map(|_| if bounded { rnd(&mut s) } else { 3.0 * rnd(&mut s) })
                .collect();
            let (q, xs) = codes(&x, bounded.then_some(1.0));
            let scales = row_scales(&w, rows, cols);
            let mut arena = HalfArena::new();
            let range = arena.push_panel_as(Encoding::I16, &w, &scales, rows, cols);
            let panel = &arena.as_ref()[range];
            let init: Vec<f32> = (0..rows).map(|_| rnd(&mut s)).collect();
            let bias = (!accumulate).then_some(&b[..]);
            let mut port = init.clone();
            gemv_q16x_portable(panel, &scales, xs, &q, cols, bias, &mut port);
            for r in 0..rows {
                let terms: Vec<i64> = (0..cols)
                    .map(|j| {
                        let code = (Encoding::I16.decode(w[r * cols + j], scales[r]) / scales[r]).round();
                        code as i64 * i64::from(q[j])
                    })
                    .collect();
                let dot: i64 = terms.iter().sum();
                let unit = f64::from(scales[r]) * f64::from(xs);
                let base = if accumulate { init[r] } else { b[r] };
                let want = dot as f64 * unit + f64::from(base);
                // Integers are exact; the fp32 flushes, lane sums and final scaling each round by
                // at most 2^-24 of a partial bounded by sum |terms| (< 32 roundings here).
                let magnitude = terms.iter().map(|t| t.abs() as f64).sum::<f64>() * unit;
                let tol = 32.0 * 2f64.powi(-24) * (magnitude + f64::from(base.abs())) + 1e-12;
                proptest::prop_assert!((f64::from(port[r]) - want).abs() < tol, "r {} {} vs {}", r, port[r], want);
                // Against the unquantized product: half a weight step and half an activation step.
                let exact: f64 = (0..cols).map(|j| f64::from(w[r * cols + j]) * f64::from(x[j])).sum();
                let bound = 0.5 * f64::from(scales[r]) * x.iter().map(|v| f64::from(v.abs())).sum::<f64>()
                    + 0.5 * f64::from(xs) * w[r * cols..(r + 1) * cols].iter().map(|v| f64::from(v.abs())).sum::<f64>()
                    + 0.25 * f64::from(scales[r]) * f64::from(xs) * cols as f64;
                proptest::prop_assert!((want - f64::from(base) - exact).abs() <= bound + 1e-6);
            }
            #[cfg(target_arch = "x86_64")]
            if available() {
                let mut simd = init.clone();
                unsafe { gemv_q16x(panel, &scales, xs, &q, cols, bias, &mut simd) };
                for r in 0..rows {
                    proptest::prop_assert_eq!(simd[r].to_bits(), port[r].to_bits());
                }
            }
        }
    }

    #[test]
    fn q16x_gemv_survives_extreme_codes() {
        // Every weight at +-max and every code at +-QX with matching signs: the largest dot.
        let (rows, cols) = (8, 512);
        let w: Vec<f32> = (0..rows * cols).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let x: Vec<f32> = (0..cols).map(|j| if (cols + j) % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let (q, xs) = codes(&x, Some(1.0));
        let scales = row_scales(&w, rows, cols);
        let mut arena = HalfArena::new();
        let range = arena.push_panel_as(Encoding::I16, &w, &scales, rows, cols);
        let b = vec![0.0f32; rows];
        let mut port = vec![0.0f32; rows];
        gemv_q16x_portable(&arena.as_ref()[range.clone()], &scales, xs, &q, cols, Some(&b), &mut port);
        for r in 0..rows {
            let want: f64 = (0..cols).map(|j| f64::from(w[r * cols + j] * x[j])).sum();
            assert!((f64::from(port[r]) - want).abs() < 1e-3, "{} vs {want}", port[r]);
        }
        #[cfg(target_arch = "x86_64")]
        if available() {
            let mut simd = vec![0.0f32; rows];
            unsafe { gemv_q16x(&arena.as_ref()[range], &scales, xs, &q, cols, Some(&b), &mut simd) };
            assert!(simd.iter().zip(&port).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn portable_projection_matches_simd_bitwise() {
        if !available() {
            return;
        }
        let mut s = 31u64;
        for (rows, cols) in [(1, 1), (20, 5), (64, 211), (256, 211), (200, 37)] {
            let w: Vec<f32> = (0..rows * cols).map(|_| rnd(&mut s)).collect();
            let x: Vec<f32> = (0..cols)
                .map(|j| if j % 3 == 1 { 0.0 } else { rnd(&mut s) * 4.0 })
                .collect();
            let cscale = column_scales(&w, rows, cols);
            let mut arena = HalfArena::new();
            let range = arena.push_columns_as(Encoding::I16, &w, &cscale, rows, cols);
            let padded = rows.div_ceil(COL_TILE) * COL_TILE;
            let b: Vec<f32> = (0..padded).map(|r| if r < rows { rnd(&mut s) } else { 0.0 }).collect();
            let (mut simd, mut port) = (vec![0.0; padded], vec![0.0; padded]);
            let mut nz = Vec::new();
            let cols_q = &arena.as_ref()[range];
            unsafe { project_as::<true>(cols_q, &cscale, &b, &x, &mut nz, &mut simd) };
            project_q16_portable(cols_q, &cscale, &b, &x, &mut port);
            assert!(simd.iter().zip(&port).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn segments_are_cache_line_aligned() {
        let mut arena = HalfArena::new();
        let a = arena.push_panel(&[1.0; 5 * 3], 5, 3);
        let b = arena.push_panel(&[1.0; 4 * 16], 4, 16);
        assert_eq!(a.len(), panel_len(5, 3));
        assert_eq!(arena.as_ref().as_ptr() as usize % 64, 0);
        assert_eq!(b.start % 32, 0);
    }
}

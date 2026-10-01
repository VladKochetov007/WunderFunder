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

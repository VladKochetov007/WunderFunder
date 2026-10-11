//! Dynamic int8 inference kernels: int8 weights, per-vector int8 activations, int32 sums.
//!
//! The recurrent matvecs of `Precision::Q8Vnni` take
//!
//! ```text
//! out[r] = (sum_k w[r][k] * q[k]) * (wscale[r] * xs) + base[r]
//! ```
//!
//! where `w` are signed bytes in [-127, 127] with one fp32 scale per output row, and `q` are
//! signed activation codes in [-127, 127] produced per call from the fp32 vector as
//! `round_ties_even(x * (127 / max|x|))` with scale `xs = max|x| / 127`. The integer sum is
//! exact in int32 (|sum| <= 127 * 127 * cols), so every instruction set returns the same
//! integers; the fp32 epilogue is one shared loop of plain IEEE operations (convert, multiply,
//! multiply, add; never fused). All three kernels therefore return the same bits:
//!
//! - AVX-512 VNNI: `vpdpbusd` multiplies unsigned bytes by signed bytes, so the activation is
//!   stored with a +128 offset (`u = q + 128`) and the kernel subtracts `128 * sum_k w[r][k]`
//!   (precomputed per row) from the int32 result. Sixty-four rows share one pass over the
//!   activation: four zmm accumulators, one broadcast of four activation bytes per k-group.
//! - AVX2: weights widen to int16 and `vpmaddwd` against signed int16 activations, exact in
//!   int32 (no `vpmaddubsw`, whose int16 saturation would change results).
//! - Portable: scalar integer loops.
//!
//! Packed layout. Rows are padded to a multiple of 64 and grouped in super-blocks of 64 rows.
//! For super-block `s`, k-group `g` (four consecutive columns) and row block `i` (16 rows) there
//! is one 64-byte line at `((s * kgroups + g) * 4 + i)`; byte `4 * j + t` of the line holds
//! `w[16 i + j][4 g + t]`. A line is exactly one `vpdpbusd` source operand, and a super-block is
//! read as one contiguous stream. Padding rows and columns hold zero codes and zero scales.

use bytemuck::{Pod, Zeroable};

/// Rows per super-block (four zmm accumulators).
pub const SUPER_ROWS: usize = 64;
/// Activation byte code that represents zero in the offset (unsigned) representation.
pub const OFFSET: u8 = 128;

/// Instruction set a kernel runs on; every choice returns identical bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// AVX-512 F/BW/VNNI `vpdpbusd`.
    Vnni512,
    /// AVX2 `vpmaddwd` on widened bytes.
    Avx2,
    /// Scalar integer loops.
    Portable,
}

/// Whether the running CPU can execute `isa`.
pub fn supported(isa: Isa) -> bool {
    match isa {
        Isa::Portable => true,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c")
        }
        #[cfg(target_arch = "x86_64")]
        Isa::Vnni512 => {
            supported(Isa::Avx2)
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni")
        }
        #[cfg(not(target_arch = "x86_64"))]
        _ => false,
    }
}

/// The fastest supported instruction set.
pub fn best() -> Isa {
    [Isa::Vnni512, Isa::Avx2]
        .into_iter()
        .find(|isa| supported(*isa))
        .unwrap_or(Isa::Portable)
}

/// One 64-byte, cache-line aligned weight line (16 rows x 4 columns).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
pub struct Line(pub [i8; 64]);

/// Packed int8 matrix for the dynamic-activation kernels.
pub struct DynQ8 {
    rows: usize,
    cols: usize,
    kgroups: usize,
    lines: Vec<Line>,
    /// Per-row weight scale, padded with zeros to whole super-blocks.
    scales: Vec<f32>,
    /// `128 * sum_k w[r][k]`, the offset the VNNI kernel's unsigned activations add.
    offset_sum: Vec<i32>,
}

/// Activation length (in bytes) a matrix with `cols` columns reads: whole k-groups.
pub fn padded_cols(cols: usize) -> usize {
    cols.div_ceil(4) * 4
}

impl DynQ8 {
    /// Import exact row-major codes in [-127, 127] and finite row scales.
    pub fn from_rows(
        codes: &[i8],
        scales: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Self, &'static str> {
        if rows == 0
            || cols == 0
            || rows.checked_mul(cols) != Some(codes.len())
            || scales.len() != rows
            || scales.iter().any(|s| !s.is_finite() || *s < 0.0)
            || codes.contains(&-128)
        {
            return Err("invalid int8 codes, scales or dimensions");
        }
        let kgroups = cols.div_ceil(4);
        let supers = rows.div_ceil(SUPER_ROWS);
        let mut lines = vec![Line([0; 64]); supers * kgroups * 4];
        let mut padded = vec![0.0; supers * SUPER_ROWS];
        padded[..rows].copy_from_slice(scales);
        let mut offset_sum = vec![0i32; supers * SUPER_ROWS];
        for row in 0..rows {
            let (sb, within) = (row / SUPER_ROWS, row % SUPER_ROWS);
            let (block, lane) = (within / 16, within % 16);
            let mut sum = 0i32;
            for col in 0..cols {
                let code = codes[row * cols + col];
                sum += i32::from(code);
                let line = (sb * kgroups + col / 4) * 4 + block;
                lines[line].0[4 * lane + col % 4] = code;
            }
            offset_sum[row] = 128 * sum;
        }
        Ok(Self {
            rows,
            cols,
            kgroups,
            lines,
            scales: padded,
            offset_sum,
        })
    }

    /// Quantize finite row-major weights with a symmetric per-row scale `max|w| / 127`,
    /// round-to-nearest-even; all-zero rows get a zero scale.
    pub fn pack(weights: &[f32], rows: usize, cols: usize) -> Result<Self, &'static str> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(weights.len()) {
            return Err("invalid matrix dimensions");
        }
        if !weights.iter().all(|v| v.is_finite()) {
            return Err("quantized weights must be finite");
        }
        let mut codes = vec![0i8; rows * cols];
        let mut scales = vec![0.0f32; rows];
        for row in 0..rows {
            let values = &weights[row * cols..(row + 1) * cols];
            let maximum = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if maximum == 0.0 {
                continue;
            }
            let scale = (maximum / 127.0).max(f32::from_bits(1));
            scales[row] = scale;
            for (code, value) in codes[row * cols..(row + 1) * cols].iter_mut().zip(values) {
                *code = (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
            }
        }
        Self::from_rows(&codes, &scales, rows, cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Streamed bytes: weight lines, row scales and offset sums.
    pub fn storage_bytes(&self) -> usize {
        self.lines.len() * 64 + self.scales.len() * 4 + self.offset_sum.len() * 4
    }

    /// The weight lines as bytes, for an optional prefetch hint.
    pub fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.lines)
    }

    /// Row-major signed codes and the row scales this matrix represents.
    pub fn to_rows(&self) -> (Vec<i8>, Vec<f32>) {
        let codes = (0..self.rows * self.cols)
            .map(|index| {
                let (row, col) = (index / self.cols, index % self.cols);
                let (sb, within) = (row / SUPER_ROWS, row % SUPER_ROWS);
                let line = (sb * self.kgroups + col / 4) * 4 + within / 16;
                self.lines[line].0[4 * (within % 16) + col % 4]
            })
            .collect();
        (codes, self.scales[..self.rows].to_vec())
    }

    fn check(&self, q: &[u8], xs: f32, bias: Option<&[f32]>, out: &[f32]) {
        assert!(q.len() >= self.kgroups * 4, "activation codes need padding");
        assert!(xs.is_finite() && xs >= 0.0, "invalid activation scale");
        assert_eq!(out.len(), self.rows, "wrong output size");
        assert!(bias.is_none_or(|b| b.len() == self.rows), "wrong bias size");
    }

    /// `out = (W q) * (row_scale * xs) + bias`, or `out += ...` when `bias` is `None`.
    /// `q` holds offset codes from `quantize` (`code + 128`), padded to `padded_cols`.
    pub fn gemv(&self, isa: Isa, q: &[u8], xs: f32, bias: Option<&[f32]>, out: &mut [f32]) {
        self.check(q, xs, bias, out);
        match isa {
            // SAFETY: `supported` is the dispatch contract; `check` covers every bound.
            #[cfg(target_arch = "x86_64")]
            Isa::Vnni512 if supported(Isa::Vnni512) => unsafe {
                self.gemv_vnni(q, xs, bias, out)
            },
            #[cfg(target_arch = "x86_64")]
            Isa::Vnni512 | Isa::Avx2 if supported(Isa::Avx2) => unsafe {
                self.gemv_avx2(q, xs, bias, out)
            },
            _ => self.gemv_portable(q, xs, bias, out),
        }
    }

    /// Scalar reference (signed activation codes recovered from the offset bytes).
    pub fn gemv_portable(&self, q: &[u8], xs: f32, bias: Option<&[f32]>, out: &mut [f32]) {
        self.check(q, xs, bias, out);
        let mut acc = [0i32; SUPER_ROWS];
        for sb in 0..self.rows.div_ceil(SUPER_ROWS) {
            acc.fill(0);
            for g in 0..self.kgroups {
                for block in 0..4 {
                    let line = &self.lines[(sb * self.kgroups + g) * 4 + block].0;
                    for lane in 0..16 {
                        let mut sum = 0i32;
                        for t in 0..4 {
                            sum += i32::from(line[4 * lane + t]) * (i32::from(q[4 * g + t]) - 128);
                        }
                        acc[16 * block + lane] += sum;
                    }
                }
            }
            self.epilogue(sb, &acc, false, xs, bias, out);
        }
    }

    /// Shared fp32 epilogue for super-block `sb`; plain IEEE operations in a fixed order.
    #[inline(always)]
    fn epilogue(
        &self,
        sb: usize,
        acc: &[i32; SUPER_ROWS],
        offset: bool,
        xs: f32,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let first = sb * SUPER_ROWS;
        let count = SUPER_ROWS.min(self.rows - first);
        for j in 0..count {
            let r = first + j;
            let dot = if offset { acc[j] - self.offset_sum[r] } else { acc[j] };
            let scaled = dot as f32 * (self.scales[r] * xs);
            out[r] = scaled + bias.map_or(out[r], |b| b[r]);
        }
    }

    /// # Safety
    /// The CPU must support AVX2 (`supported(Isa::Avx2)`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn gemv_avx2(&self, q: &[u8], xs: f32, bias: Option<&[f32]>, out: &mut [f32]) {
        use std::arch::x86_64::*;
        let kg = self.kgroups;
        let mut acc = [0i32; SUPER_ROWS];
        unsafe {
            let base = self.lines.as_ptr() as *const u8;
            let offset = _mm256_set1_epi16(128);
            for sb in 0..self.rows.div_ceil(SUPER_ROWS) {
                for block in 0..4 {
                    let mut s = [_mm256_setzero_si256(); 4];
                    for g in 0..kg {
                        // Four signed codes repeated for four rows: lanes (a0 a1)(a2 a3)...
                        let word = std::ptr::read_unaligned(q.as_ptr().add(4 * g) as *const i32);
                        let a = _mm256_sub_epi16(
                            _mm256_cvtepu8_epi16(_mm_set1_epi32(word)),
                            offset,
                        );
                        let line = base.add(((sb * kg + g) * 4 + block) * 64);
                        for (quad, sum) in s.iter_mut().enumerate() {
                            let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                line.add(16 * quad) as *const __m128i
                            ));
                            *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(w, a));
                        }
                    }
                    // Each accumulator holds (row, even/odd pair) sums for four rows.
                    let lo = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_hadd_epi32(s[0], s[1]));
                    let hi = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_hadd_epi32(s[2], s[3]));
                    if (sb + 1) * SUPER_ROWS <= self.rows {
                        // Full super-block: vector epilogue, the same operations as `epilogue`.
                        let xv = _mm256_set1_ps(xs);
                        for (half, dot) in [lo, hi].into_iter().enumerate() {
                            let r0 = sb * SUPER_ROWS + 16 * block + 8 * half;
                            let scale = _mm256_mul_ps(_mm256_loadu_ps(self.scales.as_ptr().add(r0)), xv);
                            let scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(dot), scale);
                            let base = match bias {
                                Some(b) => _mm256_loadu_ps(b.as_ptr().add(r0)),
                                None => _mm256_loadu_ps(out.as_ptr().add(r0)),
                            };
                            _mm256_storeu_ps(out.as_mut_ptr().add(r0), _mm256_add_ps(scaled, base));
                        }
                    } else {
                        _mm256_storeu_si256(acc.as_mut_ptr().add(16 * block) as *mut __m256i, lo);
                        _mm256_storeu_si256(acc.as_mut_ptr().add(16 * block + 8) as *mut __m256i, hi);
                    }
                }
                if (sb + 1) * SUPER_ROWS > self.rows {
                    self.epilogue(sb, &acc, false, xs, bias, out);
                }
            }
        }
    }

    /// # Safety
    /// The CPU must support AVX-512 F/BW/VNNI and AVX2 (`supported(Isa::Vnni512)`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vnni")]
    unsafe fn gemv_vnni(&self, q: &[u8], xs: f32, bias: Option<&[f32]>, out: &mut [f32]) {
        use std::arch::x86_64::*;
        let kg = self.kgroups;
        unsafe {
            let base = self.lines.as_ptr() as *const __m512i;
            for sb in 0..self.rows.div_ceil(SUPER_ROWS) {
                let stream = base.add(sb * kg * 4);
                // Two interleaved accumulator sets hide the 5-cycle `vpdpbusd` latency.
                let mut even = [_mm512_setzero_si512(); 4];
                let mut odd = [_mm512_setzero_si512(); 4];
                let word = |g: usize| {
                    _mm512_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(4 * g) as *const i32))
                };
                let mut g = 0;
                while g + 2 <= kg {
                    // No software prefetch: interleaved trials showed T1 hints 10-15% slower.
                    let (a0, a1) = (word(g), word(g + 1));
                    for i in 0..4 {
                        even[i] = _mm512_dpbusd_epi32(even[i], a0, _mm512_load_si512(stream.add(4 * g + i) as *const _));
                        odd[i] = _mm512_dpbusd_epi32(odd[i], a1, _mm512_load_si512(stream.add(4 * g + 4 + i) as *const _));
                    }
                    g += 2;
                }
                if g < kg {
                    let a0 = word(g);
                    for i in 0..4 {
                        even[i] = _mm512_dpbusd_epi32(even[i], a0, _mm512_load_si512(stream.add(4 * g + i) as *const _));
                    }
                }
                // Vector epilogue: the same convert, multiply, multiply, add as `epilogue`.
                let xv = _mm512_set1_ps(xs);
                for i in 0..4 {
                    let r0 = sb * SUPER_ROWS + 16 * i;
                    if r0 >= self.rows {
                        break;
                    }
                    let mask = ((1u32 << (self.rows - r0).min(16)) - 1) as u16;
                    let total = _mm512_add_epi32(even[i], odd[i]);
                    let dot = _mm512_sub_epi32(total, _mm512_loadu_si512(self.offset_sum.as_ptr().add(r0) as *const _));
                    let scale = _mm512_mul_ps(_mm512_loadu_ps(self.scales.as_ptr().add(r0)), xv);
                    let scaled = _mm512_mul_ps(_mm512_cvtepi32_ps(dot), scale);
                    let base = match bias {
                        Some(b) => _mm512_maskz_loadu_ps(mask, b.as_ptr().add(r0)),
                        None => _mm512_maskz_loadu_ps(mask, out.as_ptr().add(r0)),
                    };
                    _mm512_mask_storeu_ps(out.as_mut_ptr().add(r0), mask, _mm512_add_ps(scaled, base));
                }
            }
        }
    }
}

/// Offset activation codes for `DynQ8::gemv`: `q[j] = 128 + round_ties_even(clamp(x[j] * inv,
/// ±127))` with `inv = 127 / max|x|`; bytes past `x.len()` up to the next multiple of four are
/// set to 128 (code 0). Returns the dequantization scale `max|x| / 127`, or 0 with all-zero
/// codes when `x` is all zero or non-finite. NaN maps to code -127 on every path (`max` returns
/// its non-NaN operand, like `maxps`). Plain IEEE operations, so all builds agree.
#[inline(always)]
pub fn quantize(x: &[f32], q: &mut [u8]) -> f32 {
    let m = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    finish_quantize(x, m, q)
}

#[inline(always)]
fn finish_quantize(x: &[f32], m: f32, q: &mut [u8]) -> f32 {
    let padded = padded_cols(x.len());
    let q = &mut q[..padded];
    if !(m > 0.0 && m.is_finite()) {
        q.fill(OFFSET);
        return 0.0;
    }
    let inv = 127.0 / m;
    for (o, v) in q.iter_mut().zip(x) {
        *o = ((v * inv).max(-127.0).min(127.0).round_ties_even() as i32 + 128) as u8;
    }
    q[x.len()..].fill(OFFSET);
    m / 127.0
}

/// `quantize` on 16 lanes at a time; the same codes and scale bit for bit.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn quantize_avx2(x: &[f32], q: &mut [u8]) -> f32 {
    use std::arch::x86_64::*;
    let m = unsafe {
        let mut acc = _mm256_setzero_ps();
        let sign = _mm256_set1_ps(-0.0);
        for c in x[..x.len() / 8 * 8].chunks_exact(8) {
            acc = _mm256_max_ps(_mm256_andnot_ps(sign, _mm256_loadu_ps(c.as_ptr())), acc);
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let m = lanes.iter().fold(0.0f32, |m, v| m.max(*v));
        x[x.len() / 8 * 8..].iter().fold(m, |m, v| m.max(v.abs()))
    };
    let padded = padded_cols(x.len());
    let q = &mut q[..padded];
    if !(m > 0.0 && m.is_finite()) {
        q.fill(OFFSET);
        return 0.0;
    }
    let inv = 127.0 / m;
    let main = x.len() / 16 * 16;
    unsafe {
        let (vi, lo, hi) = (_mm256_set1_ps(inv), _mm256_set1_ps(-127.0), _mm256_set1_ps(127.0));
        let code = |p: *const f32| {
            let v = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(p), vi), lo), hi);
            _mm256_cvtps_epi32(_mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v))
        };
        let offset = _mm256_set1_epi16(128);
        for i in (0..main).step_by(16) {
            let packed = _mm256_packs_epi32(code(x.as_ptr().add(i)), code(x.as_ptr().add(i + 8)));
            let ordered = _mm256_add_epi16(_mm256_permute4x64_epi64::<0b11_01_10_00>(packed), offset);
            let bytes = _mm256_packus_epi16(ordered, ordered);
            _mm_storel_epi64(q.as_mut_ptr().add(i) as *mut __m128i, _mm256_castsi256_si128(bytes));
            _mm_storel_epi64(
                q.as_mut_ptr().add(i + 8) as *mut __m128i,
                _mm256_extracti128_si256::<1>(bytes),
            );
        }
    }
    for (o, v) in q[main..].iter_mut().zip(&x[main..]) {
        *o = ((v * inv).max(-127.0).min(127.0).round_ties_even() as i32 + 128) as u8;
    }
    q[x.len()..].fill(OFFSET);
    m / 127.0
}

/// `half::project_as::<true>` on sixteen lanes: the same per-lane multiply then add in the same
/// column order, so every output equals the AVX2 version's bits.
///
/// # Safety
/// The CPU must support AVX-512F (and BW for the surrounding wrapper).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn project_zmm(
    cols: &[u16],
    scales: &[f32],
    b: &[f32],
    x: &[f32],
    nz: &mut Vec<u32>,
    scaled_x: &mut [f32],
    out: &mut [f32],
) {
    use crate::half::COL_TILE;
    use std::arch::x86_64::*;
    let n_cols = x.len();
    let tiles = out.len() / COL_TILE;
    assert!(out.len().is_multiple_of(COL_TILE) && b.len() >= out.len());
    assert!(cols.len() >= tiles * n_cols * COL_TILE && scales.len() >= n_cols);
    assert!(scaled_x.len() >= n_cols);
    nz.clear();
    nz.extend((0..n_cols as u32).filter(|&j| x[j as usize] != 0.0));
    for &j in nz.iter() {
        let j = j as usize;
        scaled_x[j] = unsafe { *x.get_unchecked(j) * *scales.get_unchecked(j) };
    }
    unsafe {
        for t in 0..tiles {
            let bt = b.as_ptr().add(t * COL_TILE);
            let mut acc: [__m512; 4] = std::array::from_fn(|i| _mm512_loadu_ps(bt.add(16 * i)));
            let tile = cols.as_ptr().add(t * n_cols * COL_TILE);
            for &j in nz.iter() {
                let xj = _mm512_set1_ps(*scaled_x.get_unchecked(j as usize));
                let col = tile.add(j as usize * COL_TILE);
                for (i, a) in acc.iter_mut().enumerate() {
                    let v = _mm256_loadu_si256(col.add(16 * i) as *const __m256i);
                    let w = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                    *a = _mm512_add_ps(*a, _mm512_mul_ps(w, xj));
                }
            }
            let o = out.as_mut_ptr().add(t * COL_TILE);
            for (i, a) in acc.iter().enumerate() {
                _mm512_storeu_ps(o.add(16 * i), *a);
            }
        }
    }
}

/// Optional per-stage cycle accounting of the dynamic-int8 step (cargo feature `profile`);
/// compiles to nothing otherwise. Slots: 0 projection + GELU, 1 quantize z, then per layer k
/// `2 + 4k` ih matvec, `3 + 4k` hh matvec, `4 + 4k` gate update, `5 + 4k` requantize h; after the
/// last layer (2 layers: 10..) head.0 matvecs, head GELU, head.2.
pub mod prof {
    #[cfg(feature = "profile")]
    pub static CYCLES: [std::sync::atomic::AtomicU64; 32] =
        [const { std::sync::atomic::AtomicU64::new(0) }; 32];

    /// Time stamp, or 0 when profiling is off.
    #[inline(always)]
    pub fn tick() -> u64 {
        #[cfg(all(feature = "profile", target_arch = "x86_64"))]
        // SAFETY: rdtsc has no preconditions.
        return unsafe { std::arch::x86_64::_rdtsc() };
        #[cfg(not(all(feature = "profile", target_arch = "x86_64")))]
        0
    }

    /// Charge the time since `*t` to `slot` and restart the clock.
    #[inline(always)]
    #[allow(unused_variables)]
    pub fn lap(slot: usize, t: &mut u64) {
        #[cfg(feature = "profile")]
        {
            let now = tick();
            CYCLES[slot].fetch_add(now - *t, std::sync::atomic::Ordering::Relaxed);
            *t = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / 16777216.0 - 0.5) * 2.0
        }
    }

    fn matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng(seed);
        (0..rows * cols).map(|_| r.next() * 0.3).collect()
    }

    fn isas() -> Vec<Isa> {
        [Isa::Portable, Isa::Avx2, Isa::Vnni512]
            .into_iter()
            .filter(|i| supported(*i))
            .collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn zmm_i16_projection_precompute_matches_avx2_bits() {
        if !supported(Isa::Vnni512) {
            return;
        }
        let (rows, cols) = (384, 37);
        let mut rng = Rng(41);
        let weights: Vec<f32> = (0..rows * cols).map(|_| rng.next() * 0.7).collect();
        let scales = crate::half::column_scales(&weights, rows, cols);
        let mut arena = crate::half::HalfArena::new();
        let range = arena.push_columns_as(
            crate::half::Encoding::I16,
            &weights,
            &scales,
            rows,
            cols,
        );
        let x: Vec<f32> = (0..cols)
            .map(|j| if j % 5 == 0 { 0.0 } else { rng.next() })
            .collect();
        let bias: Vec<f32> = (0..rows).map(|_| rng.next()).collect();
        let mut avx2 = vec![0.0; rows];
        let mut zmm = vec![0.0; rows];
        let mut avx2_nz = Vec::new();
        let mut zmm_nz = Vec::new();
        let mut scaled_x = vec![0.0; cols];
        unsafe {
            crate::half::project_as::<true>(
                &arena.as_ref()[range.clone()],
                &scales,
                &bias,
                &x,
                &mut avx2_nz,
                &mut avx2,
            );
            project_zmm(
                &arena.as_ref()[range],
                &scales,
                &bias,
                &x,
                &mut zmm_nz,
                &mut scaled_x,
                &mut zmm,
            );
        }
        assert_eq!(avx2_nz, zmm_nz);
        assert_eq!(
            avx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            zmm.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn pack_round_trip_keeps_codes() {
        for (rows, cols) in [(1, 1), (5, 7), (64, 8), (70, 13), (192, 256)] {
            let m = DynQ8::pack(&matrix(rows, cols, 5), rows, cols).unwrap();
            let (codes, scales) = m.to_rows();
            let again = DynQ8::from_rows(&codes, &scales, rows, cols).unwrap();
            assert_eq!(again.to_rows(), (codes, scales));
        }
    }

    #[test]
    fn quantize_paths_agree_and_handle_edges() {
        let mut r = Rng(9);
        for len in [0usize, 1, 3, 7, 8, 15, 16, 17, 31, 33, 256, 257] {
            let mut x: Vec<f32> = (0..len).map(|_| r.next() * 3.0).collect();
            if len > 2 {
                x[1] = -4.0; // exact extreme
            }
            let (mut a, mut b) = (vec![7u8; len + 8], vec![9u8; len + 8]);
            let sa = quantize(&x, &mut a);
            let sb = unsafe { quantize_avx2(&x, &mut b) };
            assert_eq!(sa.to_bits(), sb.to_bits(), "len {len}");
            assert_eq!(a[..padded_cols(len)], b[..padded_cols(len)], "len {len}");
            assert!(a[..len].iter().all(|c| (1..=255).contains(c)));
        }
        let mut q = vec![0u8; 8];
        assert_eq!(quantize(&[0.0; 5], &mut q), 0.0);
        assert!(q.iter().all(|c| *c == OFFSET));
        assert_eq!(quantize(&[f32::INFINITY, 1.0], &mut q), 0.0);
        // The extreme maps to +-127 exactly and dequantization is within half a step.
        let x = [0.5f32, -1.0, 0.25, 0.0];
        let s = quantize(&x, &mut q);
        assert_eq!(q[1], 1);
        for (c, v) in q.iter().zip(x) {
            assert!((f32::from(*c) - 128.0) * s - v <= s / 2.0 + 1e-7);
        }
    }

    #[test]
    fn kernels_match_integer_oracle_bitwise() {
        let mut r = Rng(3);
        for (rows, cols) in [(1, 1), (3, 5), (64, 4), (65, 9), (130, 33), (768, 256), (1152, 384)] {
            let m = DynQ8::pack(&matrix(rows, cols, 17), rows, cols).unwrap();
            let x: Vec<f32> = (0..cols).map(|_| r.next()).collect();
            let bias: Vec<f32> = (0..rows).map(|_| r.next()).collect();
            let mut q = vec![0u8; padded_cols(cols)];
            let xs = quantize(&x, &mut q);
            // Independent oracle from the exported codes.
            let (codes, scales) = m.to_rows();
            let want: Vec<f32> = (0..rows)
                .map(|row| {
                    let dot: i32 = (0..cols)
                        .map(|k| i32::from(codes[row * cols + k]) * (i32::from(q[k]) - 128))
                        .sum();
                    dot as f32 * (scales[row] * xs) + bias[row]
                })
                .collect();
            for isa in isas() {
                let mut out = vec![0.0f32; rows];
                m.gemv(isa, &q, xs, Some(&bias), &mut out);
                assert_eq!(
                    out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{isa:?} {rows}x{cols}"
                );
                // Accumulating form: out += W q.
                let mut acc = bias.clone();
                m.gemv(isa, &q, xs, None, &mut acc);
                for row in 0..rows {
                    let plain = want[row] - bias[row];
                    assert!((acc[row] - (bias[row] + plain)).abs() < 1e-4, "{isa:?}");
                }
            }
        }
    }

    #[test]
    fn extreme_codes_do_not_overflow() {
        // Every weight -127 or 127 against saturated activations: largest-magnitude sums.
        let (rows, cols) = (64, 512);
        let weights: Vec<f32> = (0..rows * cols).map(|i| if (i % cols) % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let m = DynQ8::pack(&weights, rows, cols).unwrap();
        let x: Vec<f32> = (0..cols).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let mut q = vec![0u8; cols];
        let xs = quantize(&x, &mut q);
        let reference = {
            let mut out = vec![0.0f32; rows];
            m.gemv_portable(&q, xs, Some(&vec![0.0; rows]), &mut out);
            out
        };
        assert!(reference.iter().all(|v| (*v - cols as f32).abs() < 1e-2));
        for isa in isas() {
            let mut out = vec![0.0f32; rows];
            m.gemv(isa, &q, xs, Some(&vec![0.0; rows]), &mut out);
            assert_eq!(out, reference, "{isa:?}");
        }
    }

    #[test]
    fn rejects_invalid_import() {
        assert!(DynQ8::from_rows(&[1, 2], &[1.0], 1, 3).is_err());
        assert!(DynQ8::from_rows(&[-128], &[1.0], 1, 1).is_err());
        assert!(DynQ8::from_rows(&[1], &[f32::NAN], 1, 1).is_err());
        assert!(DynQ8::pack(&[f32::INFINITY], 1, 1).is_err());
    }
}

//! Experimental signed-byte weights with the existing int16 activation interface.
//!
//! This module is deliberately separate from submission precision selection. It tests
//! the bandwidth premise before a full recurrent export or quality claim. Four rows
//! share each 16-column panel, using 64 bytes instead of the int16 path's 128 bytes.
//! Integer products are summed in pairs of chunks, converted to eight fp32 lanes,
//! then reduced in the same fixed order as q16x. The AVX2 and scalar paths agree
//! bitwise; only weight quantization differs from the existing q16x representation.

const ROWS: usize = 4;
const COLS: usize = 16;
const TILE: usize = ROWS * COLS;

/// Portable packed matrix; scales are per output row, weights lie in [-127, 127].
pub struct PackedQ8 {
    rows: usize,
    cols: usize,
    scales: Vec<f32>,
    codes: Vec<i8>,
}

impl PackedQ8 {
    /// Quantize finite row-major weights, with round-to-nearest-even and zero padding.
    pub fn pack(weights: &[f32], rows: usize, cols: usize) -> Result<Self, &'static str> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(weights.len()) {
            return Err("invalid matrix dimensions");
        }
        if !weights.iter().all(|v| v.is_finite()) {
            return Err("quantized weights must be finite");
        }
        let chunks = cols.div_ceil(COLS);
        let panels = rows.div_ceil(ROWS);
        let capacity = panels
            .checked_mul(chunks)
            .and_then(|n| n.checked_mul(TILE))
            .ok_or("packed matrix dimensions overflow")?;
        let mut scales = vec![0.0; panels * ROWS];
        let mut codes = vec![0; capacity];
        for row in 0..rows {
            let values = &weights[row * cols..(row + 1) * cols];
            let maximum = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if maximum == 0.0 {
                continue;
            }
            // Retain subnormal rows instead of dividing by an underflowed zero scale.
            let scale = (maximum / 127.0).max(f32::from_bits(1));
            scales[row] = scale;
            for (col, value) in values.iter().enumerate() {
                let index =
                    ((row / ROWS) * chunks + col / COLS) * TILE + (row % ROWS) * COLS + col % COLS;
                codes[index] = (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
            }
        }
        Ok(Self {
            rows,
            cols,
            scales,
            codes,
        })
    }

    /// Number of columns, before zero padding.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of output rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Stored code and scale bytes, excluding vector metadata.
    pub fn storage_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * size_of::<f32>()
    }

    /// Reconstruct the exact represented weights for model-side quantization diagnostics.
    pub fn dequantize(&self) -> Vec<f32> {
        let chunks = self.cols.div_ceil(COLS);
        (0..self.rows * self.cols)
            .map(|index| {
                let (row, col) = (index / self.cols, index % self.cols);
                let packed =
                    ((row / ROWS) * chunks + col / COLS) * TILE + (row % ROWS) * COLS + col % COLS;
                f32::from(self.codes[packed]) * self.scales[row]
            })
            .collect()
    }

    fn validate(&self, q: &[i16], xs: f32, bias: Option<&[f32]>, out: &[f32]) {
        assert!(
            q.len() >= self.cols.div_ceil(COLS) * COLS,
            "activation codes need padding"
        );
        assert!(xs.is_finite() && xs >= 0.0, "invalid activation scale");
        assert_eq!(out.len(), self.rows, "wrong output size");
        assert!(bias.is_none_or(|b| b.len() == self.rows), "wrong bias size");
    }

    /// Compute the scalar reference, with bias or accumulation into the existing output.
    pub fn gemv_portable(&self, q: &[i16], xs: f32, bias: Option<&[f32]>, out: &mut [f32]) {
        self.validate(q, xs, bias, out);
        let chunks = self.cols.div_ceil(COLS);
        for row in 0..self.rows {
            let mut lanes = [0.0f32; 8];
            for chunk in (0..chunks).step_by(2) {
                for (lane, sum) in lanes.iter_mut().enumerate() {
                    let mut integer = 0i32;
                    for c in chunk..(chunk + 2).min(chunks) {
                        let pos =
                            ((row / ROWS) * chunks + c) * TILE + (row % ROWS) * COLS + 2 * lane;
                        let col = c * COLS + 2 * lane;
                        integer += i32::from(self.codes[pos]) * i32::from(q[col])
                            + i32::from(self.codes[pos + 1]) * i32::from(q[col + 1]);
                    }
                    *sum += integer as f32;
                }
            }
            let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
                + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
            out[row] = sum * (self.scales[row] * xs) + bias.map_or(out[row], |b| b[row]);
        }
    }

    /// Dispatch to AVX2 byte widening/integer madd, or the bitwise scalar fallback.
    pub fn gemv(&self, q: &[i16], xs: f32, bias: Option<&[f32]>, out: &mut [f32]) {
        self.validate(q, xs, bias, out);
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature check; validate covers every load/store bound.
            unsafe { self.gemv_avx2(q, xs, bias, out) };
            return;
        }
        self.gemv_portable(q, xs, bias, out);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn gemv_avx2(&self, q: &[i16], xs: f32, bias: Option<&[f32]>, out: &mut [f32]) {
        use std::arch::x86_64::*;
        let chunks = self.cols.div_ceil(COLS);
        // Even full-range i16 activations are safe: four products have magnitude
        // at most 4 * 127 * 32768 = 16,646,144, below i32::MAX and 2^24.
        unsafe {
            for panel in 0..self.rows.div_ceil(ROWS) {
                let mut lanes = [_mm256_setzero_ps(); ROWS];
                for chunk in (0..chunks).step_by(2) {
                    let mut integers = [_mm256_setzero_si256(); ROWS];
                    for c in chunk..(chunk + 2).min(chunks) {
                        let x = _mm256_loadu_si256(q.as_ptr().add(c * COLS).cast());
                        for (row, sum) in integers.iter_mut().enumerate() {
                            let pos = (panel * chunks + c) * TILE + row * COLS;
                            let bytes = _mm_loadu_si128(self.codes.as_ptr().add(pos).cast());
                            let w = _mm256_cvtepi8_epi16(bytes);
                            *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(w, x));
                        }
                    }
                    for row in 0..ROWS {
                        lanes[row] = _mm256_add_ps(lanes[row], _mm256_cvtepi32_ps(integers[row]));
                    }
                }
                let halves = _mm256_hadd_ps(
                    _mm256_hadd_ps(lanes[0], lanes[1]),
                    _mm256_hadd_ps(lanes[2], lanes[3]),
                );
                let sums = _mm_add_ps(
                    _mm256_castps256_ps128(halves),
                    _mm256_extractf128_ps(halves, 1),
                );
                let start = panel * ROWS;
                let scales = _mm_mul_ps(
                    _mm_loadu_ps(self.scales.as_ptr().add(start)),
                    _mm_set1_ps(xs),
                );
                let products = _mm_mul_ps(sums, scales);
                let mut final_lanes = [0.0; ROWS];
                _mm_storeu_ps(final_lanes.as_mut_ptr(), products);
                for row in start..(start + ROWS).min(self.rows) {
                    out[row] = final_lanes[row - start] + bias.map_or(out[row], |b| b[row]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn rejects_invalid_weights_and_shape() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(PackedQ8::pack(&[value], 1, 1).is_err());
        }
        assert!(PackedQ8::pack(&[], 0, 3).is_err());
        assert!(PackedQ8::pack(&[1.0], 2, 2).is_err());
        assert!(PackedQ8::pack(&[], usize::MAX, 2).is_err());
    }

    #[test]
    fn rounding_zeros_and_subnormals() {
        let weights = [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5];
        let matrix = PackedQ8::pack(&weights, 1, weights.len()).unwrap();
        assert_eq!(matrix.dequantize(), [127.0, 0.0, 2.0, 2.0, 0.0, -2.0, -2.0]);
        let matrix = PackedQ8::pack(&[f32::from_bits(1), 0.0], 1, 2).unwrap();
        assert_eq!(matrix.dequantize(), [f32::from_bits(1), 0.0]);
        let matrix = PackedQ8::pack(&[0.0; 9], 3, 3).unwrap();
        let mut out = [42.0; 3];
        matrix.gemv(&[i16::MIN; 16], 1.0, None, &mut out);
        assert_eq!(out, [42.0; 3]);
    }

    #[test]
    fn full_range_integer_bound_matches_oracle_and_padding_is_ignored() {
        for cols in [1usize, 15, 16, 17, 31, 32, 33, 256, 384, 768] {
            let weights = vec![127.0; 5 * cols];
            let matrix = PackedQ8::pack(&weights, 5, cols).unwrap();
            for code in [i16::MIN, i16::MAX] {
                let q = vec![code; cols.div_ceil(COLS) * COLS];
                let mut scalar = [0.0; 5];
                let mut dispatched = [0.0; 5];
                matrix.gemv_portable(&q, 1.0, None, &mut scalar);
                matrix.gemv(&q, 1.0, None, &mut dispatched);
                assert_eq!(scalar.map(f32::to_bits), dispatched.map(f32::to_bits));
                let exact = (127i64 * i64::from(code) * cols as i64) as f32;
                assert!((scalar[0] - exact).abs() <= exact.abs() * 2e-6);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn quantization_bound_and_bitwise_dispatch(
            rows in 1usize..17, cols in 1usize..100,
            seed in prop::collection::vec(-10000i16..10000, 1..80),
            code in any::<i16>(),
        ) {
            let weights: Vec<f32> = (0..rows * cols).map(|i| f32::from(seed[i % seed.len()]) / 911.0).collect();
            let matrix = PackedQ8::pack(&weights, rows, cols).unwrap();
            let dequantized = matrix.dequantize();
            for i in 0..weights.len() {
                prop_assert!((dequantized[i] - weights[i]).abs() <= matrix.scales[i / cols] * 0.50001 + 1e-6);
            }
            let q: Vec<i16> = (0..cols.div_ceil(COLS) * COLS).map(|i| if i % 3 == 0 { code } else { seed[i % seed.len()] }).collect();
            let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.01).collect();
            let mut scalar = bias.clone();
            let mut dispatched = bias.clone();
            matrix.gemv_portable(&q, 1.0 / 16383.0, Some(&bias), &mut scalar);
            matrix.gemv(&q, 1.0 / 16383.0, Some(&bias), &mut dispatched);
            prop_assert_eq!(scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            matrix.gemv_portable(&q, 0.007, None, &mut scalar);
            matrix.gemv(&q, 0.007, None, &mut dispatched);
            prop_assert_eq!(scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }
    }
}

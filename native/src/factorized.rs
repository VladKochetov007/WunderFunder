//! Exact imported FP32 low-rank factors; two fixed-order matrix-vector products.

pub struct FloatFactors {
    rows: usize,
    cols: usize,
    rank: usize,
    left: Vec<f32>,
    right: Vec<f32>,
}

impl FloatFactors {
    pub fn new(
        left: Vec<f32>,
        right: Vec<f32>,
        rows: usize,
        cols: usize,
        rank: usize,
    ) -> Result<Self, &'static str> {
        if rows == 0
            || cols == 0
            || rank == 0
            || rank > rows.min(cols)
            || rows.checked_mul(rank) != Some(left.len())
            || rank.checked_mul(cols) != Some(right.len())
            || left.iter().chain(&right).any(|value| !value.is_finite())
        {
            return Err("invalid finite FP32 factor dimensions or values");
        }
        Ok(Self {
            rows,
            cols,
            rank,
            left,
            right,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn left(&self) -> &[f32] {
        &self.left
    }
    pub fn right(&self) -> &[f32] {
        &self.right
    }
    pub fn storage_bytes(&self) -> usize {
        (self.left.len() + self.right.len()) * size_of::<f32>()
    }

    pub fn gemv(
        &self,
        x: &[f32],
        scratch: &mut [f32],
        bias: &[f32],
        out: &mut [f32],
        portable: bool,
    ) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(scratch.len(), self.rank);
        assert_eq!(bias.len(), self.rows);
        assert_eq!(out.len(), self.rows);
        matvec(&self.right, x, None, scratch, portable);
        matvec(&self.left, scratch, Some(bias), out, portable);
    }
}

pub fn matvec(weights: &[f32], x: &[f32], bias: Option<&[f32]>, out: &mut [f32], portable: bool) {
    assert!(!x.is_empty());
    assert_eq!(x.len().checked_mul(out.len()), Some(weights.len()));
    assert!(bias.is_none_or(|values| values.len() == out.len()));
    #[cfg(target_arch = "x86_64")]
    if !portable && std::is_x86_feature_detected!("avx2") {
        // SAFETY: feature checked; the caller validated the complete matrix dimensions.
        unsafe { matvec_avx2(weights, x, bias, out) };
        return;
    }
    let _ = portable;
    for (index, (row, target)) in weights.chunks_exact(x.len()).zip(out).enumerate() {
        let mut lanes = [0.0f32; 8];
        let full = x.len() / 8 * 8;
        for column in (0..full).step_by(8) {
            for lane in 0..8 {
                lanes[lane] += row[column + lane] * x[column + lane];
            }
        }
        let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for column in full..x.len() {
            total += row[column] * x[column];
        }
        *target = total + bias.map_or(0.0, |values| values[index]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matvec_avx2(weights: &[f32], x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let columns = x.len();
    let full = columns / 8 * 8;
    unsafe {
        for first in (0..out.len()).step_by(4) {
            let rows = (out.len() - first).min(4);
            let mut accumulators = [_mm256_setzero_ps(); 4];
            for column in (0..full).step_by(8) {
                let input = _mm256_loadu_ps(x.as_ptr().add(column));
                for row in 0..rows {
                    let values =
                        _mm256_loadu_ps(weights.as_ptr().add((first + row) * columns + column));
                    accumulators[row] =
                        _mm256_add_ps(accumulators[row], _mm256_mul_ps(values, input));
                }
            }
            for row in 0..rows {
                let mut lanes = [0.0f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), accumulators[row]);
                let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
                    + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
                for column in full..columns {
                    total += weights[(first + row) * columns + column] * x[column];
                }
                out[first + row] = total + bias.map_or(0.0, |values| values[first + row]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_import_without_transforming_factor_bytes() {
        let left = vec![0.0, -0.0, f32::from_bits(1), 2.0];
        let right = vec![1.0, 0.0, 0.0, 1.0];
        let matrix = FloatFactors::new(left.clone(), right.clone(), 2, 2, 2).unwrap();
        assert_eq!(
            matrix
                .left()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            left.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(matrix.right(), right);
        assert_eq!(matrix.storage_bytes(), 32);
        assert!(FloatFactors::new(vec![f32::NAN], vec![1.0], 1, 1, 1).is_err());
        assert!(FloatFactors::new(vec![1.0], vec![1.0], 1, 1, 0).is_err());
        assert!(FloatFactors::new(vec![1.0], vec![1.0], 1, 1, 2).is_err());
    }

    #[test]
    fn scalar_and_simd_factor_products_are_bitwise_equal() {
        for (rows, columns, rank) in [(3, 3, 1), (33, 17, 9), (48, 33, 17), (1152, 384, 128)] {
            let values = |length: usize, offset: usize| {
                (0..length)
                    .map(|i| ((i + offset) % 31) as f32 / 31.0 - 0.5)
                    .collect::<Vec<_>>()
            };
            let left = values(rows * rank, 1);
            let right = values(rank * columns, 7);
            let matrix = FloatFactors::new(left, right, rows, columns, rank).unwrap();
            let x = values(columns, 11);
            let bias = values(rows, 3);
            let (mut a, mut b) = (vec![0.0; rows], vec![0.0; rows]);
            let (mut sa, mut sb) = (vec![0.0; rank], vec![0.0; rank]);
            matrix.gemv(&x, &mut sa, &bias, &mut a, false);
            matrix.gemv(&x, &mut sb, &bias, &mut b, true);
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(sa, sb);
        }
    }
}

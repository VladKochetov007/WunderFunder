//! Dense kernels written so LLVM vectorizes them without `unsafe` or fast-math.
//!
//! `dot` accumulates into 16 independent lanes; each lane is a plain in-order
//! float sum, so vectorizing across lanes needs no reassociation.

const LANES: usize = 16;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for k in 0..LANES {
            acc[k] += x[k] * y[k];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// out = W x + b for a row-major `rows x cols` matrix.
#[inline]
pub fn gemv(w: &[f32], bias: &[f32], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    assert_eq!(w.len(), out.len() * cols);
    assert_eq!(bias.len(), out.len());
    for ((o, row), b) in out.iter_mut().zip(w.chunks_exact(cols)).zip(bias) {
        *o = dot(row, x) + b;
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch.nn.GELU() (exact erf form).
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn dot_oracle(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum()
    }

    proptest! {
        #[test]
        fn dot_matches_f64_oracle(v in prop::collection::vec((-4.0f32..4.0, -4.0f32..4.0), 0..300)) {
            let (a, b): (Vec<f32>, Vec<f32>) = v.into_iter().unzip();
            let want = dot_oracle(&a, &b);
            let got = f64::from(dot(&a, &b));
            prop_assert!((got - want).abs() <= 1e-4 * (1.0 + want.abs()), "got {got} want {want}");
        }

        #[test]
        fn gemv_matches_rowwise_oracle(rows in 1usize..20, cols in 1usize..40, seed in any::<u64>()) {
            let mut s = seed | 1;
            let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s % 2001) as f32 / 1000.0) - 1.0 };
            let w: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
            let b: Vec<f32> = (0..rows).map(|_| next()).collect();
            let x: Vec<f32> = (0..cols).map(|_| next()).collect();
            let mut out = vec![0.0; rows];
            gemv(&w, &b, &x, &mut out);
            for r in 0..rows {
                let want = dot_oracle(&w[r * cols..(r + 1) * cols], &x) + f64::from(b[r]);
                prop_assert!((f64::from(out[r]) - want).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn gelu_reference_points() {
        // torch.nn.GELU()(torch.tensor([-3., -1., 0., 0.5, 2.]))
        let want = [-0.004_049_71, -0.158_655_25, 0.0, 0.345_731_14, 1.954_499_7];
        for (x, w) in [-3.0f32, -1.0, 0.0, 0.5, 2.0].iter().zip(want) {
            assert!(
                (gelu(*x) - w).abs() < 1e-6,
                "gelu({x})={} want {w}",
                gelu(*x)
            );
        }
    }
}

//! Branchless activation approximations that LLVM can vectorize.
//!
//! Each keeps the exact function's meaning (same asymptotes, symmetric,
//! monotone) with small bounded error, so trained weights need no retraining.
//! Errors are pinned by tests against libm.

/// exp as 2^n * e^r with n = round(x / ln 2), |r| <= ln2/2, degree-7 Taylor
/// core (truncation < 6e-9) and an exact power of two built from bits, so
/// rounding is not amplified. Input clamped to [-40, 40].
#[inline(always)]
pub fn exp_fast(x: f32) -> f32 {
    const LN2: f32 = std::f32::consts::LN_2;
    let x = x.clamp(-40.0, 40.0);
    let n = (x / LN2).round();
    let r = x - n * LN2;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    // |n| <= 58, so the biased exponent stays in the normal range.
    let pow2 = f32::from_bits(((n as i32 + 127) as u32) << 23);
    p * pow2
}

#[inline(always)]
pub fn sigmoid_fast(x: f32) -> f32 {
    1.0 / (1.0 + exp_fast(-x))
}

#[inline(always)]
pub fn tanh_fast(x: f32) -> f32 {
    2.0 * sigmoid_fast(2.0 * x) - 1.0
}

/// erf via Abramowitz & Stegun 7.1.26 (|error| <= 1.5e-7 in exact arithmetic).
#[inline(always)]
pub fn erf_fast(x: f32) -> f32 {
    let s = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    s * (1.0 - poly * exp_fast(-a * a))
}

/// torch.nn.GELU() exact form, evaluated with erf_fast.
#[inline(always)]
pub fn gelu_fast(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_fast(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// GELU(approximate="sigmoid"): x * sigmoid(1.702 x). Different function:
/// only for models trained with the same approximation.
#[inline(always)]
pub fn gelu_sigmoid(x: f32) -> f32 {
    x * sigmoid_fast(1.702 * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> impl Iterator<Item = f32> {
        (-12_000..=12_000).map(|i| i as f32 * 1e-3)
    }

    fn max_err(f: impl Fn(f32) -> f32, g: impl Fn(f64) -> f64) -> f64 {
        grid()
            .map(|x| (f64::from(f(x)) - g(f64::from(x))).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn sigmoid_tanh_errors_are_bounded() {
        let e_sig = max_err(sigmoid_fast, |x| 1.0 / (1.0 + (-x).exp()));
        let e_tanh = max_err(tanh_fast, f64::tanh);
        assert!(e_sig < 2e-6, "sigmoid err {e_sig}");
        assert!(e_tanh < 4e-6, "tanh err {e_tanh}");
    }

    #[test]
    fn gelu_error_is_bounded_and_asymptotes_hold() {
        let e = max_err(gelu_fast, |x| {
            0.5 * x * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
        });
        assert!(e < 5e-6, "gelu err {e}");
        assert_eq!(sigmoid_fast(100.0), 1.0);
        assert!(sigmoid_fast(-100.0) < 1e-17);
        assert_eq!(tanh_fast(50.0), 1.0);
        assert_eq!(tanh_fast(-50.0), -1.0);
        assert!((gelu_fast(-8.0)).abs() < 1e-6);
        assert!((gelu_fast(8.0) - 8.0).abs() < 1e-5);
    }

    #[test]
    fn monotone_on_grid() {
        let xs: Vec<f32> = grid().collect();
        for f in [sigmoid_fast as fn(f32) -> f32, tanh_fast, erf_fast] {
            assert!(xs.windows(2).all(|w| f(w[1]) >= f(w[0])));
        }
    }
}

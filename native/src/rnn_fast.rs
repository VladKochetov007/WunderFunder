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

use std::ops::Range;

use bytemuck::{Pod, Zeroable};

use crate::approx::{gelu_fast, sigmoid_fast, tanh_fast};
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

struct LayerIdx {
    w_ih: Range<usize>,
    b_ih: Range<usize>,
    w_hh: Range<usize>,
    b_hh: Range<usize>,
}

pub struct FastRnn {
    n_in: usize,
    hidden: usize,
    n_out: usize,
    act: Activations,
    weights: Arena,
    state: Arena,
    w_in_cols: Range<usize>, // column-major hidden x n_in, folded normalization
    b_in: Range<usize>,
    layers: Vec<LayerIdx>,
    w1: Range<usize>,
    b1: Range<usize>,
    w2: Range<usize>,
    b2: Range<usize>,
    h: Range<usize>,
    z: Range<usize>,
    gi: Range<usize>,
    gh: Range<usize>,
    cat: Range<usize>,
    a: Range<usize>,
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
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

/// Four output rows per pass: each x chunk is loaded once and feeds eight independent FMA
/// chains, so the kernel is throughput- rather than latency-bound. Remaining rows use `dot`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
fn gemv(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
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
                    acc[2 * r + 1] = _mm256_fmadd_ps(_mm256_loadu_ps(wr.add(8)), x1, acc[2 * r + 1]);
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
        out[r] = dot(&w[r * n..(r + 1) * n], x) + b[r];
    }
}

impl FastRnn {
    pub fn new(w: RnnWeights, act: Activations) -> Result<Self, ShapeError> {
        let n_in = w.mean.len();
        let hidden = w.b_in.len();
        let n_out = w.b2.len();
        // Validate by constructing the exact model's shape checks.
        let check = RnnWeights {
            mean: w.mean.clone(),
            std: w.std.clone(),
            w_in: w.w_in.clone(),
            b_in: w.b_in.clone(),
            layers: w
                .layers
                .iter()
                .map(|l| crate::rnn::GruLayer {
                    w_ih: l.w_ih.clone(),
                    w_hh: l.w_hh.clone(),
                    b_ih: l.b_ih.clone(),
                    b_hh: l.b_hh.clone(),
                })
                .collect(),
            w1: w.w1.clone(),
            b1: w.b1.clone(),
            w2: w.w2.clone(),
            b2: w.b2.clone(),
        };
        crate::rnn::RnnModel::new(check)?;

        // Fold normalization: W'[r][c] = W[r][c] / std[c]; b'[r] = b[r] - sum_c W'[r][c] mean[c].
        let mut cols = vec![0.0f32; hidden * n_in];
        let mut b_in = w.b_in.clone();
        for r in 0..hidden {
            for c in 0..n_in {
                let v = w.w_in[r * n_in + c] / w.std[c];
                cols[c * hidden + r] = v;
                b_in[r] -= v * w.mean[c];
            }
        }
        let mut weights = Arena::new();
        let w_in_cols = weights.push(&cols);
        let b_in = weights.push(&b_in);
        let layers = w
            .layers
            .iter()
            .map(|l| LayerIdx {
                w_ih: weights.push(&l.w_ih),
                b_ih: weights.push(&l.b_ih),
                w_hh: weights.push(&l.w_hh),
                b_hh: weights.push(&l.b_hh),
            })
            .collect::<Vec<_>>();
        let w1 = weights.push(&w.w1);
        let b1 = weights.push(&w.b1);
        let w2 = weights.push(&w.w2);
        let b2 = weights.push(&w.b2);

        let mut state = Arena::new();
        let h = state.zeros(layers.len() * hidden);
        let z = state.zeros(hidden);
        let gi = state.zeros(3 * hidden);
        let gh = state.zeros(3 * hidden);
        let cat = state.zeros(2 * hidden);
        let a = state.zeros(hidden);
        debug_assert!(weights.len >= weights.as_ref().len());
        Ok(Self {
            n_in,
            hidden,
            n_out,
            act,
            weights,
            state,
            w_in_cols,
            b_in,
            layers,
            w1,
            b1,
            w2,
            b2,
            h,
            z,
            gi,
            gh,
            cat,
            a,
        })
    }

    pub fn n_in(&self) -> usize {
        self.n_in
    }

    pub fn n_out(&self) -> usize {
        self.n_out
    }

    pub fn reset(&mut self) {
        let h = self.h.clone();
        self.state.as_mut()[h].fill(0.0);
    }

    pub fn step(&mut self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.n_in, "input width");
        assert_eq!(out.len(), self.n_out, "output width");
        let n = self.hidden;
        let fast = self.act == Activations::Fast;
        let wts = self.weights.as_ref();
        let st = self.state.as_mut();

        // Sparse input projection (column saxpy, zero inputs skipped).
        {
            let z = &mut st[self.z.clone()];
            z.copy_from_slice(&wts[self.b_in.clone()]);
            let cols = &wts[self.w_in_cols.clone()];
            for (col, &xj) in cols.chunks_exact(n).zip(x) {
                if xj != 0.0 {
                    for (o, w) in z.iter_mut().zip(col) {
                        *o += w * xj;
                    }
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
            // State layout is [h (all layers) | z | gi | gh | cat | a].
            let (lo, hi) = st.split_at_mut(self.gi.start);
            let (gi, rest) = hi.split_at_mut(3 * n);
            let gh_off = self.gh.start - self.gi.end;
            let gh = &mut rest[gh_off..gh_off + 3 * n];
            let (inp, h): (&[f32], &mut [f32]) = if inp_start < h_start {
                let (a, b) = lo.split_at_mut(h_start);
                (&a[inp_start..inp_start + n], &mut b[..n])
            } else {
                let (a, b) = lo.split_at_mut(inp_start);
                (&b[..n], &mut a[h_start..h_start + n])
            };
            gemv(&wts[li.w_ih.clone()], &wts[li.b_ih.clone()], inp, gi);
            gemv(&wts[li.w_hh.clone()], &wts[li.b_hh.clone()], h, gh);
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

        let last = self.h.start + (self.layers.len() - 1) * n;
        let (lo, hi) = st.split_at_mut(self.cat.start);
        let (cat, rest) = hi.split_at_mut(2 * n);
        cat[..n].copy_from_slice(&lo[last..last + n]);
        cat[n..].copy_from_slice(&lo[self.z.clone()]);
        let a = &mut rest[self.a.start - self.cat.end..self.a.start - self.cat.end + n];
        gemv(&wts[self.w1.clone()], &wts[self.b1.clone()], cat, a);
        if fast {
            a.iter_mut().for_each(|v| *v = gelu_fast(*v));
        } else {
            a.iter_mut().for_each(|v| *v = gelu(*v));
        }
        gemv(&wts[self.w2.clone()], &wts[self.b2.clone()], a, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rnn::{GruLayer, RnnModel};

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
        for r in [&m.w_in_cols, &m.b_in, &m.layers[1].w_hh, &m.w2] {
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
                        + (0..n).map(|j| w[r * n + j] as f64 * x[j] as f64).sum::<f64>();
                    assert!((out[r] as f64 - want).abs() < 1e-4, "rows {rows} n {n} r {r}");
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
}

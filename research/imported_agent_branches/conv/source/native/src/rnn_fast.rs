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
//! - Conv encoder (enc="conv"): the last k z rows sit in an aligned ring buffer;
//!   taps are stored lag-major so the depthwise conv runs as 16-lane register
//!   blocks over channels; the pointwise projection is stored column-major and
//!   accumulated in 32-lane register blocks (no per-row horizontal sums).

use std::ops::Range;

use bytemuck::{Pod, Zeroable};

use crate::approx::{gelu_fast, sigmoid_fast, tanh_fast};
use crate::kernels::{gelu, sigmoid};
use crate::rnn::{RnnWeights, ShapeError};

const LANES: usize = 16;
/// Output block of the column-major pointwise projection (4 AVX2 registers).
const PW_BLOCK: usize = 32;

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

struct ConvIdx {
    k: usize,
    stride: usize, // hidden rounded up to LANES; padding lanes hold zero taps and values
    taps: Range<usize>, // k x stride, lag-major: taps[j][c] = conv_dw.weight[c, 0, j]
    b_dw: Range<usize>,
    pw_stride: usize,   // hidden rounded up to PW_BLOCK
    w_pw: Range<usize>, // hidden x pw_stride, column-major: w_pw[j][i] = conv_pw.weight[i, j]
    b_pw: Range<usize>, // pw_stride
    ring: Range<usize>, // k x stride pre-residual z rows; slot `pos` receives the next row
    c: Range<usize>,
    p: Range<usize>,
    pos: usize,
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
    conv: Option<ConvIdx>,
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

#[inline(always)]
fn gemv(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    for ((o, row), bias) in out.iter_mut().zip(w.chunks_exact(x.len())).zip(b) {
        *o = dot(row, x) + bias;
    }
}

impl FastRnn {
    pub fn new(w: RnnWeights, act: Activations) -> Result<Self, ShapeError> {
        let n_in = w.mean.len();
        let hidden = w.b_in.len();
        let n_out = w.b2.len();
        // Validate by constructing the exact model's shape checks.
        crate::rnn::RnnModel::new(w.clone())?;

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
        let mut conv = w.conv.as_ref().map(|c| {
            let stride = hidden.div_ceil(LANES) * LANES;
            let mut taps = vec![0.0f32; c.k * stride];
            for ch in 0..hidden {
                for j in 0..c.k {
                    taps[j * stride + ch] = c.w_dw[ch * c.k + j];
                }
            }
            let mut b_dw = c.b_dw.clone();
            b_dw.resize(stride, 0.0);
            let pw_stride = hidden.div_ceil(PW_BLOCK) * PW_BLOCK;
            let mut w_pw = vec![0.0f32; hidden * pw_stride];
            for i in 0..hidden {
                for j in 0..hidden {
                    w_pw[j * pw_stride + i] = c.w_pw[i * hidden + j];
                }
            }
            let mut b_pw = c.b_pw.clone();
            b_pw.resize(pw_stride, 0.0);
            ConvIdx {
                k: c.k,
                stride,
                taps: weights.push(&taps),
                b_dw: weights.push(&b_dw),
                pw_stride,
                w_pw: weights.push(&w_pw),
                b_pw: weights.push(&b_pw),
                ring: 0..0,
                c: 0..0,
                p: 0..0,
                pos: 0,
            }
        });
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
        // Conv state goes last: `step` splits the arena at the ring start.
        if let Some(cv) = &mut conv {
            cv.ring = state.zeros(cv.k * cv.stride);
            cv.c = state.zeros(cv.stride);
            cv.p = state.zeros(cv.pw_stride);
        }
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
            conv,
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

    pub fn has_conv(&self) -> bool {
        self.conv.is_some()
    }

    pub fn reset(&mut self) {
        let h = self.h.clone();
        self.state.as_mut()[h].fill(0.0);
        if let Some(cv) = &mut self.conv {
            self.state.as_mut()[cv.ring.clone()].fill(0.0);
            cv.pos = 0;
        }
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
        if let Some(cv) = &mut self.conv {
            conv_residual(cv, wts, st, self.z.clone(), fast);
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

/// z += gelu(W_pw dwconv(ring) + b_pw) after pushing z into the ring.
#[inline(always)]
fn conv_residual(cv: &mut ConvIdx, wts: &[f32], st: &mut [f32], z: Range<usize>, fast: bool) {
    let (k, stride, n) = (cv.k, cv.stride, z.len());
    let (lo, hi) = st.split_at_mut(cv.ring.start);
    let z = &mut lo[z];
    // State layout after `a`: [ring | c | p], each segment block-aligned.
    let (ring, rest) = hi.split_at_mut(cv.c.start - cv.ring.start);
    let (c, p) = rest.split_at_mut(cv.p.start - cv.c.start);
    let (c, p) = (&mut c[..stride], &mut p[..cv.pw_stride]);
    ring[cv.pos * stride..][..n].copy_from_slice(z);
    // Tap j multiplies row t-(k-1)+j, which lives in slot (pos+1+j) mod k.
    let taps = &wts[cv.taps.clone()];
    let b_dw = &wts[cv.b_dw.clone()];
    let first = if cv.pos + 1 == k { 0 } else { cv.pos + 1 };
    for blk in (0..stride).step_by(LANES) {
        let mut acc = [0.0f32; LANES];
        acc.copy_from_slice(&b_dw[blk..blk + LANES]);
        let mut slot = first;
        for j in 0..k {
            let w = &taps[j * stride + blk..][..LANES];
            let r = &ring[slot * stride + blk..][..LANES];
            for l in 0..LANES {
                acc[l] += w[l] * r[l];
            }
            slot += 1;
            if slot == k {
                slot = 0;
            }
        }
        c[blk..blk + LANES].copy_from_slice(&acc);
    }
    cv.pos = first;
    // p = b_pw + sum_j W_pw[:, j] c[j], one register block of outputs at a time.
    let (ps, cols, b_pw) = (cv.pw_stride, &wts[cv.w_pw.clone()], &wts[cv.b_pw.clone()]);
    for blk in (0..ps).step_by(PW_BLOCK) {
        let mut acc = [0.0f32; PW_BLOCK];
        acc.copy_from_slice(&b_pw[blk..blk + PW_BLOCK]);
        for (col, &cj) in cols.chunks_exact(ps).zip(&c[..n]) {
            let col = &col[blk..blk + PW_BLOCK];
            for l in 0..PW_BLOCK {
                acc[l] += col[l] * cj;
            }
        }
        p[blk..blk + PW_BLOCK].copy_from_slice(&acc);
    }
    if fast {
        z.iter_mut()
            .zip(p.iter())
            .for_each(|(zj, pj)| *zj += gelu_fast(*pj));
    } else {
        z.iter_mut()
            .zip(p.iter())
            .for_each(|(zj, pj)| *zj += gelu(*pj));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rnn::{ConvEncoder, GruLayer, RnnModel};

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
            conv: None,
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

    fn with_conv(mut w: RnnWeights, k: usize) -> RnnWeights {
        let hidden = w.b_in.len();
        let mut s = 0xA5A5_5A5A_1234_5678u64 ^ k as u64;
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
        w.conv = Some(ConvEncoder {
            k,
            w_dw: v(hidden * k, 0.5),
            b_dw: v(hidden, 0.2),
            w_pw: v(hidden * hidden, 0.4),
            b_pw: v(hidden, 0.2),
        });
        w
    }

    fn run_both(act: Activations, n_in: usize, hidden: usize, layers: usize) -> f32 {
        let w = weights(n_in, hidden, layers, 7);
        run_pair(w, act)
    }

    fn run_pair(w: RnnWeights, act: Activations) -> f32 {
        let n_in = w.mean.len();
        let mut exact = RnnModel::new(w.clone()).unwrap();
        let mut fast = FastRnn::new(w, act).unwrap();
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
    fn conv_encoder_matches_reference_model() {
        // k = 1 (current row only), k < hidden, odd hidden (padding lanes), k = 16 as trained.
        for (n_in, hidden, layers, k) in [
            (5, 3, 1, 1),
            (40, 33, 2, 5),
            (37, 20, 2, 16),
            (30, 16, 1, 16),
        ] {
            let w = with_conv(weights(n_in, hidden, layers, 11), k);
            let err = run_pair(w, Activations::Exact);
            assert!(err < 2e-5, "n_in {n_in} hidden {hidden} k {k}: {err}");
        }
        let err = run_pair(with_conv(weights(40, 33, 2, 11), 16), Activations::Fast);
        assert!(err < 1e-4, "fast: {err}");
    }

    #[test]
    fn conv_reset_clears_the_ring() {
        let mut m = FastRnn::new(with_conv(weights(12, 20, 2, 5), 16), Activations::Fast).unwrap();
        let x = inputs(12, 40);
        let mut first = vec![[0.0f32; 2]; 40];
        let mut out = [0.0f32; 2];
        for (r, o) in x.iter().zip(first.iter_mut()) {
            m.step(r, o);
        }
        m.reset();
        for (r, want) in x.iter().zip(&first) {
            m.step(r, &mut out);
            assert_eq!(&out, want);
        }
    }

    #[test]
    fn conv_bad_shape_is_rejected() {
        let mut w = with_conv(weights(12, 8, 1, 5), 4);
        w.conv.as_mut().unwrap().w_pw.pop();
        let err = FastRnn::new(w, Activations::Fast).err().unwrap();
        assert!(err.0.contains("conv_pw.weight"), "{}", err.0);
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

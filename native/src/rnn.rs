//! Stateful single-row twin of `wnn.nn.RNNModel`:
//! z = gelu(W_in (x - mean)/std + b); GRU stack; y = W2 gelu(W1 [h_last, z] + b1) + b2.

use crate::kernels::{gelu, gemv, sigmoid};

#[derive(Debug)]
pub struct GruLayer {
    pub w_ih: Vec<f32>,
    pub w_hh: Vec<f32>,
    pub b_ih: Vec<f32>,
    pub b_hh: Vec<f32>,
}

#[derive(Debug)]
pub struct RnnModel {
    n_in: usize,
    encoder_width: usize,
    hidden: usize,
    readout_width: usize,
    inv_std: Vec<f32>,
    mean: Vec<f32>,
    w_in: Vec<f32>,
    b_in: Vec<f32>,
    layers: Vec<GruLayer>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w_mid: Vec<f32>,
    b_mid: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    // State and scratch, allocated once.
    h: Vec<f32>,
    x_norm: Vec<f32>,
    z: Vec<f32>,
    gi: Vec<f32>,
    gh: Vec<f32>,
    cat: Vec<f32>,
    a: Vec<f32>,
    mid: Vec<f32>,
    y: Vec<f32>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ShapeError(pub String);

pub struct RnnWeights {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub w_in: Vec<f32>,
    pub b_in: Vec<f32>,
    pub layers: Vec<GruLayer>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// Optional second hidden readout matrix; empty for the historical one-hidden-layer head.
    pub w_mid: Vec<f32>,
    pub b_mid: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

fn expect(name: &str, got: usize, want: usize) -> Result<(), ShapeError> {
    if got == want {
        Ok(())
    } else {
        Err(ShapeError(format!(
            "{name}: expected {want} values, got {got}"
        )))
    }
}

impl RnnModel {
    pub fn new(w: RnnWeights) -> Result<Self, ShapeError> {
        let n_in = w.mean.len();
        let encoder_width = w.b_in.len();
        let hidden = w
            .layers
            .first()
            .map(|layer| layer.b_hh.len() / 3)
            .unwrap_or(0);
        let readout_width = w.b1.len();
        let n_out = w.b2.len();
        expect("std", w.std.len(), n_in)?;
        expect("inp.weight", w.w_in.len(), encoder_width * n_in)?;
        if w.layers.is_empty() {
            return Err(ShapeError("at least one GRU layer required".into()));
        }
        for (k, l) in w.layers.iter().enumerate() {
            expect(
                &format!("weight_ih_l{k}"),
                l.w_ih.len(),
                3 * hidden * if k == 0 { encoder_width } else { hidden },
            )?;
            expect(
                &format!("weight_hh_l{k}"),
                l.w_hh.len(),
                3 * hidden * hidden,
            )?;
            expect(&format!("bias_ih_l{k}"), l.b_ih.len(), 3 * hidden)?;
            expect(&format!("bias_hh_l{k}"), l.b_hh.len(), 3 * hidden)?;
        }
        expect("head.0.weight", w.w1.len(), readout_width * (hidden + encoder_width))?;
        if w.w_mid.is_empty() != w.b_mid.is_empty() {
            return Err(ShapeError("head.extra weight and bias must be present together".into()));
        }
        if !w.w_mid.is_empty() {
            expect("head.extra.weight", w.w_mid.len(), readout_width * readout_width)?;
            expect("head.extra.bias", w.b_mid.len(), readout_width)?;
        }
        expect("head.2.weight", w.w2.len(), n_out * readout_width)?;
        let layers = w.layers.len();
        let has_mid = !w.w_mid.is_empty();
        Ok(Self {
            n_in,
            encoder_width,
            hidden,
            readout_width,

            inv_std: w.std.iter().map(|s| 1.0 / s).collect(),
            mean: w.mean,
            w_in: w.w_in,
            b_in: w.b_in,
            layers: w.layers,
            w1: w.w1,
            b1: w.b1,
            w_mid: w.w_mid,
            b_mid: w.b_mid,
            w2: w.w2,
            b2: w.b2,
            h: vec![0.0; layers * hidden],
            x_norm: vec![0.0; n_in],
            z: vec![0.0; encoder_width],
            gi: vec![0.0; 3 * hidden],
            gh: vec![0.0; 3 * hidden],
            cat: vec![0.0; hidden + encoder_width],
            a: vec![0.0; readout_width],
            mid: vec![0.0; if has_mid { readout_width } else { 0 }],
            y: vec![0.0; n_out],
        })
    }

    pub fn n_in(&self) -> usize {
        self.n_in
    }

    pub fn widths(&self) -> (usize, usize, usize) {
        (self.encoder_width, self.hidden, self.readout_width)
    }

    pub fn reset(&mut self) {
        self.h.fill(0.0);
    }

    pub fn step(&mut self, x: &[f32]) -> &[f32] {
        assert_eq!(x.len(), self.n_in, "input width");
        let n = self.hidden;
        for ((o, xi), (m, s)) in self
            .x_norm
            .iter_mut()
            .zip(x)
            .zip(self.mean.iter().zip(&self.inv_std))
        {
            *o = (xi - m) * s;
        }
        gemv(&self.w_in, &self.b_in, &self.x_norm, &mut self.z);
        self.z.iter_mut().for_each(|v| *v = gelu(*v));

        for (k, layer) in self.layers.iter().enumerate() {
            let (before, rest) = self.h.split_at_mut(k * n);
            let h = &mut rest[..n];
            let inp: &[f32] = if k == 0 {
                &self.z
            } else {
                &before[(k - 1) * n..]
            };
            gemv(&layer.w_ih, &layer.b_ih, inp, &mut self.gi);
            gemv(&layer.w_hh, &layer.b_hh, h, &mut self.gh);
            let (gi_r, gi_rest) = self.gi.split_at(n);
            let (gi_u, gi_c) = gi_rest.split_at(n);
            let (gh_r, gh_rest) = self.gh.split_at(n);
            let (gh_u, gh_c) = gh_rest.split_at(n);
            let gates = gi_r
                .iter()
                .zip(gh_r)
                .zip(gi_u.iter().zip(gh_u))
                .zip(gi_c.iter().zip(gh_c));
            for (hj, (((ir, hr), (iu, hu)), (ic, hc))) in h.iter_mut().zip(gates) {
                let r = sigmoid(ir + hr);
                let u = sigmoid(iu + hu);
                let c = (ic + r * hc).tanh();
                *hj = (1.0 - u) * c + u * *hj;
            }
        }
        let last = &self.h[(self.layers.len() - 1) * n..];
        self.cat[..n].copy_from_slice(last);
        self.cat[n..].copy_from_slice(&self.z);
        gemv(&self.w1, &self.b1, &self.cat, &mut self.a);
        self.a.iter_mut().for_each(|v| *v = gelu(*v));
        if self.w_mid.is_empty() {
            gemv(&self.w2, &self.b2, &self.a, &mut self.y);
        } else {
            gemv(&self.w_mid, &self.b_mid, &self.a, &mut self.mid);
            self.mid.iter_mut().for_each(|v| *v = gelu(*v));
            gemv(&self.w2, &self.b2, &self.mid, &mut self.y);
        }
        &self.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(n_in: usize, hidden: usize, layers: usize) -> RnnWeights {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut v = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s % 2001) as f32 / 1000.0 - 1.0) * 0.3
                })
                .collect()
        };
        RnnWeights {
            mean: v(n_in),
            std: vec![1.5; n_in],
            w_in: v(hidden * n_in),
            b_in: v(hidden),
            layers: (0..layers)
                .map(|_| GruLayer {
                    w_ih: v(3 * hidden * hidden),
                    w_hh: v(3 * hidden * hidden),
                    b_ih: v(3 * hidden),
                    b_hh: v(3 * hidden),
                })
                .collect(),
            w1: v(hidden * 2 * hidden),
            b1: v(hidden),
            w_mid: Vec::new(),
            b_mid: Vec::new(),
            w2: v(2 * hidden),
            b2: v(2),
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        let mut w = weights(5, 4, 1);
        w.w1.pop();
        assert!(RnnModel::new(w).unwrap_err().0.contains("head.0.weight"));
    }

    #[test]
    fn reset_is_deterministic_and_state_matters() {
        let mut m = RnnModel::new(weights(7, 6, 2)).unwrap();
        let x = [0.3f32, -1.0, 2.0, 0.0, 0.5, -0.2, 1.1];
        let first = m.step(&x).to_vec();
        let second = m.step(&x).to_vec();
        assert_ne!(first, second, "hidden state must carry over");
        m.reset();
        assert_eq!(m.step(&x), first.as_slice());
    }
}

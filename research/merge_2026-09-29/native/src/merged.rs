//! Streaming GRU twin for the compact raw-plus-difference competition model.

use crate::kernels::{gelu, gemv, sigmoid};

#[derive(Debug)]
pub struct GruLayer {
    pub input_width: usize,
    pub w_ih: Vec<f32>,
    pub w_hh: Vec<f32>,
    pub b_ih: Vec<f32>,
    pub b_hh: Vec<f32>,
}

#[derive(Debug)]
pub struct MergedWeights {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub input_w: Vec<f32>,
    pub input_b: Vec<f32>,
    pub input_norm_w: Vec<f32>,
    pub input_norm_b: Vec<f32>,
    pub layers: Vec<GruLayer>,
    pub head_norm_w: Vec<f32>,
    pub head_norm_b: Vec<f32>,
    pub head_w: Vec<f32>,
    pub head_b: Vec<f32>,
    pub output_w: Vec<f32>,
    pub output_b: Vec<f32>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ShapeError(pub String);

fn expect(name: &str, got: usize, want: usize) -> Result<(), ShapeError> {
    if got == want {
        Ok(())
    } else {
        Err(ShapeError(format!(
            "{name}: expected {want} values, got {got}"
        )))
    }
}

fn layer_norm(values: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / n;
    let scale = 1.0 / (variance + eps).sqrt();
    for ((value, weight), bias) in values.iter_mut().zip(gamma).zip(beta) {
        *value = (*value - mean) * scale * weight + bias;
    }
}

#[derive(Debug)]
pub struct MergedGru {
    n_input: usize,
    hidden: usize,
    input_clip: f32,
    input_norm_eps: f32,
    head_norm_eps: f32,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    input_w: Vec<f32>,
    input_b: Vec<f32>,
    input_norm_w: Vec<f32>,
    input_norm_b: Vec<f32>,
    layers: Vec<GruLayer>,
    head_norm_w: Vec<f32>,
    head_norm_b: Vec<f32>,
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    output_w: Vec<f32>,
    output_b: Vec<f32>,
    previous: Vec<f32>,
    state: Vec<f32>,
    input: Vec<f32>,
    projected: Vec<f32>,
    input_gates: Vec<f32>,
    hidden_gates: Vec<f32>,
    head_input: Vec<f32>,
    head_hidden: Vec<f32>,
    output: [f32; 2],
}

impl MergedGru {
    pub fn new(weights: MergedWeights) -> Result<Self, ShapeError> {
        let n_input = weights.mean.len();
        let projection = weights.input_b.len();
        let hidden = weights.head_b.len();
        expect("std", weights.std.len(), n_input)?;
        expect("input norm weight", weights.input_norm_w.len(), projection)?;
        expect("input norm bias", weights.input_norm_b.len(), projection)?;
        expect(
            "input projection weight",
            weights.input_w.len(),
            projection * 2 * n_input,
        )?;
        expect(
            "head norm weight",
            weights.head_norm_w.len(),
            hidden + projection,
        )?;
        expect(
            "head norm bias",
            weights.head_norm_b.len(),
            hidden + projection,
        )?;
        expect(
            "head weight",
            weights.head_w.len(),
            hidden * (hidden + projection),
        )?;
        expect("head output weight", weights.output_w.len(), 2 * hidden)?;
        expect("head output bias", weights.output_b.len(), 2)?;
        if weights.layers.is_empty() {
            return Err(ShapeError("at least one GRU layer required".into()));
        }
        for (index, layer) in weights.layers.iter().enumerate() {
            expect(
                &format!("rnn.weight_ih_l{index}"),
                layer.w_ih.len(),
                3 * hidden * layer.input_width,
            )?;
            expect(
                &format!("rnn.weight_hh_l{index}"),
                layer.w_hh.len(),
                3 * hidden * hidden,
            )?;
            expect(
                &format!("rnn.bias_ih_l{index}"),
                layer.b_ih.len(),
                3 * hidden,
            )?;
            expect(
                &format!("rnn.bias_hh_l{index}"),
                layer.b_hh.len(),
                3 * hidden,
            )?;
            let expected_width = if index == 0 { projection } else { hidden };
            expect(
                &format!("rnn input width l{index}"),
                layer.input_width,
                expected_width,
            )?;
        }

        let layers = weights.layers.len();
        let input_norm_w = weights.input_norm_w;
        let input_norm_b = weights.input_norm_b;
        let head_norm_w = weights.head_norm_w;
        let head_norm_b = weights.head_norm_b;
        Ok(Self {
            n_input,
            hidden,
            input_clip: 10.0,
            input_norm_eps: 1e-5,
            head_norm_eps: 1e-5,
            mean: weights.mean,
            inv_std: weights.std.iter().map(|std| 1.0 / std).collect(),
            input_w: weights.input_w,
            input_b: weights.input_b,
            input_norm_w,
            input_norm_b,
            layers: weights.layers,
            head_norm_w,
            head_norm_b,
            head_w: weights.head_w,
            head_b: weights.head_b,
            output_w: weights.output_w,
            output_b: weights.output_b,
            previous: vec![0.0; n_input],
            state: vec![0.0; layers * hidden],
            input: vec![0.0; 2 * n_input],
            projected: vec![0.0; projection],
            input_gates: vec![0.0; 3 * hidden],
            hidden_gates: vec![0.0; 3 * hidden],
            head_input: vec![0.0; hidden + projection],
            head_hidden: vec![0.0; hidden],
            output: [0.0; 2],
        })
    }

    pub fn reset(&mut self) {
        self.previous.fill(0.0);
        self.state.fill(0.0);
    }

    pub fn n_input(&self) -> usize {
        self.n_input
    }

    pub fn step(&mut self, raw: &[f32]) -> &[f32; 2] {
        assert_eq!(raw.len(), self.n_input, "input width");
        for index in 0..self.n_input {
            let current = ((raw[index] - self.mean[index]) * self.inv_std[index])
                .clamp(-self.input_clip, self.input_clip);
            self.input[index] = current;
            self.input[self.n_input + index] = current - self.previous[index];
            self.previous[index] = current;
        }

        gemv(
            &self.input_w,
            &self.input_b,
            &self.input,
            &mut self.projected,
        );
        self.projected
            .iter_mut()
            .for_each(|value| *value = gelu(*value));
        layer_norm(
            &mut self.projected,
            &self.input_norm_w,
            &self.input_norm_b,
            self.input_norm_eps,
        );

        for (index, layer) in self.layers.iter().enumerate() {
            let (before, rest) = self.state.split_at_mut(index * self.hidden);
            let hidden = &mut rest[..self.hidden];
            let input: &[f32] = if index == 0 {
                &self.projected
            } else {
                &before[(index - 1) * self.hidden..index * self.hidden]
            };
            gemv(&layer.w_ih, &layer.b_ih, input, &mut self.input_gates);
            gemv(&layer.w_hh, &layer.b_hh, hidden, &mut self.hidden_gates);
            let (ir, rest_i) = self.input_gates.split_at(self.hidden);
            let (iz, inn) = rest_i.split_at(self.hidden);
            let (hr, rest_h) = self.hidden_gates.split_at(self.hidden);
            let (hz, hn) = rest_h.split_at(self.hidden);
            for index in 0..self.hidden {
                let reset = sigmoid(ir[index] + hr[index]);
                let update = sigmoid(iz[index] + hz[index]);
                let candidate = (inn[index] + reset * hn[index]).tanh();
                hidden[index] = (1.0 - update) * candidate + update * hidden[index];
            }
        }

        let last_start = (self.layers.len() - 1) * self.hidden;
        self.head_input[..self.hidden]
            .copy_from_slice(&self.state[last_start..last_start + self.hidden]);
        self.head_input[self.hidden..].copy_from_slice(&self.projected);
        layer_norm(
            &mut self.head_input,
            &self.head_norm_w,
            &self.head_norm_b,
            self.head_norm_eps,
        );
        gemv(
            &self.head_w,
            &self.head_b,
            &self.head_input,
            &mut self.head_hidden,
        );
        self.head_hidden
            .iter_mut()
            .for_each(|value| *value = gelu(*value));
        gemv(
            &self.output_w,
            &self.output_b,
            &self.head_hidden,
            &mut self.output,
        );
        &self.output
    }
}

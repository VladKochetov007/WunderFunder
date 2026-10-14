//! Isolated native minGRU prototype for the 211-feature student input.
//!
//! This is not a replacement for `FastRnn` or the shipped GRU. It preserves the
//! student's normalized GELU encoder and skip readout while using input-only
//! candidate/gate projections and the explicitly zero-initialized minGRU state.

use crate::approx::{gelu_fast, sigmoid_fast};
use crate::kernels::{gelu, gemv, sigmoid};
use crate::rnn_fast::Activations;
use crate::vnni::{DynQ8, Isa, padded_cols, quantize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinGruPrecision {
    F32,
    Q8Vnni,
    Q8Portable,
}

impl MinGruPrecision {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "f32" => Ok(Self::F32),
            "q8vnni" => Ok(Self::Q8Vnni),
            "q8portable" => Ok(Self::Q8Portable),
            _ => Err(format!(
                "unsupported minGRU precision {value:?}; expected f32, q8vnni or q8portable"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8Vnni => "q8vnni",
            Self::Q8Portable => "q8portable",
        }
    }

    fn isa(self) -> Option<Isa> {
        match self {
            Self::F32 => None,
            Self::Q8Vnni => Some(Isa::Vnni512),
            Self::Q8Portable => Some(Isa::Portable),
        }
    }
}

#[derive(Debug)]
pub struct MinGruCellWeights {
    pub input_size: usize,
    /// Row-major `[2 * recurrent_width, input_size]`; candidate rows precede gate rows.
    pub projection: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Debug)]
pub struct MinGruWeights {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// Row-major `[encoder_width, input_size]`.
    pub inp_weight: Vec<f32>,
    pub inp_bias: Vec<f32>,
    pub cells: Vec<MinGruCellWeights>,
    /// Row-major `[readout_width, recurrent_width + encoder_width]`.
    pub head0_weight: Vec<f32>,
    pub head0_bias: Vec<f32>,
    /// Row-major `[2, readout_width]`; kept FP32 in Q8 mode.
    pub head2_weight: Vec<f32>,
    pub head2_bias: Vec<f32>,
}

enum Matrix {
    F32(Vec<f32>),
    Q8(DynQ8),
}

struct Linear {
    rows: usize,
    cols: usize,
    weights: Matrix,
    bias: Vec<f32>,
    activation_codes: Vec<u8>,
}

impl Linear {
    fn new(
        weights: Vec<f32>,
        bias: Vec<f32>,
        rows: usize,
        cols: usize,
        precision: MinGruPrecision,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(weights.len()) {
            return Err(format!(
                "linear matrix shape mismatch: {rows}x{cols} requires {} weights, got {}",
                rows.saturating_mul(cols),
                weights.len()
            ));
        }
        if bias.len() != rows {
            return Err(format!(
                "linear bias shape mismatch: expected {rows}, got {}",
                bias.len()
            ));
        }
        if weights.iter().chain(&bias).any(|value| !value.is_finite()) {
            return Err("linear weights and bias must be finite".into());
        }
        let (weights, activation_codes) = match precision.isa() {
            Some(_) => (
                Matrix::Q8(
                    DynQ8::pack(&weights, rows, cols)
                        .map_err(|error| format!("could not pack minGRU Q8 matrix: {error}"))?,
                ),
                vec![128; padded_cols(cols)],
            ),
            None => (Matrix::F32(weights), Vec::new()),
        };
        Ok(Self {
            rows,
            cols,
            weights,
            bias,
            activation_codes,
        })
    }

    fn apply(&mut self, input: &[f32], precision: MinGruPrecision, output: &mut [f32]) {
        assert_eq!(input.len(), self.cols, "minGRU linear input width");
        assert_eq!(output.len(), self.rows, "minGRU linear output width");
        match &self.weights {
            Matrix::F32(weights) => gemv(weights, &self.bias, input, output),
            Matrix::Q8(weights) => {
                let scale = quantize(input, &mut self.activation_codes);
                weights.gemv(
                    precision.isa().expect("Q8 matrix needs an ISA"),
                    &self.activation_codes,
                    scale,
                    Some(&self.bias),
                    output,
                );
            }
        }
    }

    fn matrix_bytes(&self) -> usize {
        match &self.weights {
            Matrix::F32(weights) => weights.len() * std::mem::size_of::<f32>(),
            Matrix::Q8(weights) => weights.storage_bytes(),
        }
    }

    fn nonmatrix_bytes(&self) -> usize {
        self.bias.len() * std::mem::size_of::<f32>()
    }

    fn scratch_bytes(&self) -> usize {
        self.activation_codes.len()
    }
}

/// Stateful, single-row model stepper with the same normalized encoder and skip readout as the
/// student GRU. The layer state is initialized to actual zero and cleared by `reset`.
pub struct MinGruModel {
    n_in: usize,
    recurrent: usize,
    mean: Vec<f32>,
    std: Vec<f32>,
    inp: Linear,
    cells: Vec<Linear>,
    head0: Linear,
    head2: Linear,
    state: Vec<f32>,
    normalized: Vec<f32>,
    projection: Vec<f32>,
    cell_projection: Vec<f32>,
    head_input: Vec<f32>,
    head_hidden: Vec<f32>,
    activations: Activations,
    precision: MinGruPrecision,
}

impl MinGruModel {
    pub fn new(
        weights: MinGruWeights,
        activations: Activations,
        precision: MinGruPrecision,
    ) -> Result<Self, String> {
        let n_in = weights.mean.len();
        let encoder = weights.inp_bias.len();
        let head_input = weights.head0_weight.len() / weights.head0_bias.len().max(1);
        if n_in != 211 || weights.std.len() != n_in {
            return Err(format!(
                "native minGRU requires 211-feature mean/std vectors, got {n_in}/{}",
                weights.std.len()
            ));
        }
        if weights
            .mean
            .iter()
            .chain(&weights.std)
            .any(|value| !value.is_finite())
        {
            return Err("minGRU normalization vectors must be finite".into());
        }
        if weights.std.iter().any(|value| *value <= 0.0) {
            return Err("minGRU normalization standard deviations must be positive".into());
        }
        if encoder == 0 || weights.inp_weight.len() != encoder * n_in {
            return Err("minGRU inp.weight must have shape [encoder_width, 211]".into());
        }
        if weights.cells.is_empty() {
            return Err("native minGRU requires at least one recurrent layer".into());
        }
        if weights.cells[0].bias.is_empty() || weights.cells[0].bias.len() % 2 != 0 {
            return Err("minGRU cell projection bias must have shape [2 * recurrent_width]".into());
        }
        let recurrent = weights.cells[0].bias.len() / 2;
        if recurrent == 0 {
            return Err("minGRU recurrent width must be positive".into());
        }
        for (index, cell) in weights.cells.iter().enumerate() {
            let expected_input = if index == 0 { encoder } else { recurrent };
            if cell.input_size != expected_input
                || cell.bias.len() != 2 * recurrent
                || cell.projection.len() != 2 * recurrent * expected_input
            {
                return Err(format!(
                    "minGRU cell {index} must project {expected_input} inputs to {recurrent} candidate and gate values"
                ));
            }
        }
        if weights.head0_bias.is_empty() || head_input < encoder {
            return Err("minGRU head.0 has invalid shape".into());
        }
        let readout = weights.head0_bias.len();
        if weights.head0_weight.len() != readout * head_input || head_input - encoder != recurrent {
            return Err(
                "minGRU head.0 must have shape [M, recurrent_width + encoder_width]".into(),
            );
        }
        if weights.head2_bias.len() != 2 || weights.head2_weight.len() != 2 * readout {
            return Err("native minGRU head.2 must have shape [2, readout_width]".into());
        }

        let inp = Linear::new(
            weights.inp_weight,
            weights.inp_bias,
            encoder,
            n_in,
            precision,
        )?;
        let mut cells = Vec::with_capacity(weights.cells.len());
        for cell in weights.cells {
            cells.push(Linear::new(
                cell.projection,
                cell.bias,
                2 * recurrent,
                cell.input_size,
                precision,
            )?);
        }
        let head0 = Linear::new(
            weights.head0_weight,
            weights.head0_bias,
            readout,
            recurrent + encoder,
            precision,
        )?;
        // Like the current native Q8 GRU path, the final two-output projection stays FP32.
        let head2 = Linear::new(
            weights.head2_weight,
            weights.head2_bias,
            2,
            readout,
            MinGruPrecision::F32,
        )?;
        let layers = cells.len();
        Ok(Self {
            n_in,
            recurrent,
            mean: weights.mean,
            std: weights.std,
            inp,
            cells,
            head0,
            head2,
            state: vec![0.0; layers * recurrent],
            normalized: vec![0.0; n_in],
            projection: vec![0.0; encoder],
            cell_projection: vec![0.0; 2 * recurrent],
            head_input: vec![0.0; recurrent + encoder],
            head_hidden: vec![0.0; readout],
            activations,
            precision,
        })
    }

    pub fn step(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.n_in, "minGRU model input width");
        assert_eq!(output.len(), 2, "minGRU output width");
        for (normalized, ((value, mean), std)) in self
            .normalized
            .iter_mut()
            .zip(input.iter().zip(&self.mean).zip(&self.std))
        {
            *normalized = (*value - *mean) / *std;
        }
        self.inp
            .apply(&self.normalized, self.precision, &mut self.projection);
        match self.activations {
            Activations::Exact => self
                .projection
                .iter_mut()
                .for_each(|value| *value = gelu(*value)),
            Activations::Fast => self
                .projection
                .iter_mut()
                .for_each(|value| *value = gelu_fast(*value)),
        }

        let recurrent = self.recurrent;
        for (index, cell) in self.cells.iter_mut().enumerate() {
            let start = index * recurrent;
            let end = start + recurrent;
            let prior_layer: &[f32] = if index == 0 {
                &self.projection
            } else {
                &self.state[start - recurrent..start]
            };
            cell.apply(prior_layer, self.precision, &mut self.cell_projection);
            let (candidate, gate) = self.cell_projection.split_at(recurrent);
            for (hidden, (&candidate, &gate)) in self.state[start..end]
                .iter_mut()
                .zip(candidate.iter().zip(gate))
            {
                let candidate = if candidate >= 0.0 {
                    candidate + 0.5
                } else {
                    match self.activations {
                        Activations::Exact => sigmoid(candidate),
                        Activations::Fast => sigmoid_fast(candidate),
                    }
                };
                let write = match self.activations {
                    Activations::Exact => sigmoid(gate),
                    Activations::Fast => sigmoid_fast(gate),
                };
                *hidden = (1.0 - write) * *hidden + write * candidate;
            }
        }

        let state_start = self.state.len() - recurrent;
        self.head_input[..recurrent].copy_from_slice(&self.state[state_start..]);
        self.head_input[recurrent..].copy_from_slice(&self.projection);
        self.head0
            .apply(&self.head_input, self.precision, &mut self.head_hidden);
        match self.activations {
            Activations::Exact => self
                .head_hidden
                .iter_mut()
                .for_each(|value| *value = gelu(*value)),
            Activations::Fast => self
                .head_hidden
                .iter_mut()
                .for_each(|value| *value = gelu_fast(*value)),
        }
        self.head2
            .apply(&self.head_hidden, MinGruPrecision::F32, output);
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    pub fn hidden_state(&self) -> &[f32] {
        &self.state
    }

    pub fn layers(&self) -> usize {
        self.cells.len()
    }

    pub fn recurrent_width(&self) -> usize {
        self.recurrent
    }

    pub fn precision(&self) -> MinGruPrecision {
        self.precision
    }

    pub fn fast_activations(&self) -> bool {
        self.activations == Activations::Fast
    }

    pub fn storage_bytes(&self) -> MinGruStorage {
        let linears = std::iter::once(&self.inp)
            .chain(self.cells.iter())
            .chain(std::iter::once(&self.head0))
            .chain(std::iter::once(&self.head2));
        let matrix_weights = linears.clone().map(Linear::matrix_bytes).sum();
        let biases = linears.clone().map(Linear::nonmatrix_bytes).sum();
        let quantizer_scratch: usize = linears.map(Linear::scratch_bytes).sum();
        let normalization = (self.mean.len() + self.std.len()) * std::mem::size_of::<f32>();
        let recurrent_state = self.state.len() * std::mem::size_of::<f32>();
        let working_scratch = (self.normalized.len()
            + self.projection.len()
            + self.cell_projection.len()
            + self.head_input.len()
            + self.head_hidden.len()
            + 2)
            * std::mem::size_of::<f32>();
        MinGruStorage {
            matrix_weights,
            biases,
            normalization,
            recurrent_state,
            scratch: quantizer_scratch + working_scratch,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinGruStorage {
    pub matrix_weights: usize,
    pub biases: usize,
    pub normalization: usize,
    pub recurrent_state: usize,
    pub scratch: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|index| (((index * 37 + seed * 17) % 101) as f32 - 50.0) / 500.0)
            .collect()
    }

    fn fixture() -> MinGruWeights {
        let (input, encoder, recurrent, readout, layers) = (211, 12, 8, 6, 2);
        let mut cells = Vec::with_capacity(layers);
        for index in 0..layers {
            let cols = if index == 0 { encoder } else { recurrent };
            cells.push(MinGruCellWeights {
                input_size: cols,
                projection: values(2 * recurrent * cols, 20 + index),
                bias: values(2 * recurrent, 30 + index),
            });
        }
        MinGruWeights {
            mean: values(input, 2),
            std: (0..input)
                .map(|index| 0.75 + index as f32 / 2000.0)
                .collect(),
            inp_weight: values(encoder * input, 3),
            inp_bias: values(encoder, 4),
            cells,
            head0_weight: values(readout * (recurrent + encoder), 5),
            head0_bias: values(readout, 6),
            head2_weight: values(2 * readout, 7),
            head2_bias: values(2, 8),
        }
    }

    fn zero_fixture() -> MinGruWeights {
        MinGruWeights {
            mean: vec![0.0; 211],
            std: vec![1.0; 211],
            inp_weight: vec![0.0; 211],
            inp_bias: vec![0.0],
            cells: vec![MinGruCellWeights {
                input_size: 1,
                projection: vec![0.0; 2],
                bias: vec![0.0; 2],
            }],
            head0_weight: vec![0.0; 2],
            head0_bias: vec![0.0],
            head2_weight: vec![0.0; 2],
            head2_bias: vec![0.0; 2],
        }
    }

    #[test]
    fn exact_zero_state_is_not_transformed_before_the_first_write() {
        let mut model =
            MinGruModel::new(zero_fixture(), Activations::Exact, MinGruPrecision::F32).unwrap();
        let mut output = [0.0; 2];
        for expected in [0.25f32, 0.375, 0.4375] {
            model.step(&[0.0; 211], &mut output);
            assert_eq!(model.hidden_state(), &[expected]);
        }
        model.reset();
        assert_eq!(model.hidden_state(), &[0.0]);
    }

    #[test]
    fn exact_model_is_causal_chunkable_and_reset_replays() {
        let rows = 257;
        let inputs: Vec<f32> = (0..rows * 211)
            .map(|index| ((index * 13 % 89) as f32 - 44.0) / 80.0)
            .collect();
        let mut full =
            MinGruModel::new(fixture(), Activations::Exact, MinGruPrecision::F32).unwrap();
        let mut expected = vec![0.0; rows * 2];
        for (row, output) in inputs.chunks_exact(211).zip(expected.chunks_exact_mut(2)) {
            full.step(row, output);
        }
        let final_state = full.hidden_state().to_vec();
        let mut chunked =
            MinGruModel::new(fixture(), Activations::Exact, MinGruPrecision::F32).unwrap();
        let mut actual = vec![0.0; rows * 2];
        for (input_chunk, output_chunk) in inputs
            .chunks_exact(99 * 211)
            .zip(actual.chunks_exact_mut(99 * 2))
        {
            for (row, output) in input_chunk
                .chunks_exact(211)
                .zip(output_chunk.chunks_exact_mut(2))
            {
                chunked.step(row, output);
            }
        }
        if rows % 99 != 0 {
            let start = rows / 99 * 99;
            for (row, output) in inputs[start * 211..]
                .chunks_exact(211)
                .zip(actual[start * 2..].chunks_exact_mut(2))
            {
                chunked.step(row, output);
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(chunked.hidden_state(), final_state);
        let prefix_rows = 99;
        let mut prefix =
            MinGruModel::new(fixture(), Activations::Exact, MinGruPrecision::F32).unwrap();
        let mut prefix_output = vec![0.0; prefix_rows * 2];
        for (row, output) in inputs[..prefix_rows * 211]
            .chunks_exact(211)
            .zip(prefix_output.chunks_exact_mut(2))
        {
            prefix.step(row, output);
        }
        assert_eq!(&expected[..prefix_rows * 2], prefix_output);
        full.reset();
        let mut replay = vec![0.0; rows * 2];
        for (row, output) in inputs.chunks_exact(211).zip(replay.chunks_exact_mut(2)) {
            full.step(row, output);
        }
        assert_eq!(replay, expected);
    }

    #[test]
    fn q8_uses_exact_packed_matrices_and_keeps_finite_state() {
        let mut model =
            MinGruModel::new(fixture(), Activations::Fast, MinGruPrecision::Q8Portable).unwrap();
        let mut output = [0.0; 2];
        for index in 0..2_000 {
            let input = std::array::from_fn::<_, 211, _>(|column| {
                ((index * 7 + column * 11) % 97) as f32 / 32.0 - 1.5
            });
            model.step(&input, &mut output);
        }
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(model.hidden_state().iter().all(|value| value.is_finite()));
        assert!(model.storage_bytes().matrix_weights > 0);
        model.reset();
        assert!(model.hidden_state().iter().all(|value| *value == 0.0));
    }
}

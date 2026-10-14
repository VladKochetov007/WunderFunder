//! Causal FP32 initial-state policies for the unchanged streaming GRU backbone.

use crate::kernels::{gelu, gemv};
use crate::rnn_fast::FastRnn;

const PREFIX_ROWS: usize = 99;
const CONTEXT_WIDTH: usize = 64;

pub enum StateInitializerConfig {
    Zero,
    Constant {
        bound: f32,
        values: Vec<f32>,
    },
    FirstRow {
        bound: f32,
        input_weight: Vec<f32>,
        input_bias: Vec<f32>,
        output_weight: Vec<f32>,
        output_bias: Vec<f32>,
    },
    Prefix {
        bound: f32,
        input_weight: Vec<f32>,
        input_bias: Vec<f32>,
        output_weight: Vec<f32>,
        output_bias: Vec<f32>,
    },
}

#[derive(Clone, Copy)]
enum InitializerKind {
    Zero,
    Constant,
    FirstRow,
    Prefix,
}

/// Sequence-local initializer parameters and causal prefix moments.
pub struct StateInitializer {
    kind: InitializerKind,
    constant: Vec<f32>,
    bound: f32,
    n_in: usize,
    input_weight: Vec<f32>,
    input_bias: Vec<f32>,
    output_weight: Vec<f32>,
    output_bias: Vec<f32>,
    normalized: Vec<f32>,
    first_hidden: Vec<f32>,
    prediction: Vec<f32>,
    context: Vec<f32>,
    mean: Vec<f32>,
    m2: Vec<f32>,
    last: Vec<f32>,
    position: usize,
}

impl StateInitializer {
    pub fn new(
        config: StateInitializerConfig,
        n_in: usize,
        layers: usize,
        hidden: usize,
    ) -> Result<Self, String> {
        if n_in == 0 || layers == 0 || hidden == 0 {
            return Err("initializer dimensions must be positive".into());
        }
        let state_width = layers
            .checked_mul(hidden)
            .ok_or_else(|| "initializer state size overflow".to_owned())?;
        let (kind, bound, constant, input_weight, input_bias, output_weight, output_bias) =
            match config {
                StateInitializerConfig::Zero => (
                    InitializerKind::Zero,
                    0.0,
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                ),
                StateInitializerConfig::Constant { bound, values } => {
                    validate_bound(bound)?;
                    expect_len("constant", &values, state_width)?;
                    ensure_finite("constant", &values)?;
                    let values = values.iter().map(|value| bound * value.tanh()).collect();
                    (
                        InitializerKind::Constant,
                        bound,
                        values,
                        vec![],
                        vec![],
                        vec![],
                        vec![],
                    )
                }
                StateInitializerConfig::FirstRow {
                    bound,
                    input_weight,
                    input_bias,
                    output_weight,
                    output_bias,
                } => {
                    validate_bound(bound)?;
                    validate_context_weights(
                        n_in,
                        state_width,
                        &input_weight,
                        &input_bias,
                        &output_weight,
                        &output_bias,
                    )?;
                    (
                        InitializerKind::FirstRow,
                        bound,
                        vec![],
                        input_weight,
                        input_bias,
                        output_weight,
                        output_bias,
                    )
                }
                StateInitializerConfig::Prefix {
                    bound,
                    input_weight,
                    input_bias,
                    output_weight,
                    output_bias,
                } => {
                    validate_bound(bound)?;
                    validate_context_weights(
                        3 * n_in,
                        state_width,
                        &input_weight,
                        &input_bias,
                        &output_weight,
                        &output_bias,
                    )?;
                    (
                        InitializerKind::Prefix,
                        bound,
                        vec![],
                        input_weight,
                        input_bias,
                        output_weight,
                        output_bias,
                    )
                }
            };
        for (name, values) in [
            ("input weight", &input_weight),
            ("input bias", &input_bias),
            ("output weight", &output_weight),
            ("output bias", &output_bias),
        ] {
            ensure_finite(name, values)?;
        }
        let prefix = matches!(kind, InitializerKind::Prefix);
        Ok(Self {
            kind,
            constant,
            bound,
            n_in,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
            normalized: vec![0.0; n_in],
            first_hidden: vec![0.0; CONTEXT_WIDTH],
            prediction: vec![0.0; state_width],
            context: vec![0.0; if prefix { 3 * n_in } else { n_in }],
            mean: vec![0.0; if prefix { n_in } else { 0 }],
            m2: vec![0.0; if prefix { n_in } else { 0 }],
            last: vec![0.0; if prefix { n_in } else { 0 }],
            position: 0,
        })
    }

    pub fn is_zero(&self) -> bool {
        matches!(self.kind, InitializerKind::Zero)
    }

    pub fn reset(&mut self) {
        self.position = 0;
        self.mean.fill(0.0);
        self.m2.fill(0.0);
        self.last.fill(0.0);
    }

    /// Apply the current row's initializer action before the backbone consumes the row.
    pub fn before_step(&mut self, model: &mut FastRnn, input: &[f32]) -> Result<(), String> {
        if input.len() != self.n_in || model.n_in() != self.n_in {
            return Err("initializer input width differs from the backbone".into());
        }
        match self.kind {
            InitializerKind::Zero => {}
            InitializerKind::Constant if self.position == 0 => {
                if self.constant.iter().any(|value| *value != 0.0) {
                    model
                        .set_hidden_state(&self.constant)
                        .map_err(|error| error.0)?;
                }
            }
            InitializerKind::FirstRow if self.position == 0 => {
                self.normalize(model, input)?;
                self.context.copy_from_slice(&self.normalized);
                self.predict_context()?;
                if self.prediction.iter().any(|value| *value != 0.0) {
                    model
                        .set_hidden_state(&self.prediction)
                        .map_err(|error| error.0)?;
                }
            }
            InitializerKind::Prefix if self.position < PREFIX_ROWS => {
                self.normalize(model, input)?;
                let count = (self.position + 1) as f32;
                for feature in 0..self.n_in {
                    let row = self.normalized[feature];
                    let delta = row - self.mean[feature];
                    self.mean[feature] += delta / count;
                    self.m2[feature] += delta * (row - self.mean[feature]);
                    self.last[feature] = row;
                }
            }
            InitializerKind::Prefix if self.position == PREFIX_ROWS => {
                for feature in 0..self.n_in {
                    let offset = 2 * self.n_in + feature;
                    self.context[feature] = self.mean[feature];
                    self.context[self.n_in + feature] =
                        (self.m2[feature] / PREFIX_ROWS as f32).max(0.0).sqrt();
                    self.context[offset] = self.last[feature];
                }
                self.predict_context()?;
                if self.prediction.iter().any(|value| *value != 0.0) {
                    model
                        .add_hidden_state(&self.prediction)
                        .map_err(|error| error.0)?;
                }
                self.mean.fill(0.0);
                self.m2.fill(0.0);
                self.last.fill(0.0);
            }
            _ => {}
        }
        self.position = self.position.saturating_add(1);
        Ok(())
    }

    fn normalize(&mut self, model: &FastRnn, input: &[f32]) -> Result<(), String> {
        model.normalize_input(input, &mut self.normalized);
        if self.normalized.iter().any(|value| !value.is_finite()) {
            return Err("normalized initializer input must be finite".into());
        }
        Ok(())
    }

    fn predict_context(&mut self) -> Result<(), String> {
        let context = &self.context;
        gemv(
            &self.input_weight,
            &self.input_bias,
            context,
            &mut self.first_hidden,
        );
        self.first_hidden
            .iter_mut()
            .for_each(|value| *value = gelu(*value));
        gemv(
            &self.output_weight,
            &self.output_bias,
            &self.first_hidden,
            &mut self.prediction,
        );
        let bound = self.bound;
        self.prediction
            .iter_mut()
            .for_each(|value| *value = bound * value.tanh());
        if self.prediction.iter().any(|value| !value.is_finite()) {
            return Err("initializer prediction must be finite".into());
        }
        Ok(())
    }
}

fn validate_bound(bound: f32) -> Result<(), String> {
    if bound.is_finite() && bound > 0.0 {
        Ok(())
    } else {
        Err("initializer residual bound must be finite and positive".into())
    }
}

fn expect_len(name: &str, values: &[f32], expected: usize) -> Result<(), String> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "{name} length mismatch: expected {expected}, got {}",
            values.len()
        ))
    }
}

fn ensure_finite(name: &str, values: &[f32]) -> Result<(), String> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(format!("{name} values must be finite"))
    }
}

fn validate_context_weights(
    context_width: usize,
    state_width: usize,
    input_weight: &[f32],
    input_bias: &[f32],
    output_weight: &[f32],
    output_bias: &[f32],
) -> Result<(), String> {
    expect_len("input weight", input_weight, CONTEXT_WIDTH * context_width)?;
    expect_len("input bias", input_bias, CONTEXT_WIDTH)?;
    expect_len("output weight", output_weight, state_width * CONTEXT_WIDTH)?;
    expect_len("output bias", output_bias, state_width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_stats_reset_with_the_sequence_state() {
        let mut initializer = StateInitializer::new(
            StateInitializerConfig::Prefix {
                bound: 1.0,
                input_weight: vec![0.0; CONTEXT_WIDTH * 3],
                input_bias: vec![0.0; CONTEXT_WIDTH],
                output_weight: vec![0.0; 2 * CONTEXT_WIDTH],
                output_bias: vec![0.0; 2],
            },
            1,
            1,
            2,
        )
        .unwrap();
        assert_eq!(initializer.position, 0);
        initializer.position = 100;
        initializer.mean[0] = 3.0;
        initializer.m2[0] = 4.0;
        initializer.last[0] = 5.0;
        initializer.reset();
        assert_eq!(initializer.position, 0);
        assert_eq!(initializer.mean, [0.0]);
        assert_eq!(initializer.m2, [0.0]);
        assert_eq!(initializer.last, [0.0]);
    }
}

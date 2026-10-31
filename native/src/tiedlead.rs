//! Causal state and nonlinear encoder for the `raw112-event99-lead52-v1` model input.

use crate::kernels::{gelu, gemv};

pub const RAW_WIDTH: usize = 112;
pub const EVENT_WIDTH: usize = 99;
pub const INPUT_WIDTH: usize = RAW_WIDTH + EVENT_WIDTH + 52;
pub const LEAD_WIDTH: usize = 52;
const I0_PRICE_EVENT_INDEX: usize = 66;
const SHARED_WIDTH: usize = 128;
const BRANCH_WIDTH: usize = 64;
const LEAD_RAW_START: usize = 52;
const LEAD_RAW_STOP: usize = 104;

/// Weights for the residual branches attached to the ordinary linear encoder.
pub struct TiedLeadWeights {
    encoder_width: usize,
    tied_w1: Vec<f32>,
    tied_b1: Vec<f32>,
    tied_w2: Vec<f32>,
    tied_b2: Vec<f32>,
    tied_out_w: Vec<f32>,
    tied_out_b: Vec<f32>,
    lead_out_w: Vec<f32>,
    lead_out_b: Vec<f32>,
}

impl TiedLeadWeights {
    /// Validate and construct weights copied from the PyTorch tiedlead module.
    pub fn new(
        encoder_width: usize,
        tied_w1: Vec<f32>,
        tied_b1: Vec<f32>,
        tied_w2: Vec<f32>,
        tied_b2: Vec<f32>,
        tied_out_w: Vec<f32>,
        tied_out_b: Vec<f32>,
        lead_out_w: Vec<f32>,
        lead_out_b: Vec<f32>,
    ) -> Result<Self, String> {
        let expected = [
            (
                "tied.0.weight",
                tied_w1.len(),
                SHARED_WIDTH * (2 * LEAD_WIDTH),
            ),
            ("tied.0.bias", tied_b1.len(), SHARED_WIDTH),
            ("tied.2.weight", tied_w2.len(), BRANCH_WIDTH * SHARED_WIDTH),
            ("tied.2.bias", tied_b2.len(), BRANCH_WIDTH),
            (
                "tied_out.weight",
                tied_out_w.len(),
                encoder_width * (2 * BRANCH_WIDTH),
            ),
            ("tied_out.bias", tied_out_b.len(), encoder_width),
            (
                "lead_out.weight",
                lead_out_w.len(),
                encoder_width * BRANCH_WIDTH,
            ),
            ("lead_out.bias", lead_out_b.len(), encoder_width),
        ];
        if let Some((name, got, want)) = expected.into_iter().find(|(_, got, want)| got != want) {
            return Err(format!("{name}: expected {want} values, got {got}"));
        }
        if encoder_width == 0 {
            return Err("tiedlead encoder width must be positive".into());
        }
        Ok(Self {
            encoder_width,
            tied_w1,
            tied_b1,
            tied_w2,
            tied_b2,
            tied_out_w,
            tied_out_b,
            lead_out_w,
            lead_out_b,
        })
    }

    pub fn encoder_width(&self) -> usize {
        self.encoder_width
    }
}

/// One tiedlead model's reusable scratch and current-sequence feature state.
pub struct TiedLeadEncoder {
    weights: TiedLeadWeights,
    first: bool,
    previous_raw: [f32; LEAD_RAW_STOP],
    latest_i1: [f32; LEAD_WIDTH],
    shared_input: [f32; 2 * LEAD_WIDTH],
    shared_hidden: [f32; SHARED_WIDTH],
    branch_output: [f32; BRANCH_WIDTH],
    i0: [f32; BRANCH_WIDTH],
    i1: [f32; BRANCH_WIDTH],
    lead: [f32; BRANCH_WIDTH],
    branch_pair: [f32; 2 * BRANCH_WIDTH],
    lead_projection: Vec<f32>,
}

impl TiedLeadEncoder {
    /// Create a resettable tiedlead encoder for one model.
    pub fn new(weights: TiedLeadWeights) -> Self {
        Self {
            lead_projection: vec![0.0; weights.encoder_width],
            weights,
            first: true,
            previous_raw: [0.0; LEAD_RAW_STOP],
            latest_i1: [0.0; LEAD_WIDTH],
            shared_input: [0.0; 2 * LEAD_WIDTH],
            shared_hidden: [0.0; SHARED_WIDTH],
            branch_output: [0.0; BRANCH_WIDTH],
            i0: [0.0; BRANCH_WIDTH],
            i1: [0.0; BRANCH_WIDTH],
            lead: [0.0; BRANCH_WIDTH],
            branch_pair: [0.0; 2 * BRANCH_WIDTH],
        }
    }

    /// Reset history to the start of a complete sequence.
    pub fn reset(&mut self) {
        self.first = true;
        self.previous_raw.fill(0.0);
        self.latest_i1.fill(0.0);
    }

    /// Assemble the input using the causal latest-i1 cache before model normalization.
    pub fn assemble_input(
        &mut self,
        raw: &[f32],
        event: &[f32],
        input: &mut [f32],
    ) -> Result<(), String> {
        if raw.len() != RAW_WIDTH || event.len() != EVENT_WIDTH || input.len() != INPUT_WIDTH {
            return Err("tiedlead input must be raw112 + event99 + lead52".into());
        }
        if self.first || event[I0_PRICE_EVENT_INDEX] > 0.5 {
            self.latest_i1
                .copy_from_slice(&raw[LEAD_RAW_START..LEAD_RAW_STOP]);
        }
        input[..RAW_WIDTH].copy_from_slice(raw);
        input[RAW_WIDTH..RAW_WIDTH + EVENT_WIDTH].copy_from_slice(event);
        input[RAW_WIDTH + EVENT_WIDTH..].copy_from_slice(&self.latest_i1);
        Ok(())
    }

    /// Compute the tiedlead encoder residual from the model's normalized-then-restored input.
    pub fn encoder_residual(
        &mut self,
        restored_input: &[f32],
        residual: &mut [f32],
    ) -> Result<(), String> {
        if restored_input.len() != INPUT_WIDTH {
            return Err("restored tiedlead input must have 263 columns".into());
        }
        if residual.len() != self.weights.encoder_width {
            return Err(format!(
                "tiedlead residual must have encoder width {}",
                self.weights.encoder_width
            ));
        }

        let model_raw = &restored_input[..RAW_WIDTH];
        for index in 0..LEAD_WIDTH {
            self.shared_input[index] = model_raw[index];
            self.shared_input[LEAD_WIDTH + index] = if self.first {
                0.0
            } else {
                model_raw[index] - self.previous_raw[index]
            };
        }
        self.i0 = self.shared_forward();

        for index in 0..LEAD_WIDTH {
            self.shared_input[index] = model_raw[LEAD_RAW_START + index];
            self.shared_input[LEAD_WIDTH + index] = if self.first {
                0.0
            } else {
                model_raw[LEAD_RAW_START + index] - self.previous_raw[LEAD_RAW_START + index]
            };
        }
        self.i1 = self.shared_forward();

        self.branch_pair[..BRANCH_WIDTH].copy_from_slice(&self.i0);
        self.branch_pair[BRANCH_WIDTH..].copy_from_slice(&self.i1);
        gemv(
            &self.weights.tied_out_w,
            &self.weights.tied_out_b,
            &self.branch_pair,
            residual,
        );

        for index in 0..LEAD_WIDTH {
            let current = model_raw[LEAD_RAW_START + index];
            self.shared_input[index] = current;
            self.shared_input[LEAD_WIDTH + index] =
                current - restored_input[RAW_WIDTH + EVENT_WIDTH + index];
        }
        self.lead = self.shared_forward();
        gemv(
            &self.weights.lead_out_w,
            &self.weights.lead_out_b,
            &self.lead,
            &mut self.lead_projection,
        );
        for (total, addition) in residual.iter_mut().zip(&self.lead_projection) {
            *total += addition;
        }

        self.previous_raw
            .copy_from_slice(&model_raw[..LEAD_RAW_STOP]);
        self.first = false;
        Ok(())
    }

    fn shared_forward(&mut self) -> [f32; BRANCH_WIDTH] {
        gemv(
            &self.weights.tied_w1,
            &self.weights.tied_b1,
            &self.shared_input,
            &mut self.shared_hidden,
        );
        self.shared_hidden
            .iter_mut()
            .for_each(|value| *value = gelu(*value));
        gemv(
            &self.weights.tied_w2,
            &self.weights.tied_b2,
            &self.shared_hidden,
            &mut self.branch_output,
        );
        self.branch_output
            .iter_mut()
            .for_each(|value| *value = gelu(*value));
        self.branch_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_weights(encoder_width: usize) -> TiedLeadWeights {
        TiedLeadWeights::new(
            encoder_width,
            vec![0.0; SHARED_WIDTH * (2 * LEAD_WIDTH)],
            vec![0.0; SHARED_WIDTH],
            vec![0.0; BRANCH_WIDTH * SHARED_WIDTH],
            vec![0.0; BRANCH_WIDTH],
            vec![0.0; encoder_width * (2 * BRANCH_WIDTH)],
            vec![0.0; encoder_width],
            vec![0.0; encoder_width * BRANCH_WIDTH],
            vec![0.0; encoder_width],
        )
        .unwrap()
    }

    #[test]
    fn lead_cache_tracks_latest_event_and_resets() {
        let mut encoder = TiedLeadEncoder::new(zero_weights(3));
        let mut raw = [0.0; RAW_WIDTH];
        raw[LEAD_RAW_START..LEAD_RAW_STOP].fill(3.0);
        let mut event = [0.0; EVENT_WIDTH];
        let mut input = [0.0; INPUT_WIDTH];
        let mut residual = [0.0; 3];

        encoder.assemble_input(&raw, &event, &mut input).unwrap();
        encoder.encoder_residual(&input, &mut residual).unwrap();
        assert!(
            input[RAW_WIDTH + EVENT_WIDTH..]
                .iter()
                .all(|value| *value == 3.0)
        );

        raw[LEAD_RAW_START..LEAD_RAW_STOP].fill(4.0);
        encoder.assemble_input(&raw, &event, &mut input).unwrap();
        encoder.encoder_residual(&input, &mut residual).unwrap();
        assert!(
            input[RAW_WIDTH + EVENT_WIDTH..]
                .iter()
                .all(|value| *value == 3.0)
        );

        raw[LEAD_RAW_START..LEAD_RAW_STOP].fill(9.0);
        event[I0_PRICE_EVENT_INDEX] = 1.0;
        encoder.assemble_input(&raw, &event, &mut input).unwrap();
        encoder.encoder_residual(&input, &mut residual).unwrap();
        assert!(
            input[RAW_WIDTH + EVENT_WIDTH..]
                .iter()
                .all(|value| *value == 9.0)
        );

        encoder.reset();
        event[I0_PRICE_EVENT_INDEX] = 0.0;
        raw[LEAD_RAW_START..LEAD_RAW_STOP].fill(2.0);
        encoder.assemble_input(&raw, &event, &mut input).unwrap();
        encoder.encoder_residual(&input, &mut residual).unwrap();
        assert!(
            input[RAW_WIDTH + EVENT_WIDTH..]
                .iter()
                .all(|value| *value == 2.0)
        );
    }

    #[test]
    fn first_row_changes_are_zero_and_event_flag_is_row_local() {
        let weights = TiedLeadWeights::new(
            1,
            vec![0.0; SHARED_WIDTH * (2 * LEAD_WIDTH)],
            vec![0.0; SHARED_WIDTH],
            vec![0.0; BRANCH_WIDTH * SHARED_WIDTH],
            vec![0.0; BRANCH_WIDTH],
            vec![0.0; 2 * BRANCH_WIDTH],
            vec![0.0],
            vec![0.0; BRANCH_WIDTH],
            vec![0.0],
        )
        .unwrap();
        let mut encoder = TiedLeadEncoder::new(weights);
        let mut raw = [1.0; RAW_WIDTH];
        let event = [0.0; EVENT_WIDTH];
        let mut input = [0.0; INPUT_WIDTH];
        let mut residual = [0.0; 1];
        encoder.assemble_input(&raw, &event, &mut input).unwrap();
        encoder.encoder_residual(&input, &mut residual).unwrap();
        assert!(
            encoder.shared_input[LEAD_WIDTH..]
                .iter()
                .all(|value| *value == 0.0)
        );

        raw[0] = 2.0;
        encoder.assemble_input(&raw, &event, &mut input).unwrap();
        encoder.encoder_residual(&input, &mut residual).unwrap();
        assert_eq!(encoder.shared_input[0], 1.0);
        assert_eq!(encoder.shared_input[LEAD_WIDTH], 0.0);
    }
}

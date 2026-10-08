//! Native rank-32 multiplicative readout used by the H272 parent follow-up.
//!
//! This is deliberately a small inference primitive rather than a generic head
//! framework. It consumes FastRnn's existing [recurrent, projection]
//! representation and the causal event features already produced by
//! StreamFeatures.

use crate::kernels::gelu;

pub const HIDDEN_WIDTH: usize = 512;
pub const CONTEXT_WIDTH: usize = 99;
pub const RANK: usize = 32;
pub const OUTPUT_WIDTH: usize = 2;
const FUSION_WIDTH: usize = 3 * RANK;

/// Frozen FP32 parameters for one H272 multiplicative control.
#[derive(Clone, Debug)]
pub struct MultiplicativeReadout {
    hidden_projection: Vec<f32>,
    context_projection: Vec<f32>,
    fusion: Vec<f32>,
    fusion_bias: Vec<f32>,
    output: Vec<f32>,
    output_bias: Vec<f32>,
}

impl MultiplicativeReadout {
    pub fn new(
        hidden_projection: Vec<f32>,
        context_projection: Vec<f32>,
        fusion: Vec<f32>,
        fusion_bias: Vec<f32>,
        output: Vec<f32>,
        output_bias: Vec<f32>,
    ) -> Result<Self, String> {
        let expected = [
            (
                "hidden_projection",
                hidden_projection.len(),
                RANK * HIDDEN_WIDTH,
            ),
            (
                "context_projection",
                context_projection.len(),
                RANK * CONTEXT_WIDTH,
            ),
            ("fusion", fusion.len(), RANK * FUSION_WIDTH),
            ("fusion_bias", fusion_bias.len(), RANK),
            ("output", output.len(), OUTPUT_WIDTH * RANK),
            ("output_bias", output_bias.len(), OUTPUT_WIDTH),
        ];
        for (name, actual, wanted) in expected {
            if actual != wanted {
                return Err(format!("{name}: expected {wanted} values, got {actual}"));
            }
        }
        if [
            &hidden_projection,
            &context_projection,
            &fusion,
            &fusion_bias,
            &output,
            &output_bias,
        ]
        .into_iter()
        .any(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err("readout weights must be finite".into());
        }
        Ok(Self {
            hidden_projection,
            context_projection,
            fusion,
            fusion_bias,
            output,
            output_bias,
        })
    }

    /// Apply parent * (1 + tanh(output(GELU(fusion(hidden_rank * context_rank)))))
    pub fn apply(
        &self,
        anchor: &[f32],
        hidden: &[f32],
        context: &[f32],
    ) -> Result<[f32; OUTPUT_WIDTH], String> {
        if anchor.len() != OUTPUT_WIDTH {
            return Err(format!(
                "anchor: expected {OUTPUT_WIDTH} values, got {}",
                anchor.len()
            ));
        }
        if hidden.len() != HIDDEN_WIDTH {
            return Err(format!(
                "hidden: expected {HIDDEN_WIDTH} values, got {}",
                hidden.len()
            ));
        }
        if context.len() != CONTEXT_WIDTH {
            return Err(format!(
                "context: expected {CONTEXT_WIDTH} values, got {}",
                context.len()
            ));
        }

        let mut hidden_rank = [0.0f32; RANK];
        let mut context_rank = [0.0f32; RANK];
        let mut product = [0.0f32; RANK];
        for row in 0..RANK {
            let hidden_weights =
                &self.hidden_projection[row * HIDDEN_WIDTH..(row + 1) * HIDDEN_WIDTH];
            let context_weights =
                &self.context_projection[row * CONTEXT_WIDTH..(row + 1) * CONTEXT_WIDTH];
            let mut h = 0.0f32;
            for (&weight, &value) in hidden_weights.iter().zip(hidden) {
                h += weight * value;
            }
            let mut c = 0.0f32;
            for (&weight, &value) in context_weights.iter().zip(context) {
                c += weight * value;
            }
            hidden_rank[row] = h;
            context_rank[row] = c;
            product[row] = h * c;
        }

        let mut fused = [0.0f32; RANK];
        for row in 0..RANK {
            let weights = &self.fusion[row * FUSION_WIDTH + 2 * RANK..(row + 1) * FUSION_WIDTH];
            let mut value = self.fusion_bias[row];
            for (&weight, &input) in weights.iter().zip(&product) {
                value += weight * input;
            }
            fused[row] = gelu(value);
        }

        let mut result = [0.0f32; OUTPUT_WIDTH];
        for output in 0..OUTPUT_WIDTH {
            let weights = &self.output[output * RANK..(output + 1) * RANK];
            let mut correction = self.output_bias[output];
            for (&weight, &value) in weights.iter().zip(&fused) {
                correction += weight * value;
            }
            result[output] = anchor[output] * (1.0 + correction.tanh());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_readout() -> MultiplicativeReadout {
        MultiplicativeReadout::new(
            vec![0.0; RANK * HIDDEN_WIDTH],
            vec![0.0; RANK * CONTEXT_WIDTH],
            vec![0.0; RANK * FUSION_WIDTH],
            vec![0.0; RANK],
            vec![0.0; OUTPUT_WIDTH * RANK],
            vec![0.0; OUTPUT_WIDTH],
        )
        .unwrap()
    }

    #[test]
    fn zero_initialized_head_returns_parent_pair_exactly() {
        let head = zero_readout();
        let anchor = [0.25, -1.75];
        assert_eq!(
            head.apply(&anchor, &[1.0; HIDDEN_WIDTH], &[2.0; CONTEXT_WIDTH])
                .unwrap(),
            anchor
        );
    }

    #[test]
    fn constructor_and_forward_reject_wrong_contract_shapes() {
        let invalid = MultiplicativeReadout::new(
            vec![0.0; RANK * HIDDEN_WIDTH - 1],
            vec![0.0; RANK * CONTEXT_WIDTH],
            vec![0.0; RANK * FUSION_WIDTH],
            vec![0.0; RANK],
            vec![0.0; OUTPUT_WIDTH * RANK],
            vec![0.0; OUTPUT_WIDTH],
        );
        assert!(invalid.is_err());
        assert!(
            zero_readout()
                .apply(&[1.0], &[0.0; HIDDEN_WIDTH], &[0.0; CONTEXT_WIDTH])
                .is_err()
        );
    }

    #[test]
    fn nonzero_head_is_multiplicative_and_saturates_at_tanh_bounds() {
        let mut output = vec![0.0; OUTPUT_WIDTH * RANK];
        output[0] = 100.0;
        let mut fusion = vec![0.0; RANK * FUSION_WIDTH];
        fusion[2 * RANK] = 1.0;
        let mut hidden = vec![0.0; RANK * HIDDEN_WIDTH];
        hidden[0] = 1.0;
        let mut context = vec![0.0; RANK * CONTEXT_WIDTH];
        context[0] = 1.0;
        let head = MultiplicativeReadout::new(
            hidden,
            context,
            fusion,
            vec![0.0; RANK],
            output,
            vec![0.0; OUTPUT_WIDTH],
        )
        .unwrap();
        let result = head
            .apply(&[2.0, -3.0], &[1.0; HIDDEN_WIDTH], &[1.0; CONTEXT_WIDTH])
            .unwrap();
        assert!(result[0] > 3.999 && result[0] <= 4.0);
        assert_eq!(result[1], -3.0);
    }
}

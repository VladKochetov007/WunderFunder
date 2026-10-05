# Activation × optimizer screen: mechanism critique and bounded follow-ups

The completed v2 screen has 16 fits: eight pointwise choices at the input embedding and residual-head hidden layer, crossed with AdamW and MuonAdamW. It is a useful mechanism pilot, but it changes activation placement at both sites together, uses one training seed, one epoch (474 updates), and the same previously exposed 16-component diagnostic panel. Treat score patterns as hypotheses. They do not estimate seed uncertainty or fresh-component generalization.

## What the scores say

| Activation | AdamW WP | AdamW minus tanh | MuonAdamW WP | MuonAdamW minus tanh |
|---|---:|---:|---:|---:|
| tanh | 0.482259 | reference | 0.505498 | reference |
| GELU | 0.498239 | +0.015980 | 0.489122 | −0.016377 |
| ReLU | 0.486599 | +0.004340 | 0.481744 | −0.023755 |
| LeakyReLU(0.01) | 0.479897 | −0.002363 | 0.481479 | −0.024019 |
| GELU + 0.1 sin | 0.499963 | +0.017704 | 0.497938 | −0.007560 |
| GELU + 0.1 sin + 0.1(cos−1) | 0.498653 | +0.016393 | 0.497789 | −0.007710 |
| GELU × (1 + 0.5 sinc) | 0.494680 | +0.012421 | 0.491209 | −0.014290 |
| split GELU/LeakyReLU | 0.486659 | +0.004399 | 0.488666 | −0.016832 |

The apparent GELU-family advantage depends on the optimizer. The GELU-plus-sine variant is the top AdamW arm, but it is 0.007560 below tanh under MuonAdamW. Tanh itself is 0.023239 higher under MuonAdamW than AdamW, while the optimizer difference is small and negative for GELU-plus-sine (−0.002025). This is a large point-estimate activation-by-optimizer interaction in this single screen, not evidence that one optimizer or activation wins generally. The nominal learning rate and weight decay were equal, but that does not make AdamW and Muon’s parameter-wise update magnitudes comparable.

The target-specific scores also move differently. Tanh+AdamW has target WP `[0.498794, 0.465725]`, while tanh+Muon is nearly balanced at `[0.505566, 0.505430]`. Most Muon arms remain closer across targets than their AdamW counterparts. That could reflect optimization or target-specific fit behavior; the pooled metric and one fit do not identify the cause.

The initial results do not show that sinusoidal terms themselves caused the AdamW lift: plain GELU is close, the two sine variants are close to each other, and the sinc variant is somewhat lower. There are no paired component intervals in this result artifact, and a single initialization cannot establish a reliable ordering among these small differences. The channel-split arm is another parameter-free control, not a learned mixture: half the existing channels receive each function.

## Where a mechanism could enter

The embedding activation is applied after the 42-to-32 linear projection, before the GRU. Tanh bounds its input to `[-1,1]`; GELU keeps a roughly linear positive tail; ReLU discards negative values; LeakyReLU preserves a small negative tail; the sine residual adds bounded output and derivative oscillations; and the normalized sinc factor modulates GELU mostly near the origin. These choices change the amplitude, sign balance, and derivative distribution of the GRU input. Even though every model retains the same internal GRU sigmoid and tanh equations, its input and recurrent gate preactivations can therefore enter more or less saturated regions. Neither gate preactivations nor gate saturation, hidden-state norms, activation occupancy, or gradient norms were logged in this fit.

The head activation operates after the GRU state in the MLP correction branch. It cannot directly change the current forward pass’s recurrent gate values. It can change the correction features and gradients, which then alter shared GRU and head parameters during training and affect later states. The correction output layer is zero-initialized, so the correction starts at zero. On the first update, the correction output weights receive gradients based on the activated hidden features, while the correction hidden-layer weights receive zero gradient through that branch because the output weights are zero. The activation’s effect on hidden-layer learning is therefore delayed until the output layer has moved. The linear anchor and its recurrent gradient path are present from the start.

Because the screen used the same activation in the embedding and head, it cannot assign the observed score change to input-drive geometry, residual-head geometry, or their interaction. The unchanged GRU gates protect the gate formula, but not gate saturation induced by changed embedding amplitudes.

There is also a small formula-reproduction nuance. The code uses PyTorch `torch.nn.functional.gelu(x)` with its default exact-erf mode. The cited [paper appendix](https://arxiv.org/html/2602.05688v1) writes `jax.nn.gelu(x) + 0.1*sin(x)` and `jax.nn.gelu(x)*(1+0.5*sinc(x))`; as documented in [JAX's GELU API](https://docs.jax.dev/en/latest/_autosummary/jax.nn.gelu.html), its default GELU mode is approximate. The current screen therefore tests the stated function families, but it is not a bit-for-bit reproduction of those appendix expressions. A matched exact/approximate control should quantify whether this implementation detail matters.

## Eight bounded follow-ups

Each proposal changes one identifiable factor or measures an existing checkpoint. Keep the current cohort, objective, width, clipping, update count, zero-init anchor, and fixed fit-order recipe unless the proposal says otherwise. Do not select arms using this exposed test panel; use it only for the predeclared mechanism comparisons. These are study designs, not launch instructions.

1. **Measure activation and gate pathways on the saved checkpoints (no fitting).** On fit sequences, record embedding pre/postactivation RMS, sign/zero occupancy, derivative quantiles, GRU update/reset gate preactivation and sigmoid saturation fractions, candidate-tanh derivative, state norms, and per-family gradient/update RMS. Also report the correction-to-anchor output ratio over sequence position. Implement a diagnostic GRU equation path only after verifying its outputs against `nn.GRU`. This can distinguish changed drive amplitude from nonlinear shape and head scale; it cannot prove which pathway caused an exposed-panel score difference.

2. **Localize embedding versus head placement (eight fits).** Use tanh and GELU-plus-sine in a 2×2 embedding/head factorial, crossed with both existing optimizers: `(tanh,tanh)`, `(sine,sine)`, `(sine,tanh)`, `(tanh,sine)` × `{AdamW, MuonAdamW}`. This directly identifies which placement carries the current interaction while preserving matched widths, initialization, and update count. Keep both optimizer strata because the first screen reverses the activation ordering by optimizer.

3. **Match exact and approximate GELU formulas (six new fits).** For each optimizer, add tanh-approximate versions of plain GELU, GELU-plus-sine, and GELU-times-sinc, using the current exact-GELU fits as controls. Use the same formula at both placements and keep every other setting fixed. This answers whether the paper-default nuance explains any small difference; it is not a reason to relabel the present results as invalid.

4. **Test whether a small learnable activation mixture is useful (four fixed variants per optimizer).** In each placement use `tanh(x) + α·(GELUSine(x)−tanh(x))`. Compare fixed `α=0`, fixed `α=1`, one learned scalar shared by both placements, and separate learned embedding/head scalars, initialized at zero and unconstrained or explicitly bounded in the preregistration. Keep scalar parameters out of weight decay. This asks whether the model benefits from choosing a mixture, instead of forcing the same hand-picked function everywhere. Match total parameter counts in fixed controls by retaining the same scalar parameters but freezing them.

5. **Separate activation shape from embedding amplitude (four fits plus fixed calibration).** Estimate each activation’s output RMS from fit rows only, before training. Compare the unscaled activation and a fixed scalar gain that matches tanh’s RMS, for tanh and GELU-plus-sine. Apply the same gain to both placements only if the question is global amplitude; otherwise restrict it to the embedding to isolate gate drive. The gain is fixed, not batch-statistic based, so streaming outputs stay independent of future rows and unrelated sequences.

6. **Keep a KAN-inspired local head in the comparison.** With embedding activation and optimizer held fixed, compare the current hidden-width-13 MLP correction (457 parameters) with a seven-knot additive spline correction (448 parameters), then test a fixed additive hybrid: a four-knot spline (256 parameters) plus a hidden-width-5 MLP (177 parameters), for 433 correction parameters. Initialize both residual branches at zero. This keeps the total model sizes within about 0.3% while testing whether localized edge functions complement a smaller dense path. The activation screen contains no spline and cannot reject that direction. Describe the spline accurately as piecewise-linear and KAN-inspired, not as the original cubic KAN.

7. **Compare locked and row-wise dropout without adding confounds (six fits per chosen optimizer stratum).** On a fixed, predeclared tanh/GELU-plus-sine pair, compare no dropout, channel mask locked over each sequence, and independent row-wise masks at the same `p=0.1`, applied only to the embedded input before the GRU. Keep head dropout off. Locked masks preserve a feature channel’s identity over the stream; row-wise masks inject temporal noise and may broaden gate preactivations. Repeat by optimizer as a separate stratum if comparing optimizer interactions; do not select an optimizer from these same exposed test scores.

8. **Factor schedule from optimizer update scale (eight fits).** Cross AdamW and MuonAdamW with constant versus warmup-cosine LR and with nominal versus predeclared recurrent-update-RMS-matched scaling. Estimate the scaling from a fit-only calibration batch, before seeing test scores, and apply it to the GRU gate matrices only; report both actual update RMS and integrated LR. Match schedule area under the LR curve, keep nonrecurrent groups and weight decay fixed, and preserve 474 updates. This tests whether Muon’s large tanh result or the activation interaction is an effective-step-size artifact rather than a function-family effect.

The saved outputs support paired component resampling for descriptive intervals within this one seed, but they do not add training seeds. Any follow-up that appears promising should be repeated over predeclared seeds and, for a stronger generalization claim, on an independently frozen component panel before it is described as a reliable performance gain.

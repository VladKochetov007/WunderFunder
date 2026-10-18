# State geometry proposals: novelty attack R2

These are seven additional proposals after comparing the prior 25 cards, their R2 critique, and the frozen `state_geometry.md`. They are not empirical findings. No source search, data inspection, fitting, inference, or implementation was done for this round. Any serving state must use only current/past rows within a sequence, reset at sequence boundaries, and avoid `seq_ix`, labels, and `is_scored`.

## 1. Shared sparse measurement operator with a current-row inverse

**Mechanism.** For each documented feature family, model each instrument’s current vector as `x_i = D_g z_i + r_i`: `D_g` is a shared low-rank measurement dictionary, `z_i` a latent coordinate, and `r_i` a preserved residual. Obtain `z_i` by a fixed ridge or sparse inverse of `D_g` on the current row; the temporal model receives both `z_i` and `r_i`. Fit `D_g` with same-row reconstruction plus supervised loss, without future-row targets.

**Assumption / prediction.** Stable anonymous channels may mix a smaller basis shared across instruments. Latent coordinates should align better than raw slots while residuals retain exceptions. **Cheapest falsifier:** verify recovery on symbolic known mixtures; stop if the inverse is non-identifiable or unstable. A future matched-capacity fit tests alignment and endpoint quality. **Nearest prior:** shared instrument encoders and masked-current-feature reconstruction; this adds a generative measurement operator, current-row inverse, and residual escape. **Alternative:** a direct row encoder learns useful mixing. Cost: moderate; precompute the inverse map for serving.

## 2. Input-only conditional innovation structure

**Mechanism.** Build a directed graph over documented input families: do past residuals from A improve one-step prediction of B after conditioning on B’s own past and a shared component? Keep edges stable across sequence-disjoint training partitions. Use the graph as a fixed sparsity mask on cross-family update weights; add no lag bank or hand-authored market graph.

**Assumption / prediction.** Dependencies may be sparse after common variation is removed. Prediction: selected edges remain predictive under degree-matched shuffled masks. **Cheapest falsifier:** a future input-only screen compares edge stability across sequence halves with shuffled and common-factor controls; labels do not select edges. A later matched fit tests the mask. **Nearest prior:** code-transition topology and HLOB-style graphs; this estimates raw-family conditional predictability, not code cycles or semantic message passing. **Alternative:** dense recurrence handles dependencies better. Cost: moderate. These are predictive edges, not physical causes.

## 3. Jordan-split state for positive and negative innovations

**Mechanism.** From a compact, target-free input representation, compute a causal innovation against a simple running predictor, `e_t = z_t - zhat_t`. Split it into `e_t+ = max(e_t, 0)` and `e_t- = max(-e_t, 0)`, then maintain separate bounded accumulators with independently learned decay. Expose their difference and sum: net direction and total up/down variation remain distinct, so an up-then-down path does not collapse to a zero net move.

**Assumption / prediction.** Positive and negative input excursions may have different persistence or predictive roles even when net displacement is identical. Prediction: opposite-order or reversal paths with matched endpoint and net change produce distinct states. **Cheapest falsifier:** no-fit toy paths verify this separation and sequence-reset behavior; a future fit compares split accumulators with a single signed-innovation accumulator at equal state size. **Nearest prior:** event-paced write/decay and path-area ideas; every row still updates, with no event trigger, lag bank, or path signature. **Alternative:** a standard recurrent update already learns directional memory. Cost: low-to-moderate.

## 4. Target orientation and amplitude as separate predictions

**Mechanism.** For each target, predict an orientation score `q = 2*sigmoid(a)-1` and a nonnegative clipped amplitude `m = 2*sigmoid(b)`, then emit `p = q*m`. Train orientation against the sign of the clipped target with loss weight `|clip(y)|`; train amplitude against `|clip(y)|` on nonzero targets. The scoring weight therefore affects orientation errors on influential rows, while amplitude learns a separate conditional quantity. No target-derived rule enters inference.

**Assumption / prediction.** Direction and size may have different input structure, and a direct scalar head may entangle them. Prediction: the factorized head should change sign accuracy and magnitude calibration independently, especially across high- and low-weight targets. **Cheapest falsifier:** analytic toy distributions check that the head can recover asymmetric sign probabilities and bounded amplitudes; a future matched-parameter fit compares it with direct regression. **Nearest prior:** scalar OOF calibration and distributional distillation; this changes the supervised output parameterization, not a post-hoc map or teacher target. **Alternative:** the scalar head already optimizes the useful product. Cost: low. Speculative; the target’s exact semantics remain unknown.

## 5. Separate state prediction from current-row assimilation

**Mechanism.** Use a two-stage observer update. First evolve a prior state `h_t^- = F(h_{t-1})`; then encode the current input, compute its innovation `e_t = E(x_t) - H h_t^-`, and correct with `h_t = h_t^- + K(e_t)`. Read out from both corrected memory and `e_t`, so a useful new observation can affect the current prediction before its consequences are fully stored in the recurrent state. Apply the same stages on every row; no hold, event gate, or mid-sequence reset.

**Assumption / prediction.** The prediction movement seen in the earlier audit could reflect useful new input, delayed processing by state, or both. This decomposition makes those pathways explicit. **Cheapest falsifier:** a synthetic delayed-step stream should distinguish direct innovation readout from state-only readout; future controlled fit ablates each path separately. **Nearest prior:** soft state-write gates and CUSUM gates; this uses a prediction-error correction operator rather than activity-controlled update rate. **Alternative:** an ordinary GRU jointly learns assimilation and memory. Cost: moderate, with a small observation map.

## 6. Frozen warm-up context alongside within-sequence dynamics

**Mechanism.** Encode the 99 legal warm-up rows into a compact sequence context `c` using only those observed inputs. At the first required prediction, initialize dynamic state from `c` and retain a direct context path to the readout; subsequent rows update the dynamic state normally while `c` stays fixed. Reset both at the sequence boundary. Do not estimate target means, use sequence IDs, or adapt context with future rows.

**Assumption / prediction.** Some useful sequence-level environment information may be observable during warm-up and remain distinct from later within-sequence innovations. Prediction: context should contribute beyond a dynamic state trained to process the same warm-up, particularly in sequence-level target covariance. **Cheapest falsifier:** symbolic paired sequences with identical post-warm-up paths but different warm-up context verify that the context branch can preserve this distinction; a future matched fit compares ordinary warm-up processing with context initialization plus skip. **Nearest prior:** prefix normalization, which changes feature scales; this adds a target-free static state component and preserves the original dynamic path. **Alternative:** recurrently processing warm-up already stores enough. Cost: low.

## 7. Passivity-bounded hidden-state energy with an innovation ledger

**Mechanism.** Constrain the recurrent memory map’s symmetric part to be contractive and bound state-energy growth by observed input innovation: `||h_t||² <= rho*||h_{t-1}||² + c*||e_t||²`, with `0 < rho <= 1`. Track the nonnegative residual between this bound and realized energy as a scalar dissipation/instability channel to the readout. This imposes a balance on model state only; it does not assert conservation of market price, volume, or flow.

**Assumption / prediction.** Unforced hidden-state amplification may add unstable prediction motion, while legitimate changes remain explainable by current input innovations. Prediction: small input perturbations produce bounded state differences and any quality gain should coincide with lower unexplained energy growth. **Cheapest falsifier:** a no-fit bound check on symbolic recurrence matrices; future fit compares constrained and unconstrained models at equal state size and exposure. **Nearest prior:** generic architecture swaps and output smoothing; this constrains the state transition itself and still processes every row. **Alternative:** abrupt market information requires unconstrained state amplification. Cost: moderate; speculative and easy to overconstrain.


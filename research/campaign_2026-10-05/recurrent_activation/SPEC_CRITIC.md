# Independent critic: bounded recurrent-candidate activation

## Decision and scope

The proposed function is a defensible bounded curvature perturbation:

```text
u = tanh(input_candidate + reset * recurrent_candidate)
n = u + alpha * sin(pi*u)
h_next = update * h_prev + (1-update) * n
```

For `alpha ∈ {-0.1, 0, +0.1}`, `dn/du = 1 + alpha*pi*cos(pi*u)` is strictly positive, and `n(±1)=±1`; with reset/update gates in `[0,1]` and an initial state in `[-1,1]`, the hidden-state interval is invariant. This supports a bounded candidate nonlinearity experiment. It does not support a market-periodicity claim or a general gradient-stability claim.

The fixed six-arm `alpha × optimizer` matrix is an acceptable first mechanism screen if alpha-zero backend parity and identical initial conditions are established first. The AdamW/Muon comparison at equal nominal LR is only an interaction screen: equal LR, betas, decay, and step count do not mean equal parameter-wise update scale. Do not describe it as an optimizer ranking. The activation-within-optimizer contrasts are the cleaner primary contrasts.

## Required equations and PyTorch parity

PyTorch GRU gate chunks are ordered reset, update, new (`r,z,n`). For each timestep and layer, match these equations and bias placement exactly:

```text
r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
a = W_in x + b_in
q = W_hn h + b_hn
u = tanh(a + r * q)
n = u + alpha * sin(pi*u)
h' = z * h + (1-z) * n
```

In the displayed definition, input and recurrent candidate components must include their separate biases; specifically, `reset` multiplies the recurrent affine result including `b_hn`, after the recurrent matrix multiply. Applying reset to `h` before `W_hn`, applying reset to the combined candidate, dropping either bias, or swapping `z` with `1-z` breaks alpha-zero PyTorch parity. The new activation replaces only the final candidate `tanh`; reset/update sigmoids and all gate equations remain unchanged. Respect PyTorch's per-layer input widths, directionality (unidirectional only unless specified), stacked `r,z,n` row slicing, and gate biases.

Parity fixtures should compare alpha-zero output and final hidden state against `torch.nn.GRU` for one and multiple layers, batch size greater than one, uneven lengths, nonzero initial states, and both `h=None` and explicit zero `h`. Compare full sequence to chunked evaluation with state carry and to row-at-a-time stepping. Add forward-gradient parity against the reference for alpha zero, and finite/nonzero gradients for alpha ±0.1. Use fixed weights copied parameter-for-parameter from a PyTorch GRU; comparing two separately randomized backends is not a parity check.

## Initialization, carry, and optimization contract

- Initialize or load one baseline parameter set once, then clone it exactly to all six arms. A candidate-alpha-zero model must match the source baseline before any update. Do not independently draw six GRU initializations or recreate initialization using a different RNG call order.
- Initial sequence state is zero, as in `nn.GRU(..., h=None)`. Carry hidden state across TBPTT chunks of the *same* sequence, detach at the same chunk boundary as the baseline trainer, and reset to zero at every new sequence. No state may cross batch slots or distinct sequences. Preserve source sampling order, row subsampling, padding/masking, and stateful chunk boundaries.
- All arms receive the same minibatch sequence and exactly the same number of optimizer updates. Log actual update count, scheduler steps, skipped/nonfinite batches, and any gradient clipping. A nominal epoch is not a matched budget if an arm skips updates or receives a different data order.
- Keep architecture, all non-GRU activations, output heads, objective, target clipping/weights, feature normalization, dropout, and residual/auxiliary losses identical. The only scientific factor within each optimizer stratum is fixed alpha.
- The custom GRU must expose gate matrices using audited names and shapes. If using `muon_param_groups`, the recurrent three-gate matrices must be the only Muon tensors, with three row blocks per GRU matrix; input projection, heads, biases, and any scalar parameters stay in AdamW. Assert a disjoint exact partition of all trainable parameters, assert every intended tensor is assigned once, and verify no GRU weight/bias is accidentally omitted or assigned to both optimizers. Custom parameters must not be hidden outside `model.gru.*` if the existing group selector relies on that prefix.
- Save seed, initialization/checkpoint provenance, alpha, optimizer group names, per-group LR/weight decay, scheduler, batch order identifier, effective step count, and pre/post-clipping grad/update RMS. No test-panel metric may affect checkpoint selection.

For alpha zero, compare one forward/backward/optimizer step to the existing PyTorch implementation with the same loss and optimizer. AdamW should be exact within deterministic floating-point tolerance. Muon should receive identical gradient tensors, selected group membership, and one-step updates; this also tests the custom name adapter. Then test all six fixed arms. Parameter counts should be identical across alpha within each architecture; optimizer state counts should also match.

## Adversarial CPU checks before the independent code review

Run only under the campaign's CPU limiter. The smallest useful fixtures are deterministic synthetic tensors, not parquet scans:

1. **Reset-after-affine sentinel:** choose nonzero `b_hn`, nontrivial `r`, and hand-set `W_hn`; demonstrate custom alpha-zero output equals `nn.GRU`. A faulty `W_hn(r*h)` or forgotten recurrent candidate bias must fail.
2. **Gate-order sentinel:** distinct constant gate matrices/biases for r/z/n; compare output and hidden state to PyTorch. This catches swapped chunk order and update interpolation.
3. **Chunk/state tests:** sequence split at irregular boundaries, explicit incoming nonzero state, exact full-vs-chunk output/state parity at alpha zero; independent sequence resets and batch permutation tests ensure no state leak.
4. **Gradient test:** compare alpha-zero input, recurrent matrix, and both candidate-bias gradients with PyTorch. At ±0.1, assert finite gradients and show candidate function and derivative stay in stated ranges on a dense grid over `[-1,1]`.
5. **Optimizer partition and update:** enumerate groups by parameter name, prove disjoint/exhaustive membership and exact expected Muon matrices, then compare one-step baseline/custom updates. Test wrong names and duplicate group assignment fail closed.
6. **Equal-budget/control test:** instrument the fixed-fit loop to assert equal sampled batches, updates, alpha=0 control, and schedule calls across all arms. A deliberately skipped step should fail the receipt.

If the candidate is initialized from a trained checkpoint, include the checkpoint hash, exact state-dict key mapping, optimizer-state policy (fresh optimizers or restored states), and a pre-update prediction parity receipt. A trained PyTorch GRU has no compatible parameterization guarantee with a custom candidate-cell implementation until that receipt passes.

## Interpretation of E470 and stronger-data follow-up

E470 is useful for mechanism generation, not this recurrent hypothesis: its sixteen 8,235-parameter fits use one seed/474 updates and the same historically exposed 16-component diagnostic test set (out of 40 selected sequences). The observed tanh optimizer reversal and the weak overall small-model metrics make optimizer-by-activation, effective step size, and capacity plausible alternatives to activation shape. E470 changes embedding and readout activations and does not test the GRU candidate equation. Its displayed differences should not be used to pick alpha or claim a better recurrent model.

The discriminating next comparison should use the already-defined strong two-output causal GRU continuation architecture and a larger component-respecting evaluation, with the original regression anchor/control held constant. Start every alpha arm by cloning the same full strong-model checkpoint; load its standard PyTorch GRU tensors directly into the alpha-zero-compatible custom cell, verify pre-update full-sequence predictions match, and use fresh optimizer state for every arm. This tests whether the altered recurrence improves continuation from an existing strong solution instead of confounding the question with random initialization or a weak 8K-parameter toy model. Also fit a matched alpha-zero continuation from the same checkpoint. Freeze a deployment-relevant train recipe, feature/target normalization, labels/masks, checkpoint epoch, initialization policy, and scorer before fitting. Prefer several predeclared chain-disjoint component folds with out-of-fold predictions and component-level paired bootstrap; if only the 40-sequence E470 cohort is available locally, rotate all eligible components through validation and call this a historically exposed robustness screen. A larger count of correlated rows from the same 40 sequences is not stronger independent evidence. Recompute all fold-specific transforms/propensities from training rows only and exclude training row groups connected to evaluation components under the predeclared feature-only graph; do not use cached validation labels to select initialization, checkpoint, or normalization.

Use a staged but balanced factorial: `alpha ∈ {0,-0.1,+0.1}` × `placement ∈ {candidate only, embedding only, candidate+embedding}` × `optimizer ∈ {AdamW, MuonAdamW}`, with at least three paired seeds on identical fold/data orders for any follow-up that survives the six-arm candidate-only screen. Include a plain `alpha=0` baseline for every placement/optimizer cell. The embedding perturbation must use the same bounded map on the 32-wide pre-GRU embedding and be labeled a distinct factor; do not silently reinterpret the first six candidate-only arms. For optimizer effects, compare both the normal schedule and a predeclared update-RMS-matched schedule calibrated on fit batches only. Freeze the calibration before evaluating held-out components, keep schedule area and updates matched, and publish both nominal LR and measured groupwise update RMS. This separates candidate-shape response from the optimizer's effective step scale and from embedding placement without searching the evaluation panel.

Report all paired candidate-minus-alpha-zero deltas within each optimizer/placement/fold/seed, the activation-by-optimizer and activation-by-placement interactions, per-target metrics, chain/component bootstrap intervals, and the unpooled fold/seed outcomes. Do not choose alpha, placement, optimizer, checkpoint, or schedule from the exposed 16-component panel. A result on the same panel, even with component bootstrap, remains a conditional one-seed or small-cohort diagnostic; confirmation needs a frozen independent component panel or repeated leakage-safe folds/seeds. No public-score expectation follows from this screen.

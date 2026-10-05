# Blind E481 design critique

## Decision

The fixed 18-fit comparison is a useful, falsifiable candidate-shape study. The most important interpretation constraint is already recognized in the design: sine and cubic share central slope and endpoints, but their endpoint derivatives differ. Results can support a local shape comparison at this scale; they cannot isolate “periodicity” or prove a market process. Keep the baseline, sine, and cubic arms and report all cells.

## Mathematical checks

For `u=tanh(v) in [-1,1]`, the sine map is `f_s(u)=u+0.1 sin(pi u)`. Its derivative is `1+0.1 pi cos(pi u)`, with minimum `1-0.1 pi > 0`; because the map is increasing and fixes `-1` and `+1`, its output stays in `[-1,1]`. The cubic map `f_c(u)=u+0.1 pi u(1-u^2)` has derivative `1+0.1 pi(1-3u^2)`, whose minimum is `1-0.2 pi > 0`, and it fixes the same endpoints. Both maps have derivative `1+0.1 pi` at zero. The cubic's endpoint derivative is `1-0.2 pi`, while the sine's is `1-0.1 pi`; therefore the experiment changes saturation-near-endpoint slope as well as off-center higher-order shape. It should be described as a matched-center/endpoints shape comparison, with “curvature” as shorthand.

The candidate correction is inside the tanh output domain. It is not an unbounded periodic embedding, and the scalar monotonicity proof does not establish stability of either the GRU recurrence or its gradients. Tests should evaluate both formulas and their derivatives over the entire closed interval, include exact endpoints and center, verify the stated derivative minima, and use finite-difference/autograd checks away from extrema. Check finite behavior for representative finite preactivations near both tanh saturation tails.

## Implementation invariants and likely failure cases

- Compute the same `u=tanh(v)` in all three arms; apply baseline `u`, sine `u+0.1*sin(pi*u)`, or cubic `u+0.1*pi*u*(1-u*u)` only to the candidate state. Do not alter reset/update gates, input and hidden bias placement, gate order, state initialization, or layer/head wiring.
- Preserve the E478 reset-after GRU equation, including `tanh(x_n + r * h_n)` with the hidden candidate bias inside `h_n`. A reset-before implementation can look plausible while changing the method.
- Match the source baseline checkpoint exactly for both seeds: every parameter and identity normalization buffer, 61,058 trainable parameters, same seed-specific shuffled fit order, and same externally fit-normalized features. The 18 fits must be ordered and represented as the full 3-shape × 2-optimizer × 3-seed matrix.
- Reset state at each B4 fit batch and each evaluation component; carry the tensor state through all 79 contiguous 256-row chunks and detach only at the registered truncated-BPTT boundary. Test chunk concatenation against a full sequence, future isolation, batch/component reset, and both layers' final-state placement.
- Muon must select only `gru.weight_ih_l{0,1}` and `gru.weight_hh_l{0,1}`, each with three gate blocks. Every other parameter stays in the inherited AdamW group. Check disjoint exact parameter coverage, inherited LR/WD/betas, one global norm clip over the whole model, and finite parameters after the step.
- The common IEEE FP32 policy must wrap parity and update comparisons for both cuDNN and scripted recurrence paths. The B4×256 CUDA probe needs finite outputs, state, loss, gradients and changed parameters for all 18 cells, plus teardown/resource receipts. Preserve original tolerances and record failures before any retry.
- Verify gradients and clipped-update parity for both AdamW and Muon at the baseline shape. A passing AdamW parity case does not validate Muon's orthogonalization/update branch. For the sine and cubic maps, confirm nonzero candidate-path input/recurrent gradients and exact formula parity in the CPU suite.
- Fit-only loading should see precisely the 24 fit groups. Recompute feature normalization from those rows, keep internal model buffers at identity, and prove no test rows are read before fit completion. Evaluation clipping must follow a raw finite check; use exactly the common official score mask and preserve the complete selected raw target/prediction export.

## Matching and result interpretation

The registered pairwise effects are correctly oriented as sine-minus-baseline, cubic-minus-baseline, and cubic-minus-sine, separately for each seed and optimizer. The nine Muon-minus-Adam contrasts compare optimizers within each shape/seed. The three optimizer-by-shape interactions per seed should be computed as paired differences of the corresponding shape contrasts with a single shared component-multiplicity vector; retain all three even though one is algebraically dependent on the other two. They are not three independent confirmations.

Use a single numeric-sorted component order and the same 2,000 draw matrix within a seed across every pair and interaction. Recompute global WP from pooled sufficient statistics in each draw; do not average component-level WPs. Keep t0/t1 score reporting aligned to the same fixed mask. Primary interval scope is conditional on these already exposed 16 components and the fitted weights. Do not pool seeds, infer a public score, or select the best arm from this screen.

The design would be easier to interpret with a small train-only diagnostic of the candidate preactivation distribution (for example, `u` quantiles and the share near `|u|>0.9` at fixed telemetry steps). This is optional descriptive context, not an adaptive signal or a new fit cell. Without it, a sine/cubic score difference cannot be linked to where their maps differ. Their endpoint slopes are already unequal, so any “curvature-only” explanation should be avoided.

## Falsifiers

The implementation is invalid if source baseline parity fails; the output leaves `[-1,1]`; either derivative becomes nonpositive; gradients vanish/nonfinite through either map; optimizer groups include any non-GRU matrix in Muon; initialization/order differs within a seed; a fit or scorer reads test labels early; or an actual cell fails finite-update/resource checks. Preserve those failures.

The scientific hypothesis that a sine-shaped candidate helps beyond a cheap matched-center alternative weakens if cubic ties or beats sine consistently, the sine/cubic difference changes sign across seeds or optimizers, or both differ from baseline only through unstable single-cell results. A gain for either nonlinear map alone does not establish a periodic inductive bias. A cubic win can motivate a separate inference-cost measurement, but not deployment without native/PyTorch parity and runtime checks.

## Dated correction — 2026-10-05 03:04 UTC

The original critic text at SHA-256 `3d51fa884a9c0613017250ed06199d22a7fa5206c397a067fba25799e65b077b` misread the registered interaction count as three per seed. This correction supersedes the earlier sentence calling for three interactions per seed. The design instead specifies **three total**, one per seed: `(WP_cubic,Muon - WP_sine,Muon) - (WP_cubic,AdamW - WP_sine,AdamW)`. The complete count remains 30: 18 pairwise shape effects (three shapes, two optimizers, three seeds), nine Muon-minus-AdamW effects (three shapes, three seeds), and three joint difference-in-differences (one per seed). Keep all baseline, sine, and cubic arms. Do not add the other two dependent optimizer-by-shape contrasts; this correction changes only the critique's interpretation, not the registered design.

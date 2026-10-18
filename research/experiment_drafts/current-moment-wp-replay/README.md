# Current-moment weighted-Pearson replay

Proposal-only continuation of **H-055**, with experiment ID **E-CURRENT-MOMENT-WP-20261018**, pinned to source `1d57ed815c18eb278bcdd0207369b200970b28fb`. It tests a current, fixed-superbatch objective with exact output-gradient replay; it does not revive H-055's decayed moment memory or erase E-291's negative result.

No adapter, clean model/input/LR registry binding, device profile, fit, or grade is included. `training_authorized=false` and `launch_ready=false`. Reuse the canonical GRU factory; do not infer a new architecture or trainer from this proposal.

## Packet files

- [PR narrative](PR.md) answers the repository's eleven-section experiment template.
- [Machine recipe](recipe.json) binds the two-arm proposal and lists remaining work. Its plan-only command is documented but was not run.

## Mechanism and prior evidence

The treatment computes clipped weighted-Pearson moments over all eligible rows in the current 64-sequence superbatch, then replays the same FP32, K=500 truncated chunks to accumulate the chain-rule gradient before one optimizer update. Its moments are current to one fixed parameter state: no EMA/decay, no per-chunk correlation objective, no update between passes. The gradient is exact only for that superbatch's specified truncated graph; it is not a full-training-set or full-BPTT gradient.

The [H-055 metric-objective audit](../../unified/ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md) preserves E-291's rejected pooled-memory comparison and H-018/E-023's per-chunk constant-target failure. The [shared gradient-estimator continuation note](../../unified/ideas/GRADIENT_ESTIMATOR_CONTINUATIONS_20261018.md) freezes the CP30 family, effect/cost gates, and analytical derivative. The [literature note](../../unified/sources/GRADIENT_ESTIMATOR_LITERATURE_20261018.md) records the source review and transfer limits.

## Read sources

- [Gao et al., 2021, Gradient Cache paper](https://arxiv.org/html/2101.06983v2), sections 3.2–3.3: graphless forward, loss gradients with respect to cached outputs, then replayed chunk backpropagation. It studies contrastive retrieval, not recurrent forecasts.
- [Author implementation](https://github.com/luyug/GradCache/blob/main/src/grad_cache/grad_cache.py), `forward_no_grad`, `build_cache`, `forward_backward`, and RNG-state replay. This supports the chain-rule construction only; GRU hidden-state/mask/reset parity remains unimplemented here.
- [Wang, Fang & Liu, 2014, SCGD](https://arxiv.org/abs/1411.3803), abstract only: context for nonlinear functions of expected moments, not a guarantee for this finite-batch nonconvex GRU objective and not a reason to reuse EMA moments.

The only next step proposed is a no-fit synthetic gradient/replay audit. Any later profile or CP30 fit requires the missing bindings and separate training authority. No forecast gain, public-score change, runtime result, or financial causal effect is claimed.

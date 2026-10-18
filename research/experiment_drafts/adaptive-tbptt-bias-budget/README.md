# Adaptive TBPTT with an empirical bias budget

**Status:** proposal only. No controller, trainer adapter, B300 profile, run, or score is present. This packet is a continuation of H-RR17-19, not a new ARTBP claim.

- Hypothesis: H-RR17-19; variant: adaptive_bias_budget.
- Experiment: E-ADAPTIVE-TBPTT-20261018.
- Frozen research source: 1d57ed815c18eb278bcdd0207369b200970b28fb.
- launch_ready=false; training_authorized=false.

The candidate estimates recurrent loss-to-state gradient tails from training-only probes and selects a detach horizon from 500, 1000, or 2000 rows. The control fixes the horizon at 2000. Both use 500-row microchunks and update after every 2000 input rows. This separates graph horizon from optimizer cadence. The empirical estimate is a controller signal, not a certified or exact bound on true gradient error.

The proposed no-fit gate compares the empirical proxy with exact full-versus-truncated gradients on controlled synthetic recurrences, including geometric tails and deliberately delayed, sparse, or nonmonotone tails. It flags every observed underestimation; passing such a finite gate would not prove a universal guarantee.

## Prior work and local overlap

Aicher, Foti, and Fox, “Adaptively Truncating Backpropagation Through Time to Control Gradient Bias,” [PMLR paper](https://proceedings.mlr.press/v115/aicher20a.html) is the primary method source. It defines lagged hidden-state gradient norms, assumes geometric decay in expectation after a lag, estimates a relative-bias proxy, and selects the shortest horizon meeting a tolerance. The theorem depends on its assumptions; its empirical proxy is not a universally certified bound. The authors report synthetic and language-model experiments, not this task or trainer. Evidence level: primary paper with conditional theory and external empirical results.

The authors’ [reference implementation](https://github.com/aicherc/adaptive_tbptt/blob/master/tbptt/adaptive_truncation.py) is first-party code evidence for estimating gradient norms, decay rate, tail error, and the selected K. It is a method reference, not a drop-in adapter for this repository.

The [ARTBP paper by Tallec and Ollivier](https://arxiv.org/abs/1705.08209) is the distinct randomized compensated estimator already represented by H-RR17-19. This packet does not add ARTBP compensation or call ordinary adaptive detachment unbiased.

Local overlap and evidence:

- [H-RR17-19 card](../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md#h-rr17-19) and its [existing proposal packet](../../campaign_2026-10-17/research_reset_r1/drafts/rr17-19/README.md) cover ARTBP, not adaptive biased truncation.
- [Unified ideas index](../../unified/ideas/README.md) records earlier fixed 500-versus-2000 detach preparation and its profile-only state. Existing fixed-horizon screens were small or inconclusive; they do not establish a quality result for this proposal.
- The prior [finite-graph ARTBP oracle report](../../campaign_2026-10-03/continuation/round6/artbp_credit/REPORT.md) is a synthetic compensated-gradient check, not a trained adaptive-TBPTT result.
- Shared research notes: [gradient-estimator continuations](../../unified/ideas/GRADIENT_ESTIMATOR_CONTINUATIONS_20261018.md) and [gradient-estimator literature](../../unified/sources/GRADIENT_ESTIMATOR_LITERATURE_20261018.md).

## Packet contents

- [PR.md](PR.md) answers the repository’s 11-section experiment template and states the blocked adapters and evidence gates.
- [recipe.json](recipe.json) freezes the two-arm plan and plan-printer fields. It is not a runnable trainer recipe.

No model, training command, data/cache path, ancestry, B300 topology/profile, or output directory is bound yet. The plan printer may be run later from a complete checkout; it only prints a matrix and is not executed as part of this proposal.

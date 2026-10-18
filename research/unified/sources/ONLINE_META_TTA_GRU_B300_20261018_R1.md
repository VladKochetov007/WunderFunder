# Literature basis: meta-trained input-only online adaptation

**Date checked:** 18 October 2026
**Related hypothesis:** `H-ONLINE-META-TTA-GRU-B300-20261018-R1`
**Status:** proposal-level source synthesis; no local task fit or score.

## Primary sources and transfer limits

1. Sun et al., [*Test-Time Training with Self-Supervision for Generalization
   under Distribution Shifts*](https://arxiv.org/abs/1909.13231), arXiv:1909.13231.
   It motivates online self-supervised updates and presents a positive
   gradient-alignment condition in a convex analysis. Its empirical tasks are
   image corruptions; neither its assumptions nor its gains establish value
   for these noisy sequences.
2. Kim et al., [*Battling the Non-stationarity in Time Series Forecasting via
   Test-time Adaptation*](https://arxiv.org/abs/2501.04970), arXiv:2501.04970.
   TAFAS includes feedback from partially observed future ground truth. The
   competition callback provides no targets or mask, so that feedback
   mechanism cannot be copied. Its causal scheduling and stabilization
   questions are useful design prompts only.
3. Wu et al., [*Test-Time Adaptation for Non-stationary Time Series: From
   Synthetic Regime Shifts to Financial Markets*](https://arxiv.org/abs/2602.00073),
   arXiv:2602.00073. Results vary by regime and financial task; some adaptation
   policies help while others hurt. This supports testing mechanisms and
   regimes separately, not expecting a positive score.
4. Tandon et al., [*End-to-End Test-Time Training for Long Context*](https://arxiv.org/abs/2512.23675),
   arXiv:2512.23675. It meta-learns initializations for continuing
   self-supervised language-model updates. Its objective, data and scale differ
   substantially; the transferable question is whether an outer task objective
   can shape useful online updates.

## Local mechanism derived from the sources

Use observed next-input prediction for the online inner update because
competition labels and scoring masks are unavailable at inference. Use the
next target-block loss only as a fitting-only outer meta-objective. This tests
the gradient-alignment premise instead of assuming that input reconstruction
is a useful target proxy. The fast adapter is low-rank and reset per sequence;
its exact 99-row warm-up and 256-row schedule is frozen in the
[experiment packet](../../experiment_drafts/online_meta_tta_gru_b300/README.md).

The nearest project comparison is the
[static predictive-state GRU handoff, PR #18](https://github.com/VladKochetov007/WunderFunder/pull/18).
That PR's auxiliary loss does not perform test-time gradient updates; this
proposal measures the update/outer-meta-learning interaction. A separate
[Kalman-feature handoff, PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17)
already studies causal state estimation as an input feature. It should not be
duplicated or treated as evidence for TTA.

No paper result above is local experimental evidence. The linked packet is not
launch-ready; actual B300 profiling, complete trainer integration, clean
fold/teacher ancestry and student transfer evaluation remain open.

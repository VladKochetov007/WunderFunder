---
kind: idea
status: proposed
created_utc: 2026-10-17T23:42:00Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - ../geometry_variations_r3/geometry_r3_grade.json
  - ../geometry_variations_r3/RELEASE.json
  - ../geometry_adapter_probe_r2/PUBLIC_RESULT.json
  - DATA_AUDIT.md
---

# What to take from the previous competition discussion

## Question or claim

The user supplied a December 2025–March 2026 participant discussion and requests an independent reassessment, especially online learning, feature extraction and model diversity. These are participant reports, not controlled experiments. The prior task reportedly used 517 sequences and 32 raw features with R²-like scores; our current task has 112 inputs, 12,480 supplied sequences and global weighted Pearson. Numerical gains are not transferable estimates.

## What tonight actually established

The original short continuation and frozen feature adapter both regressed against their common component-excluded parent. Separating moments and dictionary inputs did not recover that parent score. Reversing and halving the existing learned feature correction ranked best on the exposed F0: FP32 0.68142211 versus parent 0.68122651; conditional component 95% interval for the difference[−0.00002480, +0.00042143]. The actual native q16 endpoint scored 0.68142340. This is adaptive one-fit evidence; the sign reversal does not establish the reason training went wrong. The inherited auxiliary objectives and their influence on the new input weights are a concrete hypothesis to test, alongside sampling/noise and feature transfer mismatch.

The first R2 submission scored 0.66428 public, below the reported team best TL3 0.66832. We started from a weaker local component to isolate the representation, then used the deadline quota exception; that is not a test that the representation improves TL3. The first operational improvement is to measure contributions to a complete strong endpoint, while retaining clean fitting ancestry for scientific comparisons. TL3's actual ancestry is still unknown; its embedded all-data description makes clean-CV claims inappropriate until verified.

## Transferable mechanisms and cheapest meaningful comparisons

1. **Online learning from observable inputs.** At test time the true targets and scoring mask never arrive. The admissible first design is a deterministic within-sequence state update that learns input dynamics or relationships, then exposes that state or its surprises to a supervised predictor trained on supplied labeled fitting groups. Examples to investigate are regularized recursive linear predictors, fast/slow moments, cross-instrument lagged relationships and learned recurrent feature encoders. Reset every adaptive state between sequences; retain raw inputs. Compare to frozen-state and ordinary statistic controls, and report fixed early/middle/late bins alongside the official pooled metric. A better input reconstruction error alone is not a target-quality gain.

2. **Learned feature extraction.** A GRU trained to predict future observable inputs can provide a dynamics representation to a stronger predictor, provided its pretraining excludes the evaluation components for held-out claims. Contrast frozen learned features with equal-width raw/handcrafted controls and a jointly trained encoder at the same compute. Pretraining on all data followed by evaluation on part of that same data is fitting exposure, even without target labels. Final all-data training is a separate permitted stage.

3. **SWA.** Weight averaging of late checkpoints along one compatible training trajectory adds no recurrent serving work. The trainer already exposes `--swa-frac`; inspect its actual update schedule and parent recipe before treating it as new. Compare fixed checkpoint selection with a predeclared late-trajectory average under matched seeds. Do not average unrelated seed models or incompatible architectures and call it ordinary SWA; parameter permutations and different normalization buffers can invalidate that operation. Averaging the frozen parent with its new feature adapter only shrinks the added columns; tonight's positive-alpha grid already tests that particular interpolation and did not beat the parent.

4. **Residual boosting and highway readout.** A small second stage may predict residual structure the frozen recurrent representation misses; a gated skip can preserve direct current-state information without widening recurrent state. Measure the combined endpoint, not the residual model's standalone score. Keep losses, target clipping, fitting membership and teacher ancestry explicit. Crossfit residual targets where practical and compare to an equally sized direct head before attributing gains to boosting.

5. **Diversity under a fixed runtime.** The weak-model anecdote is a useful reminder to measure covariance and conditional contribution. TL3 plus the new model must be compared to fixed TL3, using actual deterministic caches and measured merged serving costs. A full extra recurrent model is not the only route: checkpoint averaging, a distilled diverse teacher target, a small residual head, or a small fraction of a cheaper complementary member may preserve the useful difference. The public score alone cannot choose a mixture coefficient. Unknown/all-data-fitted local diagnostics must remain labeled as such.

6. **Memory timescales and causal windows.** Chrono initialization, short-window bidirectional teachers and attention over recurrent state are plausible ways to preserve or selectively retrieve information. A bidirectional encoder over a window ending at the current step can be causal; an encoder seeing later steps cannot. Its CPU cost may make it a teacher rather than a submitted model. Chrono initialization is a fresh-training intervention, not a justification to alter trained recurrent biases. Claims that one architecture inherently cannot forget need direct state/gradient evidence and matched training before acceptance.

7. **Feature banks without destroying useful levels.** The w3/w13/w21 volatility, lag-difference, rank, trend and regime statistics are concrete baselines; selective groups and fast/slow disagreements may suffice. Keep the strong raw representation and append features, then use controls to identify useful information. A raw correlation cutoff ≥0.9 is not a universal feature-removal rule: nearly identical book levels can encode a small but critical difference, and nonlinear/time-dependent uses can differ. Standardization or variance augmentation must preserve training/serving semantics and be validated on clean fitting membership. Whole-sequence test normalization uses future information and is invalid.

## Decision and next step

ITERATE. First repair or falsify the narrow auxiliary-objective explanation with an explicit target-only frozen adapter. In parallel obtain deterministic TL3 predictions and honest complementarity diagnostics. A fresh source-grounded proposal round will select a causal online-state representation for matched training. The data audit is complete; new code/profile/fit admission remains separate. Larger interesting directions get runnable B300 draft PRs with 30 paired seed/fold cells per arm, not a local promise of the previous competition's gains.

## Source records

The original discussion was supplied by the user in this session; no message URL or attached figures were provided. Reports include rolling features, highway heads, two-stage boosting, SWA, variance augmentation, chrono initialization, causal-window bidirectional models, adaptive weights, masked pretraining, hybrid attention/recurrent encoders, and learned GRU feature extraction. The author descriptions of score transfer, synthetic-data origin, optimizer behavior and architecture limitations remain external hypotheses. Current measured evidence is linked in the frontmatter and [campaign navigation](README.md); primary literature verification is a separate source register.

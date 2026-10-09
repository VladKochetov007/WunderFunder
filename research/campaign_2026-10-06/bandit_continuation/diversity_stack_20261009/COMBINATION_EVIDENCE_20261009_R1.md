# Combining models: evidence and next experiments

The results justify continued combination research, but diversity by itself has not guaranteed a gain. Keep three comparisons separate: they use different training budgets and evaluation panels.

| Comparison | Observed change | What it supports |
|---|---:|---|
| Four earlier frozen endpoints, equal blend, exposed fold0 | +0.0002533 versus the strongest member; interval [-0.0003159, +0.0008066] | A bounded follow-up on the complete blend; no public forecast |
| GRU384 seed3/seed4, equal blend, exposed fold1 | +0.0015120 versus seed3; interval [-0.0003661, +0.0032792] | Seed averaging helped this fitted pair; uncertainty still includes zero |
| AdamW-WMSE/Muon-WMSE, equal blend, 350-sequence exposed fold0 screen | -0.0018598 versus AdamW-WMSE; interval [-0.0026176, -0.0011248] | This exact small-budget, matched-schedule equal blend loses |

The intervals are paired whole-component resampling intervals conditional on the fitted predictions. They do not measure variation from retraining or public transfer. No local delta is added to a public score. The fold1 pair is a baseline diagnostic, not yet a test of heterogeneous teacher distillation.

The optimizer/loss quartet now has four independent fits, all six fixed pair blends and the fixed four-way blend evaluated. Its AdamW-WMSE control is strongest among the four single models; none of the declared equal blends beats it overall. The loss-only AdamW pair previously improved target1 while losing target0. These observations motivate target-specific and calibrated combination policies as separate hypotheses, rather than silently replacing the registered equal-weight endpoint.

The immediate finite queue is R/Q (Huber delta4/delta2), then I/U (fixed activation mixture with WMSE/Huber), then J/K (WSD/SGDR). Preserve the same training membership, fixed checkpoint policy and output identities. Record actual learning-rate, momentum, gradient norms, clipping and update counts. Reuse the parsed cache while distinguishing per-process loading from operating-system cache reuse.

Later optimizer work should distinguish a faithful matched-schedule comparison from an optimizer-specific tuning arm. A different Muon learning-rate multiplier, momentum policy or clipping rule changes the recipe and deserves its own parent link and budget. The current small screen is insufficient to reject those descendants or larger independently trained teachers.

For stacking or boosting, fit combination parameters only within declared training components and evaluate on disjoint components. Start with simple fixed or component-cross-fitted policies before another neural meta-model. A residual learner must use out-of-fold teacher predictions for fitting; ordinary in-sample residuals would confound the comparison. Previously inspected folds remain exposed after repartitioning.

Retain both weights and aligned predictions: weights support serving, gradients and later exports; predictions make blend and correlation analysis cheap. Current output paths preserve optimizer/loss, run identity, source/recipe pins, checkpoint epoch and evaluation membership. Training traces and resumable states are retained separately. SSD space is currently sufficient after the verified archival move.

Evidence: [full quartet and uncertainty](grading_r2/D02_QUARTET_FINDINGS_R2.md), [root independent primary calculation](grading_r2/ROOT_QUARTET_PRIMARY_REVIEW_R1.json), [fold1 pair](../../../../scratch/fold1_gru384_successor_20261009/grading_r1/BASELINE_PAIR_R1.md), [multidimensional matrix](matrix/experiment_matrix.html).

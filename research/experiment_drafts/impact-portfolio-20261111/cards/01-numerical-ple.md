# 01. Continuous numerical embeddings before temporal attention

**Status: proposed, not implemented or run.** IDs: `H-IMPACT-20261111-01` / `E-IMPACT-20261111-01`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **T** from [CONTRACT.md](../CONTRACT.md); priority 1.

## Mechanism and prior work

Representation continuation of H-RR17-07 discrete-level encoding; continuous interpolation, not categorical level indicators. Not a repeat of PR74 FIR filters.

A flat linear input map may force temporal blocks to learn scalar thresholds. Add continuous train-quantile numerical embeddings so the same temporal backbone can use nonlinear scalar structure.

## Frozen candidate design

**Control:** Unchanged raw112 linear projection and Transformer.

**Treatment:** Keep the same raw projection. Add 16 train-only quantile knots per scalar, a learned 4-dimensional embedding at each knot, continuous linear interpolation, concatenate 112 embeddings and project to H768. Initialize this final added projection to zero. Clamp the added branch outside the outer knots; preserve raw values through the unchanged bypass. Deduplicate coincident knots; a constant feature contributes zero.

The scientific design values above are proposed for implementation. Source/config/runtime/receiver identities still need binding in a separate mechanism-specific PR; this file is not a runnable config. Apart from the declared changes, inherit the source recipe, full data, precision, optimizer, schedule, update budget, stream semantics and fixed final endpoint from the shared contract.

## Prediction, alternatives and falsifier

Useful gains across both seeds with nonzero branch contribution, especially in fixed train-defined scalar regions, would support nonlinear value encoding.

The branch remains unused, improves training only, or produces mixed/negative primary WP. Extra nonlinear capacity is an alternative explanation; a capacity-matched MLP branch is a separately budgeted follow-up, not an inferred mechanism result.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

Fit and hash knots using train membership only. Never recompute from a whole valid sequence or current test batch. Transformation reads x[t] only.

Adds current-row interpolation/projection; no extra sequence state. Large teacher still needs separate serving/transfer work.

## Implementation and evidence

Missing: Train-only quantile artifact, scalar interpolation module and checkpoint/export integration.

No launch command exists yet. Receiver-only correctness/profile gates and exact ownership/timeout/artifact bindings are required after implementation. Do not run tests, models or inference on the Mac.

Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus:

- knot occupancy and duplicate counts.
- PLE/raw projection norm ratio.
- per-target endpoint WP.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

Conceptual primary sources (not results on this task):

- [https://arxiv.org/abs/2203.05556](https://arxiv.org/abs/2203.05556)

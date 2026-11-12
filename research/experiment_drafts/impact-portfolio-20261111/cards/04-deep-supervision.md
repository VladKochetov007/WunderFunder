# 04. Direct label supervision at intermediate Transformer depths

**Status: implemented; receiver acceptance pending; zero fits executed.** IDs: `H-IMPACT-20261111-04` / `E-IMPACT-20261111-04`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **T** from [CONTRACT.md](../CONTRACT.md); priority 4.

## Mechanism and prior work

Different from PR58 filtered residual targets: same labels and mask, attached to layers6/12/18. Conceptual precedent is Deeply-Supervised Nets; its vision results are not evidence here.

A final-only objective may provide a weak or conflicting learning signal to early temporal blocks. Small intermediate heads could make useful signal accessible at several depths.

## Frozen candidate design

**Control:** Current final-head WMSE only; common backbone initialization unchanged.

**Treatment:** Add independent two-output linear heads after blocks6,12,18. L=L_primary+0.1*mean(L_WMSE_6,L_WMSE_12,L_WMSE_18), each using the exact primary training mask/weight reduction. Final-head loss weight stays1. Discard intermediate heads at serving.

The user requested implementation in this same PR. [Implementation amendment](../IMPLEMENTATION_04_06.md) and [B300 commands](../B300_COMMANDS.md) bind the frozen design to an isolated runner. Existing pipeline files and the native optimization recipe remain unchanged.

## Prediction, alternatives and falsifier

Stronger early-block gradients and better final fixed-E4 WP, rather than only better intermediate heads.

Intermediate heads improve but final WP does not, or the added objectives constrain deeper features. Increased total gradient scale is an alternative explanation and must be logged, not hidden by LR retuning.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

Train labels only; intermediate valid metrics are diagnostics and cannot select the serving head or coefficient.

No additional inference branch after export; final predictor parity still needs receiver verification.

## Implementation and evidence

`depth_models.py` provides explicit intermediate heads; `auxiliary_objectives.py` and `auxiliary_backward.py` add exact global-window-normalized loss. Export strictly loads then removes training heads.

Use [the prepared receiver sequence](../B300_COMMANDS.md). No checks, profiles, fits or inference have run locally or on B300 for this card.


Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus the following optional receiver analysis (per-depth/head diagnostics are not automatically emitted):

- per-depth preclip gradient norms.
- each auxiliary loss and head WP diagnostic.
- final-only primary WP.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

Conceptual primary sources (not results on this task):

- [https://arxiv.org/abs/1409.5185](https://arxiv.org/abs/1409.5185)

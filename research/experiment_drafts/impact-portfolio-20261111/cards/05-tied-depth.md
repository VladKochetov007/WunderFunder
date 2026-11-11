# 05. Twenty-four Transformer applications with twelve shared blocks

**Status: implemented; receiver acceptance pending; zero fits executed.** IDs: `H-IMPACT-20261111-05` / `E-IMPACT-20261111-05`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **T** from [CONTRACT.md](../CONTRACT.md); priority 9.

## Mechanism and prior work

Weight sharing is distinct from PR48 width/context scaling and PR72 gradient horizon. ALBERT motivates sharing, not a claimed market-series transfer.

Repeated composition may retain useful depth with fewer independent parameters and less overfitting.

## Frozen candidate design

**Control:** 24 independent blocks, but initialize blocks13..24 as exact value copies of blocks1..12. The copies are independent parameters after initialization. This special fresh control is not a cached PR75 control.

**Treatment:** Use the same initial function, but tie each pair (l,l+12) to one parameter set; execute 24 logical layers. Keep 24 separate layer-position KV caches, including at reset/resume. Shared parameters appear once in the optimizer; autograd sums both uses. Do not halve the execution depth or share KV caches.

The user requested implementation in this same PR. [Implementation amendment](../IMPLEMENTATION_04_06.md) and [B300 commands](../B300_COMMANDS.md) bind the frozen design to an isolated runner. Existing pipeline files and the native optimization recipe remain unchanged.

## Prediction, alternatives and falsifier

Less train-valid separation and better fixed-E4 WP would support the shared-parameter recipe.

Underfitting or no stable gain. Fewer parameters and altered accumulated gradients are inherent coupled changes; this is not proof of an isolated regularization mechanism. No latency saving is assumed.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

Preserve causal mask and independent logical-layer cache histories; parameter sharing cannot alias stream state.

Fewer weight tensors, same24 block applications/KV cache count. Not automatically a small or fast submission.

## Implementation and evidence

`depth_models.py` implements the copied independent control and twelve unique shared blocks with24 logical caches. The unchanged Muon builder owns each unique tensor once; reconstruction and gradient-sharing probes are prepared.

Use [the prepared receiver sequence](../B300_COMMANDS.md). No checks, profiles, fits or inference have run locally or on B300 for this card.


Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus the following optional receiver analysis (per-depth/head diagnostics are not automatically emitted):

- unique versus logical parameter counts.
- gradient norm by shared block.
- train/full-valid curve and fixed endpoint.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

Conceptual primary sources (not results on this task):

- [https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)

# 03. Joint Transformer with nonlinear target-private late adapters

**Status: proposed, not implemented or run.** IDs: `H-IMPACT-20261111-03` / `E-IMPACT-20261111-03`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **T** from [CONTRACT.md](../CONTRACT.md); priority 3.

## Mechanism and prior work

Adjacent to PR77 specialists and PR55 multi-head mixtures. This retains joint supervision and one shared trunk; there are no independent teacher fits, gates or target-selective trunk losses.

The two targets may need different nonlinear readout features even when temporal representation should remain shared.

## Frozen candidate design

**Control:** Shared final hidden state and existing two-output head.

**Treatment:** For target k, use h_k=h+U_k GELU(V_k h), rank8; keep that target's original output row. Initialize U_k=0 and V_k with its own RNG. Both losses update the common backbone. A linear-linear adapter is forbidden because it can collapse into the existing linear head.

The scientific design values above are proposed for implementation. Source/config/runtime/receiver identities still need binding in a separate mechanism-specific PR; this file is not a runnable config. Apart from the declared changes, inherit the source recipe, full data, precision, optimizer, schedule, update budget, stream semantics and fixed final endpoint from the shared contract.

## Prediction, alternatives and falsifier

Nonzero target-private corrections and stable paired WP without sacrificing one target support limited late specialization.

Adapters collapse, one target worsens enough to erase the other gain, or effects vary by seed. Extra capacity remains an alternative; an equal-parameter shared nonlinear head is a separate follow-up.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

No target values, validity flags, validation-derived gates or future inputs enter adapters.

One shared stream, two small nonlinear readouts. Added operations and bytes require eventual CPU measurement.

## Implementation and evidence

Missing: Target-private adapter factory, exact per-target output mapping and export/reload support.

No launch command exists yet. Receiver-only correctness/profile gates and exact ownership/timeout/artifact bindings are required after implementation. Do not run tests, models or inference on the Mac.

Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus:

- adapter contribution norms by target.
- shared and private gradient norms.
- t0/t1 and pooled WP.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

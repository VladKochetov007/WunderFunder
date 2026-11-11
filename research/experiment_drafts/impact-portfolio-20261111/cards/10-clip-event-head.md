# 10. Auxiliary prediction of the official clipping-boundary events

**Status: proposed, not implemented or run.** IDs: `H-IMPACT-20261111-10` / `E-IMPACT-20261111-10`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **G** from [CONTRACT.md](../CONTRACT.md); priority 10.

## Mechanism and prior work

Adjacent to PR62 auxiliary/target-clipping experiments and PR63 robust losses; keeps WMSE, sample weights and clipping unchanged, adding a distinct event label instead of a new primary clipping threshold.

Explicitly recognizing signed high-magnitude events may improve representation of the rows receiving the largest official weights.

## Frozen candidate design

**Control:** K16 GRU primary WMSE with the unchanged official training recipe.

**Treatment:** From the shared hidden state add one three-logit head per target for raw training label classes y<-2, -2<=y<=2, y>2. Add0.05 times mean unweighted cross-entropy over both targets to unchanged WMSE. Before any fit require each tail class for each target to occur in>=100 eligible train rows across>=32 fitting sequences, counting only finite target rows under the same all_after_warmup loss mask. If not, mark this fixed recipe infeasible; do not retune thresholds on valid.

The scientific design values above are proposed for implementation. Source/config/runtime/receiver identities still need binding in a separate mechanism-specific PR; this file is not a runnable config. Apart from the declared changes, inherit the source recipe, full data, precision, optimizer, schedule, update budget, stream semantics and fixed final endpoint from the shared contract.

## Prediction, alternatives and falsifier

Event discrimination improves and translates into better full pooled WP without offsetting center-row damage.

CE improves but official WP does not, classes are too rare, or center performance regresses. Easy class separation alone is not evidence of useful regression.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

Class labels require pre-clipping raw train targets in a loss-only path. Train class counts cannot use valid; no event labels or score mask at callback.

Auxiliary heads are discarded. Main predictor remains the K16 GRU.

## Implementation and evidence

Missing: Explicit raw-target loss view, fixed prevalence feasibility receipt, CE heads and export exclusion.

No launch command exists yet. Receiver-only correctness/profile gates and exact ownership/timeout/artifact bindings are required after implementation. Do not run tests, models or inference on the Mac.

Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus:

- per-target class prevalence and sequence counts.
- per-class CE/confusion on fixed train panel.
- tail/center diagnostic coverage and full pooled WP.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

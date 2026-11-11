# 08. GRU sensitivity regularization with a perturbed carried state

**Status: implemented; receiver acceptance pending, not run.** IDs: `H-IMPACT-20261111-08` / `E-IMPACT-20261111-08`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **G** from [CONTRACT.md](../CONTRACT.md); priority 8.

## Mechanism and prior work

Different from channel dropout PR59 and input-stem PR79/80. Lipschitz RNN is motivation for examining stability, not the implementation or a proof of stability for this GRU.

A recurrent predictor may amplify small errors in carried state. A training-only noisy-state branch could reduce brittleness while preserving the useful clean memory.

## Frozen candidate design

**Control:** K16 GRU primary WMSE, same clean state trajectory.

**Treatment:** Every fourth eligible chunk, at offset128 fork the current carried GRU state. Add independent Gaussian noise with coordinate std0.01*max(detached hidden RMS,1e-3). Advance the fork through the same next32 inputs; compare its predictions to stop-gradient clean predictions. Add0.01*mean_squared_prediction_difference/(0.01^2+1e-8). Keep primary predictions and all future carried state from the clean branch. Never replace the stream with the noisy state. Persist the separate perturbation RNG.

The design is implemented in this same draft PR; [the implementation amendment](../IMPLEMENTATION_07_10.md) supersedes the earlier separate-PR proposal. Source/config pins and receiver commands are prepared, but acceptance has not run. Apart from the declared changes, inherit the source recipe, full data, precision, optimizer, schedule, update budget, stream semantics and fixed final endpoint from the shared contract.

## Prediction, alternatives and falsifier

Reduced train-panel perturbation sensitivity without losing long-memory usefulness, plus paired fixedE6 WP gain.

Sensitivity falls but clean quality declines: the regularizer may erase useful memory. It imposes no global contraction guarantee. Added forward/backward work is a real cost.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

At each suffix position use only the inputs observed up to that position. Fork remains in the same training sequence and is discarded; no noisy state enters evaluation or later chunks.

No additional model or operations after training-only branch removal.

## Implementation and evidence

Implemented in the packet-local runtime. [B300 commands](../B300_CARDS07_10.md) select the correct pinned family, prepare train-only artifacts, run receiver checks and a per-cell profile before fitting. Local tests, model imports, training and inference remain prohibited.

Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus:

- clean/noisy prediction difference at offsets1/8/32.
- primary and auxiliary loss.
- clean-state endpoint WP and runtime overhead.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

Conceptual primary sources (not results on this task):

- [https://arxiv.org/abs/2006.12070](https://arxiv.org/abs/2006.12070)

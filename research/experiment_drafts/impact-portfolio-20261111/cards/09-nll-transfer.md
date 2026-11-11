# 09. Variance-head auxiliary on K16 GRU: continuation of the earlier NLL idea

**Status: proposed, not implemented or run.** IDs: `H-IMPACT-20261111-09` / `E-IMPACT-20261111-09`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **G** from [CONTRACT.md](../CONTRACT.md); priority 7.

## Mechanism and prior work

Explicit recipe continuation, no weight transfer: Oct21 NLL-Control+0.000738421, conditional95% interval[-0.000292725,+0.001869540] on one exposedF0/seed1001/fixedE5 panel. Different historical targets/normalization/initializer; not a replication or confirmed effect. Both arms start from scratch; no historical checkpoint or parent fit is required. The auxiliary keeps WMSE as its anchor.

A variance head may separate predictable mean signal from input-dependent noise without discarding high-error examples from the primary objective.

## Frozen candidate design

**Control:** K16 GRU with unchanged mean head and primary WMSE.

**Treatment:** Add a two-output log-variance head s=clamp(raw_s,-5,3), initialized to0. On eligible training rows add0.05*mean(0.5*exp(-s)*(mu-yc)^2+0.5*s) to the unchanged WMSE. Use independent diagonal variances, not an unconstrained covariance matrix. No variance-based reweighting of the primary loss; discard s at inference.

The scientific design values above are proposed for implementation. Source/config/runtime/receiver identities still need binding in a separate mechanism-specific PR; this file is not a runnable config. Apart from the declared changes, inherit the source recipe, full data, precision, optimizer, schedule, update budget, stream semantics and fixed final endpoint from the shared contract.

## Prediction, alternatives and falsifier

Nontrivial finite variance structure plus paired current-target mean WP gain over K16 control.

Lower NLL without better mean WP, variance pinned at bounds, or harm to high-weight rows. The Gaussian likelihood is a surrogate on clipped labels, not calibrated market uncertainty or Bayesian confidence.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

Variance is predicted from causal hidden state; no target enters it as a feature. Fit all variance parameters only on train.

Mean-only export retains the original callback architecture; verify parity on receiver.

## Implementation and evidence

Missing: Variance head, bounded Gaussian surrogate and mean-only export. Exact earlier NLL code is not silently reused under a new name.

No launch command exists yet. Receiver-only correctness/profile gates and exact ownership/timeout/artifact bindings are required after implementation. Do not run tests, models or inference on the Mac.

Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus:

- variance histogram and bound saturation.
- WMSE/NLL and per-target endpoint WP.
- fixed train-defined magnitude slices.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

Conceptual primary sources (not results on this task):

- [https://arxiv.org/abs/1703.04977](https://arxiv.org/abs/1703.04977)

# 01. Continuous numerical embeddings before temporal attention

**Status: implemented; receiver acceptance pending. No fits executed.** IDs:
`H-IMPACT-20261111-01` / `E-IMPACT-20261111-01`. Preparation owner:
Ivan / Codex; receiving owner unassigned. Template **T** from
[CONTRACT.md](../CONTRACT.md); priority 1.

## Mechanism and prior work

Representation continuation of H-RR17-07 discrete-level encoding; continuous
interpolation, not categorical level indicators. Not a repeat of PR74 FIR
filters. A flat linear input map may force temporal blocks to learn scalar
thresholds. Continuous train-quantile embeddings let the same temporal backbone
use nonlinear scalar structure while retaining its raw feature path.

## Frozen candidate design

**Control:** Unchanged raw112 linear projection and canonical PR75 plain
Transformer.

**Treatment:** Keep the raw projection. Add 16 train-only quantile knots per
scalar, a learned 4-dimensional embedding at each knot, continuous piecewise
linear interpolation, concatenate the 112 embeddings and project to H768. The
added projection starts at zero. Clamp the added branch outside the outer knots;
preserve raw values through the unchanged bypass. Deduplicate coincident knots;
a constant feature contributes zero.

The adapter is [`PLEInputProjection`](../runtime/impact81/representations.py).
[`preprocessing.py`](../runtime/impact81/preprocessing.py) builds and binds its
train-only artifact; [`fitting.py`](../runtime/impact81/fitting.py) composes the
adapter with the canonical trainer. The original training loop and repository
pipeline remain unchanged.

## PLE preprocessing amendment

The implementation uses 64 evenly spaced row positions from each of all 10,325
train sequences, for 660,800 values per feature. It first applies the unchanged
canonical normalizer whose moments use all fit rows, then computes 16 linear
empirical quantiles on this fixed sample. This is not an exact all-row quantile
estimator. The sample is label-independent; valid/test rows and labels are
excluded. The receiver produces one artifact, shared by every arm and embedded
in checkpoints.

## Prediction, alternatives and falsifier

Useful gains across both seeds with nonzero branch contribution, especially in
fixed train-defined scalar regions, would support nonlinear value encoding.
The branch may remain unused, improve training only, or produce mixed/negative
primary WP. Extra nonlinear capacity is an alternative explanation; a
capacity-matched MLP branch is a separately budgeted follow-up, not an inferred
mechanism result.

## Fitting panel and decision

Two fresh seeds 204101/204102, each with a fresh control and treatment:
**2+2=4 planned fits**; zero executed, zero reused controls, and zero
teacher/pretraining fits. These two cells share one exposed official split; they
are not two folds. The implementation does not authorize or automatically
schedule a launch. Follow the contract's +0.0005 mean-WP allocation heuristic,
both seed deltas and per-target review; retain mixed/negative outcomes. Extra
compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

Fit and hash knots using train membership only. Never recompute from a whole
valid sequence or current test batch. Transformation reads x[t] only. The
addition is a current-row interpolation/projection and does not add temporal
state. The large Transformer still requires separate serving/transfer work.

## Implementation and receiver evidence

The server flow is in [B300_COMMANDS.md](../B300_COMMANDS.md): prepare the
artifact, run receiver acceptance, profile one cell for 50 updates, review the
profile/reload receipt, then fit under the same source and device allocation.
These commands are prepared, not executed. Receiver checks, numerical parity,
full-size profile, runtime and quality remain pending.

Canonical receipts must include shared losses/LR/preclip gradients/updates, full
valid counts and WP, source/config/data/ancestry hashes, final checkpoints and
prediction manifests. Knot occupancy and the PLE-to-raw projection norm ratio
are suggested additional analyses, not telemetry currently emitted by the
runtime; collect them only with separately identified receiver-side analysis.
Import completed-run evidence using the
[MIDAS handoff](../midas/README.md) and
[evidence exporter](../midas_evidence.py); route registration alone is not
execution evidence.

## Evidence status

Zero new fits, profiles or grades. No measured or forecast competition gain.
Local model imports, construction, tests, inference, training and profiling
were not run. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md).

Conceptual primary source, not evidence on this task:
[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2203.05556).

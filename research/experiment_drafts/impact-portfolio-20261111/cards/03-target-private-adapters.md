# 03. Joint Transformer with nonlinear target-private late adapters

**Status: implemented; receiver acceptance pending. No fits executed.** IDs:
`H-IMPACT-20261111-03` / `E-IMPACT-20261111-03`. Preparation owner:
Ivan / Codex; receiving owner unassigned. Template **T** from
[CONTRACT.md](../CONTRACT.md); priority 3.

## Mechanism and prior work

Adjacent to PR77 specialists and PR55 multi-head mixtures. This keeps joint
supervision and one shared trunk; it uses no independent teacher fits, gates or
target-selective trunk losses. The two targets may need different nonlinear
readout features even when their temporal representation should remain shared.

## Frozen candidate design

**Control:** Shared final hidden state and existing two-output linear head.

**Treatment:** For target k, use `h_k = h + U_k GELU(V_k h)` with rank8, and
retain the corresponding original output row. Both target losses continue to
update the common backbone. The two up-projections start at zero. The supplied
base head remains the shared linear readout, with each target's own row applying
its adapter correction. This is not the target-selective-loss screen from PR77.

The implementation is [`TargetPrivateHead`](../runtime/impact81/target_head.py),
composed by [`fitting.py`](../runtime/impact81/fitting.py) with the canonical
PR75 plain runtime. The existing training loop and pipeline remain unchanged.

## Prediction, alternatives and falsifier

Nonzero target-private corrections and stable paired WP without sacrificing one
target would support limited late specialization. Adapters may collapse, one
target may worsen enough to erase the other gain, or effects may vary by seed.
Extra capacity remains an alternative; an equal-parameter shared nonlinear head
is a separate follow-up.

## Fitting panel and decision

Two fresh seeds 204101/204102, each with a fresh control and treatment:
**2+2=4 planned fits**; zero executed, zero reused controls, and zero
teacher/pretraining fits. The two cells share one exposed official split; they
are not two folds. The implementation does not authorize or automatically
schedule a launch. Follow the contract's +0.0005 mean-WP allocation heuristic,
both seed deltas and per-target review; retain mixed/negative outcomes. Extra
compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

No target values, validity flags, validation-derived gates or future inputs enter
the adapters. One shared stream feeds two small nonlinear readouts. Added
operations and bytes require eventual CPU measurement; no serving claim is made.

## Implementation and receiver evidence

The server flow is in [B300_COMMANDS.md](../B300_COMMANDS.md): prepare shared
preprocessing, run receiver acceptance, profile one cell for 50 updates, review
profile/reload evidence, then fit with the approved receipt on the same source
and device allocation. These commands are prepared, not executed. Receiver
checks, numerical parity, full-size profile, runtime and quality remain pending.

Canonical receipts must include shared losses/LR/preclip gradients/updates, full
valid counts and WP, source/config/data/ancestry hashes, final checkpoints and
prediction manifests. Adapter contribution and shared/private gradient norms
are suggested analyses, not telemetry currently emitted by the runtime. Import
completed-run evidence using the
[MIDAS handoff](../midas/README.md) and
[evidence exporter](../midas_evidence.py); route registration alone is not
execution evidence.

## Evidence status

Zero new fits, profiles or grades. No measured or forecast competition gain.
Local model imports, construction, tests, inference, training and profiling
were not run. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md).

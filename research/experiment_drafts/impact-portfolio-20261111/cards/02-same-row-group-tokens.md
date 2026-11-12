# 02. Same-row feature-group tokens before the temporal Transformer

**Status: implemented; receiver acceptance pending. No fits executed.** IDs:
`H-IMPACT-20261111-02` / `E-IMPACT-20261111-02`. Preparation owner:
Ivan / Codex; receiving owner unassigned. Template **T** from
[CONTRACT.md](../CONTRACT.md); priority 2.

## Mechanism and prior work

This is an explicit modern successor to PR8 grouped input encoding and adjacent
NF19-24. The earlier note proposing a separate PR is superseded: the
implementation is included in PR81 as a recipe amendment to that lineage. The
specific treatment is nonlinear current-row token mixing, not a duplicate PR8
implementation. Flattening all channels may hide structured cross-group
interactions; separate group encoders test that possibility before temporal
modeling.

## Frozen candidate design

**Control:** Raw112 linear projection into the unchanged causal Transformer.

**Treatment:** Use the canonical 16 groups: i0/i1 crossed with p22, v22, dp4,
and dv4 (eight groups), plus a0–a7 separately (eight groups). Linear+GELU maps
each group to width64; add learned group IDs; apply one current-row self-attention
block with four heads and FF128; mean-pool and project to H768; add to the
unchanged raw projection through a zero-initialized output layer. Attention uses
groups at t only, never an entire future window.

The implementation is [`GroupInputProjection`](../runtime/impact81/representations.py).
It derives ordered group slices from the canonical `FEATURE_COLUMNS` schema.
The adapter is composed through
[`fitting.py`](../runtime/impact81/fitting.py) with the pinned PR75 plain runtime;
the existing training loop and pipeline remain unchanged.

## Prediction, alternatives and falsifier

Stable WP improvement with several groups contributing would support a better
current-row interaction representation. No paired improvement, group collapse,
or an effect explained only by added capacity are falsifiers. A flat equal-
parameter nonlinear stem is the discriminating follow-up; it is not part of the
current fit plan.

## Fitting panel and decision

Two fresh seeds 204101/204102, each with a fresh control and treatment:
**2+2=4 planned fits**; zero executed, zero reused controls, and zero
teacher/pretraining fits. The two cells share one exposed official split; they
are not two folds. The implementation does not authorize or automatically
schedule a launch. Follow the contract's +0.0005 mean-WP allocation heuristic,
both seed deltas and per-target review; retain mixed/negative outcomes. Extra
compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

The group order is derived from the canonical feature schema, not inferred from
names or physical exchangeability. Attention sees only the current row and is
followed by the unchanged causal temporal attention. This adds per-row work
without extending temporal state. No latency or CPU envelope claim is made.

## Implementation and receiver evidence

The server flow is in [B300_COMMANDS.md](../B300_COMMANDS.md): prepare the
shared preprocessing artifact, run receiver acceptance, profile one cell for 50
updates, review profile/reload evidence, then fit under the same source and
device allocation. These commands are prepared, not executed. Receiver checks,
numerical parity, full-size profile, runtime and quality remain pending.

Canonical receipts must include shared losses/LR/preclip gradients/updates, full
valid counts and WP, source/config/data/ancestry hashes, final checkpoints and
prediction manifests. Per-group representation norms and attention entropy are
suggested analyses, not telemetry emitted by the current adapter; the adapter
does not currently return attention weights, so entropy needs separate
instrumentation. Import completed-run evidence using the
[MIDAS handoff](../midas/README.md) and
[evidence exporter](../midas_evidence.py); route registration alone is not
execution evidence.

## Evidence status

Zero new fits, profiles or grades. No measured or forecast competition gain.
Local model imports, construction, tests, inference, training and profiling
were not run. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md).

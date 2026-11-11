# 02. Same-row feature-group tokens before the temporal Transformer

**Status: proposed, not implemented or run.** IDs: `H-IMPACT-20261111-02` / `E-IMPACT-20261111-02`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **T** from [CONTRACT.md](../CONTRACT.md); priority 2.

## Mechanism and prior work

Explicit modern successor to PR8 grouped input encoding and adjacent NF19-24. Continue PR8 with a recipe amendment; do not create a duplicate mechanism PR. Nonlinear current-row token mixing is the proposed specific package.

Flattening all channels may hide structured cross-group interactions. Encode schema groups separately and mix them within the same observed row before temporal modeling.

## Frozen candidate design

**Control:** Raw112 linear projection into the unchanged causal Transformer.

**Treatment:** Use exactly 16 schema groups: i0/i1 times p22,v22,dp4,dv4 (eight groups), plus a0..a7 separately (eight). Linear+GELU maps each group to width64; add learned group IDs; one current-row self-attention block (4 heads, FF128, pre-LayerNorm), then mean pool, project to H768 and add to the unchanged raw projection. Zero-init the added output projection. Never turn an entire future window into a token.

The scientific design values above are proposed for implementation. Source/config/runtime/receiver identities still need binding in a separate mechanism-specific PR; this file is not a runnable config. Apart from the declared changes, inherit the source recipe, full data, precision, optimizer, schedule, update budget, stream semantics and fixed final endpoint from the shared contract.

## Prediction, alternatives and falsifier

Stable WP improvement with several groups contributing would support a better current-row interaction representation.

No paired improvement, group collapse, or an effect explained only by added capacity. A flat equal-parameter nonlinear stem is the discriminating follow-up; not counted in the initial four fits.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

Freeze explicit ordered feature indices from the canonical schema. Group names do not establish physical meanings or exchangeability. Attention here sees only groups at t, followed by unchanged causal temporal attention.

Adds per-row work without extending temporal state. No latency or CPU envelope claim.

## Implementation and evidence

Missing: Pinned group-index artifact, within-row encoder, reset-free input/export adapter.

No launch command exists yet. Receiver-only correctness/profile gates and exact ownership/timeout/artifact bindings are required after implementation. Do not run tests, models or inference on the Mac.

Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus:

- per-group representation norms.
- group attention entropy.
- per-target endpoint WP.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

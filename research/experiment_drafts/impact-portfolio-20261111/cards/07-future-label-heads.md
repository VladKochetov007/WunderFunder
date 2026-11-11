# 07. Future-label auxiliary heads on a causal Transformer

**Status: implemented; receiver acceptance pending, not run.** IDs: `H-IMPACT-20261111-07` / `E-IMPACT-20261111-07`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **T** from [CONTRACT.md](../CONTRACT.md); priority 6.

## Mechanism and prior work

Explicit continuation of the old multi-horizon-target backlog. Different from PR78 future-input pretraining: joint supervised task training, no pretraining stage or initializer.

Predicting several nearby task horizons may make hidden states retain signal useful to the current target.

## Frozen candidate design

**Control:** Current two-target WMSE, four full epochs.

**Treatment:** From h[t], add two-target heads for offsets1,16,64. Primary current-target objective unchanged. L=L_primary+0.1*mean(three future-label WMSE losses), each with source-equivalent abs-clipped-target weights and its own valid-row denominator. Use t>=99 and t+h within the same fitting sequence; mask missing/nonfinite labels. No other fitting stage.

The design is implemented in this same draft PR; [the implementation amendment](../IMPLEMENTATION_07_10.md) supersedes the earlier separate-PR proposal. Source/config pins and receiver commands are prepared, but acceptance has not run. Apart from the declared changes, inherit the source recipe, full data, precision, optimizer, schedule, update budget, stream semantics and fixed final endpoint from the shared contract.

## Prediction, alternatives and falsifier

Future task losses improve and the unchanged current-target head improves at fixedE4.

Auxiliary tasks improve but current WP does not, or horizon interference hurts a target. Highly overlapping task labels may add redundant loss rather than useful memory. These are row offsets, not verified clock-time horizons. The initial screen tests the three-head package; attributing an effect to an individual offset requires extra separately budgeted arms.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

Only loss construction may look ahead in training labels. Model inputs, caches, normalization, feature processing and outputs at t remain prefix invariant. Across TBPTT boundaries a label-only buffer is permitted; no future feature/state encoding. Never borrow labels from valid or another sequence.

Drop auxiliary heads; no future-label or target input exists in the exported predictor.

## Implementation and evidence

Implemented in the packet-local runtime. [B300 commands](../B300_CARDS07_10.md) select the correct pinned family, prepare train-only artifacts, run receiver checks and a per-cell profile before fitting. Local tests, model imports, training and inference remain prohibited.

Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus:

- loss and eligible count per horizon/target.
- combined preclip gradient norm; separate primary/auxiliary attribution is optional receiver analysis.
- current-target fixed-E4 WP.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

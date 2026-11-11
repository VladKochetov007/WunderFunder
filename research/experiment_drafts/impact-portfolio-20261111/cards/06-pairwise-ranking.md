# 06. Within-sequence ranking auxiliary for the Transformer

**Status: implemented; receiver acceptance pending; zero fits executed.** IDs: `H-IMPACT-20261111-06` / `E-IMPACT-20261111-06`. Preparation owner: Ivan / Codex; receiving owner unassigned. Template **T** from [CONTRACT.md](../CONTRACT.md); priority 5.

## Mechanism and prior work

Different from PR70 correlation objective and PR56 teacher-delta KD: supervised pair ordering, no teacher. RankNet supplies a conceptual pairwise logistic objective.

WMSE may fit magnitude without sufficient ordering discrimination among clearly separated targets. A small pairwise term could improve ordering while retaining the original regression anchor.

## Frozen candidate design

**Control:** Exact Transformer primary WMSE.

**Treatment:** Within each fitting sequence/chunk, form only pairs (t,t-32) with both rows after warmup and absolute clipped-label difference>=0.5. For each target use sign(yc_t-yc_u) and softplus(-sign*(z_t-z_u)/1.0); weight by sqrt(abs(yc_t)*abs(yc_u)), normalize by eligible pair weight per target, then average targets. L=L_primary+0.05*L_pair; zero term when no eligible pairs. Do not use is_scored (train lacks it), cross-sequence pairs, pair labels as inputs or validation-driven pair mining.

The user requested implementation in this same PR. [Implementation amendment](../IMPLEMENTATION_04_06.md) and [B300 commands](../B300_COMMANDS.md) bind the frozen design to an isolated runner. Existing pipeline files and the native optimization recipe remain unchanged.

## Prediction, alternatives and falsifier

Better train pair concordance together with better pooled fixed-E4 WP.

Concordance improves while weighted Pearson does not or deteriorates. Rank order alone is not equivalent to Pearson: nonlinear monotone changes can hurt the official metric.

## Fitting panel and decision

Two fresh seeds204101/204102, each with a control and treatment: **2+2=4 new fits**; zero reused controls, zero teacher/pretraining fits. These two cells share one exposed official split; they are not two folds. No automatic follow-up or runtime budget is assigned. Follow the contract's+0.0005 mean-WP allocation heuristic, both seed deltas and per-target review; retain mixed/negative outcomes. Extra compute buys a matched mechanism screen, not a promised scaling law.

## Leakage and deployment

Both labels are training-loss targets only; each prediction remains prefix causal. Pair eligibility and RNG are frozen without valid feedback.

Training-only loss; identical callback architecture.

## Implementation and evidence

`auxiliary_objectives.py` implements within-chunk lag32 pairs, per-target global denominators and safe empty cases. `auxiliary_loop.py` logs losses, pair counts/mass and concordance separately from official WP.

Use [the prepared receiver sequence](../B300_COMMANDS.md). No checks, profiles, fits or inference have run locally or on B300 for this card.


Return all shared losses/LR/preclip gradients/updates, full valid counts and WP, source/config/data/ancestry hashes, final checkpoints and prediction manifests, plus the following optional receiver analysis (per-depth/head diagnostics are not automatically emitted):

- eligible pair counts by target.
- pair loss and concordance on fixed train panel.
- WMSE, clipping counts and official WP.

## Evidence status

Zero new fits/profiles/grades. Mechanism critique is recorded in [REVIEW.md](../REVIEW.md). No measured or forecast competition gain.

Conceptual primary sources (not results on this task):

- [https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)

# Creative research iteration — 2026-09-30

The bounded iteration is complete: methodology repairs, independent proposal/critique, matched-seed audit M01, and fixed teacher-mixture screen I1. I1 failed its registered allocation gate; M01 did not establish a stable positive width-recipe effect. No new GPU training, submission, or public-score improvement follows from these results.

The research method is adequate as a controller for bounded work when its contract, critique, experiment and evidence records govern actual decisions. It is not itself a statistical guarantee. For this competition, useful score or deployment improvements take priority over novelty; scientific explanations require their own evidence.

## Contract and evidence boundary

The [registered contract](contract.json) fixes a competition objective, tier C, one ideation/critique/probe/synthesis cycle, at most three CPU screen families, and zero new GPU hours. The recorded public references are candC **0.6640** and candV1 **0.6641**, not independently verified leaderboard readings. The user's subsequent request authorized [GitHub archival publication](publication-authorization.json); no competition submission was made.

All available folds have been inspected: folds 0/1 heavily, folds 2/3 in earlier width/fine-tuning comparisons, and the former holdout both inspected and used in final fitting. None is an untouched confirmation set. Pure-chain filtering removes the specified train-neighbour overlap; it does not reverse adaptive exposure or establish transfer to the hidden period. Repartitioning those observations would not create new confirmation.

The registered primary metric is pooled, clipped, target-absolute-value-weighted Pearson over the prediction/scoring mask, averaged across the two outputs. Whole continuous chains are the resampling unit. Bootstrap intervals condition on the fitted predictions. Training variability, adaptive selection and future-domain shift are separate uncertainties.

Metric-source limitation: the active research implementation switched to pooled scoring in commit `134c4e8` on September 25, while the imported `METRIC.md`, `utils.py` and local competition docs still specify per-block averaging. A September 27 report says the live organizer FAQ confirmed pooled scoring, but this iteration did not recover an archived primary-source confirmation; the live documentation response was not readable through the available fetch. M01 and I1 use their frozen pooled definition unchanged. Their results verify that declared calculation, not an independently synchronized official scorer.

## Methodology repairs

1. **Objective and novelty.** The core skill now distinguishes competition, engineering and discovery objectives. Incremental changes are eligible; plateau decisions compare expected utility, uncertainty and cost. A novel explanation is optional for a useful competition improvement.
2. **Adaptive validation and seeds.** The core skill now records exposure lineage, attempted configurations/proxies, a selection budget, frozen references/configurations and prospective confirmation. Inspected data cannot regain untouched status through new folds, new agents or new seeds. Paired training-seed effects are reported separately from chain resampling and ensemble gains. `screen_failed` is a spending decision, not a universal refutation. A compact manifest and append-only results suffice; there is no new orchestration framework.
3. **Task-specific expert checks.** Alpha Squad and MFT roles now separate causal input availability, predictive utility and causal-effect identification. OLS/LARS/Sharpe/entropy thresholds, counterparty stories, venue identification and temporal placebos are no longer universal vetoes. Roles and review cycles are bounded by the contract; agent agreement is not independent evidence.

Changed instructions: [core skill](../../../.agents/skills/conduct-creative-research/SKILL.md), [empirical validation](../../../.agents/skills/conduct-creative-research/references/empirical-validation.md), [artifact fields](../../../.agents/skills/conduct-creative-research/references/research-artifacts.md), [Alpha Squad](../../../skills/alpha-squad/README.md), [MFT Research Experts](../../../skills/mft-research-experts/README.md). The `.agents` and `.claude` core copies are identical.

The [independent critic](critic.json) reviewed the amended protocol and proposal cards before current-cycle outcomes. It accepted the exploratory protocol and required concrete comparison definitions. The critic shared a model family and historical context with the other agents; its review is a check on the design, not statistical evidence.

## M01: matched-seed audit, completed

The [frozen manifest](seed-audit.json) compares the existing GRU256 and GRU128 head-only fine-tunes for matched seeds **3, 4, 5**, on pure chains from folds **0, 1**. This is a methodology demonstration on previously studied width experiments, not a new width discovery or fresh training replication. The evaluator and manifest were committed at `6d2e963` before execution.

The original logs support fixed epoch-3 EMA parents followed by one epoch of head-only fine-tuning with `inp,gru` frozen. Candidate/control configurations match except for width and artifact identity. The seed-3 parent logs have a schema difference: `quant_i1` is absent in the older GRU128 record and false in GRU256; `quant_feats` is false in both. Equal epochs do not imply equal compute or serving budgets across widths. [Checkpoint provenance](M01-checkpoint-provenance.json) records the twelve run configurations and hashes of logs, current generation scripts and parent/head checkpoints; historical logs do not cryptographically bind each prediction cache to its checkpoint.

The one permitted execution used **860 sequences in 250 pure chains**. These counts come from the hashed current split cache and should not be confused with the older 831-sequence local audit.

| Comparison: GRU256 minus GRU128 | Pooled WP delta | Conditional chain 95% interval |
| --- | ---: | --- |
| Seed 3 | +0.000073 | [−0.001823, +0.002059] |
| Seed 4 | −0.000339 | [−0.002288, +0.001489] |
| Seed 5 | +0.000792 | [−0.001984, +0.003240] |
| Mean of the three matched seed effects | +0.000175 | [−0.001887, +0.002129] |
| Mean-prediction three-seed ensemble | +0.000696 | [−0.001315, +0.002629] |

The sample SD across the three matched seed effects is **0.000572**; two effects are positive. The mean of paired effects and the score of averaged predictions answer different questions. Predictions are averaged before final clipping, and pooled fold deltas are not averages of within-fold correlations.

**Decision: insufficient evidence for a stable positive width-recipe effect from this audit.** The observed ensemble advantage remains an exploratory fixed-artifact result. Neither these intervals nor the three reused seeds correct selection bias or establish unseen-score improvement. This does not show that width is useless; it does not select a replacement model.

Execution on `beleriand`: CPU quota 160%, numerical libraries limited to two threads, 8 GiB memory cap, 480-second timeout, GPU visibility disabled. Actual wall time **30.29 seconds**, maximum RSS **942,784 KiB** (about 0.90 GiB), exit status **0**. Cached predictions were read only. Outputs use exclusive creation and retain input hashes. [Result](M01-paired-seeds.result.json), [resource log](M01-paired-seeds.run.log).

## I1: fixed disagreement-gated teacher mixture, completed

The [committed ideator card](ideator.json) proposes one fixed causal disagreement gate between plain and distilled GRU256 teacher pairs, evaluated inside the unchanged deployed ensemble remainder. It compares the gated teacher against a uniform mixture of the same components, the distilled pair, and original C. It does not propose serving all four expensive teachers.

The critic identified that gating and average mixture composition change together. The permitted screen therefore tests the whole fixed gate-versus-mixture proposal; it cannot attribute a gain to conditional selection alone. A cached teacher-function gain also cannot establish that a same-size student will learn it, or that new training seeds will replicate it.

The [pre-outcome amendment](prereg-amendment.json) froze all seven cache identities, folds, exact formulas, coverage definition and stopping rule. The [implementation](../../../scripts/conditional_teacher_probe.py) and amendment were committed at `169ad5d` before execution. It wrote an [immutable input inventory](I1-result.inputs.json) before loading scoring arrays and verified that input filesystem fingerprints did not change during evaluation.

Here Q denotes the distilled teacher pair, U its uniform mixture with the plain pair, and G the registered disagreement-gated teacher. Each replaces the same GRU256 slot with the other three ensemble members fixed. Results cover the same **860 sequences / 250 pure chains** as M01.

| Ensemble comparison | Fold 0 delta | Fold 1 delta | Pooled delta | Conditional pooled chain 95% interval |
| --- | ---: | ---: | ---: | --- |
| G − U, primary | +0.000099 | +0.000088 | +0.000090 | [−0.000025, +0.000208] |
| G − original C | +0.000027 | +0.000095 | +0.000057 | [−0.000153, +0.000249] |
| G − Q | −0.000048 | −0.000009 | −0.000028 | [−0.000064, −0.000003] |

The pooled primary conditional chain SD is **0.000060**. Gated metric-weight coverage is only **0.8077% / 0.8033%** on folds 0/1, versus the required **5%** on each. The primary pooled gain also misses the required **+0.0004**, although its sign and the comparison with original C are nonnegative on both folds. The gate operates on roughly 0.8% of scored rows/weight, and the distilled pair Q itself scores slightly above G; this screen supplies no reason to attribute a useful gain to conditioning.

**Decision: `screen_failed`; close this exact gate.** No threshold retuning, matched-student GPU fit, or submission promotion follows. This allocation decision rejects the registered recipe as the next experiment, not conditional teaching in general. The intervals are descriptive for reused, fixed teacher artifacts and are not selection-adjusted significance tests.

Execution on `beleriand`: CPU quota 160%, two numerical threads, 8 GiB memory cap, 480-second timeout, GPU visibility disabled; **23.21 seconds** wall time, maximum RSS **1,067,048 KiB** (about 1.02 GiB), exit status **0**. No model fitting or new forward inference occurred. [Result](I1-result.json), [resource log](I1-run.log), [pre-outcome input hashes](I1-result.inputs.json).

## Deferred proposals

**I2, cost-aware runtime governor:** deferred before outcome testing. The proposed timing replay was not sufficiently specified. Literal component timings `85 + 17 + 17 + 17` microseconds total 136 microseconds/row, whereas the cited 48m52s public runtime over 39.4 million rows implies about 74.4 microseconds/row. An additive overhead alone cannot reconcile them without becoming negative; using them as ratios requires an independently fixed scale and overhead. The transition rule and mapping from a 1,970-sequence simulated run to fewer cached validation chains also need registration. This is an unresolved engineering proposal, not a failed measured improvement.

**I3, random-origin training augmentation:** deferred as proposed. Cached early/late error profiles cannot separate state age from sequence content. A controlled reset-origin forward pass would test context sensitivity, but even that would not establish that augmentation helps training. No substitute correlational screen was run.

## Next decisions

1. Evaluate the newly completed Transformer experiments using a registered matched comparison and ensemble-replacement score. Their final fold-1 artifacts were captured in the [late archive supplement](https://github.com/VladKochetov007/WunderFunder/tree/archive/2026-10-01/snapshot/2026-10-01/delta-late); ordinary training-log scores do not establish a contribution to C. Check fixed checkpoints, fold exposure and real serving cost before another architecture fit.
2. Calibrate heterogeneous ensemble-member and fixed-overhead runtime costs on the relevant CPU paths, then register a concrete governor policy and replay mapping. Treat a useful governor correction as engineering progress even if it introduces no new predictive mechanism.
3. Reconcile the authoritative metric source and identify genuinely prospective confirmation before another promotion claim. Keep I1 student training closed: its prerequisite gate failed. A different teacher proposal needs its own rationale, frozen comparison and trial allocation, rather than retuning this result.

The cycle ends here after two bounded CPU evaluations, totaling **53.50 seconds** of measured evaluator wall time. It does not open another parameter sweep, create a new holdout by repartitioning, or promote a submission automatically. Existing training jobs were left unchanged.

## Verification and reproducibility

The core skill passed `quick_validate`; its six JSON examples parse, relative references resolve, and mirrored files match byte for byte. The two evaluators have **32 passing synthetic tests**, including 22 for paired-seed evaluation. They cover paired effects/sample SD, chain-stratified resampling, rejection of chains split across folds, degenerate Pearson sides, invalid/aliased seed manifests, pure OOF selection, warm-up/scoring masks, averaging before clipping, teacher-mixture formulas, coverage and immutable output. Ruff and `git diff --check` passed for the reviewed code.

These checks establish implementation properties, not research success. The results are deliberately marked discovery-only, and there is no new public score, fresh holdout confirmation or newly trained model from this iteration.

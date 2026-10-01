# Independent post-failure scout — selection weights closed

**Decision: do not allocate this proposal, its diagnostic, or a forecasting-budget amendment tonight.** The strongest mechanism I initially identified was outcome-conditional scoring selection used only in training weights. A deeper historical search found its direct predecessor, E-103, and the subsequent conditional-moment argument and failed residual-map experiment. This is a previously investigated selection family, not an independent new high-upside direction. The initial provisional novelty assessment was wrong and was corrected before any data-backed work.

This scout read repository history, source, and existing result receipts only. It did not load competition rows, targets or prediction arrays, fit anything, execute a diagnostic, query WP, change model code, or publish anything. All existing validation folds are exposed. The controller explicitly closed the selection proposal after the E-103 finding; this document does not authorize a reopening.

## Mechanism considered, and why it initially seemed relevant

Let X denote the available causal history, Y a training target, and S the scoring indicator known only on mask-bearing training-side validation data. The considered intervention was a bounded multiplier derived from `qXY=P(S=1|X,Y)` divided by `qX=P(S=1|X)`, learned only on permitted training chains and applied to the training main loss. It would add no serving feature, hidden mask access, inference state or native operation.

The algebraic motivation is limited but real: for a pointwise loss L, `E[S L(X,Y)] = E[qXY(X,Y) L(X,Y)]`. A purely X-dependent weight does not alter an unconstrained conditional minimizer at fixed X, whereas an outcome-dependent weight can. This identity does **not** establish outcome-dependent selection in these data, identify a causal mechanism, justify transporting a selector from valid to train/test, or make a clipped ratio multiplied into the current heuristic weights equal the official Pearson objective.

E-162 reports a scored/unscored class-3 WP gap of .17–.19 at equal coarse propensity. That observation motivated the scout, but it is compatible with omitted causal information or a miscalibrated/coarse propensity representation. It does not show `S` depends on `Y` conditional on the full causal history.

## Closest prior work changes the decision

| Existing source | Actual evidence | Implication |
|---|---|---|
| `research/experiments.jsonl`, E-103 | Explicit **label-aware P(S|X,Y)** GBM, chain folds 0/1 to 2/3, 35% row sample. AUC .8957 to .8979, gain .0022 versus its .005 gate. Class-3 AUC .9088 to .9091. | Direct duplicate mechanism; the historical go/no-go failed. No claim that a full forecast fit of these weights was already performed. |
| `research/ideas/2026-09-27_metric.md`, hypothesis 1 | Already argues that a small AUC gain could conceal changes in high-weight conditional target moments. | Even the proposed “AUC is not the right diagnostic” rescue is historical. Rephrasing it would not justify a new allocation. |
| E-186 and `scripts/probe_conditional_mask.py` | Cross-fitted coarse conditional-mask residual contrasts .0596/.0934, but output correction lost on both folds: −.002182/−.000347, pooled −.001296. | A visible residual contrast did not transfer into useful prediction. This does not refute every training-weight construction, but removes the simplest positive evidence for reopening. |
| E-155 then E-170 | Sharper causal propensity gave a weak +.0010 result in an older recipe; in the EMA-4 recipe it gave −.0001 on both folds. | Historical large propensity gains cannot be carried forward as expected incremental upside in the current model. |
| E-045 and E-237 | Propensity as input did not help; scorer-mask auxiliary failed its exact pair gate. | Input/auxiliary variants are also familiar, not fresh alternatives to the failed weight proposal. |

E-103 is a limited classifier screen, not proof of conditional independence. Conversely, its limitations are not positive evidence. No new observation found in this bounded scout identifies a specific omitted operator or demonstrated implementation error that would justify overruling that stop. I therefore do not propose another classifier, another target transform, a lower diagnostic threshold, or a forecast fit.

## Source audit: the naive multiplier was underspecified

The current parent loss is not simply `qX²`.

- `scripts/train_rnn.py` first constructs post-warmup weights `qX² + .03` for the ordinary registered parent recipe.
- `true_mask_weights` replaces their **shape** on sequences carrying a true scoring mask with `c * (S + .03)`. The scalar c preserves their pre-replacement propensity-weight mass within the batch. The implementation detects mask-bearing sequences with `scored.any`, not an explicit source identifier.
- The causal since factor is then applied. With the current recipe it is `max(1/(1+since/32), .25)`.
- Main weighted MSE additionally includes each target's `abs(clip(Y,-2,2))`. The same pre-target-magnitude `main_valid` also supplies auxiliary weights and EMA consistency weights.

Consequently, inserting a ratio before true-mask replacement may have its rowwise effect overwritten on valid rows while altering their source mass. Inserting it into `main_valid` afterward also changes auxiliary and EMA optimization, not only the main estimand. Neither is the claimed isolated intervention. A legitimate future test would need an explicit main-only, train-source-only treatment, separate unchanged auxiliary/EMA tensors, and a matched no-op path; no such implementation is authorized or supplied here.

There is also a nuisance provenance issue. The current `train_propensity.py --heldout-fold` excludes that fold from gradient fitting but evaluates it each epoch and saves the best held-fold AUC checkpoint. A filename such as `prop_f0` is therefore not by itself evidence of untouched outer-fold selection. Existing copies have hashes in `sam/propensity-copy-receipt.json`; this scout did not re-audit those historical training runs. Any future stricter claim would have to document checkpoint-selection exposure as well as gradient-fit chains. This is a limitation of a possible new study, not a reason to alter already frozen comparators.

## Allocation, cost, and stopping rule

At the inspected `state.json` snapshot (01:52:46 UTC), all eight permitted scientific configurations were occupied. The recent AUX/F/T/ENS results supply no new positive selection-weight evidence. A new matched main-only weighting control and candidate would require an explicit prospective configuration amendment, plus nuisance-fit accounting and forecasting-fit reservations. Two arms × two parent seeds × two folds × parent/head stages would already cost 16 forecasting fits before final fitting; head-only adaptation would not test a change intended for train-source parent learning. This branch is not free merely because the serving cost would be unchanged.

A future successful training-only change could retain the same h128 architecture and fixed-three native runtime, but that is a systems advantage, not evidence of score improvement. No fit-time estimate is asserted: no timing of this unimplemented intervention exists.

**Stop now.** The historical direct screen failed, its main conceptual rescue was already proposed and produced a negative simple intervention, and no distinct positive discriminator was found. Do not spend a diagnostic query or amend the eight-configuration limit for this family in the current campaign. This closes this scout's proposal, not all possible research in selection bias.

The validated fallback remains `submissions/overnight_baseline3/baseline3_fixed.zip`, with its existing correctness/runtime receipts. This note makes no new SOTA, hidden-test, or leaderboard claim.

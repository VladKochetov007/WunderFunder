# Second deepening round: assumptions to attack before the EDA

Written before the new real-data analyses. The objective is to decide what evidence is worth obtaining before another forecasting-model fit, not to manufacture a statistically significant improvement. GitHub publication is paused by the user; this round stays on local disk and beleriand.

The previous round establishes three narrow observations: auxiliary/shared gradients were positively aligned at two early windows of one parent; a decoded class-3 binary clock materially changes event writes but has unresolved semantics; one global sparse-line volume representation failed its fixed engineering gate. It does not establish that auxiliary tasks help throughout training, that suppressed shape events are uninformative, or that no useful flow representation exists.

## Training and fixed-ensemble geometry

The first gradient sample was balanced by target-derived market class. That estimates neither the natural distribution of FT batches nor the contribution of each class to the pooled metric. Later sequence positions and saved head stages may have different auxiliary/main balance. The new gradient audit must keep all sampled chains/windows, show paired parents/stages, and report leave-one-chain sensitivity without pretending that many windows create many independent training seeds. Saved one-epoch heads are not V1's two-epoch heads, and fixed checkpoints are not a reconstruction of Adam's historical moments.

Positive Euclidean cosine is not a theorem about generalization; negative cosine is not a prerequisite for a useful ablation. Global clipping mixes private auxiliary-output and shared-head gradients, while adaptive moments can cancel common scaling. No diagnostic gradient is to be interpreted as a simulated optimization gain without actually evaluating the declared update process. No update is allocated in this round.

An individual member's scale invariance creates a possible mismatch with fixed-weight ensembling. It does not establish useful scale headroom. Previous class-specific scales, offsets and blend gates often failed (E-109/E-127/E-138 and the September29 journal); the oracle per-sequence fits in `eda_scale.py` use labels and are not attainable online features. The new OOF study measures analytic sensitivity at the already fixed C mixture, with no coefficient fit or candidate-score query. Small or fold-unstable derivatives would weaken a simple calibration explanation while leaving nonlinear ensemble-aware training untested.

Global Pearson is not a weighted average of per-class Pearsons. Weighted covariance decomposes into within-class covariance plus covariance of class means, and both variances have matching decompositions. That exact accounting can distinguish where C's score comes from without declaring the between-class component illegitimate. Metric-mass concentration across chains should also be reported. The concentration index `(sum W)^2/sum(W^2)` is not an estimate of the number of independent rows or a magic correction for temporal dependence.

## Event-clock reliability versus usefulness

The relevant treatment is binary recurrence update/no-update. Unweighted disagreements in signed move pairs can exaggerate instability of that binary decision; rare edges must not count like frequent events unless the estimand explicitly concerns keys. Conversely, high row coverage can conceal a few systematic high-activity keys. The new study therefore separates per-key signed agreement, row/event-weighted binary agreement, known-key versus fallback behavior, and variation across whole audit chains.

Two fresh train-only fits assess stability of a data-derived partition. They do not provide observed physical bid/ask truth. Target-jump impurity can reflect changes in future references, and stable bias can survive every fit. The old unique-key gate remains failed historically; the new binary estimand is registered prospectively and must not be represented as retroactively passing the old test.

The proposed `z_t` to `h_t` event token change also changes gradient routing through the row GRU. Counts of volume/trade updates between token writes can establish that an interval contains observations beyond its endpoint, but cannot establish that the row state encodes them well or that event memory improves future-target forecasts. All rows still reach the ordinary GRU. Changing the token, clock, state decay and FT freeze policy simultaneously would make a later result uninterpretable.

## Trade-mark predictability and its stronger controls

Observed trade-code changes are not identified exchange messages. The detector can introduce persistence, omit repeated same-code trades or include neutral-return transitions. The new analysis must specify the detector and category mapping before outcomes, count excluded/empty/ambiguous cases, and avoid full-sequence centering. Categories may be useful without being economic buy/sell signs.

A fixed global-frequency control is weak in a mixture of markets and regimes. A predictor could beat it merely by recognizing the local category mix. The primary baseline should adapt category frequencies causally inside each sequence; the candidate adds a fixed-order dependence on prior marks. Predictions are scored before updating with the observed mark. Prequential evaluation explicitly respects that predict/observe/update order ([Dawid, 1984](https://academic.oup.com/jrsssa/article/147/2/278/7106293)).

Log loss is an appropriate proper score for the categorical probability forecast ([Gneiting and Raftery, 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)). A log-loss difference compares these probability estimators; with finite fitted models it is not automatically a mutual-information estimate, evidence of economic informed trading, or a future-price alpha estimate. The outcome here is a feature mark, not t0/t1.

Full-sequence permutation retains category counts but destroys both order and slow changes in local composition. Fixed-block permutation retains more slow composition while disrupting fine order; the two nulls answer different questions. Neither is a deployable transformation or an exact randomization test under arbitrary nonstationarity. With only a few predeclared permutation seeds, report descriptive contrasts and their variability, not an invented calibrated permutation p-value. Prefix causality tests apply to the real predictor, not to the deliberately offline surrogate generator.

## Statistical reporting contract

Use whole chains as the resampling unit and keep fit/audit groups disjoint for each diagnostic. Preserve fold and market strata explicitly. Report event-weighted ratios and equal-chain means as different estimands. Chain-bootstrap intervals condition on fitted artifacts and the observed chain grouping; they do not establish physical independence between unidentified market periods, include optimizer variation, undo previous data exposure, or validate transfer to the private test.

All prescribed slices and controls must remain visible, including cold-start behavior, tails, weak folds and negative effects. A favorable class or window cannot replace the frozen primary result. This round allows descriptive uncertainty and sensitivity analysis; it does not make familywise significance claims across its four adaptively motivated analyses. Next-model decisions should reflect effect size, stability, mechanism relevance and serving cost together.

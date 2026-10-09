The six cards below are first proposals from this transcript only. No fit was run, and no other agent’s ideas are used. Tags: **EST** speaker’s own report, **RET** his retelling of the winners or organizers, **REC** his reconstruction of the encoding, **ANA** transfer to our sequences, **SPEC** a step the transcript does not pin down.

Our setting, used in every card: 112 anonymous numeric channels, two regression targets, deterministic causal row serving, full reset on each sequence, no future row or label. Official score is `global_wp_clip2_v1` (mask, clip ±2, weight by clipped-label magnitude). Training loss and training clipping stay separate. Every arm keeps the raw channels. Parameters that see labels or channel couplings are fit on a declared train-sequence block only. The already exposed validation split stays a description, not a held-out claim.

Cost tiers, with no durations: **A** train-block scan, no predictor. **B** closed-form or univariate fit on that block, inside the local 6-CPU / 20 GB / 5 GB envelope, no GPU. **C** one small CPU tree plus a matched row-wise comparison. **D** nested lookup maps for both targets, still without a neural fit.

| ID | Type | Transcript piece | Probe tier |
|---|---|---|---|
| E533-BNP-01 | Encoding-structure probe | Denominator = minimum neighbor gap; differences were the point of the search and were left unused | A |
| E533-BNP-02 | Geometry / latent space | Linear sums found by regression error toward 0; which term to drop was unresolved | B |
| E533-BNP-03 | Conditional-support probe | Fixing `v22` collapsed other columns’ level counts | A–B |
| E533-BNP-04 | Interaction feature | Pair, triple, and quadruple interactions as a separate level-1 pack | C |
| E533-BNP-05 | Evidence regularizers | Shrunk group mean, log tilt, and the unnormalized difference that sits near 0 for rare groups; nested folds | B–D |
| E533-BNP-06 | Unconventional detail | Ordinal position after sorting by a time difference | A–B |

## Source sections

| Passage | Short paraphrase of what is actually said | Tag | Cards |
|---|---|---|---|
| P1 | One stratified 5-fold was drawn once and shared. Inner folds were free: he used 5, Cazano used 10 and repeated that draw. | EST | 05 |
| P2 | About 228k rows, 130 columns named `V1`…`V130`, binary target, full obfuscation. Their plan was to let the model find the meaning. First place decoded the columns. | EST; winner = RET | 01, 03 |
| P3 | Some columns are constant. Others are sums of two, or of about five, others. He found them by regressing one column on others: error toward 0 means a real dependence. Plots agreed. Which variable to delete was unclear (`Va` versus `Vb = Va − Vc`). | EST | 02 |
| P4 | Every column lay strictly in `[0, 20]`. The histogram was a discrete comb plus spread. He reconstructs the pipeline as discrete values, then small noise, then a per-column scale (his example: 4 becomes 20). “Поиск деноминатора” is the search for the coefficient you divide by. A questioner says the method was the minimum difference between neighboring values; he answers “Ага, отлично.” On the winner call he says they wanted those denominators in order to subtract features, they never put those differences in the model, and the search remained a rudiment in the code. | Range, search, unused differences EST; generative story REC; min-gap affirmed EST | 01, 06 |
| P5 | Some columns have no missing values. On those, the number of distinct values sits around 360. He reads three of them as a day index. First place used that time axis. | Missingness and ~360 EST; day reading is his inference; winner use RET | 06 |
| P6 | They dropped about 50 of 130 with someone else’s script and kept about 80. Shifting the script by about five columns made results worse. Asked what he would have done without the script, he describes greedy deletion by importance, the reverse order, then re-adding dropped columns, each time with full fold validation. Recomputing the later stats on all 130 instead of 80 was someone else’s trial and looked similar. A listener’s PCA worry: a rotation can move columns into the wrong space. | Drop counts and ±5 EST; greedy path is hypothetical EST; 130-vs-80 secondhand; PCA is audience | 02 |
| P7 | Rows with `v22` fixed (example 0) make other columns repeat and cut their number of levels by about an order of magnitude. He concludes `v22` groups the table, and group identity lines up with the target. A separate model per `v22` value works poorly: it is a tree’s first split and throws away the other groups. | EST | 03, 05 |
| P8 | New column = target mean inside a category. Rare levels overfit (one row with target 0 forces a mean of 0). The fix uses the group count `n` and a coefficient `α` (his example is 10), plus the global train mean. Large `n` moves the value to the group mean; small `n` moves it to the global mean. `α` is chosen by ordinary CV. The test map is computed from train. | Limits and `α` example EST; algebraic form is not on a readable board | 05 |
| P9 | He names weight of evidence as a log of the group’s positive count against a total count: a tilt toward 0 or 1 with a different regularizer, and says that worked well. What he himself tried: a difference (parsing of a garbled sentence: count of ones minus count of zeros). Few rows in the group → the difference wobbles near 0. Many rows → a large positive or large negative value. “Вот такая штука работала.” | Log tilt EST; difference EST with a parse note | 05 |
| P10 | Numeric columns make too many groups, so regularization wipes the stats out. Binning and then treating bins as categories was done. A univariate linear model of the target on that one column is described as something one could fit (“ещё можно было”). | Binning EST; univariate line is described, not clearly a reported run | 03, 05 |
| P11 | Leak he names: means for folds 1–2 are computed with fold 3, then a model trained on folds 1–2 still sees fold 3 inside those mean features. He says he used that leak in earlier contests; here the local score would not move the leaderboard. Honest version: inside each outer training block, split again and compute means only on the inner complements. Test uses the whole outer block. There is no one global feature matrix. Inner scheme may be 3, 10, or five repeats of 10. Only that is “абсолютно верный.” For a new task he says a fully honest recalculation is appropriate. Level-2 and level-3 stacks still leak a little; the leaderboard moves in the same direction by less. These target features show the overfit much more plainly, so he would not rebuild all 167 models honestly for BNP. | EST | 05 |
| P12 | Faron’s add-on to gradient boosting writes an Excel report of single-column importances and of double and triple interactions, meant to be read by eye. The library name is garbled and he says he had not run it. Cazano’s script used raw columns plus pairwise, triple, and quadruple interactions, no target means, and produced 167 models. Many were useless alone; together they added something. Stas’s own feature model was separately in the top 10. The team stacked the two level-1 packs and then further levels, and finished third. He says feature engineering was the main ingredient. | EST; library name unknown | 04, 05 |
| P13 | Retelling the winners: two time columns, their difference, sort by that difference. Labels then run as zeros first, with ones appearing later, “примерно как в LSTM.” They added the ordinal index of the row in that sorted difference. | RET | 06 |

Parts not turned into cards: shared-drive team process, prize split, core-count anecdotes, and the 167-model sweep as a sweep. P11’s honest-recomputation rule and P12’s “useless alone, useful in the sum” fact are used as protocol and as a blend successor.

## E533-BNP-01 — Lattice denominator probe

**Source.** P4. **EST:** columns were stuck in `[0, 20]`; the histogram was discrete structure plus spread; the repair was a denominator search; he affirmed “минимальная разность между соседними значениями”; the denominators existed so they could subtract columns; those differences never entered the model and stayed a rudiment in code. **REC:** organizers added noise to discrete values and scaled each column (example: 4 mapped to 20).

**Structural correspondence.** Anonymous channels may be a small code, plus noise, times a per-channel scale. **ANA:** nothing here says our 112 channels share a bound of 20. The estimator transfers; the bound does not.

**Concrete intervention.** On one declared train-sequence block, per channel, using observed values only:

- `E_min`: minimum positive gap between sorted unique values (the affirmed estimator).
- `E_mode`: the gap between neighboring high-mass clumps. **SPEC**, required by a tension inside P4: under his own noise-then-scale story, the raw minimum gap is the noise width, while the denominator is the step between codes.
- Offset: train-block minimum. Grid: `offset + k · δ`.

Emit a code and a residual only when `E_min` and `E_mode` agree across two disjoint train sub-blocks. Agreement means the two steps are the same order, pre-declared as a ratio interval fixed before any look at the exposed split. Outputs beside the raw channel: `q = (x − offset) / δ`, and `r = x − nearest grid point`. Serving applies the frozen `(offset, δ)` to the current row. No rounding of the served raw value. No channel deleted. Missingness rate and unique count are diagnostics for E533-BNP-06, not features.

**Assumptions.** A lattice that is real on the train block is stable across train sub-blocks. Row order inside a sequence does not change a per-channel marginal grid. **ANA.**

**Risks.** `E_min` alone will lock onto noise. A tolerance or “agreement” rule is a researcher degree of freedom; it has to be frozen first. Unique-value sketches on a subsample can miss a rare adjacent pair and inflate the min gap. Fitting the grid on the exposed sequences would make any later held-out claim void. A global affine map fit on all channels jointly is a different recipe (P6’s PCA worry).

**Predicted observable.** Per channel: `(E_min, E_mode)`, sub-block stability, and the distribution of `r / δ`. Channels where the two estimators agree are the only code candidates. A second observable, still on the train block: a tree’s split thresholds on raw `x` versus on `q`. P4 says a tree can split the smeared column anyway (“дереву плевать”), so tree agreement is expected and does not by itself retire the code. The code is for differences and for keys.

**Cheapest probe and control.** Tier A. One pass, both estimators, both sub-blocks, diagnostics only. Control: the same estimators on a Gaussian column matched to the train mean and variance of that channel, and on a copy of the channel with an independent jitter added. A genuine grid has to separate from both controls on residual-to-step concentration.

**Falsifier.** On the train block, no channel has `E_min` and `E_mode` stable and in agreement, relative to the Gaussian and jitter controls. The recipe then emits no code and no residual. The diagnostic table is still kept. A later clump estimator of a different form is a new recipe.

**Parents and links.** No campaign parent. Source P4. Feeds E533-BNP-03 (codes as keys) and E533-BNP-06 (differences in code units, which is the rudiment they computed and did not use).

**Level-2 successors.** If `E_min` and `E_mode` disagree everywhere, stop this recipe; a successor may change the clump rule, not the raw control. If they agree on a subset, a successor emits `q_a − q_b` for a capped set of agreeing pairs, row-local, raw columns kept. Pair search itself belongs to E533-BNP-06 so this probe stays a scan.

## E533-BNP-02 — Same-row linear residual geometry

**Source.** P3. **EST:** constant columns; identities `Va = Vb + Vc`; sums of about five columns; detector = linear regression of one column on others with error toward 0; deletion target was ambiguous. P6. **EST:** a borrowed deletion script, sensitive to about five columns. **Hypothetical EST:** greedy drop, reverse drop, re-add. **Audience, not used:** PCA may move features into the wrong space.

**Structural correspondence.** **ANA:** BNP’s identities are across columns of one row. The matching object is a same-row linear form on the 112 channels, not a temporal filter. Lagged forms are a successor.

**Concrete intervention.** Do not drop channels and do not replace them with principal components. On the train block, for each channel `j`, take the four other channels with the largest absolute correlation and fit an ordinary least-squares form with an intercept, so the largest relation considered has five columns (his “суммы из пяти”). Keep the relation when the residual variance is near the numerical floor relative to the control below. Served extra channels are the frozen residuals `r_j(x) = x_j − (a0 + Σ a_k x_k)` for the admitted forms, plus a binary diagnostic “this row’s residual is at the floor.” Constants are flagged and left in the raw set. Coefficients are train-block constants. Application is the current row only, so reset is vacuous; a unit check still confirms that sequence id does not enter the residual.

**Assumptions.** Redundancy, if present, is visible in linear correlation and in subsets of size at most five. A relation that exists only inside a subgroup will be missed here (that case is a successor with E533-BNP-03).

**Risks.** Screening on the largest correlations, then testing those same correlations, overstates the number of identities. The permutation control below is part of the recipe for that reason. An ill-conditioned five-column subset can show a tiny residual by chance; the control has the same subset size. Coefficients fit on the exposed split cannot support a held-out claim. Deleting the “dependent” column, which is what they eventually did with a script, is a different recipe and is not this card.

**Predicted observable.** A list of admitted identities and the residual-variance ratio against the control. The interesting output is which parent columns the regression uses, because P3’s open question was which term is the sum and which is the part.

**Cheapest probe and control.** Tier B. Train block only. Control: permute the target channel of the regression across rows inside the train block, refit the same subset size, and record residual variance. Admission requires the real residual ratio to separate from that null by a margin frozen before the run. The probe may stop at the list. Any later model that consumes `r_j` also consumes the raw channels, and its matched control consumes a residual built from the permuted fit so the width of the feature set matches.

**Falsifier.** No subset of size at most five has a train-block residual ratio that separates from its permutation null. This recipe then adds no residual channels. It does not declare that no nonlinear or within-group identity exists.

**Parents and links.** Source P3, P6. Independent of E533-BNP-01 (scale grid versus cross-channel span). Interaction successor with E533-BNP-03 and E533-BNP-04.

**Level-2 successors.** Identities refit inside keys admitted by E533-BNP-03, still emitting residuals and still keeping raw columns. Lag-1 residuals with an explicit sequence reset, fit per sequence on the prefix only, or with frozen coefficients. Interaction mining of E533-BNP-04 repeated on the residual channels. The greedy delete/re-add story in P6 is a compression successor only after residuals exist, and only as a report of which raw columns a small model can ignore, not as a deletion from the served raw set.

## E533-BNP-03 — Group key by conditional support collapse

**Source.** P7. **EST:** conditioning on one value of `v22` made other columns repeat and reduced their number of levels by about an order of magnitude; separate models per value worked poorly; the group lined up with the target. P10. **EST:** bin a numeric column before treating it as a category. P2. **EST:** names were useless, so structure had to be measured.

**Structural correspondence.** **ANA:** one anonymous coordinate is a group label if fixing it shrinks the support of the others. The order-of-magnitude figure is a report about `v22`, not a threshold to copy. Separate per-key models are what he already found worked poorly, so the served object is one model that sees the key.

**Concrete intervention.** Candidate keys, train block only: each raw channel binned by quantile edges fit on that block, and, if E533-BNP-01 admitted a code, that integer code. For each candidate, on a fixed set of other channels, compute the mean number of distinct values inside a key level. Keep raw channels. The served feature is the discrete key index (bin id or lattice code), stateless, so sequence reset does not apply. Do not fit one model per key.

**Assumptions.** Support collapse is visible in distinct-count, which matches what he looked at (levels `1, 1, 55, 22, 33, 44`). Quantile bins are a stand-in for “a fixed value” on a continuous channel. **SPEC** relative to a lattice code, which is the closer analogue of his discrete `v22`.

**Risks.** Quantile edges fit on the exposed split leak that split into the key. A key that only collapses columns because of a shared missingness pattern should show up in the missingness diagnostic and be labeled as such. Searching 112 keys on the exposed validation and keeping the best is selection on an exposed set. Within-sequence shuffle versus across-row shuffle can disagree when a channel is a smooth time ramp; both controls are required.

**Predicted observable.** For each candidate, the ratio of conditional distinct-count to marginal distinct-count, compared with the two shuffles. The transcript’s qualitative claim is a large drop. Here the admission bar is separation from the shuffle null, not a copied factor of ten.

**Cheapest probe and control.** Tier A if bins are simple quantiles; tier B if lattice codes are attached. Controls, same bin count: permute the key across rows in the train block; permute the key inside each sequence. A key is admitted only if it separates from the within-sequence shuffle. The across-row shuffle catches a key that is only a sequence-level constant.

**Falsifier.** No candidate separates from the within-sequence shuffle on train-block conditional distinct-count. The recipe adds no key index. It does not retire target-side statistics on a pre-declared bin (E533-BNP-05 can still use one pre-declared channel).

**Parents and links.** Source P7, P10. Uses E533-BNP-01 codes when they exist, and still runs on raw bins when they do not. Supplies keys to E533-BNP-05. Does not use the target, so it can be fit without touching the label-derived cards.

**Level-2 successors.** Two-column keys, only among single columns that individually moved the distinct-count. Key-conditional linear residuals (link to E533-BNP-02). A single shared model with key-by-channel products for the few channels whose support collapsed, which is the interaction version of “use the group without splitting the training set into separate models.”

## E533-BNP-04 — Materialized interactions from a train-block tree report

**Source.** P12. **EST:** an Excel report of which columns interact, including doubles and triples, intended for visual inspection; Cazano’s level-1 pack was raw columns plus 2-way, 3-way, and 4-way interactions and did not use target means; many of the 167 models were useless alone and the sum still helped; Stas’s different feature pack also reached the top 10.

**Structural correspondence.** **ANA:** the report is a way to choose a small set of column tuples. The served objects are explicit functions of those tuples, not the tree and not a 167-model sweep. Row-local functions need no sequence state.

**Concrete intervention.** On the train block, fit one shallow gradient-boosted tree, depth limited so the first probe only sees pairwise co-splits, seed fixed, both targets fit separately. Read pairs that co-occur on a root-to-leaf path. For each retained pair `(i, j)` and each frozen threshold pair `(t_i, t_j)` from that tree, add three columns: `x_i x_j`, `x_i − x_j`, and `1{x_i > t_i and x_j > t_j}`. Cap the pair list at a count fixed before the run. Raw channels stay. The first comparison model is row-wise and linear, because these features do not use time. A sequence model is a successor, not the admission probe.

**Assumptions.** Co-splitting in one shallow tree is a cheap stand-in for the interaction report he described and did not personally run. **ANA.** Products and threshold conjunctions are the explicit features; the transcript does not say which algebra Cazano used for a “pairwise interaction,” only that he joined columns in twos, threes, and fours.

**Risks.** The pair list is a selected set. Comparing it with a single random pair understates selection. The matched control therefore draws the same number of pairs by the same extractor after the target is permuted inside the train block. Thresholds taken from a tree fit on the exposed split are not a held-out result. Depth, learning rate, and the cap are degrees of freedom; they are frozen in the probe receipt. Four-way products are the transcript’s full sweep and are out of this probe.

**Predicted observable.** On an inner slice of the train block, official `global_wp_clip2_v1` of raw-plus-selected-pairs versus raw-plus-null-pairs, separately for each target. The observable is the gap between those two, plus the list of pairs. No public-score addition.

**Cheapest probe and control.** Tier C, CPU, one tree per target on a train-block sample that fits the local RAM cap, then the linear comparison. Control: identical pipeline after permuting that target inside the train block, same pair budget, same feature width. Both arms include raw channels.

**Falsifier.** For both targets, the selected-pair arm does not separate from the target-permuted pair arm on the inner-slice official metric. This recipe stops. It does not say a different interaction extractor would fail, and it does not say a weak pair family is useless inside a later blend (P12: half the models were useless alone).

**Parents and links.** Source P12. Independent of target-mean features on purpose: that was Cazano’s pack versus Stas’s pack. Level-2 blend with E533-BNP-05 is the team’s actual construction, under P11’s honest rule. Optional input: residual channels from E533-BNP-02, as a second pair list, not as a replacement of the raw pair list.

**Level-2 successors.** Triples drawn only from pairs that beat the null. A small blend of this pack with the E533-BNP-05 maps, each pack built inside the nested split. Repeating the extractor on lattice codes (E533-BNP-01) or on group keys (E533-BNP-03). A sequence model that receives the frozen pair features, admitted only after the row-wise gap is real.

## E533-BNP-05 — Three evidence regularizers, nested by sequence

**Source.** P8, P9, P11, with keys from P7 and bins from P10. **EST limits of the shrunk mean:** group count `n`, global mean, `α` as a CV constant (example 10, not a reported shipped value); large `n` tends to the group mean; small `n` tends to the global mean. **EST:** the log tilt worked well; the formula he spoke is a log of a positive count against a total, which is less specific than a textbook weight-of-evidence ratio. **EST, parsed:** his own difference sits near 0 when the group is small and becomes large positive or large negative when the group is large, and that version worked. The near-0 behavior is what identifies an unnormalized difference. A mean of one row would be extreme, not near 0. **EST protocol:** recompute the map inside each outer training block on inner folds; a single global matrix leaks the outer evaluation fold into the training rows; for a new task, compute it honestly.

**Algebra status.** The speech does not leave a readable board equation. The shrinkage

`(n_c · group_mean + α · global_mean) / (n_c + α)`

is the linear form with the limits he stated. Tag that formula as a reconstruction of **EST** limits.

**Structural correspondence.** **ANA to regression, SPEC in the exact map:**

- `M`: the reconstructed shrinkage applied to the chosen target.
- `D`: the sum, over rows in the key, of `(y − global_mean)`. This is the behavior match to his difference: magnitude grows with evidence, and a rare key stays near 0. It is not a mean.
- `L`: `log((mass+ + β) / (mass− + β))` with `mass+` and `mass−` the summed positive and negative parts. **SPEC** formula, included because he named a log regularizer and the spoken ratio is incomplete. `β` is chosen only on inner folds.

Two variants of `y` enter the maps: metric-clipped at ±2, and unclipped. Clipping the training objective is still a separate choice. Each target has its own maps.

**Concrete intervention.** Split train sequences into an outer set of blocks once (P1). Inside each outer training block, split sequences again and fit `M`, `D`, and `L` for each key on the inner complements only. The map for an outer evaluation sequence is fit without that sequence. Unseen keys take the empty-group limit: global mean for `M`, 0 for `D` and for `L`. **ANA** of his rare-group rule to the `n = 0` case.

Serving is a frozen lookup from feature key to number. It does not read labels. Sequence reset is vacuous for a stateless lookup; a unit check confirms the table does not update at serve time. Past-label online updates are not in this recipe. They would require the inference callback to pass past targets, which this card does not assume.

The cheapest model is the lookup itself. He described that map as already a second-level model that is not a learned model. A neural student that tries to predict `y` from raw channels plus these columns is a successor.

**Assumptions.** Folds must be whole sequences. Random row folds, which are what P1 describes for BNP, would leak neighboring rows of the same sequence. That sequence rule is **ANA**, forced by our dependence structure. Bins or keys exist from E533-BNP-03 or, if that card admits nothing, from one pre-declared channel whose edges are fit inside the inner block.

**Risks.** A global map fit on all train sequences, then scored on the exposed split, is the leak he called out in P11. Using the current row’s label in its own map is a further leak. `α`, `β`, bin count, and key identity are selection; they stay inside the inner split. `D` and `M` will disagree on rare keys with large `|y|`, which is also where the official weight is large, so a mean-only encoding can hide the regularizer he said mattered. Level-2 blends that consume these columns still have the milder leak he described in P11; a successor blend has to rebuild the maps inside the outer block rather than reuse one matrix.

**Predicted observable.** On the outer train sequences, the official metric of three lookup predictors versus the constant global mean, for each target and for clipped versus unclipped inputs. Also the gap between an intentionally leaky map (fit with the outer sequences included) and the nested map. The leaky number is a diagnostic of P11, not a result to keep. A further observable: `|D − n·(M − global_mean)|` on rare keys, which measures the regularizer gap directly.

**Cheapest probe and control.** Tier B for one pre-declared key and both targets; tier D only if several keys from E533-BNP-03 are already admitted. Control: the same lookup after permuting the target across sequences inside the inner fit, which keeps key frequencies and the target marginal and breaks the key-target link. A second control is the constant global mean.

**Falsifier.** Under nested sequence splits, neither `M` nor `D` separates from the permuted-target lookup and from the global mean, on either target, on the outer train sequences, by the official metric. `L` is reported and does not rescue the recipe, because its formula is speculative. The leaky map may still separate; if only the leaky map separates, the transcript’s failure mode is reproduced and the feature is invalid. This does not retire the key itself as a non-target feature (E533-BNP-03).

**Parents and links.** Sources P8, P9, P11, P7, P10. Depends on a key, preferably from E533-BNP-03. Independent level-1 pack from E533-BNP-04 (features versus raw interactions). No parent in the campaign tree.

**Level-2 successors.** Univariate linear map from one numeric channel onto the target, the alternative P10 describes for numeric columns, fit inside the same nested sequence split, raw channel kept. A student fit on raw plus `M` and `D`, with the maps rebuilt per outer fold. A prefix-only running `D` inside the sequence, admitted only after a separate check that the official callback exposes past labels, with reset at sequence start. A blend with E533-BNP-04 under the same nested rebuild. Repeating the inner split several times, which P11 says they did, is a stability successor, not part of the cheapest probe.

## E533-BNP-06 — Prefix ordinal of a channel difference

**Source.** P13. **RET:** the winners subtracted two time columns, sorted by that difference, saw zeros and then ones, and added the row’s ordinal position in that order. P5. **EST:** never-missing columns and a distinct-count near 360 on three of them; he read them as days. P4. **EST:** denominators were computed in order to subtract columns, and those differences were not used.

**Structural correspondence.** Two different facts are easy to smear together. The ordinal is the winners’ feature (**RET**). The unused normalized difference is his own rudiment (**EST**). **ANA:** a global sort uses other rows and future rows, so it cannot be served. The servable object is the within-sequence prefix rank of a frozen pair’s difference, reset when the sequence id changes. The global sort may be computed on the train block as a diagnostic of whether P13’s phenomenon is present at all.

**Concrete intervention.** Candidate clocks, train block only: channels with no missing values whose distinct-count is stable across sequences. Do not threshold at 360. Candidate pairs: all pairs inside that clock set if it is small, otherwise a cap fixed in advance. If the clock set is empty, the pair list is a capped sample of raw pairs, plus pairs whose lattice steps agreed in E533-BNP-01 when that card admitted any.

For a pair, `d = x_a − x_b` on the current row. If both channels have an agreed denominator, also form `d_q = q_a − q_b` in code units (the rudiment). The served feature is the prefix rank of `d` among rows `1…t` of the same sequence, ties broken by the average of tied prefix positions, row index as the deterministic final tie break. The first row’s rank is the constant 1. State clears on sequence change. A unit check runs the same sequence twice in a row and requires identical ranks. The raw difference is served as well. The global sort index is not a feature.

**Assumptions.** If a pair orders the target, some of that order is visible inside a sequence from past and current rows. **SPEC.** If the order exists only across sequences, the diagnostic global rank will move and the prefix rank will not, and this servable recipe fails. Never-missing plus stable distinct-count is a candidate generator inspired by P5, not evidence that any local channel is a day count.

**Risks.** Choosing the pair as the maximum over many pairs on the exposed split is selection. The falsifier uses the maximum of an equally large null search. Prefix rank of a smooth ramp is almost a time index; the within-sequence permutation control is there to catch a feature that is only “how far into the sequence.” Lattice `d_q` inherits any wrong denominator from E533-BNP-01. Using future rows of the sequence to rank the current row breaks causality even inside the sequence.

**Predicted observable.** Train-block association of the prefix rank with each target, and the same association for the non-servable global rank. The useful pattern, if it exists, is a prefix-rank association that exceeds the null max. A global-rank association without a prefix-rank association is evidence that P13 is present only as a non-causal order.

**Cheapest probe and control.** Tier A for clock screening and a capped pair scan; tier B if monotone association is fit as a univariate slope. Score every candidate by a pre-declared rank correlation of prefix rank with the target on the train block, separately per target. Null: repeat the whole max-of-K selection after cycling the target across sequences, preserving each sequence’s internal target path and breaking pair-target alignment. A second null permutes `d` inside the sequence before ranking, which keeps the marginal of `d` and destroys order.

**Falsifier.** The selected pair’s prefix-rank association does not exceed the null maximum from the across-sequence target cycle, for both targets. The recipe serves no rank feature. If the global rank exceeds its own null and the prefix rank does not, record that as a non-servable analogue of P13 and stop this recipe anyway. A different pair family would be a new recipe.

**Parents and links.** Sources P13, P5, P4. Uses E533-BNP-01 only for the optional `d_q`. Independent of the target-encoding pack. A pair that also collapses support is a link to E533-BNP-03, checked after selection, not as a second search.

**Level-2 successors.** Prefix rank of `d_q` for agreeing lattice pairs only. Ranks of several admitted differences as a small set, each still reset per sequence. A teacher that sees the prefix rank beside the raw channels, including the case where the rank model is weaker alone and is only a blend input (P12’s blend fact, used here as a successor rule, not as a reason to skip the falsifier).

## How the six stay independent

E533-BNP-01 and E533-BNP-02 are label-free structure: a per-channel grid, and a same-row linear span. E533-BNP-03 is label-free conditional support. E533-BNP-04 is the raw-interaction pack. E533-BNP-05 is the label-derived pack, isolated by nested sequence folds, with the unnormalized difference `D` kept beside the shrunk mean because the transcript treats the regularizer as the operative detail. E533-BNP-06 is the causal ordinal of a difference, with the winners’ global sort kept as a diagnostic that is not served.

The cheapest informative path is tier A on E533-BNP-01, E533-BNP-03, and the clock scan in E533-BNP-06, then tier B on E533-BNP-02 and the single-key lookup in E533-BNP-05, then tier C on E533-BNP-04. Each falsifier retires that recipe’s emitted feature. It does not retire the other five, and it does not retire a listed successor that changes the representation.

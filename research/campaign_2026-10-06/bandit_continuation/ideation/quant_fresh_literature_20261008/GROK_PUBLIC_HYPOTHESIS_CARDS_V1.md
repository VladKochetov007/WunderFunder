## Research contract

This is a scout deliverable from native Grok 4.7, using `conduct-creative-research` and `research-reframe`. Objective mode is competition. Verification tier for these cards is C: structural analogies from one assigned source, with no fit run. Every card is `not_tested`. A later screen on the already inspected validation set is discovery evidence. It does not confirm public transfer, and it does not revive a fresh holdout.

No score definition is changed. The measured outcome for every later test is the frozen official scorer: fixed scored mask, predictions and labels clipped to ±2, weights from the original clipped-label magnitudes, globally weighted Pearson. Training losses may differ. Sequence state resets at each sequence boundary. Inputs at row t use only features at or before t, plus parameters frozen on training sequences. Labels are not inference inputs.

The assigned source was kept. Selection seed `202610080214`, pool of 351 PDFs, file then two uniform pages. Source SHA-256 `e1ea29818d0291de7b4aa4b05fe5141d7ef6b76969daf575844afe8c66c69a65`, passage SHA-256 `f332b83fb085b30f4d3062da9dad45388f59d03dfc803fae56dbe57774ebac54`, pages 4 and 10 of Quant Arb, “Small Trader Alpha - Another Real Strategy.” The extract quotes prose. The page-4 opportunity table is named and not numerically transcribed, so no table figure is used.

## What transfers from those two pages

The headline cross-exchange USDT strategy does not transfer. The source establishes operational facts about Binance, Poloniex, KuCoin, withdrawal APIs, and inventory. It establishes nothing about the 211 anonymous coordinates.

Four small components are still usable as roles, with diagnostics that can reject the roles:

- Page 4: capital in transit can arrive after the opportunity has vanished. Pre-positioned stock and in-flight stock are different states.
- Page 4: a very large apparent opportunity can be the toxic one, present because the venue is believed to fail.
- Page 10: a fill on one leg can remove the hedge on the other leg. The loss is sequential, not a simultaneous spread.
- Page 10: the same numerical gap matters when liquidity is sparse. Liquid-to-liquid trading is described there as tail inventory control. A status flag can disagree with a second observation.

Lenses used across the four families: change-time, re-encode, factor, invert, instrument, and mine-surprises.

Shared controls, frozen before any outcome is read: same sequence membership, same scored rows, same training rows, same architecture width, same step budget. The control receives the candidate state after a within-sequence circular shift of half the sequence length, so the parameter count and within-sequence marginals match while alignment breaks. One predeclared seed pair is a screen. It is not confirmation. If a declared slice has too few scored rows to estimate a direction, record `insufficient_evidence` for that slice.

## Card 1 — Delay expiry and inventory location

Mechanism label: `delay_expiry_inventory_location`. Lenses: change-time, re-encode.

Source component: page 4, waiting for USDT to arrive while the arbitrage disappears, versus keeping stock on the slow venue beforehand.

Analogy and limits: some coordinates may be on a slower update clock than their peers. A gap is usable when the slow coordinate is already fresh, and it decays while that coordinate stays stale and its peers move. This is a role for anonymous series. It is not evidence that any coordinate is a transfer, a balance, or a price. If every coordinate changes on essentially every row, there is no transit clock.

Causal update, reset at each sequence start:

- Past-only scale s_{t,j}: expanding median of |Δx_{u,j}| for u ≤ t, with a frozen train median until 32 past rows exist.
- Material move m_{t,j} = 1{|Δx_{t,j}| > 3 s_{t,j}}. The multiplier 3 is predeclared.
- Age a_{t,j} = 0 on a material move, else min(a_{t-1,j}+1, 256).
- Peers of j: the eight training-only increment-correlation neighbors, frozen, diagonal excluded.
- Stocked: fresh_{t,j} = 1{a_{t,j} ≤ 2}.
- Depleted: dep_{t,j} = 1{a_{t,j} ≥ 8 and a peer moves at t}.
- Expiry-weighted peer innovation: e_{t,j} = Σ_{k peer of j} w_{jk} Δx_{t,k} exp(−a_{t,j}/8), with w_{jk} the frozen train correlation.

The candidate augmentation is φ_t = (a_t/256, fresh_t, dep_t, e_t). The served state is φ only. The half-sequence shift of φ is the control.

Assumptions: update rates are heterogeneous; peer groups are sparse; stocked and depleted rows carry different target associations; ages are computable causally inside a 20,000-row sequence.

Observable direction: the official metric improves against the shift control, and the paired gain concentrates on scored rows with a high mean of dep_t. A gain that sits only on fresh rows supports the age channel and leaves the expiry channel unsupported.

Cheapest test: skip a neural fit. On the fixed membership, fit one predeclared ridge from (x_t, φ_t) and the same ridge from (x_t, shifted φ_t), training rows only. Report the official metric on all scored rows and on the depleted slice.

Falsifier for this recipe: the depleted-slice paired difference has the wrong sign or is flat, or fewer than about 5% of coordinates have a train material-move rate inside (0.01, 0.40). The second result means this dataset has no transit clock. Archive this recipe. Age summaries can still be reused as a compression of holding time if card 3 needs them.

Full-scale step and teacher path: one small complete-sequence recurrent student, identical budget, φ versus shifted φ. Curriculum is part of the treatment only in a second paired run: first half of the steps on low-age rows, second half on all rows, against a shuffled-order control with the same step count. A larger teacher may keep per-coordinate ages and a richer peer graph; the student receives only the sufficient statistics (a, fresh, dep, e) plus x. Distillation targets are the teacher’s two outputs, with ancestry declared. Sequence state resets in both.

Cost: the ridge screen is a small CPU analysis. One surviving recurrent screen is one short GPU pass under the small-data protocol. The teacher is a later admitted accelerator cohort.

Reusable artifacts: frozen scales, peer graph, age definition, depleted-slice membership hash.

Interaction child: `toxic_delay_gate`, with card 4. Pass e_t through only when stress and flag-disagreement are low. When they are high, zero the delay channel. That is the page-4 claim that the large opportunity may be the one created by impending failure.

## Card 2 — Isolated-shock inventory

Mechanism label: `isolated_shock_inventory`. Lenses: invert, factor.

Source component: page 10, sweep risk. One leg is hit, the hedge venue is then swept, and the exit is forced at a loss. The risk is described as small when the passive quote is wide and on a venue that is not sourced until liquid books are gone.

Analogy and limits: the two future targets can behave like sequential legs. An isolated move in one feature block leaves a committed state before any other block moves. A global jump is a different event. Nothing in the source shows that the competition targets are hedges of each other. The dual diagnostic below is the check.

Causal update:

- Freeze K = 4 blocks by train-only correlation clustering of increments, each block at least five coordinates. K is predeclared.
- Block shock z_{t,b} = median_{j in b}(Δx_{t,j} / s_{t,j}), reusing card 1 scales.
- Isolated event iso_{t,b} = 1{|z_{t,b}| > 3 and every other block has |z| < 1.5}.
- Inventory I_{t,b} = 0.9 I_{t-1,b} + iso_{t,b} z_{t,b}, reset each sequence.
- τ_{t,b} is the row count since that block’s last isolated event.

Invert diagnostic, train labels only, never a serving input: regress each target on iso_{t,b} z_{t,b}. The hedge-like signature is an opposite sign across the two targets for the same block, or opposite signs for the two targets on I_{·,b}.

Assumptions: shocks are sometimes isolated; the half-life 0.9 is a declared time scale, not a tuned one; evaluation after the shock uses the fixed mask and does not drop quiet rows.

Observable direction: official-metric gain against the shift control concentrates in the predeclared window of the next 8 scored rows after any isolated event. A global gain with no post-shock concentration supports a generic block summary, not sweep memory. A post-shock gain with a flat or worse global metric keeps the state as a conditional member.

Cheapest test: the same ridge protocol on (x_t, I_t, τ_t, iso_t) versus the shifted state. Publish the train dual-sign table before the valid metric is read.

Falsifier for this recipe: no opposite-sign pair appears in the train dual table, or the post-shock window matches the shift control. Stop this serving recipe. Keep the four block medians as a dictionary for card 3.

Full-scale step and teacher path: a recurrent teacher whose only added state is I and τ, then a student that sees those states as extra inputs. Judge the student by the official metric of the whole predictor. Also report the post-shock window so a rare-event contribution is visible. A weak full-sequence score with a real post-shock contribution admits the state into a gated mixture. It does not admit it as a standalone replacement.

Cost: ridge screen on CPU. Recurrent confirmation is one short GPU screen after the dual-sign table is non-empty.

Reusable artifacts: block dictionary, isolated-event timestamps, dual-sign table, inventory recurrence.

Interaction child: `tiered_sweep`, with card 3. Compute I inside activity tiers rather than correlation blocks. Page 10’s liquid-versus-illiquid sweep is the analogy: an isolated fast-tier jump followed by a slow-tier jump.

## Card 3 — Activity-tier quotient and sparsity-gated gap

Mechanism label: `activity_tier_gated_gap`. Lenses: factor, re-encode.

Source component: page 10, liquid-to-liquid gaps become large enough when size is huge or volatility spikes; the same spread on an illiquid venue is a different object. Page 4’s “two types, one favorite” supplies only the idea of a typed partition. The missing table contributes no rates.

Analogy and limits: coordinates can form update-rate tiers. A cross-tier gap can be predictive in sparse episodes and be noise otherwise. Tiers are activity classes, not exchanges. If within-tier residuals explain almost all variance, the quotient is only a rename of the raw vector.

Causal construction:

- Train material-move rate u_j and jump share define three frozen tiers by predeclared tertiles: slow, mid, fast.
- q_t^{(u)} is the median of robust-scaled levels inside tier u.
- Frozen train coefficient β_{uv} from past-only orthogonal regression of tier increments. Gap g_t(u,v) = q_t^{(u)} − β_{uv} q_t^{(v)}.
- Fast-tier energy RV_t = Σ_{j in fast} (Δx_{t,j}/s_j)^2.
- sparse_t = 1{RV_t exceeds the train 90th percentile of RV}. The percentile is frozen.
- Candidate channels, kept separate: tier medians, within-tier residual energy, g_t · sparse_t, and g_t · (1 − sparse_t).

Assumptions: tiers are stable between train and valid; one frozen β is enough inside a sequence; sparsity episodes are a minority of rows and still appear on the scored mask.

Observable direction: under the same training budget, the gated channel accounts for more of the official-metric gain than the always-on gap channel, with the difference larger in the high clipped-label-magnitude quintile. Matching the raw-feature model at a smaller input width is a cost result. Record it separately from a predictive claim.

Cheapest test: three ridges on identical rows: raw x; tier channels with both gap channels; tier channels with the sparse channel removed. Same penalty, same membership.

Falsifier for this recipe: dropping the sparse channel and dropping the quiet gap channel change the official metric by similar amounts, and the tier model loses to raw x on the full mask. This dataset then lacks a sparsity-conditional cross-tier gap. A later serving compression can still use the tier means if the cost result is real.

Full-scale step and teacher path: the teacher sees the raw coordinates plus the four channel groups. The student sees tier medians, residual energy, and the two gap channels. Distillation is a declared auxiliary target. The promotion comparison is the student against the raw-input student at equal step count, on the official metric.

Cost: CPU ridge lesion first. One GPU screen only if the gated channel wins the lesion on the high-magnitude slice.

Reusable artifacts: tier assignment, β, RV threshold, channel definitions.

Interaction child: `tail_inventory_control`, with card 4. Page 10 calls liquid-to-liquid trading a way to stop inventory building during tails. Treat the gated gap as a head that is allowed to be weak alone. Promote it only by its contribution to a stress-gated blend.

## Card 4 — Toxic edge and stale-flag disagreement

Mechanism label: `toxic_edge_stale_flag`. Lenses: mine-surprises, instrument.

Source component: page 4, a large opportunity is less attractive when it exists because the venue is believed to be failing. Page 10, a withdrawal-enabled flag can be wrong, so a second check is required.

Analogy and limits: the official weights emphasize large clipped-label magnitudes, so a rare joint jump can dominate the score. A second channel is a persistent coordinate that claims a state while its associated flow does not. These are instruments for anonymous vectors. A binary-looking coordinate is not an API. A large residual is not a venue failure.

Causal instruments:

- Stress Σ_t = mean_j m_{t,j}. High stress is Σ_t at or above the frozen train 95th percentile.
- Flag set: train persistence P(x_{t,j} = x_{t-1,j}) ≥ 0.98, at least two distinct values, and at most twelve distinct values. If this set is empty, the flag assumption is already rejected.
- For each flag, bind the non-flag block with the largest train |correlation| between flag changes and block shocks.
- Disagreement D_{t,j} = 1 when the bound block’s |z| > 3 and the flag is unchanged, or the flag changes and |z| < 1.5. D_t is the mean over flags.

Linear screen, still the cheapest measurement: a predeclared ridge on tier medians, Σ_t, and mean age. The candidate adds Σ_high and D_t interactions with the tier gaps from card 3. Fit on train rows. Evaluate on the same valid scored rows.

Assumptions: stress episodes and ordinary episodes have different feature-to-target maps. Flag persistence can be estimated without knowing meanings. The high-magnitude quintile is where a weight-relevant effect appears.

Observable direction: the interaction ridge improves the official metric inside the top quintile of |clipped label|. A further signature, read from the train fit before the valid number, is a sign change on the same gap coefficient between high-Σ and low-Σ rows. Improvement only in the low-magnitude quintile can still be a useful auxiliary teacher target. It is a poor reason to replace the served head, because the official weights sit on the large labels.

Diagnostic when the meaning assumption is wrong:

- The flag set is empty, or D_t is almost always 0 or almost always 1.
- High-Σ rows are almost all rows or almost none.
- Surprise audit: take scored train rows where the baseline ridge has the wrong sign and |clipped label| is in the top quintile. If those rows do not concentrate in high Σ, high D, or depleted-and-large-gap cells from card 1, the venue-failure story missed the anomaly. The dominant cell of that table becomes the next arm’s parent, and these four role names stop guiding allocation.

Falsifier for this recipe: the high-magnitude slice is flat or worse, and the train gap coefficient keeps the same sign in both stress bins. Stop the two-head predictor. If D_t is stable and related to feature self-inconsistency while being unrelated to labels, keep D_t as a data-quality instrument and drop it from the predictor.

Full-scale step and teacher path: two independently initialized recurrent teachers. The ordinary teacher is fit on low-Σ rows. The tail teacher is fit on high-Σ rows, with inputs (gated gaps, I_t from card 2, D_t). The served blend is a frozen function of causal Σ_t only. A larger pair can be distilled into one small causal student that receives x_t, Σ_t, and D_t and resets per sequence. The tail teacher is promoted by the blend’s official metric against the ordinary teacher alone, including when the tail teacher’s own metric is worse.

Cost: the ridge, the flag census, and the surprise table are CPU work and should run first. A two-teacher screen is justified only after a high-magnitude interaction or a sign change.

Reusable artifacts: flag list, flag-to-block bindings, Σ threshold, disagreement series, surprise-cell table, separate ordinary and tail predictions.

Interaction child: `tail_inventory_control`, shared with card 3. The tail head consumes card 3’s sparse gap and card 2’s inventory. Its standalone score is not the allocation statistic.

## Comparative allocation

Run the assumption screen before any accelerator fit. It is one CPU pass that emits the age histogram, tier residual share, isolated-shock rate, flag census, D_t rate, stress-bin coefficient signs, and the surprise-cell table. All four recipes read those artifacts.

| Order | What earns the next budget | What changes the allocation |
| --- | --- | --- |
| 1 | Card 4 ridge and the shared census | Empty flags, degenerate Σ, or surprises outside stress, disagreement, and depleted-gap cells: stop treating the source roles as the search space. Follow the largest surprise cell as a new parent. |
| 2 | Card 1 ridge if ages are non-degenerate | A gain only on fresh rows: keep age, drop expiry, and spend the recurrent screen on age alone. A flat depleted slice: archive this recipe and leave the peer graph for card 3. |
| 3 | Card 3 lesion if tiers compress | Gated and quiet gaps tying: archive the sparsity claim. If tier features match raw x at lower width, keep them as a student-cost arm with a predictive claim withheld. |
| 4 | Card 2 only after a non-empty dual-sign table and a non-trivial isolated-shock rate | Post-shock gain with a weaker full-mask score: move I_t into the tail blend. A failed dual-sign table: archive the sweep recipe and keep the block dictionary. |
| 5 | One interaction, `tail_inventory_control`, if both a gated gap and a stress sign change survive | Judge the blend against the ordinary head on the official metric. A weak tail-only score with a positive blend delta keeps the tail head. A weak blend delta archives the child and leaves the parents alive. |

If the census says ages, tiers, isolated shocks, and flags are all degenerate, these four analogies do not bind to this dataset. That result retires these recipes. It does not retire delay, mixture, or rare-event ideas under a different representation. The next scout should start from the surprise-cell table rather than from another exchange-arbitrage headline.

No public-score effect is forecast. No numerical prior is assigned. A low official score on one head is a reason to move that head into a conditional target or to stop that recipe. It is not, by itself, a reason to mark the mechanism family false.

## Handoff

The future lead actor owns a CPU screen only: frozen definitions above, identical membership, official scorer untouched, artifacts listed per card. This scout does not own that execution, does not approve promotion, and does not grade the result. A grader would separately report validity, full-mask official metric, high-magnitude slice, declared rare-event slice, and training-seed variability. One seed pair remains provisional.

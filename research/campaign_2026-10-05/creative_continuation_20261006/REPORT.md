# Creative continuation — 6 October 2026

Read-only continuation. No fit, upload, ledger edit, GPU job, or notebook change.
Verification tier **C**: the external metric is a public score; local validation is exposed and partly fitted. Confirmed public best is **0.6660** (complete seed-4 Muon / chunk1000 / stopped-bootstrap GRU256, portal submission `2e5j6bnt`, 37m07s / 626 MB). Desired milestone **0.7000** is unmet. Nothing below is an expected public score, and no local delta may be added to 0.6660.

Metric `global_wp_clip2_v1`: clip targets and predictions to ±2, weight by `|clip(y)|`, pool selected rows (`need_prediction AND is_scored`), centered weighted Pearson per target, then average t0 and t1. [METRIC.md](../../../METRIC.md), [src/wnn/gmetric.py](../../../src/wnn/gmetric.py).

## 1. Why the search is stuck

The loop is dense in fits and sparse in identifiable complete-predictor feedback.

**The object being optimized is not the object being scored.** Of 236 registered activation slots, 224 were graded, almost all as one-epoch models on the same 24 fitting and 16 historically exposed evaluation components ([ACTIVATION_RETROSPECTIVE_2026-10-05.md](../ACTIVATION_RETROSPECTIVE_2026-10-05.md)). The separate sequence screen is 30 width-32 fits with absolute pooled WP about 0.44–0.55 ([THREE_SEED_REPORT.md](../sequence_components/THREE_SEED_REPORT.md)). The public predictor is a two-layer GRU256 plus teachers, EMA, and a frozen-trunk global-correlation head. A +0.005 move on the H64 panel is not a draw from the same functional as 0.6660. [EXECUTION_AND_EVIDENCE_AUDIT.md](../review_20261005/EXECUTION_AND_EVIDENCE_AUDIT.md) records the only external seed contrast that exists: seed-4 minus seed-3 is **+0.001872** on the excluded local panel (component interval includes zero) and **+0.0005** publicly. That is the transfer noise floor of this campaign. Screens smaller than that floor cannot be ranked.

**The dense local number is a different sample.** The supplied validation file has 1,873 sequences; 1,505 entered final fitting and 368 were excluded ([VALIDATION_SCORECARD.md](../../campaign_2026-10-04/VALIDATION_SCORECARD.md)). Seed-4 scores 0.700771292 on the full file and 0.701694557 on the excluded 368, against public 0.6660. Previous D256 scores **0.700792587** on the full file and **0.6641** publicly, so the ~0.035 local/public level gap is not a missing in-sample fit. The excluded 368 are historically exposed and component-linked ([strong baseline report](../target_mechanics/strong_baseline/REPORT.md): 16/16 parent and 15/16 head links on its row-excluded panel). There is no fresh holdout hiding in a renamed row subset.

**Attribution is bundled.** The public parent endpoint ties the old D256 package at 0.6641. [PUBLIC_RESULTS_REFLECTION_20261004_LATE.md](../../campaign_2026-10-04/PUBLIC_RESULTS_REFLECTION_20261004_LATE.md) states that this does **not** isolate Muon, chunk1000, or bootstrap. Local factorial pieces exist (E-352, E-353, E-354, E-356 in [experiments.jsonl](../../experiments.jsonl)) but were not the submitted contrasts. Chunk size simultaneously changes step count, EMA exposure, and the chunk loss denominator (`loss_norm: chunk`, `distill_denominator: chunk` in [seed-4 provenance](../../../artifacts/runs/c20261004_combo_s4_final/provenance.json)). E-430, the cadence separation, is still drafts plus a synthetic check, not a fit.

**The public instrument is quota-limited.** Rules allow 5 submissions per day ([docs/rules.md](../../../docs/rules.md)). The late public reflection counts 24 successful own historical submissions. Selection has been on the exposed local functional; the external functional is a rare confirm.

**A provenance footgun makes “we already train the metric” false for the trunk.** Seed-4 parent `loss` is `wmse` while `gcorr_weight` is 1.0. In [scripts/train_rnn.py](../../../scripts/train_rnn.py) the `wmse` branch returns only weighted MSE; `gcorr_weight` is applied only when `loss == "wmse_gcorr"`. The head is a separate `loss: gcorr` epoch ([head provenance](../../../artifacts/runs/c20261004_combo_head_s4_final/provenance.json)). Reviews that treat another activation on the 16-component panel as a new mechanism are changing a surrogate on the wrong object.

The ab85 “overlooked 0.704” claim is not a stuck-search exception. It is leakage. See §6.

## 2. Mechanism ledger

Verdicts are about the tested treatment. “Scope” says whether a later review was entitled to close the whole family.

| Mechanism | Verdict | Evidence | What was actually rejected |
|---|---|---|---|
| Stopped-bootstrap | helped_local_only | E-353 folds 0+1, parent+head, pooled delta 0.001173 (sd 0.000803), “weakly positive; keep for the combination” ([experiments.jsonl](../../experiments.jsonl)). Inside the public 0.6660 bundle, but [public reflection](../../campaign_2026-10-04/PUBLIC_RESULTS_REFLECTION_20261004_LATE.md) refuses individual public credit. The DTW note called h=8 “unbiased” at bias +0.0001 from a per-class projection on `[1, parent, teacher]` ([dtw/objective/REPORT.md](../../campaign_2026-10-02/dtw/objective/REPORT.md)); that is not a theorem. | One alpha-0.5 / horizon-8 screen helped locally. The unbiasedness veto of other teachers was too broad. The public knob was not isolated. |
| Final-head procedure | helped_public | Seed-3 parent 0.6641, head endpoint 0.6654 (**+0.0013**), equal parent/head 0.6655 (**+0.0001**, not a weight lesson). Same file as above. E-410 complete-recipe CV, which includes the head, is +0.002414379, CI [+0.001565628, +0.003448539] ([VALIDATION_SCORECARD.md](../../campaign_2026-10-04/VALIDATION_SCORECARD.md)). | The procedure helped this recipe publicly. The 0.0001 blend increment did not justify retuning the merge. |
| Muon | helped_local_only | E-352 two-fold pooled delta 0.001595, CI [0.000252, 0.003013]. E-349 fold-1 epoch-2 screen was null. Public isolation: none (reflection above). | This optimizer screen. Not “optimizers do not matter,” and not a public attribution. |
| chunk1000 | helped_local_only | E-354 epoch-2 screen `[0.001375, 0.000731]`, described as CI excluding 0. E-356 combo-minus-Muon pooled CI [−0.000312, +0.001354] includes 0, so chunk+bootstrap on top of Muon is not a clean second gain. Cadence confound: §1. | A confounded shorter-chunk screen. Not a proof that longer credit is the cause. E-430 remains unfitted. |
| GELU/LeakyReLU mixture | helped_local_only | E488 epoch-3 fixed 50/50 minus GELU: +0.0089295, +0.0043077, +0.0086897 on three H64 seeds; recurrent-minus-static near 0 ([ACTIVATION_RETROSPECTIVE](../ACTIVATION_RETROSPECTIVE_2026-10-05.md)). No complete GRU256 arm. | Not rejected. Not transferred. Local magnitude is not a public forecast. |
| Sine activation | null | Positive sine-minus-GELU point estimates in E470/E472, but E473’s linear control and E481’s center/slope-matched cubic moved with sine. E476’s large schedule gain reversed in all eight E480 contrasts. | Periodic curvature as the cause. A bounded slope/amplitude candidate was not the same hypothesis and was not given a complete-predictor test. |
| Learned diagonal decay / splines | helped_local_only | Learned-minus-fixed decay +0.003630 / +0.003816 / +0.003043, all three conditional CIs positive. Spline-minus-MLP on that SSM narrowed from +0.038616 to +0.012520, last CI includes 0. Absolute WP still ~0.47–0.52 versus a small GRU, not versus GRU256 ([THREE_SEED_REPORT.md](../sequence_components/THREE_SEED_REPORT.md)). | The tiny diagonal SSM, not “memory cannot help a strong model.” Splines are seed-fragile on this panel. |
| Positive-kernel attention | null | `ssm_attention_mlp` minus pointwise: −0.036404, −0.020306, +0.006302. GRU attention mixed across seeds. Same report. | Additive positive-kernel memory on width 32. Not addressed erase/write or event-keyed retrieval. |
| Joint codebook head | null | E468 joint tilted −0.000618 versus parent; regression anchor −0.000555; independent marginals −0.001354. All arms, including the anchor, saw auxiliary CE ([joint REPORT](../joint_target_model/REPORT.md)). | One frozen historical GRU128, 32-atom, one-pass head. Not joint dynamics on the live trunk. The losing anchor means this is a weak family negative. |
| Target-specific EMA clocks | null | E426 aligned minus constant **−0.000045927**, CI [−0.000164898, +0.000071328], one fold/seed ([E426_CLOCK_RESULT.md](../../campaign_2026-10-04/E426_CLOCK_RESULT.md)). | Teacher-consistency EMA timing. Not a causal input/history adapter. Production `ev_i0px` is the AND of bid and ask changes ([features.py](../../../src/wnn/features.py)); the strong probe’s event is the OR of `i0_p0` and `i0_p11` ([strong_residual_probe.py](../target_mechanics/strong_residual_probe.py) lines 201–202). |
| Volume input E413 | null | Delta +0.00096179, CI [−0.00007997, +0.00205967], one exposed fold/seed (E-413). Point estimate positive, interval includes 0. | Not a win and not a family kill. Run-relative volume history versus current-input control only. |
| Width 384 | execution_censored | E-350 one fold, W384 WP 0.679214 versus S100s3 0.677206, contrast `[0.002009, 0.000653]`. E-367 width **320** pooled +0.000072, CI includes 0. W384 v7 probes passed; both paired fits stopped in epoch 0 on the host RSS guard; no checkpoint or score ([EXECUTION audit](../review_20261005/EXECUTION_AND_EVIDENCE_AUDIT.md)). | Nothing scientific. A RAM guard is not a width null. |
| INT8/QAT | null | E419 recovered recurrent QAT minus FP32 continuation **+0.000004823**, CI [−0.000021980, +0.000032119] ([E419 report](../../campaign_2026-10-04/E419_RECOVERED_QAT_REPORT.md)). E-411 fixed q8 deltas about −2e-5, CIs include 0. Public q8 ZIP `1yv88h4s` failed `libpython3.13` linkage before scoring; that censors deployment, not accuracy ([public reflection](../../campaign_2026-10-04/PUBLIC_RESULTS_REFLECTION_20261004_LATE.md)). | Accuracy preservation on one fold is measured and null as an improvement. Ensemble-enabling runtime was not measured. The withdrawn ZIP was an ABI failure. |
| EMA6 public submissions | hurt | `final_mixed_ema6_shared` public **0.6616** (`246401vj`); `final_ema6_shared` public **0.6612** (`1ykwb95u`); then-current since pair **0.6622** ([registry.json](../../lb/registry.json)). Earlier EMA-4 LP-FT public 0.6618 was a different package and did move the board in its generation. | Those submitted EMA6 blends versus the since pair. Not “averaging never helps.” |
| Online norm | null | E-118 per-sequence z-score “adds nothing.” E-185 partner positive on fold 0, negative on fold 1. E-294 merged-head onorm minus SP2 `[0.000012, 0.00039]`. No `onorm` entry in the public registry. Seed-4 provenance has `online_norm: false`. | This input view as a second member. Not a zero-cache incident dressed up as a mechanism law; the scored arms are the evidence. |
| Cross-member blend of near-copies | null | Equal seed-3/seed-4 bag: full file 0.701106873 (above seed-4 0.700771292) but excluded 368 **0.701420130 < 0.701694557**. Unsubmitted ([scorecard](../../campaign_2026-10-04/VALIDATION_SCORECARD.md)). The excluded slice is the one that ranked seed-4 above D256, matching the public order. | This near-copy bag. Not every pair of dissimilar checkpoints. |
| Original-target privileged teacher | never_tested | Seed-4 `privileged_mode: none`. E-251 was a future quote-path **auxiliary** and lost (pair delta −0.001435). The DTW report’s “cross-fitted teachers unnecessary” used the projection in the stopped-bootstrap row, not an origin-label teacher fit. | The quote-path auxiliary, and an over-broad theoretical veto. The origin-row teacher itself has no fit. |
| Event-hold | hurt | Complete predictor 0.774416 → hold 0.692929 on the fitted 16 (Δ **−0.081486**); 0.742757 → 0.666532 on the row-excluded 16 (Δ **−0.076225**). State still updated; outputs were frozen ([strong report](../target_mechanics/strong_baseline/REPORT.md)). | That hard output hold. Not a learned quiet-row update. Both panels are exposed and graph-linked. |
| Specialist routers (E464) | null | Eight final heads and 16 complete contrasts; none beat the seed-3 combo on the exposed panel. Within-family specialist minus GRU **−0.000461**, CI [−0.001956, +0.001507] ([direction audit](../../../reflect_05_day/RESEARCH_DIRECTION_AUDIT_2026-10-05.md) citing `e464_report_all_fp32_v1.json`). Route freezes after 99 rows; every expert still steps every row ([specialist.py](../../../src/wnn/specialist.py) via that audit). | Prefix-frozen routing. Not dynamic event allocation. |

## 3. Ideas rejected for nothing

“Rejected for nothing” means the sentence that closed the idea does not test the treatment that is still open. The October 5 direction audit already reopens joint-head controls, target-specific input histories, E-244’s direction auxiliary, E413, the origin teacher, the distilled normalizer, and width. Those are not repeated as if they were new. Two items below are in neither the elephants **priority table** ([ELEPHANTS_REVIEW_2026-10-05.md](../../ELEPHANTS_REVIEW_2026-10-05.md) § “New directions”) nor the direction audit’s **ranked list** (priorities 1A–5). Two more are also absent from both lists. E-244 is included only as the audit’s own uncertain stop, not as a discovery.

### 3.1 Train-decoded queue depletion, not raw volume-code shocks

Absent from both the elephants priority table and the direction-audit ranked list. E413 (already in that elephants table) is a different input.

Closing evidence, E-215 in [experiments.jsonl](../../experiments.jsonl):

> “a raw 22-column i0 volume-code displacement model also fails to add robust event-level direction information beyond the frozen pair. This does not rule out train-decoded queue refill/depletion intensities; it rejects the cheap raw-code 4/16-event shock route.”

Measured: base-pair direction AUC `[0.665461, 0.641088]`, with 4- and 16-event volume shocks `[0.625186, 0.631678]`, AUC deltas **−0.040275** and **−0.00941**. The gate that fired was “AUC gain at least +0.03 and residual correlation at least +0.08 in both folds.”

Still open: intensities decoded on training codes only (signed touch depletion / refill between causal i0 events), residualized against the incumbent’s existing `last_d_i0px*` and `ev_i0*` features, scored as a frozen correction on excluded components. Not another raw 22-column shock, and not E413’s run-relative volume channel.

### 3.2 Pooled correlation through the recurrent trunk, not per-chunk `corr`

Absent from both lists. The final head already uses `loss: gcorr` on a **frozen** trunk. That is not this treatment.

Closing evidence, E-023 / H-018:

> `corr` epoch 0/1 **0.2937 / 0.2575** versus mix **0.6036 / 0.5983**. Hypothesis note: “Per-chunk correlation is degenerate when class-3 targets are constant within a 2000-row chunk.”

The champion parent never left that failure mode’s cousin: `loss: wmse` ignores the stored `gcorr_weight: 1.0` ([train_rnn.py](../../../scripts/train_rnn.py) lines 1801–1812). `loss: wmse_gcorr` and `loss: gcorr_pool` exist in that function and are not the seed-4 parent.

Still open: one trunk comparison of `wmse` versus `wmse_gcorr` or chunk-accumulated `gcorr_pool`, same teachers, same final head. Kill it if the correlation term only rescales the head (dispersion moves, trunk features do not) or if constant-target chunks still produce a zero denominator. Do not cite E-023 as evidence against that arm. Also do not expect a new Bayes estimator: under an unconstrained model, weighted MSE and weighted Pearson share an affine-optimal conditional mean. Any gain has to be finite-sample, which the strong report already shows is possible (the head raises WP and weighted MSE at once).

### 3.3 Input-only lag-4 gate, not the corner-driven ridge

Absent from both lists. [puzzle_b/REPORT.txt](../../campaign_2026-10-03/eda/puzzle_b/REPORT.txt):

> full4 delta versus cap4 **+0.007909** (full4 t0/t1 0.451942/0.441884). “Excluding both reduces delta +0.007909 to **+0.000613**. No broad improvement claim survives this concentration check.”

That sentence killed a universal lag. The same file’s unrun next test is different: “Input-only lag4 recurrence gate on genuinely reserved components… lag7 sham and phase-only controls. Kill if benefits fail outside rg54/1385-like components or t1 harm offsets t0.” A later audit correctly says do not reopen a period scan ([MECHANISM_AUDIT.md](../review_20261005/MECHANISM_AUDIT.md)). The sham-controlled residual on new components was not that scan and was not run.

### 3.4 Cointegration residual, not univariate i1 AUC

Absent from both lists. E-017 closed H-013 with class-3 best AUC **0.525** (21–100 rows) and **0.528** (101–1000), under a falsifier of univariate AUC < 0.55. The hypothesis note in [hypotheses.jsonl](../../hypotheses.jsonl) says the encoding was sign-cumsum and “a proper cointegration residual on reconstructed levels is not yet tested (H-015).” H-015 is still `probing` on a two-sequence-per-class reconstruction (E-015, best sign-cumsum r about 0.53–0.61), not a residual-on-champion test.

Still open only as a pre-fit partial correlation: train-only reconstructed level, causal deviation from its own EMA, Frisch–Waugh against the incumbent’s price-difference features, then weighted correlation with the frozen seed-4 residual on excluded rows. A second AUC horse-race on raw i1 moves is not open.

### 3.5 Direction auxiliary, stopped 0.000032 short of its own gate

This one **is** in the direction audit’s reopening table and is listed so that table is not the whole of §3. It is not in the elephants priority table and not in the ranked 1A–5 list.

E-244: single-model epoch-3 gain **+0.000656**; pair delta **+0.000468** ± 0.000712; predeclared minimum **+0.0005**. Status text: “Stop before fold 1; do not tune CE weight on this fold.” That is a stopping rule on one fold, not a negative result. Reopen only inside a predeclared relational or teacher arm, not as a CE-weight search. The 0.000032 shortfall is smaller than the transfer noise in §1, so a tiny-panel rerun would still be the wrong resolution.

## 4. New directions

These four are not transfers of the fixed GELU/LeakyReLU readout, not an event/quiet residual cell, not an LRU/spline branch, not a repaired joint-head control, and not a privileged origin teacher. Gated delta and LRU are already named in the 5 October elephants review and are not reused.

### 4.1 Invert — what would make +0.034 possible

**Card.** Representation change: treat 0.7000 as a claim about remaining information, not as a sum of pilot deltas. Mechanism: after the type-A scorer repair, recorded improvements are variance-scale, while the local/public level gap is the size of the milestone and is not removed by a locally better model. Assumption: no further deterministic scorer mismatch of type-A size is hiding in the opened records. Breakpoint: a real mismatch, or a new observable whose complete-predictor public move is an order of magnitude above E-410, would reopen the milestone.

**Numbers already on file, and only those.**

| Quantity | Value | Source |
|---|---:|---|
| Public best | 0.6660 | portal reflection |
| Milestone gap | 0.0340 | 0.7000 − 0.6660 |
| Seed-4 full supplied validation | 0.700771292 | scorecard |
| That level minus public | 0.034771292 | arithmetic on those two |
| D256 full validation / public | 0.700792587 / 0.6641 | scorecard, registry |
| D256 level gap | 0.036692587 | arithmetic |
| Type-A public repair | 0.5935 → 0.6503 (+0.0568) | [game_changers.md](../../game_changers.md) |
| Complete-recipe CV delta | +0.002414379 | E-410 |
| Data-doubling local mean | +0.0015 | E-350 |
| One-fold W384 local contrast | +0.002009 | E-350 |
| Public seed-4 minus seed-3 | +0.0005 | reflection |
| Mix calibration | “no mix fixes the slope”; baseline residual −74e-4 | E-301 |

Algebraic range of the metric is [−1, 1] ([METRIC.md](../../../METRIC.md)), so 0.700 is not forbidden by the formula. No file opened here states a mutual-information or oracle-Bayes ceiling. That ceiling is **unknown**. Class WP and difference-mode energy are not additive pieces of the official score: 84.3% / 85.4% of a residual-energy diagnostic sits in the difference mode, and that fraction is not a WP budget ([strong report](../target_mechanics/strong_baseline/REPORT.md)).

What the numbers do rule out is the story “the model is 0.701 locally, so 0.034 of public score is sitting in the training loss.” D256 is locally as high and publicly worse. E-301 already says reweighting classes does not remove the level gap. Post-alignment public steps in the registry are order 0.001 (0.6618, 0.6622, 0.6641, 0.6660). A complete-predictor gain the size of E-410, even if it transferred 1:1, is about one fourteenth of 0.034.

**Cheapest falsifier.** Do not fit. From the existing seed-4 cached predictions and the published class-3 mix interval about 0.41–0.51 in E-301 / [class mix note](../../campaign_2026-10-01/REPORT.md), recompute pooled WP under those mixes only. If the range is a few thousandths, composition cannot be the missing 0.034 (E-301 already points this way for an earlier package; repeat on the frozen champion before citing it as the champion’s gap). **Kills the “milestone is close” claim** if that range and every later complete-predictor paired delta stay O(0.001–0.003). **Kills this impossibility claim** only with a public score ≥ 0.676 on a pre-registered package that uses no future labels. A local +0.01 does not kill it.

Cost class: **probe**. Not a duplicate of “add E488 to 0.6660.”

### 4.2 Factor — which case holds the champion’s residual

**Card.** Representation change: partition the official numerator, not the network. For the frozen seed-4 complete predictions, on selected rows, and separately for t0 and t1, attribute

`w_i (y_i − ȳ_w)(p_i − p̄_w)`

using the **global** weighted means (so cells sum to the covariance numerator), crossed with a deployable clock cell and an age bucket. Clock cell from consecutive raw rows: `(Δi0_p0, Δi0_p11) ∈ {00, 10, 01, 11}`. Age buckets: 0, 1–7, 8–31, 32–127, 128+, counted from the last OR event, and again from the last AND event (`ev_i0px`). Report weight share and covariance contribution. Class labels may be tabulated only as a non-deployable diagnostic; they must not define the arm.

**Mechanism.** The strong 16-group diagnostic already shows the miss is not uniform: pooled WP 0.8496 at event age 0 versus 0.4728 at 128+, but class 2 is 0.8781 / 0.9234 while class 3 is 0.4108 / 0.3325. Quiet class-3 updates reduce weighted squared error by 0.01394 / 0.00822 per unit weight on the fitted panel. Excluded-368 champion t0/t1 are **0.687627604 / 0.712016713**, so on that slice t0 is the weaker target. Scoring weight is almost equal across targets (E-204: 0.497078 / 0.502922), so “t1 needs the capacity” is not a loss-mass fact. Assumption: a cell that holds little weight cannot produce a +0.034 pooled move no matter how good a specialist is. Breakpoint: if the deficit is spread across cells, or lives only in a target-derived class with no input correlate, there is no deployable case split.

**Cheapest falsifier.** The table itself, CPU, cached predictions, no training. **Kills a case-specialist** if the worst deployable cell holds well under 15% of `|clip y|` weight, or if its covariance deficit per unit weight is no worse than the complement once components are resampled. **Kills “the residual is the event row”** if age-0 cells are not the deficit. This is a partition of an existing predictor, not an event/quiet network.

Cost class: **probe**.

### 4.3 Instrument — observable before any fit

**Card.** Two counts, same pass as §4.2, still no fit.

1. Contingency, scored destination rows only (`need_prediction AND is_scored`): label-pair changed or not, by clock cell `{00, 10, 01, 11}`. The 3,761/3,763 figure is the OR clock. It does not say whether production’s AND flag sees those changes. If nearly all label changes are cell `11`, the OR-versus-AND argument is empirically empty and should be dropped. If a material share is `10` or `01`, the incumbent’s `ev_i0px` is the wrong event bit even though the raw columns are present.

2. `a6` versus that clock. `a6` is a named column in [puzzle_a/scan2.py](../../campaign_2026-10-03/eda/puzzle_a/scan2.py). On the same scored rows, agreement of `{a6[t] ≠ a6[t−1]}` with OR and with AND, plus the label-change rate in the symmetric difference. [INDEPENDENT_ELEPHANTS](../../../reflect_05_day/INDEPENDENT_ELEPHANTS_2026-10-05.md) flagged this identity as unverified. It is still not a number in any file opened here. Do not assume they match.

**Mechanism.** A wrong event bit is a measurement error in every later “event model.” Fixing the bit is cheaper than fitting a new cell. Assumption: `a6` and `i0_p0`/`i0_p11` are the columns those scripts already use. **Kills the instrument** if agreement with AND is essentially perfect and label changes outside cell `11` are at the known exception count (two fitted class-1 rows, one excluded, on the 16-group panels — not yet the full 368). **Kills “add event features”** if the new bit is a function of features already in `FEATURE_NAMES`.

Cost class: **probe**. Not a model.

### 4.4 Transfer — multivariate Hawkes compensator, not gated delta and not LRU

**Card.** Representation change: five causal compensators, one per observable mark `{p0-only, p11-only, both-prices, i0-trade, i0-volume}`, each a recursive sum `Λ_k(t) = e^{−β} Λ_k(t−1) + 1_{mark k at t}` with β fixed from the training-fold inter-event age histogram (one β, not a grid). Feed `Λ(t)` only as extra inputs. Mechanism, from mutually exciting point processes (Hawkes): the hazard of the next target-relevant jump depends on which marks just occurred, on an event clock the row-wise GRU has to store in hidden state. This is not a binary event/quiet gate, not a learned diagonal SSM, and not a gated delta rule `v − Sk`. The structural correspondence is that both the book and a Hawkes process are marked point processes whose sufficient statistic is the compensator, not the raw row.

Assumption: linear excitation in these five marks is enough to show up as a partial correlation; a neural intensity is a different, later hypothesis. Breakpoint: if the compensator is collinear with `since_i0px` and `ev_i0px`, the GRU already has it.

**Cheapest falsifier (before the fit).** Frisch–Waugh: residualize each `Λ_k` on `{age_OR, age_AND, ev_i0px, since_i0px}` using fitting rows only. On excluded components, weighted correlation of those residuals with the frozen seed-4 residual `(y − p)`, per target. Placebo: the same statistic after shuffling mark times inside the sequence (preserve counts, destroy excitation). **Kills the hypothesis** if every partial correlation’s component interval covers 0, or if the shuffled placebo is as large. Only a survivor is allowed a complete one-seed fit (§5).

Cost class: the falsifier is a **probe**. The follow-up, if it survives, is **one complete fit**. Not a duplicate of positive-kernel attention (E467/E469), which accumulates a learned kernel and already went null/negative on the small panel.

## 5. Priority stack for the next window

Maximum three experiments. Ranked by information about a **complete-predictor** delta per GPU-hour. None of their local results may be added to 0.6660. A local win does not create an expected public 0.7000.

### P1 — CPU attribution and clock contingency (do this first)

Run §4.2 and §4.3 on cached seed-4 predictions. Fit nothing. Pre-declare the promotion rule before reading the table: a deployable cell is eligible for P2 only if it has at least 15% of that target’s `|clip y|` weight on the excluded 368 **and** a component-resampled covariance deficit per unit weight worse than its complement. If no cell qualifies, **stop the window**. Do not spend the GPU budget on a substitute sweep.

Information: either locates the residual the champion actually misses, or shows that the milestone is not a case split. GPU-hours: 0.

### P2 — one complete seed, and only if P1 names a cell

One seed of the current recipe (Muon, chunk1000, stopped-bootstrap, same teachers, `include_heldout_valid: false`, same final-head merge). Change only the inputs named by P1. Default candidate if the quiet or single-sided cell wins: the five Hawkes compensators from §4.4. Control: the same five series with marks shuffled inside the sequence, parameters and head matched. If P1 instead says the deficit is collinear with existing event features, the candidate is the decoded depletion intensity from §3.1, not a second Hawkes fit. Save parent and complete scores. One seed cannot be promoted. A second seed is the next window, not this one.

Information: a paired complete-predictor delta aimed at a measured cell, with a placebo control. Cost class: **one complete fit**. Expected local size is unknown and must not be forecast from E488 or E-410.

### P3 — one-CPU INT8 timing of the existing seed-4 graph

No training. Time the Python-3.11-repaired INT8 engine on the seed-4 weights under the submission callback rules, one CPU. The scientific question is whether a second member can fit in the residual budget after 37m07s of 60, not whether q8 matches FP32 (E419/E-411 already say accuracy is preserved within noise on their panels). If the timed parent is not comfortably under the residual minutes with a margin you would actually ship, the ensemble-enabler story is closed. Do not blend anything in this window.

Information: a deployability bit that twenty more accuracy-preservation folds will not answer. GPU-hours: 0.

### What not to run

- The next activation sweep, including another H64 sine / cubic / momentum / schedule seed on the 16-component panel.
- A complete-predictor GELU/LeakyReLU transfer as the default GPU spend. It is the largest local prototype and the least new information: the panel question is answered, the trunk objective and the residual cell are not, and its H64 delta is not a public delta.
- The next tiny-panel seed of E467/E469, E464’s same prefix routers, or E468’s same codebook.
- Any resubmission or blend whose justification is `ab85_final_ema6_s4_gclpall` at 0.704.
- The seed-3/seed-4 near-copy bag. The excluded slice that agrees with the public ranking already has it below seed 4.

## 6. Underestimated: runtime, QAT, and blends

**Runtime is a scientific constraint, not a packaging footnote.** The public seed-4 package used 37m07s and 626 MB of a 60-minute, one-CPU limit ([public reflection](../../campaign_2026-10-04/PUBLIC_RESULTS_REFLECTION_20261004_LATE.md), [SECOND_PAIR_OF_EYES_2026-10-05.md](../../SECOND_PAIR_OF_EYES_2026-10-05.md)). The seed-3 head endpoint was 34m01s versus the equal mixture at 37m47s, about 3m46s–3m50s less for a single endpoint. A second full GRU256 forward is **not** a measured quantity in the files opened here; doubling 37 minutes in one’s head is not evidence. What is known is that one merged model already uses most of the hour, so a two-parent ensemble does not fit unless a member is actually timed under the remaining budget. That is why P3 exists. Local E460’s 3,946 seconds is a validation-file replay on a loaded host, not the competition runtime.

**QAT was tested as accuracy preservation, not as an ensemble enabler.** E419’s recovered recurrent QAT delta is +4.8e-6 with a CI through zero. E-411’s fixed symmetric q8 deltas are about −2e-5. Those support “quantizing this fold did not move WP.” They do not support “two quantized parents fit in 60 minutes,” and they do not support a higher score. The submitted INT8 ZIP failed linkage before any public score. Compatibility of a repaired 3.11 binary is not runtime approval. Using QAT to buy a second member is **untested**, and it is the only reason to touch quantization next. Accuracy-preserving quantization alone is not a new best.

**Blends of full-data checkpoints are allowed only when the evaluation rows were not fit. The 0.704 claim fails that test.**

Verified this session, not from the review’s paraphrase:

- [scripts/ft_gclp.sh](../../../scripts/ft_gclp.sh) lines 7 and 25–26: `FOLD=finalall` passes `--include-heldout-valid` and comments that the holdout score is in-sample. `final` does not.
- [artifacts/logs/q56.sh](../../../artifacts/logs/q56.sh) launches `ab85_final_ema6_s4_gclpall` with `finalall`.
- [ab85_final_ema6_s4_gclpall.log](../../../artifacts/logs/ab85_final_ema6_s4_gclpall.log) line 1: `train_source=valid`, `include_heldout_valid=true`, `valid_train_fraction=1.0`, `eval_source=holdout`.
- Same log line 10 and [history.json](../../../artifacts/runs/ab85_final_ema6_s4_gclpall/history.json): `weighted_pearson` **0.7040365552180464**, `split` `holdout368`, 368 blocks. The correlation matches. The split label does not make it out of sample.

The champion does not share that leak. Seed-4 parent and head provenance both have `include_heldout_valid: false`. The excluded-368 score 0.701694557 is still not a fresh holdout (exposed, component-linked), but it is not the same defect as training on those rows.

Therefore: do not revive 0.704, do not blend `gclpall` with seed 4, and do not treat a blend that inherits those weights as diversity. The near-copy seed-3/seed-4 bag excludes the holdout from final fitting and still loses to seed 4 on that slice (0.701420130 < 0.701694557). A diverse blend would be interpretable only for members whose fitting rows are disjoint from the rows used to claim the gain. This session did not certify any such pair. No other full-data checkpoint was opened history-by-history; their holdout cleanliness is **unknown** until each `include_heldout_valid` flag is read. Unknown is not a license to average them.

## Contract reminder

Question: what mechanism, if any, can move a complete causal predictor under `global_wp_clip2_v1` by enough to matter relative to transfer noise, without claiming public 0.7000. Tier C. Falsifiers are the P1 gate, the shuffled-mark control, and a public score. Invariants: causal streaming, no test labels, no holdout in the fit, no local delta stacked onto 0.6660. This document does not reopen admissions.

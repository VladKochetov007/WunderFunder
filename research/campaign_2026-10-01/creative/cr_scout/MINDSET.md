# MINDSET: where winning teams dug, mapped onto Alpha Connectome (cr_scout, 2026-10-01)

Corpus: `corpus/docs.jsonl` (19 docs: url, title, ordered blocks markdown/code/output/image, comments tree where
the API returns it; comment authors are not exposed by the API). Raw cache: `corpus/raw/` (topic JSON, notebook
.ipynb + extracted images, Playwright renders). Fetcher: `scripts/cr_scout_fetch.py` (list/fetch/notebook/build,
throttled to ≤1 req/s). Flags: **[R]** = reported by the authors, not reproduced; **[R+abl]** = reported with the
authors' own ablation or late submissions; **[X]** = reproduced or measured here (only §3 probes).

## 1. Per-source decisive moves (claim, then source)

**Kaggricult-Man reverse engineering** (leoprovorov, notebook `kaggle.com/code/leoprovorov/kaggricult-man-reverse-engineering`, Kaggriculture sim)
- Reverse-engineered the world generator. Which two shops unlock by day 6 is pure RNG,
  `random.Random((seed*1_000_003)^day)`. That gives 8×8 = 64 ordered worlds, all 64 measured across 2,835 top-10
  replays (image "Field Worlds"). [R]
- Changed the unit of analysis: "a route inside a world, not one global win rate". The leading agent is a shared
  opener, one branch at step 144 on the ordered shop pair (a 14-entry table leading to 10 continuations), then a
  forced endgame at step 648 ("Field Skeleton"). [R]
- "Field Ledger": a turn-by-turn decomposition of projected terminal value g(t) and its 48-turn trend D48, used to
  find "the turn where value starts leaking". [R]
- Intervention = frozen parent plus one narrow, gated overlay. It sells 1–2 turns early only in mirror games
  (similarity ≥ 0.9 for 6 observations), with a credit ledger so the same stock is never sold twice. [R]
- Paired same-world, seat-swapped test Δ(w) = ½Σ_s(M_T − M_C): lead-1 > control 6/6, lead-2 > lead-1 6/6. The
  crop guard added +196 in 1 world and 0 in 2, so the author reported it as weak. [R; n tiny]

**NVIDIA Nemotron 4th** (dipamc77, `/writeups/4th-place-solution`, topic 712419)
- Factored the case space. Every bit-manipulation problem not solvable by 1–2 ops is one of 5 templates
  (Majority, Choice, M5, C3, C4) over three shifted leaves, with fixed operand positions. The solver reached 99.4%
  (1593/1602; the 9 misses are underdetermined). [R]
- Anchor positions collapse the search. At bit 7 every left shift is dead and at bit 0 every right shift is
  dead, so each anchor column is a 2-variable function. Shift amounts are read off two anchor tables and joined
  on the shared rotate amount. [R]
- "Trainable trace" rules: no hidden compute, locality (each line computable from the 10–20 lines above), causal
  consistency, reference integrity, few uniform rules. These were enforced by audit scripts. [R]
- Cheap proxy metric: greedy next-token match on unseen traces (80% match ≈ 95% generation match), plus a
  train→diff viewer→classify failure→reformat loop of about 1 h. [R]
- A third party reproduced it: public 0.888 / private 0.896. [R, comment]
- Nemotron 1st (topic 709231): "the core was deciding what the model should memorize through synthetic traces and
  what it should compute inside the trace". Problems were generated from train.csv statistics with rule-based
  solvers. [R]

**IceCube 3rd** (dipamc77, `/writeups/gpus-on-ice-3rd-place-attention-xgboost-ensembler`, topic 402888)
- Bitter lesson: hand features and augmentations helped small models and washed out at scale (128d/9L 1.001 →
  512d/18L 0.982). [R]
- Largest late gain: a boosted-tree ensembler over two disagreeing heads (128-bin classifier vs vMF), fed with
  their confidence signals (bin probability, kappa, event length, z). Average 0.976 → disagreement vote 0.973 →
  XGB 0.969 → plus features 0.965. [R]
- How they found it: hand-written rules of the form "if kappa > k and prob < p and |Δ| > d" kept improving the
  score, and they noticed that was a decision tree. [R, comment]
- Ordinal target smoothing (Gaussian kernel over bins). [R]
- IceCube 1st (topic 402976): matched the loss to the metric (vMF − θ), +0.005; stacking MLP, +0.003. [R]

**Pokémon TCG "All Decks on Hand"** (dipamc77, 2nd Main Track, Playwright render)
- A memoryless 4-layer pointer-attention policy plus an explicit external information tracker. The tracker
  deduces hidden facts (e.g. prized cards after a deck search) and feeds them into the next observation. [R]
- A training-only oracle critic with hidden information, to reduce value variance. [R]
- Choosing for the expected field: a deck×deck win matrix from self-play, iterated round-robin over plausible
  final ladder mixes, and a sensitivity check that shifts the field. After the deadline the matrix broadly matched
  results against the top 25. [R]
- Failed: MCTS (gains vanished after 1B steps under partial observability), dense reward shaping, extra weight
  on candidate decks. [R]

**Pokémon TCG "Archaludon Vanguard"** (lucifer19, 0 votes): mostly a plan with few results. The usable ideas are
spending search only in critical states, paired seeds, and worst-matchup as a metric. [R, low evidence]

**Optiver Realized Volatility 1st** (nyanp, topic 274970; ablation topic 302626)
- Tick-size recovery: real price = 0.01 / min tick step of the normalised prices. A 1-D t-SNE on the
  time_id×stock price matrix then recovered the hidden time order, validated against yfinance. [R]
- They used the recovered order for time-series CV and adversarial validation, not as a direct feature. [R]
- Reasoned about the data generator: 3,833 time_ids / 443 days ≈ 8.6 per day, i.e. 30-minute buckets from 10:30
  to 15:00, so the neighbouring time_id's RV is a near-future lag. [R]
- Largest gain: nearest-neighbour aggregation over similar time_ids and stocks. RMSPE 0.2249 (none) → 0.2037
  (vol+size NN, no leak) → 0.1973 (all NN). A single LGBM wins; rank normalisation had about 0 effect. [R+abl]
- Optiver RV 3rd (topic 278676): target transformation y / RV(input window), and a self-supervised "300 s model"
  trained on test inputs (first half predicts second half) used as a feature. time_id NN was "the largest
  gamechanger". [R]

**Optiver Trading at the Close 1st** (hyd, topic 487446)
- CatBoost 0.5 + time-axis GRU 0.3 + cross-sectional Transformer 0.2 (zero-mean output over stocks). [R]
- Online retraining every 12 days, 5 times: test 5.4438 → 5.4157 (once) → 5.4030 (five times). [R table]
- Post-processing that uses the metric structure: subtract the index-weight-weighted mean of the predictions,
  because the target is index-relative. [R]

**Jane Street RT 2024, 8th private** (E. Grigoreva, topic 556542) and **17th public** (topic 556541)
- Online learning: one update per day on revealed labels, +0.008 CV, the largest lever. The 17th place only made
  it work by simulating online learning on the last 20–30 validation days to tune the window and epochs. [R]
- Reverse-engineered the target: responder_6/7/8 are 20/120/4-day rolling means of one latent variable. From that
  they built derived 8- and 60-day auxiliary targets: +0.001 CV and LB. [R]
- A day-long GRU with market averages and rolling statistics (+0.002 CV); unweighted 6-model average (best single
  LB 0.0105 → 0.0112). [R]

**Jane Street 2021 1st** (Yirun, topic 224348): supervised AE trained jointly with the MLP inside each fold (an AE
pretrained on all rows leaks labels); all 5 resp horizons as multi-label; sample weight = mean |resp|; dropped the
first 85 days (different variance). [R]

**Ubiquant 1st** (topic 338220): per-time_id cross-sectional means of the 100 top-correlated features, single
LGBM CV 0.141 → 0.154, LB 0.141 → 0.149. [R]

**G-Research Crypto 2nd/3rd** (topics 323098/323703): long overlapping walk-forward folds, plus a "smooth
knee/plateau or it's noise" check on CV curves. 3rd place used only Close (the variable that defines the target)
and features relative to the market average. [R]

**DRW Crypto 1st/2nd** (topics 599068/596581): feature-cluster medoids, SHAP stability across folds, and an MLP
loss of 0.6·MSE + 0.4·Pearson. 2nd place dropped clusters of time rows when that improved validation or the
public LB (high overfit risk). [R]

**Not covered:** no authoritative writeups for earlier Wunder challenges were found, only participant repos.
Search results also list public GitHub repos that seem to be related to this competition. I did not open them:
`docs/rules.md` bans public or private solution sharing before 15 Nov, and we keep a clean room.

## 2. Transferable moves → concrete hypotheses for our data

Status key: **NEW**, **PARTIAL** (adjacent work in CONTEXT/TEAM_INTAKE, but not this angle), **TRIED**.

Shared data: `datasets/{train,valid}.parquet`, `cache/valid_targets.npz` (y, mask, row_group, market_class),
OOF `cache/pred_c20260929_{ft_since,since}_s{3,4}_f{k}[h]_valid.npy`, folds in
`research/campaign_2026-09-29/artifacts/splits/fold{k}.json` (`evaluation.valid`). Unless stated otherwise every
falsifier uses cached OOF on fold-k evaluation rows only, with |y|-weighted statistics and per-class reporting.

**M1. Use the pooling structure of the metric: separate the sequence level from the within-sequence path.**
Sources: Optiver Close 1st (subtract the metric-weighted mean), IceCube 1st (loss = metric), Ubiquant/Optiver
(cross-sectional context). Mapping: Global WP pools rows across sequences. The oracle per-sequence D level reaches
0.727 vs about 0.70, so most of the recoverable error is between sequences. Hypothesis: the between-sequence part
of our predictions is either under-confident (amplify the causal sequence level) or noisy (shrink it).
Falsifier (run, §3): decompose cov/var into between and within, then sweep p' = p − λ·causal_mean_seq(p).
Status: **NEW angle → KILLED post-hoc [X]**. Best λ = 0 for 16/16 class×fold cells. Class-3 predictions are
under-dispersed between sequences (between-sequence variance share 0.058 vs 0.115 for y; between corr ≈ 0.50), but
the level cannot be recovered from the causal prediction history (§3 F1/F1b).

**M2. Reverse-engineer the generator: treat the targets as a labelling rule and search a small rule family.**
Sources: Nemotron 4th (5 templates + anchors → 99.4%), Kaggricult (RNG seed formula → all 64 worlds enumerated),
JS2024 8th (responders = rolling means → derived targets), Optiver RV 1st ("how is the data generated?").
Mapping: `data_eval/target_law_report.md` confirms H1 (targets change iff E = a6 lower endpoint, exact) and H2
(interior t0+t1 = −s). It rejects "winner = round(mean mid over next N)" (≤ 55% exact). The best simple form
is an excursion rule (82% in class 3), and one inspected sequence looks like an exit at a running extreme or a
trailing stop. Reframe: t0/t1 are the P&L of a simulated long/short trade opened now and closed by an exit rule
(stop-out m on the exit-side quote, matching H3). That is triple-barrier / trailing-stop labelling, a small,
enumerable family.
Hypothesis: one rule from {fixed take-profit/stop (a, b) ticks, trailing stop k ticks from the running best,
optional vertical barrier T events}, per class, reproduces ≥ 95% of class-3 winners. Cheapest falsifier: reuse
`scripts/de_target_law.py` path reconstruction (move/spread tables in `data_eval/target_law_tables.npz`) on its
held-out TRAIN panel and grid the family. Kill if no rule beats the excursion rule by ≥ 10 points in class 3.
CPU < 1 h. Payoffs if it holds: (a) the exact maturity time of each label, which is the precondition for a
leak-free M3 (§3 F3b); (b) a first-passage representation: barrier-hit probabilities as auxiliary or plug-in
heads, which turns the problem into "predict which barrier is hit first" instead of a regression on a rounded
level. Status: **PARTIAL**. The law hunt exists (data_eval); the exit-rule family enumeration does not.

**M3. Online learning on causally matured labels (test-time adaptation inside one sequence).**
Sources: JS2024 8th (+0.008 CV, the largest lever), JS2024 17th (made it work by simulating online learning on
the last validation days), Optiver Close 1st (5 online retrains), Optiver RV 3rd (self-supervised model trained on
test inputs). Mapping: row r's target is fixed by the next N events after r. Once those have happened, the label
is computable from features alone, if the law is exact. The per-sequence level/scale oracles (0.727/0.719) bound
the gain. Hypothesis: a per-sequence correction fitted on matured, feature-computed labels recovers part of the
+0.02 oracle gap.
Falsifier (run, §3, upper bound): is the residual y − βp of the latest block predicted by the mean residual of
earlier blocks with a one-block gap? Near-zero persistence kills both M3 and any "sequence-level bias" head.
Status: **BLOCKED on M2 [X]**. Persistence at the block level is about 0 (F3). At event-exact lags it is large
but not trustworthy: with true labels, the matured-residual correction gains +0.005 to +0.007 at a lag of 8
events, +0.002 to +0.004 at 12, ≤ +0.0004 at 24, and 0 at 64 (F3b). Because the winner's horizon is unknown
and possibly long (H4 rejected; trailing-stop-like exits), "N events ago" does not guarantee maturity. The
smooth decay is exactly what partial leakage would produce, and it is the likely mechanism behind the teammates'
"matured labels (leak)". Only an identified exit rule (M2) can certify maturity and turn this into a legal
signal.

**M4. Factor the case space and find the cell where value leaks (a ledger, not a global score).**
Sources: Kaggricult (Field Ledger; "route inside a world"), Nemotron 4th (templates/anchors), IceCube 3rd (hand
rules → tree). Mapping: build a WP-deficit ledger over cells of class × spread code × events since sequence start
(0, 1–3, 4–15, 16+) × rows since last event (0, 1–9, 10–99, 100+) × N-window position. Hypothesis: the class-3
deficit (0.43 vs 0.74–0.81) concentrates in a few cells, for example the first events after warm-up or wide spread
codes. In those cells var_p is out of proportion to cov, so a gated shrink or a cell-specific overlay (Kaggricult-
style: frozen parent, narrow gate, credit ledger) gains.
Cheapest falsifier: one pass over the valid parquet (event/since features) joined to OOF. For each cell report
weight share, corr, and the WP change if that cell alone is scaled by its own cross-fold-optimal factor. Kill if
no cell with ≥2% of weight has a cross-fitted gain ≥ +0.0003. CPU about 20 min. Status: **KILLED as an overlay [X]**. The joint cross-fitted per-cell scale loses 0.0003–0.0008 on 4/4 folds. The
ledger shows the class-3 deficit is flat across cells (WP 0.35–0.48 at every since/step/event-rate bucket), so
there is no local cell to patch (§3 F4).

**M5. A gated non-linear stacker on member disagreement and confidence.**
Source: IceCube 3rd (0.976 → 0.965 via a GBDT over two disagreeing heads). Mapping: our members (parent head, head
FT, two-stage FT, GRU256 distilled) disagree most on class 3, which is where the parent+FT average already helped.
Hypothesis: a depth-3/4 tree over [member preds, |Δ|, class, rows since event, spread code] beats the equal mix.
Cheapest falsifier: LightGBM |y|-weighted L2 trained on 3 folds' OOF and scored on the 4th, per target, against
the RSP2 routing. Kill if < +0.0003 on 3 of 4 folds. CPU 15 min. Serving: about 50 small trees in native code is
cheap. Status: **KILLED [X]**. The cross-fitted HGB stacker loses 0.009–0.015 against the two-stage pair (class 3:
−0.024 to −0.047), and the cross-fitted linear stacker loses 0.004 on 3/4 folds. The plain six-member mean is
+0.0005 on average, which is the known parent+FT mix (§3 F5).

**M6. Transform the target by a causal scale (predict y / s_t).**
Sources: Optiver RV 3rd (target / input-window RV), G-Research 3rd (relative-to-market features). Mapping: the
per-sequence scale oracle gives 0.719. Hypothesis: s_t = causal mean |Δmid| per quote event (in ticks, last K
events) explains |y| beyond |p|. Cheapest falsifier: per class, |y|-weighted regression of |y| on |p| and on
|p|·(s_t/s̄); then the WP of p·(s_t/s̄)^γ for γ ∈ {0, .25, .5}. Kill if the best γ ≈ 0. Status: the label-free proxy p·(causal RMS_seq(p)/class RMS)^γ is **KILLED [X]** (best
γ = 0 on 16/16 class×fold; max +0.0001). The feature-based s_t version is untested but now has a low prior.

**M7. An explicit tracker instead of hidden memory (no hidden compute, locality).**
Sources: Pokémon Unown (memoryless net + deduction tracker), Nemotron 4th (every token derivable from the nearby
lines), Nemotron 1st (decide what to memorize vs compute). Mapping: events are sparse, and the GRU has to carry
the last-k event content across many non-event rows. Hypothesis: an explicit event register (last 8 signed mid
moves in ticks, rows since each, spread at each, event count) helps class 3. Cheapest falsifier: partial |y|-weighted
correlation of each register feature, and of a ridge on all of them fitted on TRAIN, with the OOF residual on
valid. Kill if the ridge partial correlation is < 0.01. Status: **TRIED-ish**. Dual-clock, event attention and
event-content memory were rejected. The teammates' "event GRU fed by the row GRU" is untested; the probe tells
whether it is worth anything.

**M8. Multi-horizon auxiliary targets derived from the generator.**
Sources: JS2024 8th (+0.001 from derived 8/60-day targets), JS2021 1st (all resp horizons as multi-label).
Mapping: compute the winner law at N′ ∈ {1, 2, 4, 8, 12} events and the loser's m as auxiliary heads.
Cheapest CPU falsifier: on TRAIN, the correlation of each N′ target with the main target, and how predictable
each is (R² of a small ridge on simple state). It is promising only if some N′ is both much more predictable and
correlated > 0.5 with the main target. The real test is a GPU run. Status: **PARTIAL** (aux struct/next exist).

**M9. Optimise for the expected test field, not the local average.**
Sources: Pokémon Unown (deck chosen against the expected field), Kaggricult (world frequencies), Optiver RV
(adversarial validation). Mapping: public ≈ 46% class-3 sequences, and single-sequence chains are 64% class 3
(0.59 vs 0.74). Status: **TRIED** (propensity rows, de_public_proxy, class mix calibration). Only remaining
angle: adversarial validation of short-chain vs long-chain class-3 sequences, to check that the public-like
subset is not a different regime inside class 3.

**M10. Recover hidden ordering and time for validation.**
Sources: Optiver RV 1st (tick size → price → t-SNE order), Santander Value (shifted series), G-Research
(walk-forward). Status: **DONE**. The feature-only successor graph (`cache/chains.npz`) and chain-excluded folds
already exist. Cross-sequence features are banned at inference (`data_overview.md`: sequences independent), so
the NN-aggregation half of the Optiver move is not transferable.

**M11. Reverse-engineer the evaluation mask.**
Sources: Kaggricult (exact agent map), Nemotron (exact answer checker). Mapping: is_scored is hidden at test.
Status: **TRIED** (propensity models of is_scored drive the row weights).

**M12. A frozen parent plus one bounded, gated overlay, judged by paired tests.**
Source: Kaggricult (lead-1/lead-2 overlay, credit ledger, seat-swapped pairs). Mapping: this is how RSP2 routing
works already. Use it as the delivery form for any M1/M4/M6 win: a per-class or per-cell post-hoc transform with
1–2 parameters, cross-fitted, scored as paired fold deltas. Status: **method, in use**.

## 3. Probes run here [X]

All probes use cached clean-fold OOF on fold-k evaluation rows only. The reference is the two-stage FT pair
`c20260929_ft_since_s{3,4}_f{k}` (fold WPs 0.6778 / 0.6630 / 0.7024 / 0.7079). Weights are |clip y|, scored with
`wnn.gmetric` sums. Outputs are next to this file. Scripts: `scripts/cr_scout_level_probe.py` (F1/F1b/F1c/F3),
`scripts/cr_scout_cell_ledger.py` (F4), `scripts/cr_scout_tree_stacker.py` (F5),
`scripts/cr_scout_matured_probe.py <lag>` (F3b).

**F1 between/within decomposition** (mean of 4 folds × 2 targets; `probe_level.json`):

| class | y var between-seq | p var between-seq | corr between | corr within | cov share between |
|---|---:|---:|---:|---:|---:|
| 0 | 0.065 | 0.059 | 0.90 | 0.80 | 0.07 |
| 1 | 0.060 | 0.054 | 0.89 | 0.77 | 0.07 |
| 2 | 0.084 | 0.070 | 0.85 | 0.73 | 0.09 |
| 3 | 0.115 | 0.058 | 0.50 | 0.43 | 0.09 |

Class 3 is weak both between sequences (corr 0.50, half the dispersion) and within them (0.43). More than 90%
of the covariance is within sequences.

**F1b** p − λ·causal_mean_seq(p), applied per class: the best λ is 0 in 16/16 class×fold cells. Class 3 at
λ = ±0.25 loses 0.0012 to 0.0018. **F1c** p·(causal RMS ratio)^γ: the best γ is 0 in 16/16, max +0.0001.
→ The sequence level is not recoverable from our own prediction history.

**F3 block persistence** (residual of 1000-row blocks vs the earlier blocks, one-block gap): corr is −0.07 to
+0.08; class 3 mean 0.017.

**F3b event-exact matured residual** (uses true y as the stand-in label, so it is an upper bound and possibly
leaky; `probe_matured_n{8,12,24,64}.json`). Sum over classes of the cross-fitted held-fold gain from
p + κ·(newest matured-segment residual), per fold:

| lag (events) | fold 0 | fold 1 | fold 2 | fold 3 | class-0 corr(resid, feature) |
|---:|---:|---:|---:|---:|---:|
| 8 | +0.0053 | +0.0052 | +0.0058 | +0.0072 | 0.25–0.28 |
| 12 | +0.0028 | +0.0026 | +0.0026 | +0.0038 | 0.16–0.21 |
| 24 | +0.0001 | −0.0000 | 0.0000 | +0.0004 | 0.03–0.09 |
| 64 | 0 | −0.0005 | 0 | 0 | ≈ 0 |

This is not evidence of a legal signal. The label horizon is unknown (H4 rejected, trailing-stop-like exits), so
a short lag can include unexited labels. The decay looks like leakage disappearing. It becomes actionable only
after M2 certifies exit times.

**F4 cell ledger** (`probe_cells.json`; class × rows-since-event × step × causal event rate, 216 cells): the
cross-fitted per-cell scale overlay loses 0.0003–0.0008 on 4/4 folds. Class-3 WP is flat across cells:
since-event buckets 0 / 1–3 / 4–15 / 16–63 / 64–255 / 256+ give 0.48 / 0.43 / 0.43 / 0.48 / 0.43 / 0.35 (weight
0.6 / 3.7 / 4.5 / 13.3 / 13.8 / 5.7% of all metric weight). Event-rate tertiles give 0.42 / 0.44 / 0.45, step
1k–5k vs 5k+ gives 0.44 vs 0.43. No scored weight sits in steps 99–999. There is no localized leak cell.

**F5 stacker** (`probe_stacker.json`): HGB on 6 members + other-target members + class/since/a6/step, |y|-weighted,
trained on 3 folds (every 16th row) and scored on the 4th, minus the pair: −0.0089 / −0.0097 / −0.0146 / −0.0113
(class 3: −0.024 to −0.047). The cross-fitted linear stacker: +0.0003 / −0.0042 / −0.0041 / −0.0040. The
six-member mean: +0.0007 / +0.0010 / +0.0004 / −0.0001 (the known parent+FT mix).

## 4. Verdict: where to dig

1. **Dead ends, now measured:** post-hoc sequence-level de-meaning, scaling or level correction (M1, M6 proxy),
   cell-gated overlays (M4), tree or linear stacking of the current members (M5). The per-sequence oracle gap
   (0.727) is future information, not something recoverable from causal history.
2. **Dig here first: identify the exit rule (M2).** This is the Kaggricult / Nemotron-4th move: the generator is
   a small rule family (trailing stop / triple barrier on the reconstructed quote path), and data_eval already
   has the path tables. It is the only lead that (a) attacks class 3, where the deficit is uniform and therefore
   structural, and (b) unlocks the only large signal seen today (F3b, +0.005 at short lags) by making maturity
   certifiable. Then re-run F3b with exit-certified labels (true y first, then law-computed y).
3. **Then:** a first-passage representation (barrier-hit probability heads or auxiliary targets, M2b/M8), and
   the explicit event register probe (M7) as a cheap check on whether the GRU under-uses event content.

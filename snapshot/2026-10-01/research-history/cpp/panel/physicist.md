# Panel: book physicist (microstructure + EDA), 2026-09-26

Scope: what in the data is still unexplained and worth modeling. All checks are read-only on cached
valid arrays (`valid_targets`, `valid_rowinfo`, `prop_v1_valid`, `heur_valid` raw rows via memmap)
plus exactly 3 parquet row groups (valid rg 829 and 783 = dense class 3, rg 1719 = sparse class 3).
Model under study: `pred_ab30_eps_aux_valid` (train-only GRU, floor + aux head), the same model
E-132..E-136 used. "Rows-to-next-move" means rows until the next i0 price-code change (cols 0-21).
It is a *diagnostic* coordinate (future), never a feature. Probe scripts are in the session
scratchpad (`.../scratchpad/phys/*.py`). Dense/mid/sparse = within-class scoring-rate terciles.

## 1. Findings

**F1. The dense gap is mostly a composition effect in *future* distance to the next price event.**
WP depends very steeply on how far the next i0 price-code change is. Dense sequences score many rows
that sit far ahead of it:

| class 3, rows-to-next-move | P(scored) sparse / dense | weight share sparse / dense | WP sparse / dense |
|---|---|---|---|
| 0 (event row) | 0.018 / 0.069 | 0.084 / 0.068 | 0.485 / 0.408 |
| 1-2 | 0.073 / 0.220 | 0.174 / 0.092 | 0.778 / 0.728 |
| 3-8 | 0.137 / 0.367 | 0.184 / 0.100 | 0.796 / 0.760 |
| 9-32 | 0.087 / 0.311 | 0.216 / 0.173 | 0.691 / 0.595 |
| 33-128 | 0.064 / 0.329 | 0.207 / 0.276 | 0.311 / 0.281 |
| 129+ | 0.038 / 0.271 | 0.135 / 0.290 | 0.030 / 0.109 |

In sparse sequences scoring peaks 3-8 rows before a move (3.6x the far rate). In dense ones it is
flat at 0.22-0.37. The weight-share sums reproduce tercile WP (0.540 vs actual 0.547 sparse; 0.384 vs
0.385 dense). Shift-share: class 3 gap 0.156 = composition 0.120 (77%) + within-bucket 0.036.
Class 2: 0.151 = composition 0.059 + within-bucket 0.092. Classes 0/1 have small gaps (0.06),
mostly within-bucket. Past recency (`since_book`, book events per 200 rows) does *not* separate the
regimes: the gap sits in every past bucket (class 3: 0.49/0.41 at since_book 0, 0.51/0.32 at >200).
That is why the causal slices of E-133 found nothing.

**F2. Physical regime of dense class-3 sequences.** Sequence level, Spearman with scoring rate:
a3_std +0.71, a0_std +0.69, a2_std +0.68, a3_mean +0.67, a2_mean +0.66, i1 price events per row
+0.46 (0.045 -> 0.088 median), a7_mean -0.47. Row level (rg 829 / 783 dense vs 1719 sparse): vol-only
rows 64% / 47% vs 78%. i1-only rows are 4.1% / 2.2% of rows, with P(scored) 0.60 / 0.50 (sparse 0.7%,
0.05). Price rows are scored at only 0.03-0.05. Put plainly: long stretches where i0's quotes stand
still while its queues churn and i1 moves, and the scorer's activity-driven selection keeps
sampling them. Mean |y| is *not* smaller in dense class 3 (0.677 vs 0.656). The regime is about
information timing, not move size.

**F3. Effective sample size is tiny and flat in density.** The target is constant between price
events, so each quiet stretch is one outcome replicated over its scored rows. Effective episodes
per sequence, n_eff = (Σw)²/Σw_ep², run 27.8 / 31.4 / 27.1 (class 3, sparse/mid/dense), although
dense sequences score 2.6x more episodes. Episodes with ≥33 scored rows carry 25% / 41% / 66% of the
weight. In rg 783, 824 scored rows at 33-128 rows-to-move come from only 46 distinct next moves.
Row-level screens on ≤45 sequences therefore have roughly 1k effective samples. This explains the
sign flips in opus_micro §0.1.

**F4. Class 3 reduces to next-move direction, and that is a queue race visible only near its end.**
An oracle that knows the sign of the first future mid move plus the spread reaches WP 0.824 (dense
rg 829) and 0.819 (sparse rg 1719); the GRU gets 0.314 / 0.514. GRU direction (p0-p1) vs first-move
sign, measured per event:
- 2 rows before a move: +0.91 (dense), +0.91 (sparse). One row before: +0.92 / +0.90 / +0.84
  (rg 783 / 829 / 1719).
- Midpoints of quiet stretches ≥33 rows: +0.154, -0.087, +0.054 (n = 80, 54, 71; SE ≈ 0.12).
- Simple proxies at those midpoints (Σbid-vol − Σask-vol, the E-100 class-3 symbolic term, and
  log activity ratio ask/bid as in TQP eq. 7.8) are all within ±0.2 and change sign across sequences.
So the GRU does *not* miss queue-depletion structure near moves. Far from moves the direction is
close to a coin flip for every signal tried. This matches the diffusive race of TQP §7.3.1 (eq. 7.6,
p.123, from Cont and de Larrard 2013). p+ depends only on the queue ratio, and a ratio near 1 with
long queues says little until one queue is small.

**F5. i1 lead in quiet stretches: weak, but where the GRU is blind.** i1's price codes were decoded
with the i0 class-3 dictionary (N3); 14-27% of i1 transitions are unknown. The feature is i1 mid
displacement since the last i0 mid move. Row level, residual after the GRU's direction: +0.14 / +0.19
at 33-128 rows-to-move (rg 829 / 783) and +0.22 / +0.13 at 129+. Near moves it is ≈0 or negative.
Event level at stretch midpoints: +0.13, +0.04 (dense), +0.26 (sparse); the GRU at the same points
gets +0.15, -0.09, +0.05. Pooled over ~205 moves in long stretches: ≈ +0.14 ± 0.07. This does not
contradict N4 (class-3 lead ≈ 0.01), which averaged over all rows. Most class-3 intervals contain no
i1 move, and the effect is confined to long i0-quiet stretches.

**F6. Scoring is causal, and propensity marks information.** Future distances to the next
i0/trade/i1 event add only +0.000-0.008 AUC over prop_v1 (0.884-0.928, GBM, sequence split), and
+0.001-0.022 inside the top propensity quintile. There is no look-ahead mask to exploit. But WP rises
with propensity even on *unscored* rows (class 2: 0.26 -> 0.74; class 3: 0.14 -> 0.37). At the same
prop_v1 bin, scored rows are much more predictable than unscored ones (class 3 top bin, sparse: 0.67
vs 0.37). Inside the top quintile, causal features still separate scored from unscored with AUC 0.75.
**prop_v1 is not a sufficient statistic for the selection.** The true mask carries causal information
that prop_v1 misses. This fits the fact that true-mask fine-tuning moved the LB 3x more than
predicted (E-144).

**F7. The F-clock is the i0 price-code event.** Targets change on 54-65% of price-code rows and on
0.000% of trade-only, i1-only or a-only rows, in every class (every 4th sequence checked). In the two
class-3 sequences, F = round(mean mid over next N events) agrees 84-95% with the quote clock at
N = 4-5, or 85-92% with the price-code clock at N = 6-8. E-079's own definition gives 97.5%.

**F8. The mirror is not a signed column permutation.** Quantile matching of bid to ask marginals
gives the *same* signed pairing for i0 and i1 in 9 of 11 pairs (0->16-, 1->12+, 2->13+, 3->19-,
5->20+, 6->15+, 8->21+, 9->11+, 10->14-). That is evidence of one shared per-side encoder. But the
functional test fails. Fit g_D on the codes and require g_D(M x) ≈ -D: the matched map scores -0.005
(class 3), +0.06 (class 2), +0.27 (class 1; random signed perms sd 0.13) and +0.52 (class 0; random
sd 0.35), against +0.39..0.63 own held-out. In the class-3 grammar every bid price column changes on
ask-only moves (11/11): the price codes mix both sides.

## 2. Short answers to the panel puzzles

(a) Dense = scoring spread uniformly over long i0-quiet stretches (activity via volumes and i1)
instead of concentrated just before moves (F1-F2). 77% of the class-3 gap is composition. The
signal that could help is information that arrives *during* the stretch without an i0 price
event: i1 displacement (F5), and possibly queue ratios if they can be decoded better than the
volume-code mixing allows (F4, E-090 AUC 0.62). About 23% of total metric weight sits in class-3
rows ≥33 rows before the next price event (0.46 × 0.50), with WP 0.03-0.31.

(b) Class-3 WP is capped by the timing of information, not by missed structure near events. The
first move alone fixes WP 0.82. Its direction is known (0.91) only in the last few rows of a race
and is near chance at stretch midpoints for every book proxy. The GRU is at that frontier near
events. The open question is whether any exogenous source (i1, flow persistence) informs far rows.

(c) The clock is established (F7): price-code events only, no real-time or i1 component. What
remains is the exact N and the mid definition for small ticks, which needs a joint dp + price-code
decoder (N1, 85-88%). Value for the score is low: classes 0-2 already reach 0.74-0.83 and the GRU
learns the mapping implicitly.

(d) The mirror cannot be identified as a column map (F8). A code-level involution would have to be
learned: an adversarial or flow map with a D-antisymmetry constraint, or a state-graph automorphism
for class 3 only (57 spread states, 320 transitions). I do not recommend it. Data volume is not the
bottleneck (epochs, width and seeds are saturated), and the payoff would reach only class 3.

(e) See §4. The single most decision-changing view is the future-distance decomposition (item 1).

## 3. Structural hypotheses

**H-P1. Queue race with information only at depletion (class 3; liquidity).**
- *Model.* Next mid move = first touch queue to empty, p+ = (2/π)·arctan(√(λa·Vb / (λb·Va)))
  (TQP eq. 7.6/7.8). Near a price event one queue is small, so p+ ≈ 0 or 1. Far from it both queues
  are long, the ratio is near 1 and p+ ≈ ½.
- *Counterparty.* The resting liquidity provider at the depleting touch. The constraint is time
  priority: cancelling loses queue position. The edge decays within the last 1-3 events.
- *Predictions.*
  1. Direction corr of *any* causal book model (GRU, symbolic, race formula on ICA queue proxies)
     vs rows-to-move falls from ≥0.85 (≤2 rows) to ≤0.10 beyond 33 rows, in every chain fold.
  2. The empirical p+(I) at stretch midpoints is flat (|p+(0.8) − p+(0.2)| < 0.1) but steep at race
     ends.
  3. So the ≥33-row class-3 bucket has a book-only ceiling ≈ its current WP (0.1-0.3).
- *Falsifier.* A decoded queue-ratio or race feature reaches direction corr ≥0.15 at stretch
  midpoints on ≥300 class-3 sequences, with the same sign on both chain-fold halves. That would mean
  the GRU misses the race far from events, and a race-probability input becomes a lever.
- *Why it matters.* If supported, stop spending on class-3 book features and spend on exogenous
  information (H-P2) and selection (H-P3).

**H-P2. Information enters class 3 through i1 while i0 is quiet (information).**
- *Model.* A Kyle-style informed trader, or cross-asset arbitrage, acts first in the more active
  instrument. i0 quotes stay stale until its touch queue is consumed. Hasbrouck information share of
  i1 is high during i0-quiet stretches.
- *Counterparty.* i0 liquidity providers with stale quotes: Glosten-Milgrom adverse selection. Their
  constraint is queue priority plus the absence of cross-asset repricing. The edge decays at the next
  i0 price event.
- *Predictions.*
  1. Event-level corr(i1 displacement since last i0 move, sign of next i0 move) at stretch midpoints
     is ≥ +0.08 pooled over all class-3 valid sequences, and grows with the i1-activity tercile.
  2. Its residual vs the GRU is > 0 on both chain-fold halves.
  3. The effect is concentrated in dense sequences (i1 price events per row ≥ 0.06).
  4. A tied i0/i1 book encoder (opus_micro H3) lifts the dense class-3 far-row WP, not the near-move
     rows.
- *Falsifier.* Pooled corr ≤ 1 SE, or a sign flip between chain halves.
- *Expected value if supported.* About +0.002..0.005 global. The effect lives in the ~23% of weight
  where WP is now 0.03-0.31, and only in stretches with i1 activity (~50% nonzero lead).

**H-P3. The scorer selects by activity, and activity is information arrival (selection process).**
- *Model.* is_scored follows a causal activity index (a2/a3, i1 activity, a5). Activity marks
  moments when information enters, so propensity equals predictability (F6). prop_v1 captures the
  index only coarsely.
- *Counterparty.* Not a market counterparty. This is the measurement process, and the "constraint"
  is the metric's row selection.
- *Predictions.*
  1. A richer causal mask model (prop_v2: per-class GBM on a0-a7 levels and deviations, since/rate
     features for i0 book, trades and i1, recalibrated for the class-3 1.42x over-prediction of
     E-099) raises AUC inside the top propensity quintile from 0.75 to ≥0.78.
  2. Used as training weights on train rows (floor kept), it improves held-out chain-fold WP over
     prop_v1 weights on both folds, most in sparse and mid sequences (like E-134).
  3. Unscored-row WP increases monotonically with prop_v2 in every class.
- *Falsifier.* Top-quintile AUC gain < 0.01, or held-out WP gain ≤ 0 on either fold.
- *Rationale.* True masks are only on 15% of rows (valid). A better causal proxy moves that benefit
  to the 6.6x larger train set. This is the one lever with LB evidence (+0.002 floor, +0.0014
  true-mask fine-tune).

## 4. Ranked EDA / feature plan

**1. Future-distance panel as a standard decide.py slice.**
- *Question.* Where does each candidate's gain or loss live on the rows-to-next-price-event axis,
  crossed with density and class?
- *Method.* The six-bucket table of F1 for every cached prediction (script `future2.py` pattern,
  mergeable wp_sums), plus a marimo view in `notebooks/eda_bins.py`: one dense class-3 sequence
  showing decoded mid, scored rows, GRU D and decoded i1 path.
- *Cost.* 5 min per model from cache; 1 h for the notebook view.
- *Decision.* Route effort. A lever that does not move the ≥33-row buckets cannot close the dense
  gap. It also gives the class-3 ceiling a number, and it would have explained E-133..E-139 without
  four slices.

**2. Full-valid i1-lead probe (H-P2).**
- *Question.* Is F5 real beyond three sequences?
- *Method.* Fit the price-code move table on train (opus_micro R1) and apply it to i1 on all 697
  class-3 valid sequences. Features: i1 displacement since the last i0 move, i1 move count, unknown
  rate. Statistic: event-level corr at stretch midpoints and residual vs GRU, by density tercile and
  chain-fold half. Extend `scripts/micro_features.py`.
- *Cost.* One parquet pass, ~20 min on 1 core, <2 GB.
- *Decision.* Go / no-go on a tied i0/i1 encoder GRU arm, the only lever aimed at the dense class-3
  far rows (~23% of weight).

**3. Scoring-propensity sufficiency (H-P3).**
- *Question.* How much selection information does prop_v1 leave on the table, and does it track
  predictability?
- *Method.* WP vs propensity decile for scored and unscored rows (F6 table). A prop_v2 GBM on causal
  features, fitted on chain folds 0-1 and scored on folds 2-3. Report top-quintile AUC and calibration
  per class.
- *Cost.* 20 min CPU from `heur_valid` + rowinfo.
- *Decision.* Build prop_v2 weights for the next train+valid fit (one GRU run judged by `decide.py`),
  or close the mask-weight branch.

**4. Queue-race curve (H-P1).**
- *Question.* Is class-3 far-row direction at its book-information ceiling?
- *Method.* A TQP Fig. 7.2 analog. Take ICA touch-queue proxies (E-090) at race start, midpoint and
  end. Plot empirical p+(I) vs the arctan law, including the λa/λb extension. Overlay the GRU's
  implied p+. Statistics per event (not per row), on 100 train + 200 valid class-3 sequences.
- *Cost.* ~40 min.
- *Decision.* If p+ is flat at midpoints, stop class-3 book-feature work (it would join E-088/E-089/
  E-112 as refuted). If it is steep, add a race-probability input.

**5. Episode and n_eff view for validation design.**
- *Question.* What is the true noise floor of dense-regime and class-3 A/B comparisons?
- *Method.* Per-sequence episode weights (F3). Bootstrap over episodes nested in chains. Rerun the
  E-132 tercile deltas with episode-level SE.
- *Cost.* 10 min, cached.
- *Decision.* The evidence standard for class-3 and dense-regime claims (a 45-sequence screen has
  ~1k effective outcomes). Optionally test per-episode weight caps in training only if the view
  shows a few episodes dominating the train loss.

**6. F-law for classes 0-2 (low priority).**
- *Question.* What is the exact N and the mid definition for small ticks?
- *Method.* Joint dp + price-code tick decoder (N1). Clock/N search as in `clock.py` (quote clock vs
  price-code clock, N = 2..10, rounding variants), on 20 sequences per class.
- *Cost.* ~45 min.
- *Decision.* Whether an event-path readout or a "next quote move" aux target is worth a head for
  classes 0-2. Expected score value is small; its main value is explanation (tree 4a).

**Not recommended now.**
- Mirror augmentation (F8: not a column map; learned involution is costly; the bottleneck is
  information, not data).
- Further row-level class-3 book screens on ≤45 sequences (F3: too few effective outcomes).
- Look-ahead or pseudo-masks built from future event timing (F6: +0.000-0.008 AUC only).

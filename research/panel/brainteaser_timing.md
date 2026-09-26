# Panel: brainteaser solver, information / timing / stopping-time lens (2026-09-26)

Scope: can the class-3 target, the scoring selection or the time-to-event structure be exploited as
a stopping-time puzzle beyond what the GRU already does? Read-only checks on cached valid arrays
(`valid_targets`, `rte_valid`, `prop_v1_valid`, `pred_ab30_eps_aux_valid`: train-only, so out of
sample on all valid) plus 16 class-3 valid row groups for the tick law. Cross-fold means: fit on chain
folds 0-1, test on 2-3, and the reverse (`decide._chains()`). Probe scripts:
`<session scratchpad>/bt/{c1_anatomy,p1_round,c2_rescale,c3_smooth,c4_select,c5_hazard}.py`.
c5 was written but not run: the user asked for lower machine load.

## 1. Puzzle framing

**The label is a first-passage outcome.** For class 3, F = round(mean of the next 4 quote-change
mids). Write the mid path in the event clock as m0 + Δ1, m0 + Δ1 + Δ2, and so on. Then
F − m0 = round-offset + Δ1 + ¾Δ2 + ½Δ3 + ¼Δ4. So the first increment carries the most weight, and
the winning side is (almost) the sign of Δ1, the first queue race to finish. Let U_t be the sign
of the first mid move after row t, with stopping time τ. Then π_t = P(U_t = + | F_t), stopped at τ,
is a bounded martingale. All rows of one quiet stretch share U, so the optimal direction output
should have uncorrelated increments inside a stretch. Smoothing can only help if the model output
carries non-informative jitter.

**Race law.** Take Cont-de Larrard with independent diffusive queues (bid x, ask y). Then
P(up) = (2/π)·arctan(x/y), the harmonic measure of the quadrant. The expected race time grows like
x·y. In the pure diffusion, direction depends only on the queue ratio, so it would be predictable
far from the end. With replenished (mean-reverting) queues, the ratio forgets its start within one
relaxation time. The outcome is then decided by the last excursion, and π_t ≈ ½ until then. The
physicist's measurements (0.91 direction corr 1-2 rows before a move, ≈0.05 at stretch midpoints)
match the replenished regime.

**Metric algebra.** Pooled |y|-weighted Pearson is maximised by
p*(X) = E[|y|·y·s | X] / E[|y|·s | X]. If the scorer is independent of y given the model's
information (E-103: labels add +0.002 AUC), then p* does not depend on s. Selection then only
decides which X count. Conditioning on "likely scored" can change the optimal output only where the
model's information is coarser than the scorer's.

## 2. Measured checks

**M1. Class-3 label anatomy** (1.72 M scored rows, ticks = y/0.65687, pairs (t0, t1)):

| outcome | rows | abs(y) weight |
|---|---|---|
| touch, F = ask or bid: (0,−1) / (−1,0) | 36.9% / 32.1% | 17.8% / 15.5% |
| 1 tick beyond: (1,−3) / (−3,1) | 9.2% / 8.3% | 17.8% / 16.0% |
| 2 ticks: (2,−3) / (−3,2) | 3.6% / 3.3% | 8.7% / 7.9% |
| 3+ ticks (both sides) | ~4.8% | ~14% |
| inside the spread (−1,−1) | 1.0% | 0.9% |

On touch rows the winning target is exactly 0, so it has zero weight. About two thirds (~64%) of
the class-3 metric weight sits on the ~29% of rows where the 4-event average ends ≥1 tick beyond
the quote. There, the loser has abs(y) ≈ 2, the clip value.

**M2. Tick law in the event clock** (16 class-3 valid sequences spanning the scoring-rate range,
44.5 k scored rows, `ticks.reconstruct`, quote-change clock, N = 4):
- Exact .5 ties in the 4-mid mean occur on **0.68%** of rows. Half-up and banker's rounding match
  equally (59% on ties). **Rounding asymmetry is not a lever.**
- **Winning side = sign of the first mid move after t on 99.0%** of touch rows and 99.2% of all
  rows with a winner. The first move is never zero.
- np.rint of the 4-mid mean matches F exactly on 85.3% of rows. The misses are in the tier
  (magnitude), not the side.
- The second mid move continues the first **84%** of the time (spread open, then same-side refill),
  and reverses 16%.

Class 3 therefore factorises into side U (a first-passage race) × tier G (how far the next 3
events carry the mid). The metric weight is concentrated on G ≥ 1.

**M3. The martingale test and within-stretch smoothing** (`c3`, all valid, EMA reset at each i0
price-code event):

| variant (direction only) | Δ all folds | Δ class 3 | Δ classes 0-2 |
|---|---|---|---|
| EMA α 0.5, reset | −0.0013 | −0.0021 | −0.0011 |
| EMA α 0.9, reset | −0.0153 | −0.0316 | −0.0111 |
| stretch mean / 50:50 with raw | −0.0405 / −0.0080 | −0.121 / −0.012 | −0.020 / −0.004 |
| EMA α 0.5, no reset | −0.0027 | −0.0049 | −0.0025 |

Both fold pairs have the same sign. The lag-1 autocorrelation of direction increments inside
stretches is **+0.008 in class 3** (−0.13 / −0.03 / −0.11 in classes 0/1/2). The class-3 GRU
direction behaves like a Doob martingale: the newest value is the best one. The small negative
autocorrelation in classes 0/2 is not worth exploiting. Even the lightest smoother loses.
**Refuted: denoising within the stretch.** This matches E-080: the model already is the
conditional expectation, so any recalibration from past labels only adds noise.

**M4. Timing as amplitude: oracle ceiling vs causal proxies** (`c2`). Per-(class, group) affine
readout of (level L, direction D), fit on scored rows of one fold pair and tested on the other:

| grouping | test 2-3 (raw 0.6980) | test 0-1 (raw 0.6644) | class 3 only |
|---|---|---|---|
| class only (control) | +0.0001 | +0.0002 | −0.0002 / −0.0002 |
| rows since last i0 event (7 bins, causal) | −0.0003 | −0.0046 | −0.0013 / −0.0026 |
| prop_v1 quintile (causal) | +0.0001 | +0.0002 | −0.0002 / −0.0003 |
| since × prop (35 bins) | −0.0011 | −0.0026 | −0.0043 / −0.0055 |
| **ORACLE rows-to-next bucket (5 bins)** | **+0.0151** | **+0.0171** | **+0.046 / +0.050** |
| ORACLE rows-to-next × prop | +0.0153 | +0.0189 | +0.046 / +0.048 |

Perfect knowledge of when the race ends, used only to rescale the existing output, is worth
+0.015..0.017 overall and +0.05 on class 3. That is the ceiling for any hazard or timing lever that
does not add new direction information. Every causal timing axis I could build from caches gives
0 or less: age of the stretch, propensity, and their product. The GRU amplitude is already
calibrated along them. The ceiling is future information, not missing calibration.

**M5. Selection vs timing** (`c4`, all valid). Question: within a prop_v1 bin, why are scored rows
more predictable than unscored ones (physicist F6)? Either scored rows sit nearer the next event
(composition) or they are more predictable at the same distance (within). Shift-share over 5
rows-to-next buckets. The prop bins use absolute edges 0.02/0.05/0.1/0.2, so the top bin holds most
scored rows.

| | WP scored / unscored | gap | composition (timing) | within bucket |
|---|---|---|---|---|
| class 3, prop < 0.02 | 0.377 / 0.161 | +0.170 | −0.054 | +0.224 |
| class 3, prop 0.05-0.1 | 0.369 / 0.186 | +0.178 | +0.023 | +0.155 |
| class 3, prop ≥ 0.2 | 0.433 / 0.231 | +0.187 | +0.022 | +0.164 |
| classes 0-2, prop ≥ 0.2 | 0.828 / 0.688 | +0.137 | +0.018 | +0.119 |

Class 3 with prop ≥ 0.2, WP scored vs unscored at equal rows-to-next: 1-2 rows 0.70 / 0.42,
3-8 rows 0.77 / 0.60, 9-32 rows 0.63 / 0.35, 33-128 rows 0.27 / 0.14, 129+ rows 0.08 / 0.06.
At equal (prop, rows since the last event), scored rows are nearer the next event: log(1+rows)
is 0.2-0.8 lower in classes 0-2 at every age, and in class 3 once the stretch is ≥3 rows old
(at the event row itself it is 0.07-0.21 higher). So the true mask does carry some hazard
information beyond prop_v1 (consistent with E-150). But **the predictability gap is ~85-100%
within-bucket.** The scorer picks informative states at a fixed time-to-event. It does not
mainly pick imminent events. Caveat: within-bucket Pearson also reflects label and level spread
inside the subgroup, so "informative" may partly mean "more heterogeneous levels".

Answer to Q2: "likely scored" does not change p*(X) for a calibrated model (M4 propensity recal:
+0.0001); it only changes which rows matter. The lever is training capacity, not the output.

## 3. Hypothesis cards (ranked by expected metric gain)

**H-T1. The scorer marks informative states: a sharper causal selection model as the training
weight** (M5; extends physicist H-P3, E-150, E-144). Expected +0.001..0.003.
- *Mechanism.* At equal time-to-event and equal prop_v1, scored rows are +0.15..0.22 WP (class 3)
  and +0.11..0.17 (classes 0-2) more predictable. The scorer's causal index (activity, i1, a5) lands
  on states where the book already says something. prop_v1 blurs these states with uninformative
  ones, so prop² weights spend capacity on unscored-like rows. True-mask fine-tuning worked on the LB
  3x beyond its fold estimate (E-144). That fits a model learning *which states the scorer
  samples*.
- *Cheapest discriminating test (no GPU).* Recompute the M5 table with the E-150 GBM propensity in
  place of prop_v1, using `c4_select.py` with a different propensity array. If the sharper model
  absorbs the selection, the within-bucket gap at equal propensity shrinks by ≥30%. Then run the
  queued q15 GRU A/B (prop_v2 weights vs prop_v1) with `decide.py`, and read the effect by
  rows-to-next with `slice_compare.py`. Expect gains at 3-32 rows, not at 129+.
- *Falsifier.* The gap does not shrink with the GBM propensity (<10%), or q15 gives ≤0 on either
  fold.

**H-T2. Hazard-gated amplitude** (Q3, M4). Ceiling +0.016. Expected +0.000..0.002.
- *Mechanism.* Direction exists only at the race end, so optimal D ≈ P(race ends soon | X) ×
  (direction given the end). The oracle rows-to-next rescale is worth +0.015..0.017 overall and
  +0.05 on class 3. The age of the stretch and the propensity give 0, and the GRU has an aux-next
  (hazard-like) head.
- *Test (script ready, not run).* `c5_hazard.py`: HGB for P(rows-to-next ≤ 8) on causal inputs
  (since-last, since-book/trade/i1, n_book, prop_v1, |D|, L, class), then a class × hazard-decile
  affine readout, cross-fold. About 10 min on 1 core, <2 GB. Stronger version: dump the trained
  aux-next head on valid and use it as the gate.
- *Falsifier.* Gain < +0.001 on either fold pair. Then the ceiling is pure future information
  and this branch closes.
- *Prior.* Low. |D| is itself the best hazard proxy, because a small touch queue signals both the
  side and the imminence.

**H-T3. Factorised class-3 readout: side (first passage) × tier** (M1, M2). Expected +0.000..0.003.
- *Mechanism.* The winning side is the sign of the first mid move (99%). The tier G (how far the
  next 3 events carry the mid) holds ~64% of class-3 weight. On touch rows (69% of rows) the winner
  target is exactly 0, so the wMSE head gets direction gradient from only one target of weight
  0.657. That wastes signal where it is scarce.
- A closed-form readout does not have this waste. Heads: P(U = + | X) (BCE, every row), and
  P(G | U, X) (4-way). Combine them as p* = E[|y|y | X] / E[|y| | X] over the ~12 (U, G, spread)
  cells. Spread is exact in class 3, since the aux head is 1.0 accurate. Continuation after a
  half-move is 84%, which gives an easy prior for G.
- *Cheapest test (no GPU).* Slice the cached class-3 WP by realized tier (touch vs G ≥ 1) and by
  side accuracy. If the GRU loses mostly on G ≥ 1 rows through tier errors with the side correct,
  the tier head is the lever. If it loses on the side, H-T3 reduces to the existing branch head.
  Then run a one-fold GRU A/B.
- *Falsifier.* The fold A/B is ≤0, or the slice shows the tier contributes <20% of the class-3
  shortfall.

**H-T4. Race-end first-passage features** (Q4). Expected +0.000..0.002.
- *Mechanism.* In a replenished queue race the end is signalled by a depletion velocity, not by
  the level. Estimated time to depletion per side: T_a = q̂_a / λ̂_a(depletion: trades at the ask
  plus cancels). Features: log(T_a/T_b) for the side and min(T_a, T_b) for the hazard. Build
  them from the E-090 ICA touch proxies and the dp trade codes. A GRU must learn a ratio of a queue
  level and an EWMA rate across scales, which is a division it approximates poorly.
- *Test.* Event-level partial corr of log(T_a/T_b) with the side given GRU D at 9-32 and 33-128
  rows, ≥300 class-3 sequences, chain halves (the `probe_i1_lead.py` pattern). About 30 min, one
  process.
- *Falsifier.* Partial corr < 0.03 or a sign flip between halves. Prior: modest, since the E-100
  formula already overlaps the GRU at 0.85.

## 4. Closed by this lens (do not reopen without new information)

- Rounding or tie asymmetry of F: 0.68% of rows (M2).
- Within-stretch smoothing or martingale denoising: the class-3 increment autocorr is +0.008, and
  every smoother loses 0.001-0.12 (M3).
- Causal amplitude recalibration by stretch age, propensity or both: 0 to −0.005 (M4). Together
  with E-109 and E-080, output calibration is exhausted.
- Event-clock resampling (E-056), long-memory level deviation (E-020/021), online label
  recalibration (E-080): all consistent with π_t already being a martingale that the GRU tracks.
- "Predict the next event instead of F": for class 3 the side *is* the next mid move (99%). The
  existing branch and aux-next heads already cover it. What remains is the tier (H-T3).

Bottom line: the class-3 gap is a timing-information limit (perfect race-end timing is worth +0.016
as amplitude; no causal proxy reaches any of it). Live levers: H-T1 and H-T3, each ≤ ~+0.003.

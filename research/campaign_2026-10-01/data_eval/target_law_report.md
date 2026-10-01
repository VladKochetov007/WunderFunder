# Target-structure laws and the spread/structure bound (Task B, 2026-10-01)

CPU only, 3 threads, at most 8 GB. Scripts: `scripts/de_target_law.py` (H1 to H4 on TRAIN) and
`scripts/de_target_state.py` (variance shares and the model bound). Outputs: `target_law.json` (summary
and ledger entries), `target_law_train.json` (every variant, fit and held-out, per class),
`target_law_tables.npz` (fitted code tables), `target_state.json`.

**Panel.** TRAIN only. 240 fit sequences and 160 held-out sequences (60 and 40 per class), taken from
component-disjoint halves of the fold-0 manifest components (211 fit and 139 held-out components; at most
3 sequences per component; hash order; no targets read for selection). Held-out components exclude the
earlier E-272 panel. Fit tables:
- spread table: code -> (class, s), where s = -(t0+t1) on both-nonpositive rows; support >= 3, confidence >= .95; 2,905 codes;
- move table: (code_prev, code_next) -> (d_ask, d_bid) at interior-to-interior events; support >= 3, confidence >= .80; 8,563 pairs.

Quote paths are cumulative sums of tabled moves, and any unknown pair breaks the path. Bid is anchored so
that ask - bid equals the tabled spread; this is consistent on 100% of anchored rows. The event marker E is
`a6 == float32(-5.199337482452393)`, a feature only (E-273). Weights are |clip(t)|. Unknown mass (an
unmapped code, a broken path or a censored window) is reported and never imputed. Selections (H3 variant,
H4 N/offset/rounding) are made on fit sequences and graded on held-out ones.

## Laws (held-out, weighted exact share of known rows; unknown share in brackets)

| law | class 0 | class 1 | class 2 | class 3 | verdict |
|---|---|---|---|---|---|
| H1: no E at r => (t0, t1) unchanged | 100.000% [0] | 100.000% [0] | 100.000% [0] | 100.000% [0] | **confirmed, exact** |
| H1: every target change is an E row | 100% | 100% | 100% | 100% | |
| H1: E row => some target changes | 100% | 100% | 100% | 99.995% | |
| class from current code | 100% [10.7%] | 100% [1.6%] | 100% [1.5%] | 100% [0.01%] | |
| H2: interior t0+t1 = -s(code) | 100% [8.5%] | 100% [2.6%] | 100% [2.1%] | 100% [0.02%] | **confirmed on known codes** |
| H3 baseline: loser = -(s+2) | 43.3% [11.4%] | 64.8% [1.4%] | 97.5% [1.3%] | 99.76% [0.01%] | |
| H3: loser = -(s + 2 clip(m,1,W)), m = first adverse cross of the exit-side quote (bid) | 72.2% [57.0%] | 83.0% [20.4%] | 97.9% [4.9%] | 99.8% [1.3%] | **partial** |
| same rows, constant offset 2 | 59.7% | 73.5% | 97.9% | 99.98% | |
| H3, m on mid (no clip) | 5.0% | 16.5% | 10.0% | 18.3% | wrong scale |
| H4: winner = round(mean mid next N quote events - entry), N = 4/6/6/8 | 2.0% [66%] | 12.8% [33%] | 22.6% [10%] | 52.9% [5%] | **rejected** |
| H4 best N <= 32 (fit-selected) | 4.2% (N32) | 19.7% (N24) | 35.3% (N24) | 55.4% (N11) | |
| alt: winner = extreme mid in next N (half-down) | 10.3% (N20) | 39.0% (N24) | 57.8% (N20) | 82.0% (N11) | best simple form; not a law |
| alt: interior common F = round(mean mid next N), N = 16/5/4/4 fit-selected | 36.1% | 65.9% | 82.9% | 96.4% | |

Notes:
- **H1 is the cleanest result.** Targets change if and only if the feature-only quote-move marker fires,
  with no unknown mass. Between events the labels are constant. Plain price-code events without E (book
  refreshes with zero quote move) never change targets.
- **H2** reproduces E-272 on a fresh panel 2.5x larger: the interior level is an exact current-code function
  wherever the code is known, and unknown mass is material only in class 0 (8.5%).
- **H3.** Doubling the loser's first adverse move, clipped to [1, winner], beats constant offset 2 on the same
  known rows in classes 0 and 1 (+12 and +10 points) and ties in classes 2 and 3. The teammate figures
  (86.0/94.5/99.3/99.8%) are not reproduced in classes 0 and 1, and our paths are unknown for 57% (class 0)
  and 20% (class 1) of the loser weight because the move table covers class 0 transitions poorly. The
  exit-side quote (bid for a long loser, ask for a short one) works best; a mid-based m has the wrong scale.
- **H4 as stated is false.** A rounded mean of the next N mids matches at most 55% of class-3 winners and
  under 25% elsewhere. Exactness rises with N and with "ceil" rounding, a sign the winner reaches further than
  a short mean. An excursion rule (best mid within the next N quote events) does better (82% in class 3) but
  is not exact. Inspection of one class-3 sequence shows a piecewise-constant future level that jumps only
  when the price leaves a band, which looks like an exit at a running extreme or with a trailing stop. That
  rule was not identified here. Interior rows share a common reference F with mean-mid N=4 to 96% in
  class 3.

## How much of the metric is current state versus future path

Fold-0 evaluation sequences of the clean manifest (468), rows with step >= 99 and is_scored, clipped
targets, weights |clip(t_k)|. With t_k = S/2 +- D (S = t0+t1, D = (t0-t1)/2):

| quantity | t0 | t1 |
|---|---:|---:|
| share of weighted variance: S/2 | 7.1% | 7.1% |
| share: D | 80.0% | 77.9% |
| share: 2 cov(S/2, D) | 12.9% | 14.9% |
| R2 of the exact current spread (-s/2, linear) | 8.6% | 8.0% |
| R2 of the TRAIN code-conditional mean (out of sample) | 20.0% | 22.2% |
| metric weight on interior rows (S = -s exactly) | 21.3% | 22.4% |

For S itself (weight |t0|+|t1|) the exact current spread explains R2 0.25 and the code-conditional mean 0.39.
About a fifth of the metric-weighted target variance is fixed by the current state. The rest is future
path, mostly direction D; the future part of S is the loser offset and the winner size on outside rows.

## Structural output bound (reference member `cache/pred_c20260929_ft_since_s3_f0_valid.npy`, fold 0)

| prediction | Global WP | delta | class 3 delta |
|---|---:|---:|---:|
| model (p0, p1) | 0.67589 | | |
| p_S replaced by -s(code), p_D kept | 0.63832 | -0.0376 | -0.0151 |
| p_S replaced by TRAIN code-mean S | 0.65686 | -0.0190 | -0.0144 |
| p_S blended 25% / 50% / 75% toward the code-mean S | 0.67488 / 0.67152 / 0.66556 | -0.0010 / -0.0044 / -0.0103 | |
| **oracle: true S on interior rows only** (s exact where S is pure state; needs the future interior flag) | 0.67638 | **+0.0005** | +0.0070 |
| oracle: true S on all rows | 0.70319 | +0.0273 | +0.0685 |
| oracle: true D, model S | 0.97434 | +0.298 | |
| code-mean only (no model) | 0.46144 | | |

The model's "error on s": on interior rows, the |y|-weighted RMSE of p_S against -s is 0.57 target units. This
is not a spread mistake. The model hedges toward the outside branch it cannot see yet, so p_S = E[S | history]
is not -s even where S turns out to equal -s. Forcing the exact spread into p_S costs 0.038.
**Verdict: a structural parameterisation p0 = S/2 + D, p1 = S/2 - D with S taken from current codes cannot
help.** Even an oracle telling which rows are interior gains only +0.0005, concentrated in class 3 (+0.007,
with class 2 at -0.0045). Every realistic substitution or blend loses. The upside left in S (+0.027 with a
perfect S) is future-path information (loser offset, winner size), not spread. Practical leads: the H1 event
structure (labels constant between quote-move events) and the class-3 interior common-F regularity. Both
are about predicting the future level, not the spread.

Caveat: exactness tables are conditional on mapped codes and paths, and class 0 has large unknown mass.
One fold-0 member is the only model graded; the bound is an oracle construction, not a candidate.

# Online-learning setups for this task

The submission callback is `predict(DataPoint)` with `seq_ix`, `step_in_seq`, `need_prediction`, and 112 features. It never receives `t0`/`t1`. Warm-up is steps 0–98: the model may update state, it may not emit a score. Any setup that needs labels at test time is a training-time recipe only.

Measured on the frozen seed-4 blend (`wave1_online.json`, `wave1_slices.json`). Full-file blend 0.700771 is the supplied file, not a public forecast.

| Setup | Test-time legal | What happened |
|---|---|---|
| Hold the first post-touch output | yes | Far-cell WP 0.291 vs 0.482. Full file 0.619. |
| Causal EMA of the blend, reset on each joint touch, decay `e^{-1/8}` | yes | Far 0.478. Full file 0.693. |
| Same EMA, decay `e^{-1/32}` | yes | Far 0.454. Full file 0.654. |
| Hold last-trade output on far rows with no trade in 8 | yes | Cell 0.319 vs 0.424 (−0.104). Full file −0.012. |
| Far rows with a trade in the last 8 | diagnostic | 0.549 (369,820 rows) |
| Far rows with no trade in the last 8 | diagnostic | 0.424 (583,044 rows) |
| Far rows with a one-sided i0 price code in the last 8 | diagnostic | 0.666 (49,039 rows) |
| Far rows with i1 motion in the last 8 | diagnostic | 0.481 (same as all far) |
| Near rows (next label change ≤ 8) | diagnostic | 0.859 |

Volume moved in the last 8 rows on 93% of far rows and 97% of near rows. The far gap is not a frozen book.

## Setups that are still coherent

1. **Training-time episode objective.** The label is constant between joint touches. Horizon-8 bootstrap already replaces `y` with the teacher on the far rows, and `since-weight` already downweights them. One loss per episode, or true `y` at full metric weight on those rows, is a training-time online recipe. It is the only setup that sits on the 0.424 no-trade far mass. Kill at +0.005 WP on that cell. It does not fit a GRU256 in 10 GB.

2. **Trade-triggered two-timescale filter.** Fast weights or extra hidden writes on trade/one-sided codes, slow or frozen writes on the no-trade far rows. The blend is already 0.12 WP better when a trade just happened, and 0.18 better when a one-sided price code just happened. That is a router, not a new residual. i1 motion is not a router.

3. **Unsupervised test-time training on next joint touch.** Legal: no labels. Fit a tiny intensity head online from features to "AND within 8 rows," then mix a far/near readout. The oracle split is 0.48 vs 0.86. A causal intensity that is only as good as "recent trade" would mix 0.42 and 0.55, which does not reach 0.86.

4. **Streaming class-3 decoder plus a residual that only updates on trades and touches.** Filter in the engineering sense, not SGD. Lattice snap of this blend already lost. A soft decoder is a different head; the linear book residual already failed.

## Setups that this evidence closes

- Supervised online SGD from revealed labels at test time: the API does not reveal them.
- Output-space tracking of a piecewise-constant level (hold, EMA, last-trade hold): the latest GRU output is already better than any causal copy of an older output.
- "Do not update on quiet rows" as an output hold: measured −0.085 to −0.125 full-file WP. Quiet rows still have volume and trades.

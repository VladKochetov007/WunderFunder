# Visual analysis

Renderer: `wnn.viz.render_sequence` (per-pixel min/max envelopes, per-pixel event
counts, mean heatmaps; ~0.9 s per full 20k-row sequence). Panels: targets in
quanta (t0 blue, t1 orange, predictions black if given), sum of sign(dp0),
event counts (i0 price, i0 trade, i1 price), i0 volume block, i0 price block,
scored rows, a0-a7.

## 01 - class 3, full sequence (valid row group 3)

Class 3 alternates active bursts (i0 price events every ~3 rows, a bid/ask
bounce with targets at -1/0 quanta) and quiet stretches where targets reach
+-4 quanta. Scored rows sit mostly in quiet stretches, and |y| weighting makes
the large quiet-stretch targets dominate the class-3 metric.

## 02 - class 3, rows 5500-6200 (quiet stretch)

One i0 price event with a trade at ~5880 moves targets from (0, -1) to
(-3, +5) while the reconstructed level moves +2. Later events shrink the
magnitude (t1 5 -> 4 -> 3) as the level reverts. This looked like
target = anchor - current price with a slow anchor (H-016), but E-021 showed
the large jumps are anchor changes carrying future information: decoded price
jumps summed over recent events score ~0, the same sums of true target jumps
score 0.82 (leaky control).

## Reproduce

    scripts/limited .venv/bin/python -c "from wnn.data import SequenceFile; from wnn.viz import render_sequence; s=SequenceFile('datasets/valid.parquet').read(3); render_sequence(s,'out.png',window=(5500,6200),width=700)"

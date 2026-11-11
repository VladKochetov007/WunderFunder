# Fresh-context reframe brief: large-tick direction prediction (2026-10-25)

You are an independent quant/ML researcher. Only this brief is available. Plain Markdown, <=1,500 words.

## Setup
Streaming causal prediction on anonymized limit-order-book data, two instruments (i0, i1),
112 features/row (rank-Gauss-like transformed; book prices are global nonlinear encodings,
all bid columns change together; sizes, trades, 8 extra features). Sequences of 20,000
irregularly spaced rows; state reset per sequence; CPU-only streaming inference.
Targets for i0: t0 = F - ask, t1 = bid - F for a near-future price F (reached about one
book event after it changes). Metric: pooled |y|-weighted Pearson, y,p clipped to ±2.

Four "market classes" by tick quantum. Class 3 (largest tick) has ~46% of metric weight on
the public test but our correlation there is ~0.44 (other classes 0.74-0.81). For class 3
the target law is exact: with spread sp (ticks), U = F - bid:
F inside [bid, ask]: t0 = U - sp, t1 = -U; F above ask: t0 = U - sp, t1 = -(sp+2);
F below bid: t1 = -U, t0 = -(sp+2). Knowing the 3-way branch gives WP 0.968 (label oracle).
Our GRU's causal branch-direction AUC is ~0.74 (≈0.55-0.60 far from a price change,
≈0.83 within 20 rows of one). Class 3: low trade rate, 34% zero labels, t0/t1 corr -0.87.
94% of residual error is a per-label-run level offset; error grows with rows since the last
label change (corr 0.82 at run start → 0.60 after 16+ rows). Label changes at book events
are trivially predictable from the quote move itself (AUC 1.0) — the hidden part is F's
move relative to the quotes. Members (all GRU-family) share 96-99.9% of residual.
Training already uses |y|-weighted MSE, propensity weighting for scored rows, heterogeneous
recurrent-teacher distillation, aux heads for spread/branch/class, event/clock features,
causal "since last price event" weighting, queue-imbalance EMAs.

## Ask
Propose 5-7 genuinely different mechanisms to raise class-3 correlation, drawing on
market microstructure (large-tick queue dynamics, queue depletion races, Hawkes order flow,
cross-instrument lead-lag, hidden/iceberg liquidity), statistics, or ML. For each: the
structural reason it carries information our current setup lacks; a zero-fit or cheap
diagnostic that would show the information exists (e.g. a probe on held-out sequences);
falsifier; serving cost. Include at least one idea that uses instrument i1, one that changes
the target/representation for class 3, and one unconventional idea. Then name the single
assumption in this brief you think is most likely wrong.

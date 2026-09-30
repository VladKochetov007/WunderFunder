# Current scoring reference: global weighted Pearson

Verified against the organizer's [data overview](https://wundernn.io/connectome/docs/data_overview) and [FAQ](https://wundernn.io/connectome/docs/faq) on 1 October 2026. The current score pools selected rows across sequences. It does not average per-sequence correlations or apply the old block-eligibility filter.

For each target separately:

1. Select `need_prediction AND is_scored` rows from the evaluated dataset.
2. Clip targets and predictions to `[-2, 2]`.
3. Use the absolute clipped target as the row weight.
4. Compute weighted Pearson correlation over all selected rows together.

The final score is the arithmetic mean of the correlations for `t0` and `t1`. Higher is better. Average ensemble predictions before the scoring clip. A pooled score cannot generally be recovered by averaging fold or sequence correlations.

The current research implementation is [src/wnn/gmetric.py](src/wnn/gmetric.py): `global_wp` for arrays and `GlobalAccumulator` for streaming. [scripts/fold_eval.py](scripts/fold_eval.py) evaluates selected validation chains using the same six sufficient statistics per target. Computation uses float64 accumulation; local implementations return zero for degenerate sides.

Model state must still reset at each new sequence and update on every row, including the 99 warm-up rows. The hidden scoring mask is unavailable at inference. These stream rules are independent of the global score aggregation.

The imported [block metric document](METRIC_BLOCK_HISTORICAL.md), `utils.BlockAccumulator`, and `src/wnn/metric.py` are retained for reproducing historical experiments. They implement a different aggregation and must not be used to label a new score as the current competition metric. Their tests and old results have not been rewritten.

See the [source-verification record](research/iterations/2026-10-01-next/metric-source.json). Documentation verifies the organizer's declared metric; it is not an audit of the platform's internal scorer.

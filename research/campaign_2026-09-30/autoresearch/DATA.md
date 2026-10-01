# Data

Root supplied TRAIN/VALID parquet, 20,000 rows per independent sequence, 112
ordered causal features. Update state on every row, including warmup0..98 and
unscored rows. No sequence ID, target, scoring-mask or future-event information
enters inference state. Train-only maps expose unknown support explicitly.

Frozen chain-excluded fold manifests and fold-specific propensity fits identify
fitting rows. Validation folds and historical holdout have been inspected before;
do not describe them as untouched. Public incumbent development predictions are
in-sample and cannot serve as a clean OOF baseline. The clean since/since panel
is a matched recipe reference, not the public archive's fitted weights.

External MD/Binance records are authorised for distribution/publication comparisons.
No external numerical transforms, records or fitted model state enter training.
The scorer uses immutable complete prediction exports and target cache identities.

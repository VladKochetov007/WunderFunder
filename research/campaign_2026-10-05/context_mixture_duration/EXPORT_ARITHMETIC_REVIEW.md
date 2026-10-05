# E488 raw-export arithmetic review

The independent arithmetic helper is approved for use on the already-created E488 score JSON and raw prediction export. It does not open Parquet, checkpoints, or fit sources. It recomputes the clipped weighted-Pearson target scores from float32 raw targets/predictions using shifted float64 component sufficient statistics, regenerates all 2,000 numeric-sorted component bootstrap draws from the registered seed, and checks every one of the 24 snapshots, 27 global contrasts, and 54 target summaries. The duration interaction retains the registered four-term sign pattern within each seed; seeds are never pooled.

The helper checks exact archive and summary inventories, finite raw values before clipping, selection masks/counts, unique component and row-group IDs, snapshot parameter/update counts, bootstrap indices/multiplicities, score/export byte hashes, and the reported numeric summaries at absolute tolerance `1e-10`. Its synthetic direct weighted-Pearson oracle and mutation tests cover clipping, nonfinite rejection, duplicate groups, omitted/altered contrasts, and the complete snapshot/contrast matrix.

Reviewed source hashes:

- `reproduce_duration_exports.py`: `2c3f606ddec0dc4ceab56cdea1a6bf7302177ae6bd68d791e442d9d1c3afab8d`
- `test_reproduce_duration_exports.py`: `1fb1940e19818672e2cf5e96306764b74c29afc96c0fda48c5bd1428e4bce6fe`

Validation passed under `scripts/limited` with 1 CPU and a 2 GiB memory cap: all 7 synthetic tests, Ruff, and `py_compile`. The tests exercise only in-memory synthetic fixtures. No actual E488 score values or raw export arrays were opened by this review.

## Separate renderer finding

Renderer v3 is not approved yet. Its global and target contrast plots pass asymmetric error distances to Matplotlib as `xerr = [[point-low], [high-point]]`. A valid percentile interval can exclude the point estimate, making one distance negative; a synthetic check produced `ValueError: 'xerr' must not contain negative values`. The current renderer tests construct intervals that contain their point estimates, so they miss this case. Preserve the percentile interval as drawn and show the point estimate independently when it lies outside the interval; do not clamp or alter the interval. Renderer v3 source/test hashes checked: `b5c5235324dcaffbde622f76ba281c974af29b1b73311f06bed2b7179720a81b` / `cbcbb304223155f754279213491841e67ff16a08499f2fbfe3d091d57821ed50`. Its three synthetic tests pass, but the adversarial interval fixture fails as described.

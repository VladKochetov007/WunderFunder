# E477 v6 fit-only cohort check

The v5 scorer fixed `load_test_sequences`, but root review caught a duplicate full-cohort equality check in `evaluate` that would still reject the fit-only receipt. The v5 source remains preserved as an unexecuted revision. The v6 scorer removes only that duplicate comparison and delegates the provenance gate to `load_test_sequences`, which requires the receipt to equal the canonical 24 fit-role nodes and obtains test16 separately from the full source cohort.

I tested the full `evaluate` provenance path against the actual `fit_only_cohort.json` and fit manifest. The test intercepted execution immediately before the first test parquet read: it confirmed all 24 fit nodes had passed the fit-normalization gate and the next node was a canonical test-role component. Prediction scoring did not run. The provenance helper also rejected the full 40-node cohort, a fit list missing one node, and a fit list contaminated by a test node. `py_compile` and Ruff passed.

v3/v4/v5 scorer revisions and the failed v4 attempt were left unchanged. v6 SHA-256: `2858751cf670c2c766a3f8f1276c2d19ca6fe5bad9d41b19cb54013d1d535239`.

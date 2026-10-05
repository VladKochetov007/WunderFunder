# E477 v5 fit-only cohort validation

The v4 CPU grade failed before evaluation because `fit_only_cohort.json` correctly contained only the 24 fit groups, while `load_test_sequences` compared it with the full 40-node cohort. The failed v4 attempt and its outputs were preserved; v3 and v4 scorer source files are unchanged.

The separate v5 scorer changes only the provenance comparison: it requires the copied cohort to equal the canonical 24 fit-role nodes in source order, validates that all copied entries have role `fit`, and obtains the 16 test nodes separately from the canonical cohort. Fit-only normalization continues to be recomputed from fit groups before test labels are read. No test predictions or scores were evaluated during this review.

Using the actual `fit_only_cohort.json`, fit manifest, and canonical cohort file, the corrected check accepted exactly 24 fit and 16 test groups. Synthetic negative cases for the full 40-node cohort, a missing fit node, and a test node mixed into the fit copy were all rejected. `py_compile` and Ruff passed.

The reviewed v5 scorer SHA-256 is `a1f5b9a44e12267cf7a4655e04520216ec94719a29797c4d8f918e903d5bc984`. The root may now seal and run this corrected scorer identity.

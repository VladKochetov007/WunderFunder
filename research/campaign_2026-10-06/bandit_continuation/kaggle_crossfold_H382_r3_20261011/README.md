# H382 Kaggle delivery R3

R3 is the identified startup-integrity correction for the three preregistered
fold-0 H382 Muon-all comparisons. It preserves the H382 fold, cohort, fitting
membership, source snapshot, trainer, recipe, timeout limits, predictor, and
full-validation endpoint from R2. Its only worker-code change verifies the
Parquet footer range using the producer's `footer_bytes + 8` contract, including
the length prefix and `PAR1` trailer. See `H382_R3_STARTUP_REPAIR.md` for the
observed R2 mismatch and exact correction.

The private kernel is
`vladkochetov07/wnn-kaggle-crossfold-h382-20261011-r3`; its unique source-bundle
dataset is
`vladkochetov07/wnn-kaggle-crossfold-h382-bundle-20261011-r3`. The builder
creates a new deterministic bundle, kernel bootstrap, Kaggle metadata, packet
manifest, and run IDs. It does not upload, create a dataset, push, or launch.

The final exact R3 packet requires root review and a fresh quota/admission
receipt. R2 remains unchanged until its exact provider terminal status and
outputs are reconciled. R3 must not be dispatched before that reconciliation.

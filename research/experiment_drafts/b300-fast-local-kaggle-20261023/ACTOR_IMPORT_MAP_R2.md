# Integration map

`SOURCE_SHA256_R2.json` is the portable R2 source pin. Keep R1 unchanged as the
earlier local implementation snapshot. Copy the authored modules to these
canonical repository paths before review or execution:

- `source/src/wnn/training/age_readout_*.py` → `src/wnn/training/age_readout_*.py`
- `source/src/wnn/training/age_readout_recipe.py` → `src/wnn/training/age_readout_recipe.py`
- `source/scripts/age_readout_screen.py` → `scripts/age_readout_screen.py`
- `source/scripts/check_age_readout_invariants.py` → `scripts/check_age_readout_invariants.py`

The authored files reuse canonical `wnn` data, feature, cache, membership,
objective, checkpoint and provenance components. `age_readout_portable_builder.py`
rebuilds a single 117-column fold from raw Parquet groups and the transported
suffix, then checks the complete original NPY SHA before fitting. The portable
validator checks all six sidecars against the frozen local receipts and root
metadata attestations. The path map keeps the logical local cache/source/output
identities beside the actual Kaggle bundle, raw mounts and output paths. The
runner releases mapped cache pages at complete sequence boundaries, closes both
cache mappings, removes each completed fold cache and records live bundle/work
file-size peaks.

The `source/src/wnn/__init__.py`, `source/src/wnn/training/__init__.py`,
`source/scripts/train_rnn.py` and `source/scripts/check_training_structure.py`
entries are links used only to inspect this isolated source packet with the
repository structure checker. Do not copy those links over project files. The
source-check harness extends the canonical package search path in memory so the
actor modules can be tested before the root integrates them; that harness is not
part of production execution.

The actor ran Ruff format/check, Python compilation, the repository training
structure checker and bounded CPU synthetic delivery invariants. It did not read
real cache arrays, rebuild a Kaggle cache, run a profile/fit, grade a candidate,
or call Kaggle. See `SOURCE_SHA256_R2.json` and the parent scout receipt for the
exact source and check identities.

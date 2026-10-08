# H217 A3 TCN fold-0 independent grade

The delivered TCN checkpoint passed its recorded fit, fixed-epoch export, and profile gate. This grade evaluates only the prespecified epoch-3 FP32 prediction on the 468 excluded fold-0 valid groups using `global_wp_clip2_v1`.

- Official pooled WP: **0.671496804** (t0 0.661932549; t1 0.681061058)
- Selected rows: 808,863 across 468 evaluation groups
- Checkpoint SHA-256: `5fbb9815b17973dd790cd10ee11af63c1a241ca716df32d44a5b56f099af7dbc`
- Prediction SHA-256: `d178d84a0bbd74158e5788713973fd7b5c16d5d6f71e6bcc976bce44e883f3aa`
- Fold-0 split SHA-256: `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`
- Actual fit membership: 10,535 train-source plus 1,405 valid-source groups; exact match to the frozen manifest, with zero fit/evaluation component overlap

The fold is exposed and this is one training seed, so the result is discovery evidence rather than fresh confirmation. The matched GRU arm did not fit because its projected four-epoch fit/export exceeded its 7,200-second arm budget; it is not used as a comparator. No blend, all-data comparison, post-hoc slice, or public forecast is reported.

Targets were read from the campaign scorer's memory-mapped stored NPY members inside `cache/valid_targets.npz`. The prediction was memory-mapped from the delivered artifact tar. No Parquet footer, compressed NPZ member, fit, GPU, or provider operation was used.

Machine-readable receipt: `H217_A3_TCN_FOLD0_INDEPENDENT_GRADE_20261008_R1.json`. Re-run from the repository root with `scripts/limited` and the same one-CPU/1.5-GB/120-second limits:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/H217_A3_GRADE_TCN_FOLD0_V1.py
```

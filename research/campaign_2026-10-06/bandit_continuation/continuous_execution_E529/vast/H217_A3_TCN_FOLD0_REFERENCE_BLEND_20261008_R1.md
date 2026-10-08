# H217 A3 TCN conditional fold-0 comparison

This continuation compares the fixed TCN against the own seed-3 `c20261003_combo_s3_f0` epoch-3 parent on the same 468 exposed fold-0 groups. The TCN standalone outcome had already been observed when this comparison was requested, so this is adaptive discovery analysis; the parent and blend outcomes were not inspected before the fixed weight set was registered. Every blend weight is reported and none is selected.

Parent score: **WP 0.680941207** (t0 0.674463034; t1 0.687419380).

| TCN weight | WP | t0 | t1 | delta vs parent | paired component 95% CI |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.680941207 | 0.674463034 | 0.687419380 | +0.000000000 | [+0.000000000, +0.000000000] |
| 0.10 | 0.680809428 | 0.674075172 | 0.687543685 | -0.000131778 | [-0.000476314, +0.000212422] |
| 0.25 | 0.680279989 | 0.673139584 | 0.687420394 | -0.000661218 | [-0.001542473, +0.000211033] |
| 0.50 | 0.678496113 | 0.670619827 | 0.686372398 | -0.002445094 | [-0.004274731, -0.000684219] |
| 1.00 | 0.671496804 | 0.661932549 | 0.681061058 | -0.009444403 | [-0.013408911, -0.005920220] |

Pair prediction relationship on the same official mask and target-derived weights:

- t0: weighted covariance 0.826440867; weighted correlation 0.985840602
- t1: weighted covariance 0.877788678; weighted correlation 0.987708508

The parent, its two own since-teacher ancestors, and its own propensity source all bind to the fold-0 split and have zero fit/evaluation component overlap. The TCN is raw-only scratch; the parent GRU uses those teacher/propensity ancestors, so this is conditional prediction complementarity, not a matched family or compute-controlled comparison. The bootstrap is conditional on fixed predictions and exposed components; it does not estimate training-seed or public-transfer uncertainty. No all-data comparison or public forecast is made.

Targets use the frozen scorer's memory-mapped stored NPY members inside `valid_targets.npz`; the parent prediction is memory-mapped directly and the TCN prediction is memory-mapped from its delivered tar member. No Parquet footer, compressed NPZ member, fit, GPU, or provider operation was used.

Machine-readable receipt: `H217_A3_TCN_FOLD0_REFERENCE_BLEND_20261008_R1.json`. Re-run with `scripts/limited` under one CPU, 1.5 GB RAM, and 120 seconds:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/H217_A3_GRADE_TCN_REFERENCE_BLEND_V1.py
```

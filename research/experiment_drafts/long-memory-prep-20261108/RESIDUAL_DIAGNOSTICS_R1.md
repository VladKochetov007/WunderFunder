# TL3 same-row residual diagnostics R1

Status: completed score-only diagnostic; see [the native results](RESIDUAL_RESULTS_R1.md). It uses
the pinned R4 role caches while the separately approved six-fit B300 campaign
continues; it does not alter that campaign or admit any L4 fit. R1–R4 grader
files remain frozen.

## Inputs and safeguards

`residual_diagnostics_r1.py` imports the exact role pins, split validator,
sidecar/provenance validator, checkpoint hash checks, cache hash checks, and
source-row alignment check from `grade_tl3_roles_r4.py`. It reads the three
FP32 memory-mapped validation arrays and `t0`, `t1`, `is_scored` columns from
the official `valid.parquet`, one complete sequence group at a time. It does
not deserialize checkpoints, execute inference, write caches, or use training
rows. The fixed 0.34/0.33/0.33 blend is the existing reference arithmetic;
there is no blend search.

The create-only output JSON binds the primary split identity and raw manifest
file hash, full `valid.parquet` hash, R4 validator SHA, diagnostic script SHA,
and the checkpoint, prediction, sidecar, and provenance identities for each
role. It rehashes these inputs after scoring and fails if bytes changed. It
reports exactly 1,873 validation sequences and 3,528,273 official scored rows
or fails. R4 already enforces that provenance membership sets match the split
and that each prediction array's `row_groups` remain in exact official source
order.

## Metric definitions

- **WP:** `wnn.gmetric.GlobalAccumulator` semantics. For each target, clip
  labels and predictions to `[-2, 2]`, weight rows by the absolute clipped
  label, and compute one pooled weighted Pearson across all selected rows.
  Report `t0`, `t1`, and their mean. Near-constant or zero-weight target
  variance follows `gmetric`'s zero-score convention.
- **WMSE:** monitor-compatible weighted squared error per target. Use raw
  predictions and clipped labels: `sum(abs(y_clip) * (prediction_raw -
  y_clip)^2) / sum(abs(y_clip))`. Also report the pooled two-target numerator
  divided by pooled weight. Denominators (target weight mass) are emitted.
- **Residual association:** for each of the three role pairs and each target,
  report weighted Pearson between residuals on identical official scored rows,
  with weights `abs(y_clip)`. To make clipping semantics explicit, the JSON
  includes two series: `prediction_raw - y_clip` (the WMSE residual) and
  `clip(prediction_raw, -2, 2) - y_clip` (the residual implied by the official
  WP clipping). Residual values themselves are not clipped. Zero-variance
  residual association is 0; no selected observations gives a null correlation.
  The selected-row count and positive-weight count are included.

The reported **fixed blend** is computed FP32, left-to-right using 0.34, 0.33,
0.33, with no clipping before WMSE. WP's `GlobalAccumulator` clips the blend
predictions as it does every role prediction.

## Fixed diagnostic regions

The validity mask is `is_scored AND position >= 99`, using zero-based row
positions. Position bins are half-open and fixed before reading targets:

| Name | Positions |
| --- | --- |
| `pos_0099_0255` | `[99, 256)` |
| `pos_0256_0999` | `[256, 1000)` |
| `pos_1000_1999` | `[1000, 2000)` |
| `pos_2000_4999` | `[2000, 5000)` |
| `pos_5000_9999` | `[5000, 10000)` |
| `pos_10000_19999` | `[10000, 20000)` |

For candidate horizons 256 and 512, a boundary lies between positions `b-1`
and `b`, where `b` is an interior multiple of the horizon. `boundary_near`
contains positions `[b-8, b+8)` (eight rows on each side); `interior` is the
remaining scored positions. This is a predeclared positional grid aligned to
upcoming Transformer horizons, not a verified TL3 inference chunk or reset
boundary. Near/interior differences are descriptive and cannot diagnose
causality or reset bugs because the row populations differ. No boundary was
selected from score data. Each result contains rows, sequences with selected
rows, per-target weight mass, and per-target WP/WMSE for every role and the
fixed blend. The position bins partition all scored rows. For each candidate
horizon, boundary-near plus interior also partitions all scored rows. Empty slices emit
`selected_rows: 0` and `wp: null`; pairwise residual correlation is also null
with no selected rows. A missing WMSE denominator yields `wmse: null`, not a
fabricated zero.

## Local checks

Focused tests verify fixed region partitioning and counts, raw-prediction
WMSE's mask/clipping/weight semantics (including ignoring unselected
nonfinite rows and rejecting selected nonfinite values), explicit empty-slice
handling, and weighted residual Pearson against an independent direct formula.
No local full-valid scoring or model execution is part of these checks.

ROOT's bounded native CPU invocation should use a fresh output path:

```bash
R=/data/epanov_s/tabdata/runtime/transformer-optimizer-horizon-20261108-r1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$R/source-r2/src" \
/usr/bin/timeout --kill-after=10s 900s \
  /data/epanov_s/tabdata/.venv/bin/python "$R/operator/residual_diagnostics_r1.py" \
  --project-root /data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/project \
  --split-manifest /data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data/chronological_primary.json \
  --output "$R/grading/tl3-residual-diagnostics-r1.json"
```

The staged operator folder must also contain the exact frozen R4 validator.
This is a one-shot create-only result path: inspect existing receipts before
invoking it; a second invocation is not an automatic retry.

The successful output is still development evidence on an exposed validation
set; it is not independent, public, or serving evidence. The historical
mixture's consumed-cache byte ancestry remains unreconstructed.

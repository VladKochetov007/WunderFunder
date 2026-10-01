# Sequence Prediction Challenge

Current results, the best recorded submission (**V1: 0.6641**), and links to the complete checkpoint/log archive are in [Research current state](research/CURRENT_STATE.md). The [completed creative-research iteration](research/iterations/2026-09-30-creative/report.md) records the methodology fixes, preregistered screens and outcomes. [METRIC.md](METRIC.md) gives the current global scoring definition, verified against organizer documentation on 1 October 2026.

The [overnight research report](research/iterations/2026-10-01-overnight/delivery-report.md) records 52 training runs and 36 registered score readouts, with no new forecasting recipe promoted. The deterministic fixed-three package passed a full runtime audit on the measured v3 host; the unchanged historical V1 failed a fresh 60-minute audit. The [local delivery folder](../delivery-2026-10-01/README.md) contains both and experimental INT8, with a portable research archive planned. Archive completion requires a PASS in its named byte-verification receipt. New work is local/on beleriand only; GitHub publication remains paused.

Predict `t0` and `t1`, two indicators of future price movements of instrument
`i0`, from the L11 book-feature representation, trade and additional features.

## Data

Each input has 112 float32 features: two 52-feature book/trade groups
(`i0`, `i1`) and eight additional columns (`a0` through `a7`).
For each instrument, L11 denotes 11 price-like and 11 volume-like features
on each side of the book, bid and ask, plus the separate trade features.
The complete column order is in [Data overview](docs/data_overview.md).

| Dataset | Sequences | Rows |
|---|---:|---:|
| Train | 10,607 | 212,140,000 |
| Validation | 1,873 | 37,460,000 |

Every sequence has 20,000 rows. Steps 0–98 are warm-up; predictions are
required at steps 99–19,999. Process rows in order and reset state between
sequences.

## Metric

The metric is global Weighted Pearson correlation (WP), using weights
`abs(clipped target)` on rows selected by `need_prediction AND is_scored`.
Pool these rows across sequences, compute one correlation per target, and
average the two target correlations. Targets and predictions are clipped
to `[-2.0, 2.0]` before scoring. Higher is better.

The validation mask is provided for local scoring. The test mask is hidden.
See [METRIC.md](METRIC.md) for the full scoring reference.

## Files

- `datasets/train.parquet`: labeled training sequences.
- `datasets/valid.parquet`: labeled validation sequences with `is_scored`.
- `datasets/valid_mask.parquet`: the validation mask as a separate table.
- `utils.py`: callback types and input-column definitions.
- `METRIC.md`: scoring specification and reference functions.
- `baseline/`: ready-to-run stateful ONNX baseline (inference only).
- `docs/`: competition website documentation.

## Baseline

The included vanilla GRU baseline scores **0.589595 WP** on the complete
validation set. It runs statefully on CPU with one ONNX Runtime compute thread.

```bash
python baseline/solution.py --validation datasets/valid.parquet
```

The baseline contains only inference code and an ONNX model. A ready-to-upload
archive is provided as `baseline/baseline_submission.zip`; no training code or
training checkpoint is included.

## Submission

Provide `solution.py` with `PredictionModel()` and `predict(data_point)`.
The callback receives the sequence identifier, step, `need_prediction` and
the current 112-feature vector. Return `None` during warm-up and two finite
predictions in `[t0, t1]` order on every required row.

Package the solution and its required model files in one ZIP, with
`solution.py` at the root. See [Quick Start](docs/quick_start.md) and
[Submission guide](docs/submission_guide.md).

## Execution limits

- **CPU:** 1 vCPU; CPU inference only, no GPU.
- **RAM:** 16 GB.
- **Time:** 60 minutes for the entire test set.
- **Submission size:** at most 20 MB for the uploaded ZIP.

The submission guide specifies Linux with Python 3.11 and no internet
access. CPU instruction-set capabilities are not specified; runtime checks
must state which native path was actually measured.

## Website pages

- [Data overview](docs/data_overview.md)
- [Quick Start](docs/quick_start.md)
- [Submission guide](docs/submission_guide.md)
- [FAQ](docs/faq.md)
- [Competition Rules](docs/rules.md)
- [Timeline](docs/timeline.md)
- [Prizes](docs/prizes.md)
- [Get Help](docs/get_help.md)

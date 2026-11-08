# Point 2 B300 receipt: component and objective observability

Status: implemented and locally checked; no fit, resume, GPU access or new
predictions. The scoring path is unchanged. New distribution diagnostics reuse
the existing epoch-monitor forwards and keep only scalar sufficient statistics.

## Producer changes

`sequence_monitor_evaluation.evaluate_dataset` remains available at its previous
import path but is now implemented in
`src/wnn/training/sequence_monitor_dataset.py`. This keeps the shared evaluator
below its 300-line structure limit and gives one place for batch-level metric
accumulation.

For configured B300 monitoring (`blend_alpha=0.1`, report-only standalone mode),
the same raw prediction tensor supplies `z`; the already-loaded reference cache
tensor supplies `reference`; and existing `compose_prediction` supplies the
fixed `mixture`. All use the same declared cohort and actual row mask. The
official generic mixture fields (`wp`, per-target `wp`, `loss`, selected rows,
sequence count, prediction mean/std and clipping fraction) use the prior
calculation path unchanged. The evaluator adds per-target raw and official-weighted
means/stds for z, reference and mixture; prediction clipping count/fraction and
official target-weight mass on clipped predictions; plus raw/clipped target
moments, absolute/official weight mass and target clipping effects.

When standalone reporting is enabled without a blend alpha, only z is reported;
the evaluator no longer labels z as a reference or mixture. The B300 monitor
always supplies alpha `0.1`, so its fixed-mixture primary output is unchanged.

Training objective and backward code were not changed. Existing per-update
`rank0.jsonl` fields are preserved: `objective_mode`, `objective_wmse`,
`objective_correlation_loss`, `wmse_objective_weight`,
`correlation_objective_weight`, `total_loss`, `objective_aggregation`, and
`label_loss_is_curvature_scaled`; treatment rows also retain
`microbatch_correlation_surrogate_t0/t1`,
`correlation_prediction_std_floor`, and `correlation_scope`. These are detached
optimizer-window measurements. Treatment correlation is an attenuated
microbatch surrogate, not the epoch-global WP; label and treatment objective
values are not directly comparable. Dashboard mapping belongs to the L4 actor.

## Checks

Pinned Python: `C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe`.
Commands ran with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`; pytest used
`PYTHONPATH=<worktree>/src;<worktree>`.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONPATH='C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108/src;C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108'
& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m pytest tests/test_sequence_monitor_diagnostics.py `
  tests/test_pr67_receiving.py::test_report_only_monitor_composes_before_clipping_and_restores_rng `
  tests/test_ensemble_objective.py -q
# 13 passed in 4.66s

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m ruff format src/wnn/training/sequence_monitor_dataset.py `
  src/wnn/training/sequence_monitor_evaluation.py `
  src/wnn/training/sequence_monitor_head_metrics.py `
  tests/test_sequence_monitor_diagnostics.py
& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m ruff check src/wnn/training/sequence_monitor_dataset.py `
  src/wnn/training/sequence_monitor_evaluation.py `
  src/wnn/training/sequence_monitor_head_metrics.py `
  tests/test_sequence_monitor_diagnostics.py
# All checks passed.

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  scripts/check_training_structure.py `
  --root C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108
# 115 files, 0 errors; module limit 300 lines, function limit 200 lines.
```

The independent synthetic fixture includes sparse masks, clipped targets and
predictions, and distinct reference/z values. It checks official mixture WP and
per-target WP against a standalone weighted-Pearson formula, and mixture WMSE
against a separate float64 formula. It also checks all existing non-timing
generic fields bit-for-bit with and without the diagnostics, verifies score
aliases, and checks per-target raw/weighted moments, weights and clipping.
Instrumented evaluation makes six model calls with or without diagnostics and
preserves RNG state, the pre-existing parameter gradient and model mode. A
no-alpha test confirms z is reported without reference or mixture keys.

## Limits

Normal primary coverage remains 1,873 official valid sequences and 3,528,273
`is_scored` rows; no production membership was evaluated in this CPU fixture.
The monitor audit reports approximately 9.47% valid row density under the
official score mask; this repair did not remeasure it. Validation remains
exposed discovery data, and no quality or serving change is established.

The legacy generic `valid/loss` is intentionally unchanged, including its
existing float32 denominator summation; the fixture verifies it remains exactly
equal to the old path. The independent float64 WMSE oracle is compared with the
component `valid/mixture/wmse`, whose accumulator uses float64 weights. This
does not alter official WP. Sufficient-statistic CPU and memory overhead was
not profiled on production cohorts; no extra forward or full-cohort prediction
buffer was added. Exact quantiles are deferred rather than retaining selected
predictions.

Post-review mask check: `tests/test_sequence_monitor_diagnostics.py` was rerun
alone with the one-thread environment above after adding a tiny masked-NaN case:

```powershell
& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m pytest tests/test_sequence_monitor_diagnostics.py -q
# 3 passed in 4.02s
```

The check injects NaNs into unselected targets and each component prediction,
then requires the WP and streaming moment dictionaries to match the clean
selected fixture exactly. Selected nonfinite targets, predictions and target
moments fail closed. This exercises only accumulators; it performs no model
forward.

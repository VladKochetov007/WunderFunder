# Point 1: L4 train and valid diagnostic comparability

Status: implemented and locally checked; no fits, profiles, or GPU operations.
Worktree base: `96823b7aa2ac90de60ca2088479da05e66517251` on
`fix/training-observability-20261108`. ROOT owns integration and commit.

## Finding and change

The prior L4 train monitor used the first 32 fitting members with every row
after warmup. Official valid covered all 1,873 supplied sequences and only
`is_scored` rows. The displayed difference mixed cohort, mask, and in-sample
effects and did not estimate a calibrated generalization gap.

Added opt-in `monitoring.train_eval_selection: "hash_rank_v1"`. It sorts only
source-qualified fitting members by the SHA256 digest of compact JSON
`["wnn.train_eval_selection.hash_rank_v1", source, group]`, then takes the
requested count. This is deterministic, input-order and model-seed independent,
and does not use a timestamp or temporal sequence order. Duplicate member keys
fail closed. Selected keys are checked against the official evaluation
membership. The old absent/`prefix` setting retains the previous prefix behavior
for historical configs.

Train-eval events and fit provenance now record the selection algorithm, exact
membership hash and count, fitting-only source scope, applied mask, evaluation
role, mask equality, and explicit non-comparability. Hash ranking makes no
representativeness claim. Valid metrics still use the complete official
evaluation dataset and its unchanged `is_scored` mask. No objective, gradient,
checkpoint, or primary metric code was changed. Same-forward dense-valid
diagnostics are deferred; this repair does not fabricate a train `is_scored`
mask or add another model evaluation pass.

Plots now label recorded cohort sizes and masks, identify train-eval as an
in-sample subset, and explain that the difference is not a calibrated gap. The
dashboard gives the same interpretation and directs readers to run provenance
for exact selection and masks. Missing metadata in historical events stays
missing.

## Future L4 monitor overlay

[`l4-monitoring-overlay-v1.json`](l4-monitoring-overlay-v1.json) selects 128
hash-ranked fitting members, carries the actual L4 `all_after_warmup` mask, and
retains the prior telemetry cadence. Merge it under `monitoring` in a newly
authorized full recipe; it is not a complete run recipe and has not been used
for a fit. The runtime currently admits at most 256 monitored sequences.

For a future admitted run, first profile the 128-member monitor. Keep it only if
the train-eval scan takes at most 10% of one fitting epoch and peak memory stays
within that run's already approved host envelope. Otherwise reduce the cohort
count while retaining `hash_rank_v1`, or disable that diagnostic. In every
case, require official valid coverage of 1,873 sequences and 3,528,273
`is_scored` rows. Do not use the diagnostic gap for model selection or promotion;
valid remains repeatedly exposed development evidence.

## Checks and artifact receipt

Ran the focused contract tests with one CPU thread and the pinned local Python:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONPATH = 'src;.'
& 'C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe' -m pytest -q tests/test_sequence_fit_monitor_setup.py tests/test_sequence_launch_monitoring_policy.py tests/test_plot_sequence_monitor.py
```

Result: 21 passed, 1 skipped. The plot-render integration test skips in this
test environment because matplotlib is absent; label and interpretation helpers
ran. The real-journal renderer was run separately with the existing plotting
environment, without model evaluation:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONPATH = 'src;.'
& 'C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/local-plot-env/Scripts/python.exe' scripts/plot_sequence_monitor.py --run-dir 'C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/runs/l4-raw112_d0_s101101-fit' --plots-dir 'C:/Users/admin/Documents/Codex/artifacts/training-diagnosis-20261108/repair-point1-plots-v2'
```

It exported 22 PNG/SVG artifacts with `model_evaluations_triggered: 0`; the
updated WP chart was inspected at
`C:/Users/admin/Documents/Codex/artifacts/training-diagnosis-20261108/repair-point1-plots-v2/train_eval_vs_evaluation_wp.png`.
The plot-source receipt is in that output folder. These source journals are
historical and unchanged; their old prefix cohort is not relabeled as
hash-ranked.

Ruff check and formatting check passed for the nine changed Python files.
`scripts/check_training_structure.py` passed with zero errors and the required
300-line module / 200-line function limits. Focused tests verify legacy prefix
compatibility; stable hash selection under reversed fit-member input; exact
namespace ranking; duplicate rejection; no validation membership; role-specific
metadata for both `valid` and inner `control`; unchanged full-valid mask,
coverage, loss, and official WP against the metric reference; and unchanged
model parameter value and preexisting gradient during diagnostic evaluation.

Empirical quality, monitor runtime at 128 sequences, GPU memory, and independent
generalization remain unmeasured. The 128-sequence overlay is a future preset,
not evidence that larger diagnostic cohorts improve prediction quality.

# Point 2: L4 diagnostics, objective reporting, and plots

Status: implemented and locally checked. ROOT owns integration and commit.
Worktree: `fix/training-observability-20261108`. No fit, profile, GPU operation,
or validation evaluation was run.

## Changes

Diagnostics now report activation RMS and backward-gradient RMS under distinct
keys. Differentiable output tensors are counted by origin so unused outputs
(including returned hidden state) are distinguishable from trainable parameters
that have no gradient. Parameter gradient presence is captured immediately after
backward and before `finish_update` clears optimizer gradients. Output hooks
detach only for measurement and return `None`; they do not replace or modify the
gradient. PyTorch documents that returning `None` leaves the gradient unchanged
[`Tensor.register_hook`](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.register_hook.html).

The rank-zero adapter passes through native raw WMSE and correlation components,
their configured weights, per-target correlation surrogates, and objective mode,
aggregation, correlation scope, and curvature-scaling metadata. Raw components
and coefficient weights use separate metric keys and dashboard groups. The
dashboard also separates per-target WP, WMSE, distribution moments, clipping
fractions, clipped counts, and clipped weight mass by units. Native objective
semantics appear with run details.

Plot export reads only recorded journals. It adds per-target z, reference, and
mixture moment plots from the same-forward metrics, and separates online logged
objective values, raw components, correlation surrogates, and active auxiliary
terms. For native rows with zero KD or hint weight, those inactive terms are
omitted from the active-component plot. Native rank-zero plots use optimizer
updates when the source only provides file modification time. In the dashboard,
selecting elapsed time while a selected stream lacks update timestamps renders
the x-axis and tooltip in optimizer steps with an explanatory note; the selector
itself may still display “Elapsed”.

No evaluation objective, gradient calculation, optimizer behavior, checkpoint,
or primary validation metric was changed. The dashboard and plot paths do not
run inference. Historical rank-zero journals remain unchanged. Per-target
distribution panels appear only for metric keys actually present in a journal.

## Checks

Focused tests ran with one CPU thread and the supplied Python environment:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONPATH = 'src;.'
& 'C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe' -m pytest -q tests/test_sequence_monitoring.py tests/test_sequence_launch_monitoring_policy.py tests/test_sequence_rpc_mirror.py tests/test_plot_sequence_monitor.py
```

Result: 37 passed, 1 skipped. The plot-render integration test skips because
matplotlib is not installed in that test environment. The rank-zero adapter
sentinel test, gradient-origin tests, gradient snapshot-before-zeroing test,
and diagnostic forward/gradient/RNG/module-mode/buffer parity tests passed.

The plotting helper was also exercised in the supplied plotting environment
with a synthetic native event: it produced three separate axes, labelled raw
WMSE and correlation values as unweighted components, showed t0/t1 correlation
surrogates separately, and omitted KD/hint lines at zero weight. This was a
small plotting check, not a model fit or evaluation.

The dashboard grouping invariant was checked by the root Node VM sentinel:

```powershell
node 'C:/Users/admin/Documents/Codex/artifacts/training-diagnosis-20261108/point2-dashboard-invariant.cjs' 'src/wnn/dashboard/static/dashboard.js'
node --check src/wnn/dashboard/static/dashboard.js
```

Result: 39 groups inspected, 16 sentinel metrics checked, no mixed-unit groups;
JavaScript syntax check passed.

Ruff formatting, lint, and the repository structure checker passed:

```powershell
& 'C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/ruff.exe' format --check scripts/sequence_rank0_view.py scripts/plot_sequence_monitor.py scripts/sequence_monitor_plot_distributions.py src/wnn/multistage_plot_fit_figures.py src/wnn/training/sequence_loop.py src/wnn/training/sequence_monitor.py src/wnn/training/sequence_monitor_diagnostics.py src/wnn/training/sequence_monitor_gradient_tracking.py src/wnn/training/sequence_monitor_probe_keys.py tests/test_sequence_launch_monitoring_policy.py tests/test_sequence_monitoring.py tests/test_sequence_rpc_mirror.py tests/test_plot_sequence_monitor.py
& 'C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/ruff.exe' check scripts/sequence_rank0_view.py scripts/plot_sequence_monitor.py scripts/sequence_monitor_plot_distributions.py src/wnn/multistage_plot_fit_figures.py src/wnn/training/sequence_loop.py src/wnn/training/sequence_monitor.py src/wnn/training/sequence_monitor_diagnostics.py src/wnn/training/sequence_monitor_gradient_tracking.py src/wnn/training/sequence_monitor_probe_keys.py tests/test_sequence_launch_monitoring_policy.py tests/test_sequence_monitoring.py tests/test_sequence_rpc_mirror.py tests/test_plot_sequence_monitor.py
& 'C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe' scripts/check_training_structure.py --root .
```

Structure result: 156 files scanned, 0 errors, maximum module 300 lines and
maximum function 200 lines. No broad repository suite was run. ROOT owns the
independent render of actual B300 journals; this receipt does not claim that
render as an L4 actor check.

## Independent root receiving check

Root converted the complete stored native B300 treatment journal (3,280 records,
last optimizer step 3,280; 4,649,063 bytes), then exported 12 PNG/SVG artifacts
with zero model evaluations. The actual objective PNG was visually inspected:
native totals, raw components and per-target microchunk surrogates are separate;
the x axis is optimizer updates. The first bounded conversion covered only one
262-KiB chunk; root completed all chunks before assessing the full trace. Original
journals remain unchanged. Receipt and figures are under
`C:/Users/admin/Documents/Codex/artifacts/training-diagnosis-20261108/repair-point2-b300-view/`.

The independent JavaScript predicate guard is preserved as
`tests/dashboard_metric_groups_invariant.cjs`, with a pytest wrapper. It executes
the real dashboard functions against 21 independent sentinel keys, checks that
none disappear, and rejects mixing score, loss, moments, fractions, mass/counts,
objective weights and correlation-surrogate families. Default charts also expose
named missing-output and pre-zero parameter-gradient coverage separately.

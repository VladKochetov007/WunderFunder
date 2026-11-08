# Point 3 L4: fixed endpoints and report-only plateau monitoring

Status: implementation and bounded CPU checks complete. No training fits,
GPU runs, throughput measurements, SSH, or authentication were performed.

## Behavior

- Missing `monitoring.early_stopping.action` still means the historical `stop`
  behavior and keeps its fixed 5/4/0.0005 policy. `report_only` accepts a
  validated min epoch, patience, and nonnegative delta, records when the
  plateau threshold is reached, and never shortens the configured schedule.
- The future L4 overlay `l4-report-only-overlay-v1.json` applies to the existing
  six-epoch `l4-raw112_d0_s101101-fit` base. It intentionally has no `epochs`
  field, retains `all_epochs`, and configures report-only 2/2/0.0001 as an
  advisory heuristic, not a statistical stopping rule. It adds no SWA policy.
- Full official-valid WP remains exposed. The best pointer uses strict raw WP
  and is labeled a best-observed exploratory pointer, not an independent
  estimate or fixed inference endpoint. The configured final epoch remains
  `model.pt` when the schedule completes. Historical explicit stop runs retain
  their last-completed-epoch endpoint semantics. Control-role monitoring is
  identified as control and is rejected as primary-valid plateau evidence.
- Existing predeclared SWA grades are unchanged. This point does not relabel
  historical predeclared SWA endpoints.
- The new all-epoch report-only policy reserves disk for atomic safe-checkpoint
  replacement during reconciliation: with snapshots remaining, it adds six
  model-state copies to the existing snapshot estimate; with no snapshots
  remaining, it adds one model-state copy for final materialization. The
  existing 64 MiB margin remains. This is a per-host estimate based on model
  state bytes, not a measured AdamW serialization bound or aggregate-host
  reservation.
- Safe-resume identity, optimizer/RNG loading, and checkpoint selection code
  were not replaced. History now persists the plateau action, threshold state,
  strict-best state, and decline from the observed peak so the policy resumes
  from its recorded state.

PyTorch's official [saving and loading models guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
describes saving optimizer state as part of a general training checkpoint when
resuming training. This supports retaining the complete safe training state
separately from immutable inference epoch snapshots.

## Verification

Interpreter: `C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe`.
The test commands used `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `PYTHONPATH=src;.`.

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:PYTHONPATH='src;.'; C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe -m pytest -q tests/test_sequence_fit_early_stop.py tests/test_sequence_fit_monitor_setup.py tests/test_sequence_epoch_preflight.py tests/test_sequence_launch_monitoring_policy.py
# 30 passed

$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:PYTHONPATH='src;.'; C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe -m pytest -q tests/test_sequence_monitoring.py -k 'checkpoint or restore or resume'
# 4 passed, 12 deselected

C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/ruff.exe format --check src/wnn/training/sequence_config_validation.py src/wnn/training/sequence_epoch_checkpoints.py src/wnn/training/sequence_epoch_fit.py src/wnn/training/sequence_fit_artifacts.py src/wnn/training/sequence_fit_epoch.py src/wnn/training/sequence_fit_full.py src/wnn/training/sequence_fit_monitor_setup.py src/wnn/training/sequence_monitor_checkpoints.py tests/test_sequence_fit_early_stop.py tests/test_sequence_fit_monitor_setup.py tests/test_sequence_epoch_preflight.py
# 11 files already formatted

C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/ruff.exe check src/wnn/training/sequence_config_validation.py src/wnn/training/sequence_epoch_checkpoints.py src/wnn/training/sequence_epoch_fit.py src/wnn/training/sequence_fit_artifacts.py src/wnn/training/sequence_fit_epoch.py src/wnn/training/sequence_fit_full.py src/wnn/training/sequence_fit_monitor_setup.py src/wnn/training/sequence_monitor_checkpoints.py tests/test_sequence_fit_early_stop.py tests/test_sequence_fit_monitor_setup.py tests/test_sequence_epoch_preflight.py
# All checks passed

C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe scripts/check_training_structure.py --root .
# 156 files, 0 errors; maximum module 300 lines, maximum function 200 lines

git diff --check
# clean
```

The focused synthetic fit test checks the old stop default, report-only
completion through the configured six-epoch schedule, best versus last
checkpoint pointers, a control-role rejection, and endpoint provenance.
Storage tests cover the unchanged legacy formula, the extra six-copy estimate,
and the one-copy final-materialization case.

## Limits

The full CLI resume integration suite was not verified on this Windows
environment: `tests/test_training_state.py` fails during collection because
the legacy `scripts/train_rnn.py` imports the Unix-only `resource` module. The
existing focused monitoring checkpoint/restore/resume tests pass, but they do
not establish a full subprocess resume run. An earlier broader focused attempt
also encountered the existing Windows directory-fsync limitation in
`tests/test_sequence_epoch_index.py` (`os.open(directory)`); that code was not
changed by Point 3. Existing all-epoch immutable-index implementation was left
untouched.

No score, generalization, or runtime improvement is claimed. Validation remains
exposed, and the report-only plateau thresholds are operational diagnostics.

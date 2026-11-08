# Point 4 L4: reference-free standalone labels WMSE

Status: implementation and bounded CPU checks complete. No training fits,
GPU runs, throughput measurements, SSH, or authentication were performed.

## Objective and admission

`objective_mode: standalone_labels_wmse_v2` is an explicit opt-in. It is absent
from `DEFAULTS`; recipes without the mode retain their previous parsed config
and identity hash. The mode validator requires `kd_weight=0`, `hint_weight=0`,
`hint_dim=0`, aligned hints, `training_mask=all_after_warmup`, and
`normalization_scope=all_fit`. Unknown objective fields, including blend and
correlation weights, continue to fail config loading.

Admission reuses the current qualified split loader, then requires the
chronological-primary schema, train-only fit membership, the full official
valid source and role, and the official `is_scored` evaluation mask. It rejects
teacher caches, initializer paths, and final-fit admission. The resulting
`data.provenance` records this admission and the objective contract, so both
become part of the immutable run identity. Historical teacher/KD recipes keep
their existing qualified checks because the restrictions apply only when this
new mode is explicitly selected.

The B300 actor confirmed and uses the same seven objective contract fields:

```json
{
  "kind": "standalone_label_weighted_mse",
  "mode": "standalone_labels_wmse_v2",
  "target_clip": [-2.0, 2.0],
  "target_weight": "abs(clipped_target)",
  "aggregation": "global_ddp_weighted_wmse_per_actual_optimizer_window",
  "pearson_term": "none",
  "reference_training_ancestry": "none"
}
```

This matches the existing L4 label path: predictions are compared with
`clip(target, -2, 2)`, weighted by the absolute clipped target, and normalized
over the actual optimizer accumulation window. The mode adds no model branch,
teacher term, blend, correlation term, gradient change, or learned
preprocessing.

## Future recipe

`l4-standalone-labels-wmse-v2.json` was cloned from
`C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/live-dashboard/runs/l4-raw112_d0_s101101-fit/run_metadata.json`.
The clone removes the runtime-injected `model.precision_mode` field so the
production loader can add it in its normal way. Its remaining model, optimizer,
seed, and six-epoch schedule match the original config. Changes are the explicit
objective mode, 128 fitting-only `hash_rank_v1` train diagnostics, report-only
2/2/0.0001 plateau metadata, and explicit `all_epochs` retention. The base
already used all-epoch retention. The overlay does not replace or edit the
historical recipe.

The original L4 raw112 config still loads to canonical SHA-256
`4c7217b49191deb7add9b93da43a76c787dc594f3d0eb7b94ac1dbec4d398883`, matching
the completed producer config. The new future recipe loads to
`994b13be963aa688945622d55f9bf39041e1258e8d81ede97119de463ca410f3`.

## Verification

Interpreter: `C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe`.
Test commands used one thread with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `PYTHONPATH=src;.`.

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:PYTHONPATH='src;.'; C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe -m pytest -q tests/test_sequence_objective_mode.py tests/test_sequence_fit_early_stop.py tests/test_sequence_fit_monitor_setup.py tests/test_sequence_launch_adapters.py tests/test_sequence_full_fit_data.py tests/test_large_teacher_runtime_guards.py
# 68 passed

C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/ruff.exe check src/wnn/training/sequence_config.py src/wnn/training/sequence_config_validation.py src/wnn/training/sequence_fit_admission.py src/wnn/training/sequence_fit.py tests/test_sequence_objective_mode.py tests/test_large_teacher_runtime_guards.py
# All checks passed

C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/ruff.exe format --check src/wnn/training/sequence_config.py src/wnn/training/sequence_config_validation.py src/wnn/training/sequence_fit_admission.py src/wnn/training/sequence_fit.py tests/test_sequence_objective_mode.py tests/test_large_teacher_runtime_guards.py
# 6 files already formatted

C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe scripts/check_training_structure.py --root .
# 156 files, 0 errors; maximum module 300 lines, maximum function 200 lines

git diff --check
# clean
```

Tests load the complete new recipe, compare it with the legacy config plus only
the allowed overlays, and guard the historical config hash. They reject KD,
hint, blend, correlation, non-primary membership, teacher/initializer ancestry,
wrong masks and prefix normalization. A synthetic CPU calculation verifies
WMSE values and gradients against the existing label-only control with clipped
targets, absolute target weights, masks, and multiple chunks in one optimizer
window.

## Next-run metrics and limits

The future run should report all six configured epochs. `model.pt` remains the
configured final-epoch endpoint; the best-observed valid pointer is exposed and
exploratory. Report the full official-valid pooled WP and per-target t0/t1 WP
using 1,873 sequences and 3,528,273 `is_scored` rows. Also report the fitting
cohort identity and mask, target-clipping/weight moments, and raw WMSE per actual
optimizer window. The 128-member dense train diagnostic is a distinct
population and must not be used to interpret a train/valid generalization gap.

The 2/2/0.0001 plateau values are advisory, not statistical quality thresholds.
These engineering checks establish config and objective semantics only; they
do not establish model quality or an improvement. Validation remains exposed,
so promotion needs an independently admitted later evaluation or public
measurement. No such run is authorized by this code change.

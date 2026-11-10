# Point 4 B300 receipt: standalone label WMSE v2

Status: implemented and locally checked. No fit, resume continuation, GPU
access or new predictions. The new labels baseline can train and report the raw
student endpoint without a TL3 cache. Existing v1 configurations remain
unchanged and retain their fixed-mixture monitoring behavior.

## Objective and identity contract

The new mode is `standalone_labels_wmse_v2`. Its validator is separate from the
frozen `labels_wmse_v1` and `mixture_wmse_pearson_v1` four-fit validator. The
new mode requires all-fit input normalization, the all-after-warmup training
mask, full official-valid report-only monitoring, zero KD and hint weights,
zero hint width, no training reference/blend and no correlation loss. Epochs,
seed, optimizer, batch/accumulation settings, model context and optional device
requirements remain governed by the generic supported-runtime checks rather
than the old four-fit numerical freeze.

Before run identity hashing, `SequenceData.provenance` records this static
`objective_contract`:

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

The numerator is weighted label squared error with weights from the absolute
clipped target. Each actual optimizer window uses its global DDP label-weight
denominator; this contract makes no dataset-global Pearson claim. Existing
native telemetry continues to describe the mean of actual optimizer-window
objectives. With no teacher/reference cache, `SequenceRecords` supplies zero
teacher tensors, and KD/hint weights are fixed to zero.

## Monitoring and preset

For v2, the monitor can be created without `--monitor-reference-root` and calls
the existing same-forward evaluator with `blend_alpha=None`. Generic
`valid/wp` is raw z, and `valid/z/wp` aliases that same value. The monitor still
checks all 1,873 official validation sequences and the 3,528,273 selected
`is_scored` rows. Training diagnostics select 128 whole fitting members with
`hash_rank_v1` and use `all_after_warmup`; their gap remains marked
non-comparable to full sparse validation. The synthetic monitor fixture checks
the full-coverage guard and raw-z primary routing; no production cohort was read
in this turn.

V2 explicitly rejects `--teacher-cache`, `--reference-cache`, and
`--monitor-reference-root` before distributed initialization or data loading.
Optional TL3 diagnostics are deferred to the existing offline fixed-readout
path; the runtime does not silently ignore a reference argument. Legacy v1
monitoring still uses alpha 0.1 and keeps its fixed-mixture fields as primary.

The new [standalone labels preset](configs/standalone_labels_wmse_v2.json)
starts from the completed four-epoch labels recipe: B128/A4, chunk 256, Muon
at 0.0003, 768 hidden × 24 layers, context 1024, two B300 ranks and
`bf16_bf16x9`. It sets 128-sequence hash-ranked monitoring and the P3
`all_epochs` / `configured_final_epoch` policy. These are editable starting
values; no data, split or source pins were added. The old v1 mode rejects the
new checkpoint overlay as before.

## Frozen legacy evidence

The checked-in normalized v1 configs still match the completed producer
`run_metadata.config` hashes supplied for regression:

- labels seed 101101: `d3ef8df541ec5ce9ac4f68b816e5ac61a59e7c1d0a181954f0fb6bdd4f8409ef`;
- mixture-correlation seed 101101: `2d9e589d65335e8231a7949902abac67d437f64886e4f2b75b22cf610076b879`.

No v1 data provenance contract was added, so this point does not change their
normalized config identity or training numerics. The existing reference cache
validator still requires the published producer's `training_predictions` to be
`in-sample` and checks its source-qualified ancestry. The legacy correlation
objective remains a microchunk surrogate; nothing here labels it OOF or
dataset-global Pearson.

## Checks

Pinned Python:
`C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe`.
Commands used one CPU thread per BLAS/OpenMP library and
`PYTHONPATH=<worktree>/src;<worktree>`.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONPATH='C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108/src;C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108'
& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m pytest tests/test_sequence_standalone_labels.py `
  tests/test_sequence_monitor_diagnostics.py `
  tests/test_ensemble_objective.py `
  tests/test_sequence_reference_cache.py -q
# 44 passed in 5.04s

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m pytest tests/test_pr67_receiving.py::test_report_only_monitor_composes_before_clipping_and_restores_rng `
  tests/test_sequence_blend_objective.py -q
# 18 passed in 4.91s

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m ruff check src/wnn/training/sequence_ensemble_config.py `
  src/wnn/training/sequence_complement_monitor.py `
  src/wnn/training/sequence_fit.py `
  src/wnn/training/sequence_fit_admission.py `
  src/wnn/training/sequence_data.py `
  tests/test_sequence_standalone_labels.py
# All checks passed.

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m ruff format --check src/wnn/training/sequence_ensemble_config.py `
  src/wnn/training/sequence_complement_monitor.py `
  src/wnn/training/sequence_fit.py `
  src/wnn/training/sequence_fit_admission.py `
  src/wnn/training/sequence_data.py `
  tests/test_sequence_standalone_labels.py
# 6 files already formatted.

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  scripts/check_training_structure.py `
  --root C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108
# 117 files, 0 errors; module limit 300 lines, function limit 200 lines.
```

The synthetic checks cover editable epochs/seed/optimizer/context, rejected
objective combinations, the exact weighted-window label loss and gradient,
same-forward raw-z monitoring without reference namespaces, full-coverage guard
routing, explicit rejection of cache flags, unchanged normalized v1 config
hashes, prior v1 mixture monitoring behavior, and source/ancestry validation.

## Limits

The monitor fixture supplies the audited official coverage counts but does not
read all validation data. No production score, throughput, memory or storage
measurement was generated. V2 report-time TL3 diagnostics are intentionally
not supported; use the separate offline fixed readout if needed. Existing
validation remains exposed development data, and no quality or serving gain is
established.

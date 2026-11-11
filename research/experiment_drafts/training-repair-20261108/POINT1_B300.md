# Point 1 B300 receipt: honest train and validation monitoring

Status: implemented and locally checked; no fit, resume, GPU access or new
predictions. This repairs monitoring provenance and cohort selection only. It
does not change the training objective, selected checkpoint, reference cache or
historical score.

## Change

`src/wnn/training/sequence_complement_monitor.py` now accepts
`monitoring.train_eval_selection` as `prefix` or `hash_rank_v1`. Missing keys
retain the legacy prefix behavior for frozen configs. `hash_rank_v1` selects
whole source-qualified members from the official fitting membership using
SHA-256 over the fixed namespace `wnn.train_eval_selection.hash_rank_v1`, source
name and group identifier. The rank is independent of model seed and caller
ordering. It never reads `seq_ix` chronology or validation members.

The monitor context records selection, fit-only population, cohort size/hash,
train and validation masks, complete-validation population, and
`generalization_gap_comparable: false` with a reason. Fit diagnostics continue
to use `all_after_warmup`; no `is_scored` mask is synthesized for train. Official
primary validation remains all 1,873 sequences and the unchanged 3,528,273
scored-row assertion under `is_scored` after warmup. It is not replaced by a
subset or a dense diagnostic.

The optional same-forward dense-valid secondary diagnostic is deferred. The
current shared evaluator is already 288 lines against the 300-line module limit;
this change does not add a parallel evaluation pass or claim that train and
validation scores form a comparable generalization gap.

Historical configs/results remain unchanged. A future recipe can opt into
`"train_eval_selection": "hash_rank_v1"`; omission preserves old cohort identity.

## Focused checks

Python: `C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe`.
Commands ran with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`; pytest used `PYTHONPATH=<worktree>/src;<worktree>`.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONPATH='C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108/src;C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108'
& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m pytest tests/test_sequence_complement_monitor.py `
  tests/test_sequence_full_fit_data.py::test_all_after_warmup_accepts_training_source_without_scoring_column `
  tests/test_pr67_receiving.py::test_report_only_monitor_composes_before_clipping_and_restores_rng -q
# 5 passed in 3.86s

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m ruff format src/wnn/training/sequence_complement_monitor.py `
  tests/test_sequence_complement_monitor.py
& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m ruff check src/wnn/training/sequence_complement_monitor.py `
  tests/test_sequence_complement_monitor.py
# All checks passed.

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  scripts/check_training_structure.py `
  --root C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108
# 114 files, 0 errors; module limit 300 lines, function limit 200 lines.
```

The focused tests cover deterministic seed-independent fitting-cohort selection,
source-qualified whole-member handling, rejection of non-train members, legacy
prefix compatibility, invalid selection rejection, the no-`is_scored` train
path, and the unchanged official mask scorer on a tiny fixture.

## Limits and interpretation

The supplied monitor audit reports that 32 legacy prefix train sequences use
all post-warmup rows while full valid uses the official sparse `is_scored` mask
(about 9.47% of valid rows). That rate is cited from the audit, not remeasured
here. The expected primary coverage remains 1,873 / 3,528,273; the full-size
assertion remains in source but was not exercised on production data in this
CPU check. Existing validation is exposed discovery data. No fresh independent
holdout, empirical score change, speedup or serving gain is established.

The `hash_rank_v1` implementation is available for a new config but is not
retroactively applied to historical configs. No future fit should infer
representativeness or generalization-gap comparability from hash selection
alone; this is deterministic sampling of fitting members, not a temporal split.

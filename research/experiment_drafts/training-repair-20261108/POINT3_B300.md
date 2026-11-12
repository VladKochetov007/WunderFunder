# Point 3 B300 receipt: checkpoint purpose and fixed endpoint

Status: implemented and locally checked. No fit or resume continuation, GPU
access or new predictions. `model.pt` remains the configured final epoch.
Retained epoch weights remain available for explicit readouts and are never
promoted by a metric-based automatic selector.

## Policy and producer changes

`sequence_checkpoint_policy.py` defines three artifact purposes:

- `retained_epoch_snapshot_no_automatic_selection` for immutable epoch weights;
- `fixed_configured_final_endpoint` for `model.pt`;
- `latest_full_safe_resume_state` for `training_state.pt`, which carries model,
  optimizer, scheduler, history, safe log offsets and rank-local RNG state.

Epoch snapshots continue to be detached, cloned CPU FP32 model state at each
safe boundary and remain content-addressed in the append-only index. The final
writer continues to save the model passed at the configured final epoch,
regardless of validation values in earlier history rows. Earlier observed
best-metric readouts are exploratory if selected after seeing those metrics.
A separately predeclared endpoint may use a pinned epoch index entry without
changing the meaning of the fixed `model.pt` endpoint.

Generic recipes may explicitly overlay
`"checkpoint_retention": "all_epochs"` and
`"checkpoint_endpoint_policy": "configured_final_epoch"`. The policy field is
optional, has no default, and is accepted only with `all_epochs`; this leaves
historical config hashes unchanged. The future standalone B labels preset can
use this overlay:

```json
{
  "checkpoint_retention": "all_epochs",
  "checkpoint_endpoint_policy": "configured_final_epoch"
}
```

The configured `epochs` value pins the final endpoint. This overlay is not
admitted by the exact four-fit objective v1 mode; that validator still requires
`final_only`. A generic labels config copy with the frozen `objective_mode`
removed accepts the overlay, while the checked-in v1 config loads unchanged and
rejects it. Point 4 must bind this policy to a distinct new standalone objective
preset before use; no frozen v1 recipe or result is being changed.

## Storage preflight

The original storage threshold is unchanged when the optional endpoint policy
is absent: for a positive remaining snapshot count it requires
`int(state_bytes * snapshots * 1.10) + 64 MiB`, and with zero snapshots it
returns zero. This preserves old `all_epochs` recipes as well as `final_only`
recipes.

With the new policy enabled, preflight adds `6 * state_bytes` when snapshots
remain, to account conservatively for the old and temporary full resume files
containing model and AdamW state during atomic replacement. When no snapshots
remain, it instead reserves one model state plus 64 MiB for the final model
materialization. A failure is coordinated before the training loop advances.
The bound uses model state bytes rather than measuring serialized optimizer
files; it is a conservative planning estimate, not a measured exact peak.

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
  -m pytest tests/test_sequence_checkpoint_policy.py -q
# 7 passed in 5.28s

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m ruff check src/wnn/training/sequence_checkpoint_policy.py `
  src/wnn/training/sequence_config.py `
  src/wnn/training/sequence_epoch_checkpoints.py `
  src/wnn/training/sequence_epoch_fit.py `
  src/wnn/training/sequence_fit_artifacts.py `
  src/wnn/training/sequence_fit_state.py `
  tests/test_sequence_checkpoint_policy.py
# All checks passed.

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  -m ruff format --check src/wnn/training/sequence_checkpoint_policy.py `
  src/wnn/training/sequence_config.py `
  src/wnn/training/sequence_epoch_checkpoints.py `
  src/wnn/training/sequence_epoch_fit.py `
  src/wnn/training/sequence_fit_artifacts.py `
  src/wnn/training/sequence_fit_state.py `
  tests/test_sequence_checkpoint_policy.py
# 7 files already formatted

& C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe `
  scripts/check_training_structure.py `
  --root C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108
# 116 files, 0 errors; module limit 300 lines, function limit 200 lines.
```

The focused fixtures check clone immutability after a later model mutation,
snapshot purpose, exact storage accounting and rejection when space fits only
snapshots, zero-snapshot final-write reserve, legacy zero-snapshot behavior,
wrong-identity resume rejection, full-resume purpose, fixed-final output in the
presence of a higher earlier metric, generic overlay admission, and unchanged
four-fit v1 retention. `git diff --check` passed.

The test adapter maps unsupported Windows directory `fsync` calls to a
disposable regular file; it does not verify directory metadata durability. The
actual tensor file flush and `fsync` operations still run.

## Documentation and limits

I reread the [official PyTorch serialization notes](https://docs.pytorch.org/docs/2.15/notes/serialization.html).
They recommend saving a module `state_dict` for compatibility, describe cloning
as producing independent tensor storage, and document that `torch.load` defaults
to `weights_only=True` from PyTorch 2.6 when no pickle module is supplied. This
runtime serializes trusted internal envelopes that include Python metadata and
rank RNG values, so the existing internal loads explicitly use
`weights_only=False`; this point did not change that behavior. The producer
clone test protects the independent-weight-snapshot contract.

Storage free-space inspection runs on the primary worker's visible output
filesystem. Multi-host per-filesystem equality and production optimizer-state
file sizes were not measured. The 64 MiB headroom and six-state estimate may
not account for filesystem-specific allocation or unrelated concurrent writes.
No training, score, memory or throughput result was generated by this point.

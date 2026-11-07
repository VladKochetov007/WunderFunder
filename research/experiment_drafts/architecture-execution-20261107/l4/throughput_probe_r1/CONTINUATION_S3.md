Historical S3 attempt: both checkpoint copies migrated, but monitored canonical
resume failed before any new update. Preserve this protocol and its exact
receipts; use [RESUME_FIX_S4_R1.md](RESUME_FIX_S4_R1.md) for the bounded repair.

This was the approved protocol for continuing the two GRU cells from a whole-epoch
checkpoint to the same configured epoch12 endpoint. Native preparation and both
profiles passed; the supervisor launch is recorded in
[NATIVE_PREPARATION_R1.md](NATIVE_PREPARATION_R1.md). That launch receipt does not
establish a successful migration or new training updates. The numerical recipe, exact
official train-only/full-valid membership, runtime/GPU identity, optimizer,
scheduler, rank RNG, history, updates and native safe log prefixes are preserved.
The explicit changes are the three S3 source files and retention
`all_epochs -> final_only`. New profiles must use each original seed and physical
GPU. The previous S1/S2 sources, parent run, inference snapshots and index chain
remain historical and byte-unchanged.

S1/S2 cannot accept final_only with extended monitoring. S3 changes only
`sequence_monitor_gates.py`, `sequence_monitor_diagnostics.py` and the narrow
condition in `sequence_config_validation.py`. The latter permits extended
final_only monitoring only without early_stopping; every other check remains.
The canonical resume validator is unchanged. The source change itself does not
authorize a continuation.

Stage these receiving scripts outside both source capsules. The builder imports
the sibling `prepare_monitor_source.py` for the previously reviewed prefix patch.
ROOT runs all native commands; these instructions do not launch anything locally.

```sh
R=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1
py=/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python
Q=$R/control/gru-monitor-transition-r1
"$py" "$Q/prepare_gru_archive_source.py" --source "$R/source-r1" \
  --inventory "$R/manifests/l4-source-r1-manifest.json" \
  --output "$R/source-gru-monitor-archive-r1"
```

The exact620-file parent inventory SHA is
`7d3324230effb050ff083b939e26ad638e5df5c5b72ea70b0f8aed169997db4a`.
The new manifest and patch receipt are sibling files named
`source-gru-monitor-archive-r1-manifest.json` and
`source-gru-monitor-archive-r1-patch-receipt.json`. Their full native hashes must
be pinned after creation. Changed source modules must have at most300 lines.

Only the final packet `continuation-s3-r3/CONTINUATION_PROFILE_PLAN_R3.json` is
eligible. Its SHA is
`e5557696a25543daf4c7febb9d8f343451ce5ec5d47c857a3f2b232d19d88b5b`.
The older r1/r2 packets are preserved failed admission preparations. Raw r3
configs omit the derived model.precision_mode field; canonical load_config
restores it, making the normalized recipe exactly the old recipe with the sole
retention amendment. Both seed-specific normalized hashes are in the final plan.
The ROOT-owned r3 profile helper independently validates canonical config loading
before any pause. Each20-update native receipt needs separate ROOT approval.

After approval, ROOT waits for a new safe epoch, terminates only the exact owned
single-GPU original trainer, and pins the now-stable training_state.pt hash and
whole-epoch counters. No old inference checkpoint or index is copied into the
new continuation directory. The request below is an explicit ROOT-reviewed JSON
artifact; every `{"path": ..., "sha256": ...}` points to actual native bytes.

```json
{
  "schema": "wnn.gru-monitor-transition-request.v1",
  "approved": true,
  "reviewer": "ROOT",
  "old_output": "R/runs/gru_s101001/fit",
  "new_output": "R/runs/gru_s101001/monitor-continuation-s3-r1",
  "old_metadata": {"path": "OLD/run_metadata.json", "sha256": "ACTUAL_SHA"},
  "old_checkpoint_sha256": "ACTUAL_STOPPED_SAFE_STATE_SHA",
  "expected_next_epoch": 6,
  "expected_global_step": 38760,
  "new_config": {"path": "R/configs/gru-monitor-continuation-r3/gru_s101001.json", "sha256": "ACTUAL_RAW_SHA"},
  "approved_profile": {"path": "ROOT_APPROVED_S3_EXACT_CELL_PROFILE.json", "sha256": "ACTUAL_SHA"},
  "old_source_root": "R/source-r1",
  "new_source_root": "R/source-gru-monitor-archive-r1",
  "old_source_manifest": {"path": "R/manifests/l4-source-r1-manifest.json", "sha256": "7d3324230effb050ff083b939e26ad638e5df5c5b72ea70b0f8aed169997db4a"},
  "new_source_manifest": {"path": "R/source-gru-monitor-archive-r1-manifest.json", "sha256": "ACTUAL_SHA"},
  "source_patch_receipt": {"path": "R/source-gru-monitor-archive-r1-patch-receipt.json", "sha256": "ACTUAL_SHA"},
  "first_profile_receipt": {"path": "R/receipts/first-profile-r1.json", "sha256": "ACTUAL_SHA"},
  "not_after_epoch": 0,
  "stopped_trainer_owner": {"pid": 0, "birth": "EXACT_CANONICAL_PROCESS_BIRTH"}
}
```

`R` and `OLD` in that schema illustration must be expanded to real absolute POSIX
paths. Counters are illustrative, never a predetermined epoch to select using
validation. `not_after_epoch` must exactly equal the original
first_profile_epoch +86400 -3600 reserve; a continuation gets no new clock. The
stopped owner must match the trainer ROOT actually terminated, and that process
must have exited. Requests for any other architecture/seed/batch, numerical
change, source path, runtime, data identity, partial log offset or unfinished
full-valid coverage are rejected. A failed preparation leaves only an unready
new directory; it never overwrites or truncates the original.

```sh
PYTHONPATH=$R/source-gru-monitor-archive-r1/src \
  "$py" "$Q/migrate_monitor_checkpoint.py" \
  --request "$Q/gru_s101001-transition-request.json" --request-sha256 ACTUAL_REQUEST_SHA
```

Review the resulting transition_proof.json and transition_ready.json before
launch. Canonical run_metadata.json contains only its canonical fields; the
separate immutable proof binds the complete parent/new identities, original
checkpoint SHA, new checkpoint SHA and copied native byte prefixes. Historical
monitor events retain their original source identity, and new events use S3.
All other checkpoint fields are compared after serialization, including the
exact tensor bytes and all rank RNG arrays.

Start the external CPU archiver before resuming; wait for archive_ready.json
with the matching proof hash and live archiver PID. ROOT's continuation
supervisor must treat archiver exit/failure as a retention failure and stop only
its own continuation trainer. No automatic retry, fresh scientific fit or seed
is admitted. The archiver itself never sends signals or launches training.

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH=$R/source-gru-monitor-archive-r1/src \
  "$py" "$Q/archive_safe_state.py" --output "$NEW_OUTPUT" \
  --transition-proof-sha256 ACTUAL_TRANSITION_PROOF_SHA --poll-seconds 1
# After readiness, use the ordinary owned-command/observer wrapper with this argv:
"$py" "$R/source-gru-monitor-archive-r1/scripts/train_sequence.py" \
  --config "$R/configs/gru-monitor-continuation-r3/gru_s101001.json" \
  --data-root "$R/project" --split-manifest "$R/project/chronological_primary.json" \
  --output "$NEW_OUTPUT" --profile-receipt "$APPROVED_PROFILE" \
  --resume "$NEW_OUTPUT/training_state.pt"
```

Use the final profile plan's exact8-thread strict-FP32 environment for training,
including removal of CUBLAS_WORKSPACE_CONFIG and rank variables and preservation
of NVIDIA_TF32_OVERRIDE=0 and TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0. The archiver
publishes immutable content-addressed **full resume states**, not inference-only
weights, with source/transition SHA and each safe epoch. It rejects a skipped
epoch, rollback or changed already-archived state and ends only after all future
epochs through12 have been archived. Native full-valid journals still require
exact1873 sequences and3528273 selected rows each epoch. Old selected rows,
validation outcomes, LR and checkpoint ranking do not control continuation.

CPU verification, no model construction or fitting:

```sh
PYTHONPATH=src python -m pytest \
  research/experiment_drafts/architecture-execution-20261107/l4/throughput_probe_r1/test_transition_constraints.py \
  research/experiment_drafts/architecture-execution-20261107/l4/throughput_probe_r1/test_monitor_transition.py -q
```

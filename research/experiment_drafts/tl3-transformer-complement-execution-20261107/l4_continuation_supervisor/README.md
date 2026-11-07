# Bounded supervisor for the two existing L4 GRU trajectories

This is code-only preparation for ROOT's explicitly authorized source/retention
transition. It creates zero scientific fits or seeds. Only GRU101001/GPU0 and
GRU101002/GPU1 may continue their existing trajectories, from a later safe epoch
to the original12E/77,520-update endpoint. ROOT owns all SSH/GPU execution.
S1/S2/S3, historical runs, inference weights and epoch indexes are unchanged.
The reviewed compact actor's CONTINUATION_S3.md, migrator, transition_support.py
and external archive_safe_state.py remain authoritative and are reused unchanged.

Stage these five Python files alongside the actor's three receiving tools at
`R/control/gru-monitor-transition-r1` (Q). Keep them outside all source capsules.
The main supervisor imports S1's exact existing ownership/command helpers.
Preflight and migration execute in fresh S3 Python processes, avoiding a cached
S1 config validator. New training uses unchanged canonical S3 train_sequence.py,
the accepted observed_worker relay and existing read-only observer.

Before any stop, build_plan.py freezes both ROOT-approved exact-cell20-update
profile receipts, configs, metadata, source inventories/patch receipt and original
clock. supervisor.py invokes full native validate_transition and checkpoint
validation, checks the exact3-file source whitelist and full-valid safe prefix.
It waits at most1,800seconds for a new atomic checkpoint inode whose next_epoch
exceeds the initial counter, without inspecting validation outcomes. A timeout
preserves the old trainer. Counters must describe exact6,460-update whole epochs.

Before termination the old receipt must bind the expected cell, GPU, output and
live leader PID/birth/PGID. Procfs PPID ancestry and birth establish exactly one
same-group actual trainer with argv[1] equal to the original train_sequence.py.
Relay/observer argv containing the nested trainer string are excluded. ROOT pins
are rechecked immediately before stopping only that receipt-owned group via S1's
helper. Both leader and actual trainer must exit. The now-stable safe checkpoint
is hashed twice and checked against the same full contract.

Only four measured request fields are filled after the authorized stop:
old_checkpoint_sha256, expected_next_epoch, expected_global_step and the actual
stopped_trainer_owner. The immutable final request invokes the unchanged migrator.
Its new output is the fresh sibling `R/runs/gru_sSEED_monitor_opt/fit`; the old
directory is never copied over, truncated or resumed. Migrator journal-prefix
and tensor/RNG/optimizer/scheduler proofs remain unchanged.

The external CPU archiver starts first as a native receipt-owned process.
archive_ready must bind the exact transition proof hash, original deadline and
live receipt PID/birth. Only then does canonical resume start. Its progress guard
fails on archiver death, timeout or nonzero exit, causing cleanup of this new
continuation's owned group only. Archiver exit0 is allowed only with matching
archive_complete12E proof during finalization. Final success requires native
completed12E/77,520 and archive_complete, whose writer independently validates
all12 full1873/3,528,273 evaluations and archives every future safe epoch.

The original first-profile clock is1794071266.717658. The inherited fit deadline
is1794154066.717658 (first+82,800); no new clock is created. Two finite workers,
one-shot phase receipts and per-cell output paths prevent automatic retry/resume.
ROOT stop/deadline ends owned new commands. Healthy foreign models and the other
GRU are excluded from cell-specific retention-failure cleanup. Training retains
the exact strict-FP32 profile8-thread environment; archival uses2 CPU threads.

ROOT substitutes actual accepted native paths below. The builder reads hashes;
it stops or launches nothing. Existing different plan/phase files are preserved.

```bash
R=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1
py=/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python
Q="$R/control/gru-monitor-transition-r1"
export PYTHONPATH="$R/source-r1/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$py" "$Q/build_plan.py" --runtime "$R" --python "$py" --tools "$Q" \
  --observed-worker "$ACCEPTED_OBSERVED_WORKER" \
  --first-profile-receipt "$R/receipts/first-profile-r1.json" \
  --approved-profile-101001 "$APPROVED_PROFILE_101001" \
  --approved-profile-101002 "$APPROVED_PROFILE_101002" \
  --new-source-manifest "$R/source-gru-monitor-archive-r1-manifest.json" \
  --patch-receipt "$R/source-gru-monitor-archive-r1-patch-receipt.json" \
  --output "$Q/supervisor-plan.json"
# Review the concrete generated plan and its emitted native SHA before launch.
"$py" "$Q/supervisor.py" --plan "$Q/supervisor-plan.json" \
  --plan-sha256 "$PLAN_SHA" --detach
```

All helper bytes are pinned in the plan, including the accepted observed_worker.
No modification of the old receiving runner or old authority is required.
Preparatory CPU checks:9 focused tests pass, Ruff/format clean, both CLI help
paths execute. The realistic relay -> observer -> trainer fixture excludes a
foreign process and reused PID. Other tests cover failed preflight preserving
the old worker, archiver death/timeout/unproved clean exit, final-only clean exit,
and cleanup excluding foreign/other-seed commands. No native stop, migration,
archival, CUDA work or resumed endpoint was executed by the preparation agent.

# Finite TL3 execution and dashboard admission

R8 adopts the completed first teacher and schedules the remaining16 frozen fits
on verified released L4 resources. Seven original GALAM jobs remain unchanged.
This snapshot establishes first-teacher completion and active train prediction
export, not a completed17-fit chain or a final blend result.

- [Receipt and dashboard views](RECEIPT.json): raw-file SHA256 values and all17
  actual dashboard entries; the first teacher has4 completed epochs,16 are queued.
- [Controller launch](controller.launch.json), [frozen plan](plan.json) and
  [initial status](initial-status.json): exact commands, detached PID/start ticks,
  source/data/recipe identities, task ownership and first-teacher adoption.
- [Source identity](source.identity.json) and [receiving tests](receiving-tests.log):
  immutable R8 flat code closure;38 receiving CPU checks pass on Torch2.8/Linux.
- [First fixed checkpoint verification](first-fixed-checkpoint-verification.json):
  fixed ep3, exact12880 completed updates, source/history/checkpoint/member hashes.
- [Dashboard observer](dashboard.launch.json): separate read-only helper, correct
  `quality8-full-20261028/runs` root and detached ownership. Local broker tests
  pass11 cases; all17 entries were independently read through the actual API.
- [Rejected R7 launch](rejected-r7-controller.launch.json),
  [status](rejected-r7-status.json) and failed
  [export](rejected-r7-export.log), [fin256_s4](rejected-r7-fin256_s4.log),
  [orgbase](rejected-r7-orgbase.log) preserve the diagnosed environment failure.
  Resolving the venv executable symlink selected base Python without NumPy;
  imports failed before any new optimizer updates. Every owned R7 process was
  confirmed exited before source correction; the unchanged scientific recipes
  were re-admitted with the intact venv path and a dependency probe before state.
- [Rejected dashboard R1](rejected-dashboard-r1.log): numeric row-group IDs were
  incorrectly compared across train/valid files. Source-qualified memberships
  fix this; source-local `(train,0)` and `(valid,0)` are distinct members. An R2
  observer then wrote to the parent directory; only that owned observer was
  replaced by the correctly configured R3. No training source or fit was changed.

Raw files remain byte-preserved. Mutable remote state continues independently
of this historical snapshot and of the SSH connection. The observer consumes
future native metrics without a second full-valid inference pass.

## Subsequent live check

[Live monitor receipt](LIVE_MONITOR_R1.json) confirms the first train prediction
export completed and **fin256_s4 is now fitting**: one teacher complete, one fit
running and15 predictors not yet started. Its actual API stream contains online
loss, learning rate, throughput and verified process/GPU memory/utilization.

[First native paired evaluation](NATIVE_FIRST_EPOCH_R1.json) verifies fin256_s4
epoch1 WP0.6748902187479032 on all3,528,273 scored valid rows, with585.1s fitting
and94.7s validation. The actual dashboard evaluation contains matching train/valid
WMSE and WP aliases with checkpoint/cohort provenance; no extra full-valid pass
was needed. This is an intermediate epoch, not a selected endpoint.

[Watcher launch](watcher.launch.json), [actual artifact check](watcher-actual-snapshot.json)
and [first GitHub posts](watcher-first-posts.json) bind the detached PR49 publisher.
It verified SemyonEpanov and posted both initial status and the first completed
teacher. Updates continue once an hour when state changes and promptly after
each verified fit or terminal DAG status. The credential and private publisher
state remain outside the artifact HTTP root and are not included here.
[Terminal-report watcher revision](watcher-final.launch.json) preserves the same
deduplication state and original deadline, and adds the validated final blend
WP/t0/t1, fixed weights, scored-row count, clean-GRU delta and score hash to the
terminal PR comment. Its focused result/watcher/final-score tests pass9 cases.
[R7 process-exit proof](rejected-r7-exits.json) confirms all rejected owned
children/controller exited before the correction.

The later R4 observer adds **tl3c_final_fixed_blend** only after the complete
fixed endpoint is scored and its terminal/artifact identities agree. It publishes
one full-valid score point and no invented training history. Its source and
launch receipt are bound in [final observer launch](dashboard-final.launch.json)
and [receiving admission](dashboard-final-admission.json). All frozen controller
source files still match the live plan. The integrated local report/observer/
watcher suite passes19 tests; actual terminal final-blend scoring is pending.

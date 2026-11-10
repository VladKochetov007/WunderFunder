# GRU recovery R2: original stopped S1E8 to S4

ROOT explicitly authorized repair after native S3 resume failed before any new
optimizer update: coordinated_call rejected the None returned by monitor offset
restoration. The former preflight only exercised load_training_state, so it did
not expose this canonical load_resume API bug. S4 adds an explicit boolean return
adapter in sequence_fit_resume.py, with the original three diagnostic/retention
changes preserved. The compact actor owns the separately pinned four-file source
whitelist and `_s4.py` migration/archive helpers; this packet changes none of them.

Both original S1 trajectories were stopped at8E/51,680. ROOT proved their stable
checkpoint hashes, exited original processes and fully absent S3 groups. Failed
S3 runs, source, control, proofs, archives and journals remain unchanged. R2 uses
the original S1 states directly, creating fresh siblings
`R/runs/gru_sSEED_monitor_opt_r2/fit`. It never signals an original or S3 group,
waits for no new original epoch and creates zero scientific fits/seeds.

| Seed | Original S1 checkpoint SHA256 |
|---|---|
|101001|06a1bc3ce96a2033576dfc53085ebbfcff06663d390a9b5e3c39a0a7e978d93d|
|101002|909654a77629a17dd99c7a615d8d1997a115c7a7aad50f372ad3f22b61e5cc9a|

The S4 native20-update profiles have separate ROOT approvals; no S3 result is
relabelled as measured S4. The plan pins those approvals, unchanged original
ROOT transition requests, cleanup evidence, historical owned receipts and all10
runtime helpers, including inherited supervisor/checks and observed_worker.
PID/birth/group admission treats zombies as exited and rejects any non-zombie
historical group. Original clock/deadline remains1794154066.717658.

Each independent worker runs unchanged S4 migration, then two actual canonical
load_resume probes on fresh scratch clones of that migrated state/safe journals.
Both build the real GRU, optimizer, scheduler, SequenceMonitor and RankJsonlLog,
load all model/optimizer/scheduler tensors, restore safe offsets and compare all
states, counters/history and RNG. Neither performs forward/backward or an
optimizer update. Original/migrated checkpoint bytes remain unchanged.

The CPU pytest probe hides all GPUs and asserts CUDA remains uninitialized.
Its sole stub is torch.cuda.set_rng_state: it checks the opaque serialized CUDA
RNG bytes while Python/NumPy/CPU Torch RNG are genuinely restored. This is not
proof of actual CUDA restoration. The following separate CUDA probe uses no
stub, reconstructs exact actual S4 source/config/software/precision/device UUID
identity, calls canonical load_resume on the real GPU and checks actual CUDA RNG.
Only after both receipts pass does the external CPU archiver start; its matching
live PID/proof readiness still gates production resume. The unchanged archive
guard and final12E/77,520/all-epochs archival gates remain.

R2 also fixes the actual shutdown hang without changing S1 helpers. The old
run_owned_command exception branch gives terminate_owned_process the whole
remaining deadline; the accepted observed_worker ignores TERM. R2 catches and
latches progress failures, kills only its exact new groups with zero grace, then
returns normally so each native Popen loop polls/reaps its own child. Errors are
raised only after native return. Cleanup precedes every executor shutdown/wait.
Source/helper failures are terminal; no automatic retry is admitted.

Stage the five runtime/builder Python files and optional selftest in
`Q2=R/control/gru-monitor-recovery-r2`, alongside the actor's already staged
migrate_monitor_checkpoint_s4.py, transition_support_s4.py and
archive_safe_state_s4.py. Keep Q1 read-only. ROOT uses the already accepted
observed_worker path from its prior plan, not a replacement relay.

```bash
R=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1
py=/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python
Q1="$R/control/gru-monitor-transition-r1"
Q2="$R/control/gru-monitor-recovery-r2"
export PYTHONPATH="$R/source-r1/src:$Q1"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$py" "$Q2/build_recovery_plan.py" --runtime "$R" --python "$py" --tools "$Q2" \
  --legacy-control "$Q1" --new-source-root "$R/source-gru-monitor-resume-r1" \
  --new-source-manifest "$R/source-gru-monitor-resume-r1-manifest.json" \
  --patch-receipt "$R/source-gru-monitor-resume-r1-patch-receipt.json" \
  --root-cleanup-proof "$Q1/ROOT-final-owned-cleanup-r1.json" \
  --observed-worker "$ACCEPTED_OBSERVED_WORKER" \
  --approved-profile-101001 "$Q2/gru_s101001-approved-profile.json" \
  --approved-profile-101002 "$Q2/gru_s101002-approved-profile.json" \
  --output "$Q2/recovery-plan.json"
# ROOT reviews the concrete emitted plan and native PLAN_SHA.
"$py" "$Q2/recovery_supervisor.py" --plan "$Q2/recovery-plan.json" \
  --plan-sha256 "$PLAN_SHA" --detach
```

Migrate/probe phases have1,800-second bounds; archival/resume use only the
remaining original fit deadline. Root stop closes admission and immediately
stops only R2-owned commands. Parallel revalidation checks freshness for its own
cell, preserving a sibling's already-created owned output. Final success
requires the same12E endpoint and every future immutable full-state archive.

Local focused CPU tests5passed; Ruff/format clean. They check the callback returns
to native polling before raising, immediate cleanup excludes foreign/other-seed
groups, CPU and real-CUDA receipts cannot be confused, and sibling migration
does not cause a false freshness failure. Actual checkpoint CPU probes and real
CUDA probes require ROOT's native pinned artifacts and are not claimed from
these local tests. No SSH/GPU/process action was performed by the preparation
agent. All native executions and final quality remain ROOT-owned.

[Independent review R2](INDEPENDENT_REVIEW_R2.md) binds the final five hashes and
independently passing5 tests. The reviewer found no remaining mechanical blocker;
actual native canonical CPU/CUDA probes remain mandatory receiving gates.

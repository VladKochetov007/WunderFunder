# Bounded B300 receiving controller, outside frozen source-r3

The root operator explicitly requested this finite four-cell controller. It
launches no GPU work locally. The scientific source stays frozen at
`6ab2f679e70928f913b97d5a29f0d1e09ceacaa3`; these helpers must be staged in a
separate receiver directory, never overlaid on that source or its manifest.

[controller.py](controller.py) reuses ROOT's known support package at
`/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/operator-r26/src`
for `run_owned_command`, immutable JSON/locks, PID/birth/PGID ownership and
detached execution. Only the controller/relay import that support package.
Every model child uses the actual frozen source-r3 `PYTHONPATH`; no training,
model, objective or controller implementation enters frozen source-r3.
[receiving_state.py](receiving_state.py) contains narrow receipt/resource checks;
[owned_phase.py](owned_phase.py) retains the receipt-owned session leader through
canonical cleanup. It adds no observer: source-r3's wrapper already observes
profiles/fits. [selftest.py](selftest.py) exercises the CPU-only boundaries.

## Fixed paths and stage graph

Let `R=/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1`.
The actual source is `R/source-r3`; its manifest is
`R/manifests-r3/b300-source-r3-manifest.json` with SHA256
`9a3b147c20e902786f026a29741cd9f5135de91183e305105b2039f735905d7a`.
Data is `R/project`, references `R/reference`, and split is the preserved
`multistage-optuna-20261031-r1/data/chronological_primary.json`, SHA256
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.

**Mandatory:** `/data/epanov_s/tdn7run` must already resolve to `R/b300`.
The wrapper receives that short path as `WNN_RUN_ROOT`, placing its scratch and
compiler caches under the short alias. This avoids the observed AF_UNIX path
limit in multiprocessing without modifying the frozen wrapper. The controller
refuses a missing or differently resolved alias.

1. Native `check` uses GPU pair 6,7, independently of ROOT's six reference exports
   on GPUs 0–5. It has a 1,800-second bound and needs the successful data check.
   An existing complete matching acceptance receipt is hash-verified and reused.
   A failed/partial check is not retried automatically.
2. Wait for `R/receipts/reference-preparation-complete-r1.json`, at most six hours
   by default. This external preparation is not relaunched by the controller.
3. Require the exact complete full-valid baseline and both reference caches.
4. Start a 24-hour authority clock immediately before the first actual profile,
   with four scientific fits, twelve profiles and a final one-hour collection
   reserve. Run each cell independently as shown below.

| Cell | Physical ordinal pair | Stages |
|---|---|---|
| labels_s101001 | 0,1 | profile-one → profile-two → profile-peak → ROOT approval → fit → evaluate → bundle |
| complement_s101001 | 2,3 | same |
| labels_s101002 | 4,5 | same |
| complement_s101002 | 6,7 | same |

Each profile has twelve updates and at most 1,800 seconds. Each fit receives only
the remaining pre-reserve budget. A profile failure is terminal for that cell;
other healthy cells continue. No recipe or seed is selected, retried or expanded.
`compare` runs only after all four complete their fixed final epoch 30 / 774,600
updates and all evaluation/bundle commands return zero.

The ROOT preparation receipt must contain:

```json
{
  "status": "completed",
  "split_sha256": "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c",
  "baseline_score_sha256": "<SHA256 of R/b300/baseline-score.json>"
}
```

The helper independently checks baseline schema, 1,873 sequences, official
metric, finite WP **0.6923774245652725 ± 2e-6**, evaluation manifest hash and
common reference identity. Native training still checks every cache/member/
ancestry array through the frozen canonical admission path.

ROOT places each separately reviewed production copy at
`R/b300/<cell>/approved_profile.json`, preserving every original profile-two
field except `approved`, setting `approved=true`, adding `reviewer="ROOT"` and
the independent review/date/reason. This helper never writes an approved file.
The canonical fit then compares the complete actual source/config/data/runtime
identity again before any scientific update. Ready cells do not wait for other
cells' approvals.

## ROOT launch command

Stage all five Python files together outside `R/source-r3`, for example at
`R/receiver-b300-r1/`. Use the actual provisioned B300 Python path:

```bash
R=/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1
PYTHON=/absolute/path/to/provisioned/B300/python
"$PYTHON" "$R/receiver-b300-r1/native_lifecycle_check.py" \
  --scratch "$R/control/b300-native-lifecycle-r1"
"$PYTHON" "$R/receiver-b300-r1/controller.py" --python "$PYTHON" --detach
```

The wrapper fixes BF16/BF16x9 as already declared. The controller changes only
compiler availability settings: `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`,
`TORCHINDUCTOR_COMPILE_THREADS=4`, plus exact receiving paths/device pairs.
It adds no new numerical precision setting.

The native lifecycle probe is Linux/Python 3.12+, CPU-only and bounded. It tests
nonzero wrapper exit and external TERM with two separate-session synthetic
descendants that explicitly ignore TERM, while an unrelated private control
process remains alive. It signals only fixture-owned identities, retains raw
ownership/descendant/result/log evidence under its scratch directory, and should
pass before relying on the receiver for GPU work.

Control files are under `R/control/b300-receiving-r1/`: implementation/support
hashes, detached owner, status, attempt ledger, per-command owned receipts/logs,
resource snapshots and authority. Existing phase receipts are never rerun, and
the controller rejects restart after phases have been admitted. Preserve failed
and partial evidence; any repair/restart needs a separately identified action.

## Deadlines, resources and explicit authority amendment

Every ten seconds the controller records receiving RSS using birth-bound process
descendants, including torchrun ranks in independent sessions. Limits are
1,024 GiB RSS, 200 GB of new outputs and at least 200 GB free on `/data`. New
outputs include `R/b300`, `R/reference`, and non-symlink files under project cache;
restored data links are not traversed. Existing ROOT-owned reference command
receipts contribute to observed receiving RSS. Termination remains limited to
this controller's exact receipt-owned commands; reference preparation ownership
stays with ROOT. Source-r3 observers supply all-device/per-rank GPU telemetry.

The reserve closes profile/fit admission and bounds active fitting. Completed
endpoints may evaluate/bundle/compare until the full campaign deadline. ROOT stop
signals or resource breaches stop only exact owned commands. No healthy foreign
process, GPU job, rental or source is modified.

If actual profiles show the fixed 30-epoch budget needs longer, ROOT may publish
`control/b300-receiving-r1/ROOT-authority-amendment.json` **before any fit is
reserved**. Preserve the original authority; include its SHA256 as
`previous_authority_sha256`, `approved_by="ROOT"`, a concrete `reason`, unchanged
`first_profile_epoch`, `source_commit`, `fit_cap=4`, `profile_cap=12`, and
`collection_reserve_seconds=3600`. Supply larger finite `deadline_epoch` and
`fit_deadline_epoch` exactly 3,600 seconds apart. The first accepted amendment
digest is pinned; later mutation is rejected. No automatic extension occurs.
Publish any needed amendment before publishing a cell's approval, since a ready
cell reserves its fit immediately after that approval appears.

The relay handles TERM before child creation and becomes a child subreaper for
its own descendants only. It journals birth/PPID/PGID ownership, terminates
proven separate-session rank groups through canonical helpers before the leader
group, and records remaining non-zombie processes. Foreign/reused PIDs are
excluded. This scoped process setting changes no model or system-wide setting.

Local checks: nine synthetic CPU gate/lifecycle tests, CLI help and Ruff.
Independent source review and all actual host execution remain ROOT's receiving
steps. This preparation contains zero GPU work or scientific endpoints.

# R01 mixed-arm density4 successor

## Deployment on 18 October 2026 UTC

Root deployed this packet to `/data/epanov_s/tabdata/runtime/r01-density4-v1`. The server passed all 23 CPU checks, including a real Linux stopped-parent/zombie canary. The first canary exposed `/proc/PID/environ` denying access after process exit; the corrected observer uses stable kernel identity for terminal Z/X processes, keeps missing metadata explicitly empty, and still rejects permission failures for a live process. Original failed and subsequent passing test logs are preserved.

The root-sealed control bundle is `89ccb760cdb05609bbce5239274c407922169dfbcfd1bccd44935fc1241ab238`; authority SHA256 is `364c1296b8f49712e2d8c5f70b9bdc18a9e806d097cb6ee8da47e3547d69f8d5`. `tabdata-r01-density4-v1` MainPID580481 first entered a read-only gate while the root watcher verified three actual cgroups. Root validated source/panel/control hashes, original 8 claims/52 untouched cells, the finite density profiles, inherited campaign/per-fit deadlines, joint projection, CPU affinity and live resources before granting the exact-authority start gate.

At approximately 22:03 UTC, the successor adopted all eight original seed1003 fits without restarting them and started eight seed1004 fits in the opposite-arm slots. The old controller MainPID520360 is intentionally stopped; its worker process groups and unit remain intact. Both admitting controllers cannot run together: the successor checks the pinned old main identity and stopped state on every loop and before submissions. The separate [PR15 successor](../../pr15_queue_v2/README.md) keeps 16 workers active. A subsequent NVSM sample contained 32 campaign CUDA actors (four per GPU) and all three known foreign vLLM actors. CUDA presence is launch evidence; sustained progress and rates are recorded separately.

Canonical remote receipts are `receipts/r01-density4-admission-v1/{AUTHORITY,UNIT_CREATION,READ_ONLY_READY,GATE_PREFLIGHT,START_GATE,LIVE_RESOURCES}.json` and `state/r01-density4-v1/{PREPARE,FREEZE}.json` plus the append-only events journal. The [deployment evidence](observed/) mirrors selected receipts. Historical source packets, original fit logs and checkpoints were not edited. The fit-only time projection below remains provisional and excludes export/scoring and scheduling overhead.

This packet is a successor design for adopting the original eight seed-1003 R01 fits without restarting them, then scheduling only the remaining zero-evidence cells. Adoption requires exact process identity, journal/process/workspace reconciliation, and root authority; the plan itself is not evidence that ownership transferred. Each GPU is limited to one wide and one deep fit, using the existing two-CPU lane plus the measured supplemental pair. It calls the frozen v4 fit validation, export, and paired-fold grading code from `runtime/b300-r01-v1`; it does not alter scientific arguments or the source packet.

## Historical preparation boundary

The following preserves the pre-deployment boundary, superseded by the deployment receipts above: at preparation time, no R01 adoption or successor `fit_started` count is claimed here. In particular, the separate PR15 v2 launch does not establish that R01 workers were adopted or that a 32-fit total is running. Verify any later R01 transfer against the root-owned authority and live receipts before updating this record.

The density evidence is a bounded engineering result, not an observed campaign duration. The two measured mixed-shape samples raised aggregate R01 service from 2.6493 to 4.6177 sequences/s on GPU2 and 3.3553 to 4.6541 on GPU7. The conservative fit-only projection is 38.46 hours versus 50.15 hours before scaleout, excluding delivery time. Resource reconciliation leaves profile peak memory and profile OOM counters explicitly unknown. A fresh root-owned campaign resource sample is required before every new fit and export.

Root seals `CONTROL_MANIFEST.json` and supplies `AUTHORITY.json` with the frozen paths and literal identity pins. Run the following from the deployed `r01_successor` directory; the read-only preview is useful before enabling execution:

```sh
AUTHORITY=/path/from/root/AUTHORITY.json
python3 successor.py --authority "$AUTHORITY" preview
python3 successor.py --authority "$AUTHORITY" prepare
python3 successor.py --authority "$AUTHORITY" freeze
python3 successor.py --authority "$AUTHORITY" run
```

`prepare` creates a fresh isolated state directory. `freeze` stops only the old R01 MainPID after rechecking its PID, start ticks, cgroup and command hash, and writes a durable receipt; if verification fails after SIGSTOP, it attempts SIGCONT through the pinned pidfd. `run` requires that receipt, re-reconciles the original cohort, adopts its eight workers, and starts only panel cells whose run directory and other evidence remain absent. The old unit stays intact and stopped until every adopted group exits and all delivery work ends. The root-owned host watcher must keep publishing a fresh aggregate resource receipt for the original R01 unit, this successor unit, and active PR15 v2, including per-unit memory/PID caps, OOM event counts, free host memory and worker-group RSS.

The controller journals each transition with fsync. On a global resource/deadline failure it closes fit admission, retains the unstarted cells as pending, and allows healthy adopted fits to finish; newly owned fit groups are stopped. It does not retry failed scientific fits. Each cell retains its checkpoint and log, and exports/grades remain in the canonical paths bound by the frozen panel.

The root-owned `resource_watch.py` writes the live receipt every two seconds and keeps its last verified sample if a procfs/cgroup read is transiently incomplete; the failed observation goes to the `.invalid.json` sidecar without refreshing the last-good timestamp. A sampled hard-cap contradiction instead publishes `verified: false` immediately. The successor cgroup must exist before the receipt becomes verified; an absent successor is reported as pending, never zero-filled. PR15 v2 may be represented by a terminal receipt after its service exits: while systemd still has the unit, it must report inactive/success with the exact caps; after systemd garbage-collects the unit, the watcher records the result as unknown and uses only its last verified live cgroup caps/events. The first terminal proof must arrive within 20 seconds of that last live sample and binds the 104-cell `RESULTS.json` and `COMPLETE.json` hashes. The watcher persists that proof, then rechecks both hashes, cgroup absence, and zero process members on every later poll. Its launch form is `python3 resource_watch.py --output <authority live_resource_receipt_path> --pr15-state /data/epanov_s/tabdata/state/pr15-execution-v2 --interval-seconds 2`.

Local checks cover scheduling uniqueness, process identity parsing, authority receipt gates, and CLI import/help. Deployment review must regenerate the flat Python file manifest after the final freeze and run the same tests in the remote Linux environment. This packet does not connect to SSH, stop a process, or start an experiment by itself.

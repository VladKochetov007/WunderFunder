# Independent sequence worker observation

Use one observer per run on the receiving Linux host. The durable controller still
owns launch authorization, deadlines, worker exit collection and GPU admission.
The observer sends no signals and never changes model state, fitting membership,
validation, checkpoint selection or optimizer policy.

Wrap an already authorized command (the wrapper adds the five-second trainer
progress heartbeat):

```bash
PYTHONPATH=src python scripts/observe_sequence.py \
  --output "$RUN" --interval 10 --max-seconds 86400 -- \
  torchrun --standalone --nproc_per_node=1 scripts/train_sequence.py \
  --config "$CONFIG" --data-root "$DATA" --split-manifest "$SPLIT" \
  --profile-receipt "$PROFILE" --output "$RUN"
```

Or attach without launching anything:

```bash
PYTHONPATH=src python scripts/observe_sequence.py \
  --output "$RUN" --pid "$WORKER_PID" --interval 10 --max-seconds 86400
```

For attach mode, start the trainer with `WNN_MONITOR_HEARTBEAT_SECONDS=5` to
retain exact successful-update and phase snapshots. Without that opt-in the
external observer still reports actual process/GPU liveness and producer-event
age; absent successful-update age remains unknown. The observer wall limit stops
observation only. A wrapped command's exit code is returned after its tracked
process family exits; attach-mode exit zero is an observation result, not proof
of a successful fit.

`monitor_progress.json` records phase, last successful finite optimizer update,
the time of that update and its forwarded rows/second. Its optional thread keeps
the heartbeat fresh during evaluation and checkpoint writes. Heartbeat freshness
does not claim optimizer progress. `monitor_observer.jsonl` independently binds
the root PID and descendants to Linux start ticks, excludes zombies and PID reuse,
samples every visible GPU by device index and UUID, and associates GPU memory with
actual compute PIDs. Unrelated processes are labelled `tracked=false` and are
never signalled. CPU/RAM/disk readings and event/checkpoint/progress ages are
reported separately. Missing NVIDIA readings remain unavailable; resource samples
do not establish peak FLOP utilization.

The observer does not create the trainer's run directory and starts its journal
only after that directory exists. Its journal is separate from resumable training
and evaluation offsets. The journal is capped at 256 MiB, each line at 128 KiB,
NVIDIA calls at three seconds, and each process scan at 16,384 proc entries.

Copy the native files unchanged to a local directory per run:
`run_metadata.json`, `monitor_events.jsonl`, `monitor_evaluations.jsonl`,
`monitor_observer.jsonl`, and the latest `monitor_progress.json` and checkpoint
indices. Keep source hashes and collection timestamps. The existing dashboard
continues to select runs/configurations and compare their native metrics:

```powershell
python scripts/research_dashboard.py --runs-root <local-mirror-parent> --port 8766
```

Open <http://127.0.0.1:8766/>. Fresh independent observations with live bound
processes keep a run live during long evaluation/checkpoint phases; stopped
observations remain stale. A dead process family is explicit even if its last
training event was recent. GPU UUID/PID mappings, phase and ages appear in the
selected run's observation details. Every observer metric is also available to
the existing graph selector. PNG/SVG rendering can run locally from unchanged
captured journals when the node has no matplotlib; it must never reevaluate a
model. Official valid and the fixed fitting-only train-eval cohort retain their
own masks and identities.

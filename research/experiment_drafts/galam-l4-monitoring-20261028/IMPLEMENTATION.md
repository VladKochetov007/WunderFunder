# Implemented monitor and receiving-host handoff

Status: engineering implementation checked locally, no scientific fits or GPU
profiles. The [contract](CONTRACT.json), [plan](PLAN.json) and [preflight](L4_PREFLIGHT.json)
separate working code from missing real-data/runtime admission. Original GALAM
dimensions, recipe and complete fusion/FiLM remain unknown.

## Code map

- [Dashboard](../../../src/wnn/dashboard/cli.py): dependency-free local server,
  offline assets, bounded workers/caches/read windows, config filtering, eight-run
  comparison, measured-point charts, hover inspection and visible stale/error,
  no-data, synthetic and history-limited status. No training controls.
- [Sequence monitor](../../../src/wnn/training/sequence_monitor.py) and
  [paired evaluator](../../../src/wnn/training/sequence_monitor_evaluation.py):
  opt-in batched fixed train-eval and complete official valid reporting before
  each safe epoch checkpoint. Both reporting cohorts use official masks, pooled
  metrics and clean resets; model mode, BatchNorm state and RNG are preserved.
  Existing fitting/optimizer/checkpoint paths remain canonical; absent monitoring
  does not inject a new legacy recipe key.
- [Sampled diagnostics](../../../src/wnn/training/sequence_monitor_diagnostics.py):
  branch activation/gradient RMS, hidden/cell/gates, actual attention entropy/lag,
  RelativePE statistics, fusion saturation and prediction distributions. Hooks
  detach observations and preserve the underlying attention calculation.
- [Resource sampler](../../../src/wnn/training/sequence_monitor_resources.py):
  process CPU/RSS, host RAM, output-disk capacity and physical-UUID GPU sensors,
  plus CUDA allocator statistics. Missing sensors stay absent. GPU utilization
  measures activity, not tensor-core throughput. Resources/diagnostics are the
  local rank-zero worker; independent one-GPU fits each have their own worker.
- [Journals](../../../src/wnn/training/sequence_monitor_events.py): safe-checkpoint
  offsets preserve orphan evidence and restore both event files after a crash.
  Epoch scores are stored separately from the dense step tail. Profiling records
  distinguish a planned validation policy from evaluation actually performed.
- [Recurrent adaptations](../../../src/wnn/galam/variants.py): GRU has no cell
  state; local-only has no unused global modules. The original LSTM path passed
  independent bitwise parity on a tiny CPU fixture. Neither adaptation is
  claimed to be original author code.

| Arm | Parameters | Fusion inputs |
|---|---:|---|
| Full LSTM schematic | 376426 | global, local, h, c |
| Full GRU adaptation | 286698 | global, local, h |
| Local-only LSTM adaptation | 195398 | local, h, c |

These are assumed-size112-to-2 configurations, not recovered historical sizes.
The GRU and branch ablations change decoder dimensions and total parameters.
[Inspection](configs/INSPECTION.json) also reports inactive fields and the active
fusion order. Local-only inactive global/ConvMixer placeholders cannot be tuned
silently; active fusion permutations follow the declared order.

## Curves and clocks

Comparable train-eval/valid weighted MSE use one panel, comparable official WP
another, and target0/target1 have separate panels. Online fitting loss is labelled
separately. The train-eval cohort has at most32fixed fitting sequences in the
prepared recipes; it is a diagnostic sample, not all train. Full valid remains
the primary exposed endpoint. Gates, branch/attention/bias/prediction diagnostics,
LR/gradient/clipping, process/GPU/disk sensors, throughput and timing are separate
panels with a generic metric explorer. Unmeasured metrics stay empty.

Epoch and optimizer step are durable progress axes. The optional elapsed axis
uses UTC wall time relative to the first retained event; pauses, resume downtime
and clock corrections are included. It is not active GPU time. Merged events
are ordered by step so clock correction cannot reorder model progress. CPU
process-time counters reset when the process restarts.

The live reader retains a bounded training tail and up to512evaluation events or
2MiB per run; visible history flags qualify truncated curves. Authoritative rank
logs, checkpoints and predictions remain separate. The producer caps events at
128KiB, epoch reports at32KiB/10000epochs and the secondary general monitor log at
1GiB. Exceeding its quota is an explicit error, not a scientific negative result.
Disk admission must budget primary logs/exports as well as secondary monitor files.

## Local dashboard

From the repository root, with Python3.11+:

```powershell
python scripts/research_dashboard.py --runs-root C:/path/to/run-parent --port 8766
```

Each actual run is a direct child directory containing immutable
`run_metadata.json`, `monitor_events.jsonl` and `monitor_evaluations.jsonl`.
Current local services are http://127.0.0.1:8766 (real run directory, currently
empty) and http://127.0.0.1:8767 (labelled synthetic demonstration). Runtime logs,
PIDs and browser captures are under
`C:/Users/admin/Documents/Codex/reviews/tabdata-monitoring-20261028/`.
These services were started separately from the CLI. No remote training exists.

To create a separate demo, never an actual fit output directory:

```powershell
python scripts/create_dashboard_demo.py --output C:/path/to/synthetic-demo
```

## Future receiver commands; not executed

After the real-data/headroom/environment/source checks, from the exact deployed
source, prepare the frozen chronological manifest and a one-GPU profile:

```bash
PYTHONPATH=src python scripts/build_chronological_primary_split.py --root DATA_ROOT --output PRIMARY_JSON
CUDA_VISIBLE_DEVICES=GPU_INDEX PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/train_sequence.py --config research/experiment_drafts/galam-l4-monitoring-20261028/configs/full-lstm-s1001.json --data-root DATA_ROOT --split-manifest PRIMARY_JSON --output NEW_PROFILE_DIR --profile-updates 50
```

Upper-case arguments are unresolved receiving-host paths/assignments. Commands
are concrete implemented interfaces, not claimed execution. The manifest producer
requires verified complete source files plus existing audited component/target
indexes. The inspected L4 project does not yet provide them.

A completed exact-identity engineering receipt needs review before admitting the
corresponding full fit; the existing runtime records this as `approved`. Do not
reuse it after changing batch, recipe, source, data or runtime. Profile loss is
not validation. Measure first-launch/warmed phases, epoch/evaluation time, monitor
cost and shared CPU/disk contention before estimating the six-fit duration.
The initial batch16is a starting candidate; actual membership must not create a
singleton training tail or BatchNorm-contaminating DDP padding. No optimal batch
or L4 ETA is currently measured. Monitored profiles exercise the actual sampled
train/diagnostic/resource callbacks and require at least one diagnostic interval
(25updates in these recipes). They do not evaluate valid or save scientific
weights; this distinction is explicit in status and profile provenance.

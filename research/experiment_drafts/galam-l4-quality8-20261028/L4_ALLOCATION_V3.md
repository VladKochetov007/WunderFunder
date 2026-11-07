# Eight-L4 allocation after the GRU-history audit

28 October 2026. Concrete plan for user review; zero dispatches, profiles or fits.
Latest read-only observation:2026-10-28T02:43:45Z through the same persistent SSH
session. Eight NVIDIA L4,23034MiB each; computation0% in snapshot; GPU4 retains
1148MiB for root-owned services. RAM728GiB total/~457GiB available, /data2~70GiB
free. CPU192logical/96physical across two NUMA nodes. Session cgroup cpu/memory
limits are max; this does not reserve shared host resources.

## One process and one scientific fit per GPU

| GPU | Fit | Initial CPU-affinity proposal |
|---|---|---|
| 0 | Fresh plain GRU256x2, historical causal211-feature system benchmark | 0-7, NUMA0 |
| 1 | Full schematic GALAM LSTM64, control for GALAM changes | 8-15, NUMA0 |
| 2 | GALAM LSTM128, capacity package | 16-23, NUMA0 |
| 3 | GALAM with two LSTM layers | 24-31, NUMA0 |
| 4 | GALAM GRU72, near-parameter-matched architecture alternative | 48-55, NUMA1 |
| 5 | GALAM LSTM64 with attention history256 | 56-63, NUMA1 |
| 6 | GALAM LSTM64 with TBPTT256 | 64-71, NUMA1 |
| 7 | GALAM LSTM64 with attention/FFN-only Muon and AdamW elsewhere | 72-79, NUMA1 |

This explicitly replaces the standalone LR arm by the plain-GRU benchmark and
the horizon10 arm by the optimizer experiment. The checked v1 JSONs remain
unchanged; this table is not a ready launcher. All eight are new fits, one seed
each, no reused trained control, teacher or KD fits. A GALAM control is itself
one candidate against the plain-GRU benchmark. Architecture-only comparisons
use GPU1 as control; the plain211-feature benchmark is a whole-system comparison
against raw112 GALAM, not pure attribution to the recurrent cell.

Common maximum20epochs and seed1001, epoch checkpoints, official chronological
train-only fitting/full-valid reporting, train-only transforms and zero old
teacher/initializer ancestry. Scheduler horizon, LR, precision, selection and
early-stop policies must be recorded explicitly in revised recipes before fits.
The GRU history does not establish a20-epoch optimum. All baseline/candidate
effects remain single-seed exposed-development evidence.

## Engineering sequence before full fits

1. Resume/reverify the missing train ranges from a verified source. The fresh
   receiver still has31/109ranges,8,321,499,136recorded completed bytes and no
   full train.parquet. Keep one raw source dataset on disk for all processes;
   hash it once against the pinned producer identity and reuse verified identity.
2. Resolve source-qualified coverage/group528 handling, train-only normalization,
   actual train masks, plain-GRU feature/monitor integration, GALAM hybrid Muon
   groups and receiving runtime. Driver570/cu128 versus the cu130 lock remains
   an actual compatibility dependency. No global driver/install shortcut.
3. On representative real shapes, start plain GRU batch32 and GALAM batch16;
   adaptively profile32/64/128 as memory and throughput allow, including full
   history/TBPTT and sampled diagnostics. Profiles are separate engineering work,
   not extra scientific fits. Fix the common GALAM batch across controlled arms;
   any per-arm exception is an explicit recipe difference.
4. Test the chosen exact eight recipes concurrently with monitored engineering
   profiles to measure shared disk/CPU contention. Record initialization,
   train/evaluation/checkpoint throughput, allocated/reserved and observed VRAM,
   data wait, CPU/RAM/disk pressure. Choose by completed rows/s and projected
   endpoint time, not max batch or nvidia-smi activity alone.
5. Publish the measured per-fit/panel ETA, freeze source/config/runtime/profile
   identities, then launch the eight independent detached owned jobs. Preserve
   checkpoint/RNG/optimizer/scheduler state and exact process ownership; verify
   jobs survive loss of the SSH client. Shared services remain outside ownership.

CPU affinity above is a starting locality allocation, not an exclusive reservation
or measured optimum. Begin with up to8framework threads/fit and bounded prefetch
of1-2batches, then reduce/increase measured thread work. GPU0-3 CPU locality is
node0 (0-47,96-143); GPU4-7 is node1 (48-95,144-191). Match host buffers and decode
work to the fit's node where feasible. Bounded asynchronous pinned host-to-device
copies/prefetch need explicit implementation; current direct sequence collation
is synchronous. Do not create eight full decoded dataset copies in RAM or disk.

For batch128, one raw112 float32 full-sequence input batch is1,146,880,000bytes;
211features is2,160,640,000bytes, before targets and extra copies. Two queued input
batches across seven112-feature fits plus the211-feature benchmark total roughly
20.4GB before current batches, pinned buffers and overhead. Allocate a bounded
host pipeline budget and monitor the shared host rather than assume all457GiB
available is ours. One OS page cache can serve the immutable Parquet sources.

The GALAM time loop and evaluation's frequent CPU copies can dominate small
models. If the profile identifies launch/synchronization overhead, optimize that
path with state/BatchNorm/causality/gradient parity checks before long runs.
LSTM per-time cell-state outputs and per-time normalization must survive any
chunk optimization; replacing them by a different graph is a new experiment.
Reduced precision is another explicit implementation/profile treatment, not an
unsupported flag applied to the currently FP32-only GALAM adapter.

## Monitoring and output budget

Full official validation and fixed train-eval after each epoch, separate loss/WP
plots, t0/t1, LR/gradients, sampled model diagnostics and CPU/GPU/RAM/I/O telemetry.
Detect persistent development degradation and apply only the predeclared early-stop
policy. Keep all epoch checkpoints and scalar evaluation records. Export full
predictions only for selected/final endpoints:1873*20000*2*4=299,680,000bytes per
full-valid float32 array, or4,794,880,000bytes for16exports. Exporting all8*20epochs
would instead consume47,948,800,000bytes before metadata, leaving little working
headroom after train completion. Include final export time in the ETA.

The present monitor writes each update and caps an event log at1GiB. Twenty-epoch
runs can exceed that cap; sampling/aggregation or replay-safe rotation must be
implemented and bounded before admission. Never remove sparse full-validation
events when thinning training telemetry. Training jobs, sensors and dashboard
collection need explicit independent ownership so an SSH disconnect does not
terminate the fits. The current local dashboard is not already a remote collector.

No measured ETA yet. Per-fit time is the measured sum of fit/evaluation/checkpoint
and export phases; parallel panel time follows the slowest lane plus shared setup.
Maximum budget160fit-epochs has unequal per-arm cost, and early stopping does not
guarantee a deadline. A one-GPU-per-fit plan avoids multi-GPU synchronization for
these small models; alternative concurrency is a future measured choice, not
extra hidden fits in this eight-fit allocation.

[PyTorch performance guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
supports measuring loader concurrency, pinned/asynchronous copies and NUMA/thread
locality. Exact thread counts, batches and gains above are proposals to profile.

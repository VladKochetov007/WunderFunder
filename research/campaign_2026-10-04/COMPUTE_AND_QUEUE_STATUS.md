# Compute budget and experiment count — 2026-10-04

The user clarified that the accepted research plan included an 80-device-GPU-hour campaign budget. Treat that as the cumulative campaign ceiling across local and Kaggle runs, including failures and repetitions. It is not an 80-hour Kaggle reservation. The active execution contract records provider-specific admission so consumption can be reconciled without confusing local block hours, Kaggle quota, or TPU hours.

## Current capacity

| Resource | Current allowance/status | Headroom policy |
|---|---|---|
| Local GPU | 24-GPU-job-hour block; 12,820.7 seconds charged and 4,530 seconds reserved, leaving 69,049.3 seconds (19.18 hours) unreserved | 8 worker CPUs; 24 GiB worker RAM; one GPU worker; 4.2-GB allocator and 4.8-GB observed stop; parent slice 36 GiB/12 CPUs. Current fit is near its worker RAM ceiling; do not overlap another local job. |
| Kaggle GPU | 25.59/30 device-GPU-hours remain at12:00Z after E-419 v4 launch; provider charge may lag | Two T4s; begin below 12 GiB per GPU, stop at 13 GiB; v4 cap 8.5 device-GPU-hours. Both T4s are occupied by E-419 v4. |
| Kaggle TPU | At 12:33Z, 0/20 hours used; `wnn-tpu-probe-20261004` remains QUEUED after about 11 hours | No TPU model or HBM topology has been assigned/reported; don't book TPU throughput or fit hours until allocation |

The current single local block plus Kaggle balance is 46.41 device-GPU-hours before the planned Kaggle width work and six-hour reproduction/recovery reserve. Renewed local blocks and Kaggle's Oct 10 quota refresh make the cumulative 80-hour ceiling feasible across cycles. Check provider charges after every remote run; do not assume a dual-T4 wall hour equals one quota hour.

## Count without conflating ideas and jobs

- **21** Oct 4 continuation experiment records, E-407–E-427: **14 complete, 2 partial, 1 queued, 1 running, 3 prepared**. E-424 is running; E-425–E-427 are critic-generated designs with implementation and resource admission still pending.
- **44** H-065–H-108 hypothesis entries. These are idea records, not training jobs; many are bounded screens or implementation work.
- **Immediate active tranche:** E-424 has 8 planned fit arms; its first H-093 learned-specialist fit is running locally. E-419 v4 is running remotely with two QAT arms. These are two concurrently active programs on separate machines. The original static controller has a20:00 hard-stop timer armed; the source now requires a sealed job's full timeout plus a stop margin before launch.

## First broad research tranche under the 80-hour ceiling

The first slate is now twelve registered programs, approximately **35 new fitted models**, with exact shared controls reused where provenance permits. E-425 through E-427 came from independent implementation critiques. This is a staged run slate, not a requirement to burn all 80 hours or to reject later ideas:

| Program | Planned fits | Main question |
|---|---:|---|
| E-424 H-093/H-094 | 8 | Do causal prefix specialists/adapters beat parameter-matched recurrent controls? |
| E-419 H-069/H-105 | 2 QAT continuations; reuse recovered FP32 controls | Does training quantization-aware students improve score/runtime? |
| H-073 | 3 | Do aligned chunk-mass weights beat unchanged and shuffled multipliers on the actual distilled objective? |
| H-079 | 2 | Does random-origin training improve fixed-schedule forecasting over a same-seed control? |
| E-413/H-078 | 2 | Does the volume-history lead repeat on a second fold/seed against its matched input control? |
| H-072 | 2 | Does connected macrostep credit help against midpoint-detached credit with equal forward history and pooled normalization? |
| H-066 | 4 | Does width interact with optimizer on the combo at matched settings? Benchmark inference cycles before fit; a slow model can still be a teacher. |
| H-098 | 3 | Do cross-fitted future-feature teachers transfer to causal students while preserving origin labels/weights? |
| H-067 | 2 | Does the second full-data seed improve the complete fixed predictor/bag? |
| E-425 / H-106 | 1 new fit | Does a soft prefix mixture improve expert gradient support and prediction over learned/universal routes? Reuse E-424 controls. |
| E-426 / H-107 | 3 | Does an event-density consistency-teacher EMA help when the evaluation EMA is held fixed? Compare aligned and shuffled event clocks. |
| E-427 / H-108 | 3 | Can endpoint-relative volume history gate recurrent state updates? Compare aligned, stale-cycle and sham gates. |
| **Total** | **35** | Twelve programs, with controls/reuse where provenance allows; additional CPU reset-lag and sequence-gain probes are tracked separately. |

Before each program, freeze its matched contrasts, inputs, expected cost, score gate, and output paths in the experiment ledger. A fit that fails a scientific gate remains evidence; historical results alone do not bar a bounded retest. Use the unallocated portion of the 80-hour ceiling for independent seed/fold replication, promising mechanism reframes, and other still-open review proposals (including causal transition features, i1 innovations, event-age interactions, and the origin-reset/gain probes). Every such run must be entered and sealed before its GPU allocation.


### TPU utilization gate

The submitted private TPU probe requests Kaggle `TpuV5E8` with `enable_tpu: true`; the active E-419 training kernel is a separate T4-only kernel with `enable_tpu: false`. As of 12:33Z the TPU probe has no startup log, device inventory, memory/HBM report, or TPU-hour charge. Its bounded payload is an inventory plus one tiny JAX GRU update and one hidden-16 repository-model Torch/XLA update; each framework runs in its own five-minute subprocess. Passing that probe establishes runtime availability only, not useful training throughput or score parity. The requested machine shape is metadata, not evidence that Kaggle assigned it.

Do not route current fits to TPU unchanged: `scripts/train_rnn.py` exposes only `cpu`/`cuda` and has CUDA-stream, CUDA-timing, and CUDA synchronization paths. The existing optional JAX feature-screen backend only solves small fixed ridge systems; that is not a meaningful TPU workload. Once a device is assigned, first record its topology and HBM, then run a separately admitted full-shape forward/backward/optimizer throughput pilot with fixed shapes, bfloat16 and fp32 checks, peak-memory capture, and a small score-parity comparison. If that pilot passes, add a Torch/XLA training adapter and schedule an existing matched experiment on TPU in parallel with T4 jobs, charging its hours to the separate 20-hour TPU pool. If the assigned runtime fails the real-model path, preserve the probe result and investigate JAX/XLA batched sequence work rather than claiming TPU acceleration.

E-424 and E-419 v4 are running in parallel. The active E-424 worker peaked at25.76/25.77GB cgroup RAM; current use is about25.54GB, mostly reclaimable file cache, with zero worker OOM events. Parent memory use is about35.86/38.65GB. Local current GPU memory is3.84/6.14GiB and observed peak is4.46GiB, below the4.8-GB stop. Keep one local job only and wait for memory headroom before any CPU probe. Kaggle E-419 still has no worker logs/telemetry; both T4s remain occupied. Preserve Kaggle's nine-hour width allowance and at least six GPU-hours for recovery/reproduction.

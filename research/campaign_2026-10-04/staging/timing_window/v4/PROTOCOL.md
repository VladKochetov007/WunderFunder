# Loaded-host paired callback timing v4

## Question and limits of the claim

Measure the full per-row native callback path for three fixed comparisons on one pinned logical CPU: seed-4 i16 versus seed-3 i16; q8r versus seed-3 i16; and q8x versus seed-3 i16. Every call processes the same first three complete E457 streams in the same sequence order: 60,000 callback rows, with the 99-row warmup in each stream retained and 59,703 score rows validated. Each pair uses two fresh package predictors, reverses the A/B order on alternating repeats, and repeats five times. The comparison phases run in the declared order (seed comparison, q8r, q8x). Each candidate therefore receives five complete paired trials; there are 30 full callback invocations total.

This v4 protocol treats host load as a measured covariate. It records system load averages, CPU pressure, memory pressure, aggregate/core CPU counters, and VM fault/reclaim counters before and after every trial. It never fails, discards a trial, or retries solely because another host process raises load. A callback/input/provenance error still fails closed, with completed trial rows preserved. CPU15 affinity, source/data/package hashes, complete callback row counts, and the one-CPU/2-GiB worker resource cap remain hard requirements.

For every timed callback window, record elapsed wall time, process CPU time, thread CPU time, and the existing Linux `PERF_COUNT_HW_CPU_CYCLES` user-thread event, each normalized per callback row. The event is a perf hardware-cycle count, not a literal RDTSC instruction. If the host denies the event, record it as unavailable; do not substitute a fabricated cycle rate. The process/thread CPU clocks are sampled at the same two `perf_counter` boundaries used by the frozen callback helper, excluding package loading and setup just as its wall timer does.

The worker validates the frozen E450 seed-4 native-parity readiness and attempt, and the E462 q8 build/parity/score chain and attempt bindings. It verifies the exact seed-3, seed-4, q8r, q8x ZIP and 20-stream callback-data hashes, then reuses the frozen E462 stream loader and complete callback trial function. It changes no package, callback, score, or timing helper. There is no new export, training run, holdout score, or readiness gate.

## Summary estimands and uncertainty

Report all five paired ratios for each contrast and their geometric mean. Report a two-sided 95% interval from a Student-t interval on the five log ratios (df=4, critical value 2.7764451052); label the small-sample/model-based interval clearly. Show the five raw pairs, medians, means, and pair order. Do not remove outliers or choose a best trial. Also report the five per-candidate 60k wall times and use the same log-t construction for geometric-mean callback rate and absolute projection. These three contrasts are descriptive measurements, not multiplicity-corrected hypothesis tests.

For a scaled comparison, multiply the seed-3 public runtime anchor, 2,267 seconds (37m47s, user-reported for the seed-3 i16 ZIP), by the measured paired geometric-mean runtime ratio and its interval. This ratio estimate assumes relative timing transfers across hardware and load; the public evaluator's hardware and competing work are unknown. It is a projection only, never an official runtime result or readiness claim.

Separately project each candidate's measured geometric-mean callback seconds/row over 37,460,000 rows (1,873 streams × 20,000 rows) and propagate the log-t interval. This absolute projection covers only the measured row callback path. It omits archive/model setup, evaluator I/O and scoring overhead, and hardware differences; it is not an estimate of measured public runtime. Present ratio-based and absolute projections side by side; do not reconcile them by adjusting the timing sample.

## Interference and memory-bandwidth interpretation

Pairwise reversal and adjacency reduce first-order order/cache drift, while all raw pairs, system-load snapshots, per-thread CPU/wall ratios, and cycle-per-row values make remaining interference visible. If an apparent difference is similar to within-pair dispersion or its interval spans 1, describe it as unresolved at this sample size. If host CPU pressure or the pinned core's busy fraction changes alongside process CPU/wall, scheduling contention is a plausible contributor. If per-row cycles and wall time move together while process CPU/wall remains near one, extra work, frequency changes, cache misses, or memory stalls remain possible.

No memory-controller/uncore bandwidth counter is requested or assumed. PSI and VM counters reveal pressure/reclaim, not bandwidth throughput. Therefore the trial can flag a slowdown compatible with memory/cache contention, but cannot identify or quantify bandwidth contention. Distinguish repeated paired effects from sampling noise; do not claim a memory-bandwidth mechanism from host load or cycles alone.

## Admission and resources

The distinct v4 runner is a CPU-only measurement under 1 CPU, 2 GiB RAM, no swap, and no GPU. It pins the process to CPU15 and limits BLAS/OpenMP/PyTorch threads to one. The worker timeout is 1,200 seconds. It does not wait for a quiet host, impose a global load threshold, start automatically after another service, or launch any GPU/TPU work. Root separately admits the exact source/input packet and execution timing.

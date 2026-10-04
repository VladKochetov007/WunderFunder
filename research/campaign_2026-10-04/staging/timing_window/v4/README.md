# Timing v4 source packet

This is a new loaded-host comparison instrument. It does not alter the quiet-window v3 controller, timer scripts, sealed specs, or prior attempts. It validates the same frozen parity/data/package evidence, then measures the unchanged E462 full callback under host load as recorded telemetry. There is no quiet-load wait, global load threshold, retry, export, training, score, or readiness/public-runtime claim.

The exact protocol and interpretation bounds are in `PROTOCOL.md`; a path-only campaign draft is in `specs/e465_loaded_host_paired_callback_timing_v4.draft.json`. The worker pins CPU15, performs five alternating 60k-row pairs for each of the three declared contrasts, records callback wall/process/thread CPU/perf-cycle timing and per-trial host snapshots, and reports log-scale uncertainty with paired-ratio and callback-only absolute projections.

The perf cycle count is the same Linux `PERF_COUNT_HW_CPU_CYCLES` current-thread event used by the frozen timer, not a literal RDTSC instruction. PSI, VM counters, system load, and cycle counts do not directly measure memory-controller bandwidth. No remote or scientific work has been launched from this packet.

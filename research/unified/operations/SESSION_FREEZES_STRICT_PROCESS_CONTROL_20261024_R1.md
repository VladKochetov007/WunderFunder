---
kind: operations
status: active-posture
created_utc: 2026-10-24T20:05:00Z
---

# Session freezes and strict process control — 2026-10-24 evening

## What happened

During the evening independent-review/optimizer session the laptop froze and
rebooted **twice** (once around 19:5x UTC during the first optimizer grid, once
earlier during the review-subagent phase). The first optimizer grid was running
under `scripts/limited` with one CPU core and a 900 MB cgroup cap when the second
freeze hit; only `results/adamw_s0.json` survived on disk, `adamw_s1` died
mid-fit (partial log preserved). No other research process was running at either
freeze; user-reported system state said "RAM nearly full" while the research
processes themselves were far below the user's later "up to 25 GB" allowance —
so the freezes are attributed to overall system state (memory pressure from
other applications and/or sustained-load thermals), not to a single runaway
research process.

## User instruction

"it restarted again. control subagents and code all processes strictly", with an
earlier allowance of up to 25 GB RAM. Effective immediately for this session:

1. **Zero subagents.** No Agent spawns; all work happens in the main thread.
2. **One bounded process at a time**, synchronous (foreground), never background
   batches.
3. Every process runs under `scripts/limited` with explicit caps: 1 CPU core
   (`CPUQuota=100%`), `MemoryMax` ≤ 700 MB for screen fits (far below the 25 GB
   allowance — deliberately conservative after freezes), `MemorySwapMax=0`,
   `RuntimeMaxSec` per task, physical-disk `TMPDIR`
   (`/home/vlad/development/wunder/research_tmp`).
4. **Pre-run gates** before each process: hottest `/sys/class/thermal/thermal_zone*`
   < 84 °C and `MemAvailable` ≥ 8 GB; abort with a recorded reason otherwise.
5. **60 s cool-down** between sequential fits, with post-run temperature logged
   ([strict_fit.sh](../../experiment_drafts/own-optimizer-muon-20261024/experiments/strict_fit.sh)).
6. Preserve all partial outputs and logs from killed runs as evidence; never
   delete a failed run's artifacts.

## Scope

This posture binds this session's local execution. It does not reopen scientific
admission for competition-data fits, provider jobs or uploads; the evening's
work remains analysis + the bounded synthetic optimizer screen. Historical
resource ADRs (0021/0006/0039) remain in force; this note records the stricter
session-level posture requested after the freezes.

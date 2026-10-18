# Persistent job monitoring

`wnn-research-monitor.timer` is enabled in the systemd user manager. It checks
local controlled attempts, live/failed WNN services, cgroup RAM/CPU/swap, GPU
memory, and active pinned Kaggle jobs every five minutes. Kaggle discovery
(up to100 most recent kernels) and GPU/TPU quota refresh every15 minutes.
A bounded tick checks at most20 statuses using two provider calls concurrently;
pinned/known-active jobs are first. Deferred checks are explicitly reported.

Register every successfully launched Kaggle handle in `MONITORED_KERNELS.json`.
Pinned and cached active handles remain observed even after leaving discovery.
Terminal statuses are reused only for an unchanged provider run timestamp.
Network/CLI observation failures remain separate from a scientific ERROR.
Missing/future start timestamps, long queues, >8h runs, stale worker heartbeats,
ended active services and changed resource limits produce alerts.

Latest health: `monitor_runtime/STATUS.json`; alert transitions:
`monitor_runtime/ALERT_HISTORY.jsonl`; compact new error logs are collected at
most two per tick. These mutable files are gitignored. Dated incident reports,
source, install receipt and unit templates are tracked. Initial historical
errors remain visible; they are not all newly failed research jobs.

The checker uses a20% CPU quota and512MiB RAM, no swap, a300s timeout and a
nonoverlap lock. The quota was lowered on 18 October 2026 after the latest
two-core allocation change. Its preceding run used3.69 CPU-seconds over49s, so
the20% cap retains margin while leaving room for the single-core GPU worker.
The five-minute timer remains enabled. The checker never restarts, launches,
cancels or retries experiments.
The existing E463 fifteen-minute collector separately retrieves compact terminal
JSON/log/text/CSV outputs. Local scientific workers retain the faster existing
resource watchdogs; the five-minute checker supplements them.

Inspect: `systemctl --user status wnn-research-monitor.timer` and
`journalctl --user -u wnn-research-monitor.service`. The timer persists between
agent turns and is enabled for future user-manager sessions.

Current local caps follow [ADR0039](../../../specs/adr/0039-two-core-local-budget-and-disk-scratch.md):
two aggregate CPU cores,20GB parent RAM maximum,18GB high watermark, no swap,
and one GPU worker. The active R4 task has a separate5GB hard cap. Historical
runner constants and old resource snapshots retain their dated values; current
admission reads the live parent cgroup and applies the current user allocation.
Existing OOM counters are baselines; only new increments are reported as new
resource failures.

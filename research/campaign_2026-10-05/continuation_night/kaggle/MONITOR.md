# Background provider monitoring

Private W384 serial v2 and E463 TPU v3 pushes accepted as provider version 1. Each has an immutable pre-push budget reservation in reservations/. W256 has not been pushed and remains deferred until enough unreserved provider budget is available.

As observed at 00:00 Kyiv on October 6, W384 v2 is ERROR and TPU v3 is QUEUED. W384 failed preserved synthetic trainer tests because the bounded loader captured a stale 20,000-row constant before the fixtures changed their contract to 8 rows. No OOM was recorded; child teardown completed. See [failure evidence](W384_V2_FAILURE.md). A narrow correction must preserve the full test suite and the production 20,000-row contract before any independently reviewed retry.

The enabled user timer `wnn-research-monitor.timer` invokes `wnn-research-monitor.service` every five minutes, even after this chat turn. It runs in the aggregate campaign slice with 1 CPU, 512 MiB RAM, no swap and a 300-second ceiling. It refreshes provider quota/catalog every fifteen minutes, monitors pinned kernels until terminal, records observation errors separately, saves error logs, and downloads terminal diagnostic JSON/JSONL/log/TXT/CSV files (one provider download per invocation, 45-second timeout). It also checks both local campaign states, stale heartbeats, memory/OOM changes, service exits and VRAM.

Snapshots: ../../checks/monitor_runtime/STATUS.json
Incident history: ../../checks/monitor_runtime/ALERT_HISTORY.jsonl
Provider diagnostics: ../../checks/monitor_runtime/provider_outputs/

The monitor never launches, cancels or blindly retries experiments. Failed runs need a recorded diagnosis and mitigated retry within the contract. Full checkpoint downloads are separate bounded collection work. Timer execution continues after the local scientific launch cutoff because it observes and collects only.

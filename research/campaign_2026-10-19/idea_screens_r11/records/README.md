# R11 execution records

[Startup preparation](startup_v1/README.md) records the pinned transport, attempt-2 local cold-import smoke, and create-only assembly. [Attempt V1](attempt_v1/README.md) preserves its pre-output import-inventory failure and confirms it was not restarted or stopped. [Attempt V2](attempt_v2/README.md) records the new service invocation and completed run. Its [terminal capture and resource review](attempt_v2/terminal_capture/README.md) preserve raw-byte metadata, exact unit stop, and resource-scope reconciliation.

The V2 `.npz` payload was not copied to the Mac; its inventory size and producer SHA are retained in the raw result/capture metadata. Scientific acceptance and interpretation are maintained separately.

The V2 resource record reports systemd unit timing separately from `DONE.json` elapsed values; the earlier metadata bytes are preserved as `*_V1.json`.

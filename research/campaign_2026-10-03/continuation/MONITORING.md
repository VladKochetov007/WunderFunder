# Crash monitoring and research continuity

`wnn-codex-research-watch.service` runs every minute and restarts if the watcher
itself exits unexpectedly. It checks user/kernel journals, GPU terminal job
records, aggregate OOM counters, total device memory and monitoring access.
Durable evidence is in `logs/incidents.jsonl`, `watch_memory.json`,
`watch_status.json`, job JSON and per-job logs. A new incident or degraded
monitoring publishes an admission barrier; existing successful evidence remains.

The controller reads each new incident before acknowledging the barrier. Preserve
the failed job, inputs, partial captures and source/runtime hashes; diagnose the
exit code against journal/cgroup/GPU evidence. Retry under a fresh job/output name
only after correcting the cause. Memory failures require lower measured peak or
less concurrency within the existing caps. Protocol/scientific gate failures
require a separately frozen correction and independent review, not a looser
tolerance or the same unchanged rerun. A changed training recipe is new evidence,
not a comparable resume of the failed recipe. EMA-only exports cannot restore
lost raw optimizer/hidden/RNG training state.

At20:26 Kyiv the watcher was healthy, barrier=false, monitoring_degraded=[];
historical aggregate OOM count11 was unchanged, and no GPU/training jobs remained.
Queue and fold23 score units ended successfully. Original capture v1 and v2
reviews retain their failures; v3 is being prepared separately. Numerical archive
fixture retry corrected its missing staging directory and verified the unchanged
payload, with both failure and successful retry logs retained.

Effective limits:38,654,705,664bytes aggregateRAM,12CPU,Swap0; one GPU job8CPU/
24GiB RAM,4.8decimalGB allocator and5.5decimalGB observed device cap; four CPU
slots each1CPU/2GiB/900s. All launchers use admission wrappers. No blind automatic
training restart occurs; the watcher preserves evidence and the controller makes
the diagnosed retry decision throughout the active research goal until23:00 Kyiv.

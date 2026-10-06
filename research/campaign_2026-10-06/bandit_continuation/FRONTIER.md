# Live frontier

Updated 2026-10-06T21:15:36.412501+00:00. Contract: [CONTRACT.md](CONTRACT.md).

- Own public best remains **0.6660**; Maxim's reported **0.6674** is the source reference. No new submission or scored candidate.
- E501 resource profile completed: exit0, no OOM, peak job RAM4.52GB, aggregate5.70GB, observed VRAM2.16GB. Two post-warmup batches give stage timings but no valid whole-step throughput estimate. The first admission failure and controller fix are preserved.
- E502 C0 seed3 completed both fixed epochs and wrote ep1; child exit0, no OOM, peak job RAM8.00GB (including file cache), aggregate11.90GB, observed VRAM2.60GB. A systemd reload restored legacy saved parent limits at completion, causing the guard receipt to remain failed and the sequence to stop. Correct limits are now persisted and verified across another reload. Independent artifact acceptance is recorded separately; original receipts remain unchanged. [Incident](local/E502_RESOURCE_RELOAD_INCIDENT.json).
- Seed4 is training. The new frozen v2 plan runs through `wnn-bandit-sequence-c0-exports-20261006-v3.service`; six prediction exports follow. The first recovery runner failed its environment-marker check before any scientific attempt existed; its receipt is preserved. The active runner includes the required WNN_SLICE environment. [Independent acceptance](local/E502_C0_S3_GUARD_INCIDENT_ATTESTATION.json).
- Family-distillation code passed independent criticism and 13 focused tests after correcting a random-generator confound that changed batch order when the extra head was enabled. [Implementation review](DISTILL_IMPLEMENTATION_REVIEW.md). C1–C4 have not been fitted.
- Private Kaggle GPU v3 is running; TPU v1 remains queued. GPU v1 failed on a missing sibling source file; v2 failed on a hard-coded checkpoint mount path. Both terminal logs are preserved. V3 inventories mounts before framework loading and distinguishes device-only random-init checks from trained-checkpoint/data verification. No scientific cloud fit has run in this continuation.
- A bounded systemd Kaggle poller now records provider status/quota every minute, including errors, without resubmitting. Production raw-baseline teacher packet preparation continues with its own actual-trainer profile gate and separate cloud headroom.
- Fifteen independent literature proposals are registered as H181–H195: DeepSeek V4.1 Flash and Solar Pro4, from two random library sources and directed relational distillation. Net2Net remains pending after Laguna returned no usable final answer and then a provider capacity error. OpenCode's403 and the interrupted GLM attempt remain evidence of access failures, not scientific failures.
- [EDA notes and saved plots](EDA_NOTES.md) describe historical teacher disagreement on exposed data. Disagreement was not monotonically related to error; it is not a calibrated hardness signal.

Next: complete the paired C0 and fixed evaluation; train the two clean raw baseline teachers and export missing caches; then admit matched C1–C4 in small blocks. Unknown public gains stay unknown until prospective public feedback.

Tree generated files are point-in-time catalogs. Structure/raw-row preservation is checked separately from freshness while live logs change. Original records remain preserved.

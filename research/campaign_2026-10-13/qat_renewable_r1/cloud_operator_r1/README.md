# Kaggle cloud delivery monitor handoff R1

This handoff follows two root-owned, already-pushed private kernels using bounded read-only Kaggle status checks and typed quota reads. It does not dispatch, retry, cancel, or create provider versions. The root admission receipts remain authoritative.

At the final check in this operator window on 2026-10-13 18:17 UTC, `wnn-qat-f0-s4-profile-r1-runner` version1 remained `RUNNING` and the TPU V8 kernel remained `QUEUED`. Both are nonterminal states. Profile status persisted for about33 minutes from its push timestamp, but this alone does not establish when Kaggle started its execution-time clock. The status endpoint exposes no hardware assignment or worker phase. No outputs/logs were collected because neither kernel was terminal; no profile acceptance, TPU execution, scientific result, or failure is inferred. Root owns any further read-only polling and terminal collection.

The typed quota snapshot at 17:48:04 UTC reported GPU 73,773.641 seconds used, zero reserved, 108,000 allowed, 34,226.359 remaining; TPU zero used/reserved, 72,000 allowed and remaining. Quota refresh is 2026-10-17 00:00 UTC. Recheck before any new admission; this monitor task has no authority to admit the fit kernel.

Root has already configured both kernel IDs in the shared provider target registry. Direct checks queried only these two targets to avoid the legacy registry's 57-target burst and its historical429 responses. The append-only [provider observation log](PROVIDER_OBSERVATIONS_R1.jsonl) preserves the exact checked state and source. [Handoff snapshot](HANDOFF_R1.json) binds admission paths, latest states, quota values, timeouts and collection limits.

The approved output parents resolve to physical `/dev/nvme0n1p7` Btrfs on `/home` with 61.9 GB available at 17:55 UTC; neither attempt subdirectory existed at inspection time. Create the relevant attempt directory only after that kernel becomes terminal. After terminal status, download each kernel's output and logs into the designated MAIN paths:

- Profile: `research/campaign_2026-10-13/qat_renewable_r1/kaggle_f0_s4_r1/collected/profile_attempt_001`, at most100,000,000 bytes.
- TPU V8: `research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/collected/attempt_001`, at most100,000,000 bytes.

Preserve terminal output inventory, `kaggle kernels logs` response, byte count and exact artifact hashes. For the seed4 profile, accept results for root review only when all three full-shape profiles have valid throughput evidence, exactly100 completed finite FP32 optimizer updates per condition, and complete `GradStats`; the profile command now uses2 warmup plus3 measured full-sequence batches (40 warmup and60 timed updates). The fit bundle remains unpushed. For TPU, follow V8's manifest and verify actual-data update, state reload/export, full-sequence prediction and cleanup receipts before reporting a successful bridge. If TPU remains queued when this operator window closes, preserve that as delayed provider scheduling and leave the admitted kernel running.

## Navigation

- Root seed4 profile admission, push and asset-ready receipts are at `research/campaign_2026-10-13/qat_renewable_r1/KAGGLE_S4_PROFILE_{ADMISSION,PUSH,READY}_R1.json` in MAIN. Root TPU V8 admission and push receipts are at `research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/{ADMISSION,PUSH}_R1.json` in MAIN; the V8 packet is `research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/README.md`.
- [Observation log](PROVIDER_OBSERVATIONS_R1.jsonl) and [latest handoff snapshot](HANDOFF_R1.json).

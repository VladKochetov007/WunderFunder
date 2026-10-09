# E529 live cloud status and runtime evidence

Observed **2026-10-07T22:25:06+00:00** through read-only Kaggle quota/status calls (CLI 2.2.4). No provider mutation occurred during the poll.

GPU quota: 50575.862/108000 used, 0 reserved; the poller ledger reports 43251.975 campaign seconds remaining. TPU quota: 293.075/72000 used, 0 reserved; the campaign ledger reports 7041.548 seconds remaining. These are raw provider and ledger values; no new allocation is inferred.

H232 V1 is terminal `ERROR`, with logs and outputs collected in [terminal receipt](h232_profile_terminal_poll_receipt.json). Its 442.820-second profile attempted 230 updates, completed 229, and recorded one AMP skip; all losses were finite, but throughput is invalid for admission. The skip location cannot be recovered from V1. It took 363.459 seconds to hash the raw parquet inputs.

The V2 profile-only packet is assembled at [the packet directory](h232_cloud_profile_v2/packet), pending independent root review and fresh quota admission. It lowers the explicit FP16 GradScaler initial scale to 4,096, adds a per-update trace, and reuses V1 raw digests only when canonical path, full stat tuple, and read-only mount all match; otherwise it does the original full hash. Expected savings and total wall time are unmeasured. The 2,400-second cap is an operational limit, not an ETA. No fit has been authorized.

E520 V12 is terminal `COMPLETE`, profile-only, at 977.101 seconds. Its measured p50 step interval was 3,399.341 ms, p50 loader wait 3,247.042 ms, and reported profile rate 4.707 sequences/s. E463 V4 failed before TPU compute after 24.387 seconds because its helper import was unavailable. E463 V5 remains `QUEUED` after 3,300 seconds at this poll; provider TPU reservation is zero and its 1,650-second worker cap begins only after an actual start. Queue/start and terminal ETA remain unknown.

Next gates: independently review V2 and refresh GPU quota before one push; collect and inspect its trace at terminal; do not infer fitting from RUNNING. Keep V5 as the only pending TPU request until terminal outputs arrive. H232 fitting still requires a frozen matched B8/B16 comparison and root admission.

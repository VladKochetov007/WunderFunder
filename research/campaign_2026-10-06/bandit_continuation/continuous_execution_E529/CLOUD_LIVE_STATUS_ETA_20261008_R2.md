# E529 live cloud status and runtime evidence

Observed **2026-10-07T22:30:03+00:00** through read-only Kaggle quota/status calls. The refresh confirms the prior state: H232 V1 `ERROR` with terminal outputs collected; E520 V12 `COMPLETE` with terminal outputs collected; TPU V4 `ERROR` with terminal outputs collected; TPU V5 still `QUEUED`. No reservation is active. V5 has now been queued for 3597s, with no provider start/terminal ETA available.

The account quota snapshot remains unchanged from R35: GPU 50575.862/108000 used, 0 reserved; TPU 293.075/72000 used, 0 reserved. The campaign ledger reports GPU headroom 43251.975s and TPU 7041.548s.

H232 V2 remains unsubmitted pending independent root review and a fresh admission poll. Its profile is bounded at 2,400 worker seconds; actual queue/start time remains unknown. A valid V2 profile does not authorize a fit; the matched B8/B16 comparison and root fit admission remain required. Keep TPU V5 as the only pending request until terminal evidence is downloaded and reconciled.

The preceding runtime measurements and exact V1 AMP failure diagnosis are retained in [R1](CLOUD_LIVE_STATUS_ETA_20261008_R1.md) and the linked terminal receipts.

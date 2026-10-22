---
kind: incident
status: complete
created_utc: 2026-10-18T08:30:52Z
updated_utc: 2026-10-18T08:37:51Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - ../../campaign_2026-10-18/online_features_r1/runtime/STOP_AUTHORITY_20261018_R1.json
  - ../../campaign_2026-10-06/bandit_continuation/runtime/STOP_ADMISSION.json
---

# Research stop and graceful drain — 2026-10-18

## Request and scope

At 08:27:06 UTC the user stopped research admission, prohibited new jobs, requested detailed notes for completed work, and asked that current work finish gracefully. This supersedes the earlier overnight-continuation authorization. The stop closes admission; it is not an instruction to kill a healthy worker.

## Local drain

The durable controller barrier `15e691414aa14bb4a3d888b841d85ae3` closed at 08:27:11 UTC. The controller reconciled 537 attempt records and reported a completed drain. No local research worker remained active and no healthy process was terminated. A local process/GPU inspection also found no active model-fit process.

## External run left to finish

The last available Beleriand heartbeat, recorded at 08:20:52 UTC, showed one already-admitted F1 / seed-1001 wide run at epoch 0, batch 940/1493. Its last reported peaks were approximately 2.40 GB host memory and 1.77 GB VRAM, with no OOM/max counters. The terminal state was not known when this note was written. Leave that run alone; collect its terminal checkpoint, logs and receipts after it exits. Do not enqueue a retry or successor.

## Submission correction

The user says the two geometry packages already have been submitted. At the time this stop note was first written, the local receipt confirmed R2 portal ID `14s6l2z7` at 0.66428; the R3 portal ID had not yet been supplied. A later user-pasted portal update reports R3 ID `1wmshmaz` at **0.66456**, duration **30m32s**, memory **627 MB**. The exact R3 package identity is bound in the [public result receipt](../../campaign_2026-10-18/geometry_variations_r3/PUBLIC_RESULT.json). Both scores are below the reported 0.66832 team best. These are historical submissions, not new candidates to offer again.

## Resumption condition

No further research, fitting, scoring, packaging, or PR publication is authorized by this stop record. Only passive completion and terminal collection for work already in flight are allowed. A later explicit user instruction is required to reopen admission.

## Source records

- [Machine-readable stop authority](../../campaign_2026-10-18/online_features_r1/runtime/STOP_AUTHORITY_20261018_R1.json) binds the timestamp, barrier, last known remote state and no-successor rule.
- [Controller stop barrier](../../campaign_2026-10-06/bandit_continuation/runtime/STOP_ADMISSION.json) is the durable local admission record.
- [Online-feature closeout](../../campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md) consolidates the evidence and current candidate interpretation.

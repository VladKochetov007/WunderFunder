# Continuation after access recovery, 18 October 2026, 14:40 UTC

This supersedes the access failure and pending diagnostic state in the
[14:00 handoff](CONTINUATION_20261018T1400Z.md). Root owns continued execution.
B is `/home/grinchenkoeg/computer-vision-homework-20261018` on Beleriand.

## Existing F1 remains active

[SSH recovery](ACCESS_RECOVERY_20261018T1426Z.json) and the
[raw 14:26 health receipt](../priority25_execution_r4/receipts/F1_DEEP_PROGRESS_20261018T1426Z.json)
confirm the original recovery controller at index 3, deep continuation
`computer-vision-homework-r01-deep-f1s1001-ep25-v1`, actual epoch 5 batch 460/1493,
fresh heartbeats and zero new memory events. A [follow-up at 14:40:50 UTC](../priority25_execution_r4/receipts/F1_DEEP_PROGRESS_20261018T1441Z.json) observes epoch 5 batch 1140/1493, fresh heartbeat and all memory-event deltas still zero. The recorded epoch 4 producer
metadata has 74,650 completed updates / 7,465 batches / zero skips; it was not
newly archived or tensor-verified. No restart or duplicate worker was needed.
The historical SSH outage did not establish a training failure or its cause.

Preserve the one campaign GPU worker, shared foreign processes and aggregate
2 CPU / 18 GB high / 20 GB max / zero swap. Parent cumulative memory max 306
predates F1; watch new deltas. The existing frozen sequence still owns both F1
arms, their exports and grade. Deep terminal and F1 quality remain pending.
Use the exact reviewed V2 deep terminal collector/verifier and both V2 stage1
archives after finished plus explicit exit status 0; do not repeat wide terminal
or either stage1 verification. See [current runbook](../priority25_execution_r4/RUNBOOK.md).
The reviewed F1 result collector awaits the actual deep terminal seal and
fresh resource admission; do not fabricate the arriving receipt SHA.

## Completed work: do not repeat

- [Residual outputs](diagnostics/results/README.md), [execution](diagnostics/EXECUTION_V1.json)
  and [independent interpretation review](diagnostics/RESULT_INTERPRETATION_REVIEW_V1.json):
  all 808,863 existing compact F0 rows processed without replay, training,
  parquet scan or blend/calibrator search. Exit 0 in 6.274616 seconds under
  50% CPU / 1.5 GB / 300 seconds, CUDA hidden. Raw output hashes and additive
  metric identities pass. Its own exited unit was captured and stopped.
- [Historical plot](plots/README.md): PNG/SVG/raw receipt captured, size/SHA
  verified, PNG visually inspected. Exit 0 in 9.868649 seconds; own exited
  unit stopped. Twelve old aggregates, not a new experiment.
- [Propensity recovery](propensity_recovery_r1/RESULT.md): two exact files,
  187,249 bytes, restored to `B/quality-gap-r1/restored-propensity-v1`.
  Exit 0 in 2.831190 seconds under 25% CPU / 192 MB / 120 seconds. The model and
  provenance match the archive full SHA; no tensors loaded. Its own exited
  unit was captured and stopped. The earlier ten-file student/parent/teacher
  restoration also remains complete.

[Measured findings](FOLLOWUP_RESULTS.md): FP32 75/25 mix gains 0.00027519325
versus wide; Q8 costs 0.00002458657 locally. Contributions cancel across
components: 74 positive and 67 negative. Component 2799 contributes about 75%
of the small net gain but only 9.66% of gross absolute changes; it does not
dominate total movement. High-magnitude clipped-label rows carry much of the
metric weight/covariance, a retrospective diagnostic rather than a serving
feature. No strong-control predictions are in this analysis, so its relative
error advantage and the public regression cause remain unmeasured.

## Next work and resource order

The isolated strong-control compatibility preparation lives in
[export_preparation](export_preparation/README.md). Keep the old split bytes,
checkpoint provenance and historical sources frozen. The root restored both
propensity files with model SHA b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7
and provenance SHA f6d6cf68db53c728556c1750e869e65d5d4813d622399aec768809a47985e812.
The exact old chain cache remains absent. Canonical cache a112f809... is the
only proposed compatibility target; the unrelated existing cache 315b84f3...
is not acceptable. The isolated overlay is sealed at manifest SHA
7328138a7d82e9a23e3bebe4a1e61c979a26e8e2b119a1605cac820c1dd6ee77
and has seven passing small CPU tests; independent review is in progress.
Check that review before any staging/admission; preparation is not an executed
loader or export.

Do not admit the 37.1 GB full-source hash/loader preflight while healthy F1
fits, exports or grade occupy the lane. File cache can create child/parent
memory pressure despite bounded Python objects. Run it after those jobs release
resources with fresh admission; do not skip hashes or silently weaken checks.
Only then admit no-training strong-control export on the available single GPU,
followed by independently checked predictions on exactly matching rows.
Historical prediction byte SHA is a reproducibility reference, not a forced
cross-runtime exact-byte acceptance gate.

Keep F0 build/replay/grade complete and immutable. The new ZIP's user-reported
public score remains 0.65839 versus team best 0.66832; no promotion or public
weight tuning. DataSphere and cluster observations retain their earlier
observation times; no new paid test or cell allocation occurred here. Before
new admission, refresh provider/claims/reservations without taking over the
user's active Zen browser. Mac work remains light orchestration/documents.

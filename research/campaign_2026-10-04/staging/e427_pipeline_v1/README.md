# E427 continuation pipeline packet (source only)

This packet queues the fixed E427 aligned, stale, and no-op volume-context arms
after E464 completes. It does not authorize or start any work. The existing
production-shape smoke drafts remain the exact v1 copies in
`staging/e427_smoke_review/specs/`; this packet consumes their runner records
and writes its own durable controller snapshot.

The sequence is fail-closed: verify E464 success and the already-completed
fit-only statistics, run the three B32 × 20,000-row × two-batch CUDA smokes,
require all three smoke reports to pass, then run three identical one-epoch
continuations serially. Each fit has a 3,600-second immutable timeout; the
controller reserves that full timeout before each new run and stops when the
24-hour campaign cap would be exceeded. The requested GPU allocator/stop
limits remain 4.2/4.8 GB, with 8 CPUs and 24 GiB host RAM. There are no blind
retries or automatic model/epoch choices.

Each successful fit is audited against the exact epoch-3 public combo parent,
the fixed E427 context-statistics artifact, common normalization, fit/evaluation
membership, fixed epoch-0 policy, one-epoch optimizer/EMA/projection counters,
and runner artifact hashes. `--no-eval-during-fit` is explicit: the configured
holdout IDs are not loaded as features or targets and no fit-time holdout metric
is produced. Only then are all three fixed epoch-0 checkpoints
exported in FP32 over the exact ordered VALID domain `[0, ..., 1872]`, shape
`[1873, 20000, 2]`. Scoring starts only after all three exports pass their
checkpoint, provenance, sidecar, dtype, shape, and content-hash checks.

The scoring report freezes three panels before any fit or score: fold0's 468
groups (374 overlap E427 fit VALID and 94 do not), the 368 groups excluded by
the E427 statistics preparation (none overlap the continuation fit), and the
94-group fold0 nonfit diagnostic subset. It reports only aligned-minus-stale
and aligned-minus-noop component-bootstrap contrasts (2,000 resamples, seed
271828), plus per-arm global WP and class summaries. All panels are historically
exposed. The 368 groups were excluded from the continuation and parent final
fit, but prior teacher lineage and campaign exposure preclude an untouched
holdout claim. The 94-group subset is descriptive and uncertain.

Root review and admission are required for every sealed execution spec and the
controller service. No training, export, score, provider operation, or sealed
spec is part of this source-only packet.

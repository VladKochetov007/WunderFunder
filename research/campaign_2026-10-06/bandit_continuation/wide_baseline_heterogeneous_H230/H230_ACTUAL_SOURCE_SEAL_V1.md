# H230 sealed source artifact handoff

The source-seal V3 worker completed on 2026-10-07 at 15:40:47 UTC. Its output
`H230_SOURCE_HASH_BUNDLE_V2.json` has SHA-256
`c9f4235994e9fc85f0f732fd74852e1c28b1d5c7efc8a1ebefbc71f6dc673340`. The
durable progress receipt `H230_SOURCE_SEAL_PROGRESS_V4.json` has SHA-256
`b21ac766f66dfe662b0f9315462257bb6b850dc50fac500c7d95c4070f33c411` and says
`finished`, with eight completed teachers and sixteen verified prediction
exports.

The worker used the pinned worktree cache-builder checks, including checkpoint
and provenance binding, learned ancestry, fold-safe fitting membership,
ordered target groups, sidecar identity, full prediction hashes, expected
`(row_groups, 20000, 2)` shape, float32 dtype, and finite values. It did not
materialize predictions, build the H230 cache, fit, or score a student. The
current E521 train-export receipt hash, E521 V6 artifact report/spec hashes,
all 32 pinned worktree source hashes, split hash, target-row-group hashes, and
all eight source records are captured in the V2 bundle.

Two snapshot details are explicit in the bundle. The `ft_since_s4` source is
labeled `seed=4` by its run/source identity but its training provenance and
checkpoint arguments record seed 3; the output reports both without changing
the source label. The V1 bundle's raw seed-4 provenance hash was stale because
the snapshot predates the completed fit; V3 reconciled it only to the exact
current hash in the completed E521 V6 artifact map. No score from that E521
report was used to validate or select H230 sources.

Resource observations: the scoped cgroup peak was 1,499,996,160 bytes under
the 1,500,000,000-byte hard memory cap; final current usage was 1,475,993,600
bytes. The process `ru_maxrss` reading converted by the worker was
2,585,739,264 bytes. These two observations differ; the handoff preserves both
without attributing a cause. Independent resource review should account for
the file-backed memory maps before cache admission.

This is not cache admission. The cache stage still requires independent review
of the exact source bundle/map, a fresh live data/fold identity receipt, fresh
resource and disk admission with the 40,216,345,600-byte free-space gate, and
rehashing the 32 pinned worktree files before execution. The source seal does
not compute a fresh full SHA-256 of the 35 GB raw parquet files.

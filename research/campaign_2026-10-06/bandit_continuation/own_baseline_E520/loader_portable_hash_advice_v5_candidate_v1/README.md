# E520 profile v5 loader-hash diagnostics candidate

This isolated profile-only candidate diagnoses the cgroup file-cache change
around E520's two raw-parquet verification phases and advises the kernel to
drop pages after the extra loader verification pass. It preserves the v4
packet, all prior attempts and receipts, the pinned trainer archive, split and
portable-registry identities, fold membership, profile recipe, grader, and
resource thresholds.

The v4 worker already hashes `train.parquet` and `valid.parquet` sequentially
with page-aligned `POSIX_FADV_DONTNEED` advice. A later source audit found that
the frozen fold-0 raw identities have no `sha256` key, so
`load_campaign_split()` does not take its `wnn.splits.file_sha256` branch for
those files. The one extra full pass is
`verify_portable_source()` → `wnn.portability._content_digest`. V5 wraps only
that pinned symbol and only for the exact staged train and valid parquet
paths. It keeps the verifier's original stat and byte-count checks, registry
identity comparison, complete SHA-256 comparison, and 64-entry cache
semantics. Non-parquet sources use the pinned original function unchanged.
Advice failure aborts with a durable record; it never falls back to a plain
hash.

`memory_stat_phases.json` records bounded cgroup-v2 `memory.stat` snapshots
before and after each initial raw hash, each loader portable-content hash,
and the complete `load_campaign_split()` call. The snapshots include the most
constrained visible cgroup path, current/limit/available bytes, and parsed
memory.stat counters. If memory.stat is unavailable, the receipt says so;
the existing continuous 20% and 4 GiB resource gates still apply. V4 did not
capture memory.stat, so page-cache pressure remains a hypothesis rather than
a diagnosed root cause.

Headroom admission is fail-closed if either host or cgroup available bytes are
below 4 GiB **or** their available fraction is below 20%. `resource_abort.json`
preserves the observed resource snapshot and names each failed predicate, so a
single-threshold failure is distinguishable from a joint failure.

The candidate retains the exact three warmup plus 20 timed outer-batch profile,
4,000-row real-input exporter shape probe, input and fold checks, no-fit/no-
prediction-export behavior, one-GPU runtime identity, 8 GB RSS cap, 1,800 s
worker cap, and existing CUDA/disk/headroom guards. It adds no training or
scoring path.

## Focused checks

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/pytest -q \
  research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/loader_portable_hash_advice_v5_candidate_v1/tests
```

The synthetic hash tests use small temporary files and prove that both raw
identities take the advised loader path, digests and registry identity checks
remain unchanged, advice ranges stay page-aligned, cgroup memory snapshots are
written, and advice failure aborts without an unadvised fallback. No parquet,
model arrays, or cloud resources are read by these tests.

## Review target only

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/loader_portable_hash_advice_v5_candidate_v1/prepare_profile_packet.py \
  --review-targets-only
```

The builder's `--build` path was not run. This directory contains a code
candidate and review hash map only. Independent review and fresh controller
admission are required before an operator may build a packet or retry the
profile. No provider calls, fits, or scoring were performed here.

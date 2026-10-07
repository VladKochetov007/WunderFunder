# E-522 verified-digest handoff

E-522 targets one measured E-520 profile cost: after each raw parquet is fully
hashed and checked, `load_campaign_split()` invokes
`verify_portable_source()` → `wnn.portability._content_digest` and reads that
same whole file again. The new helper is worker-local and memory-only. It does
not read, stage, or persist a sidecar digest across attempts.

## E-520 v6 evidence

The collected v6 profile receipts show:

- Initial train SHA-256: 29,240,219,011 bytes in 795.473306 s.
- Initial valid SHA-256: 5,615,509,128 bytes in 188.747190 s.
- Portable verifier repeated train SHA-256: 29,240,219,011 bytes in 748.676424 s.
- Portable verifier valid hashing started for 4.448242 s and the worker hit its
  1,800 s wall cap. The progress record had not reached its 256 MiB receipt
  interval, so the exact partial valid bytes are unknown; they are not zero.
- At abort, cgroup available memory was 29,812,908,032 / 32,212,254,720 bytes
  (92.55%), host available memory was 32,573,460,480 bytes (96.78%), RSS was
  about 73 MB, and CUDA was explicitly idle. The snapshots do not support a
  memory-pressure explanation. No profile batches or fit ran.

The raw inputs total 34,855,728,139 bytes. The v6 run confirms one complete
redundant 29.24 GB train pass and the start of the valid repeat. If split
loading completes, the duplicated pair pass is 34,855,728,139 bytes by source
size. E-520 fit code also hashes both inputs, then verifies them again in its
first `load_campaign_split()` call. It calls both seeds' trainer `main()` in
the same Python process; the pinned portability LRU then serves those later
unchanged-key calls without another raw read. The improvement therefore removes
the initial duplicate pair pass per profile/fit worker, not one pass per seed.
The recovered v6 diagnosis confirms exact input identity match, zero fadvise
failures, and no CUDA/GPU profile reached. It does not include device inventory
or GPU utilization, so those remain unknown. The v6 wall cap (1,800 s) expired
while repeating the valid digest; it is not evidence of RAM pressure or a
completed profile.

The retained v6 evidence is bound in `E522_PLAN.json`: memory-stat receipt
SHA-256 `da06cceaa903fa5f9556c7dbb61c1b88d4e16cd21a449c9e331778e9dd2a9e91`,
hash advice `1d275f3deb0a1899b7d4dfbfb398f77d798e8f9201cee39ecbfabe51eb7466fd`,
input identity `3481cfeaa53d9b6a981a1aefc7d87456cb24d040513c9f063b12ecfc1680ed01`,
resource abort `7c71a356a6ce897b6fc120bbb603b41b6c01ef80636d8d6e2435cb48a24cc109`,
and worker log `be960e6d6af450fab994a5e5a9e70d04ef39927e6525c0b167674e7ad3c48f44`.

## Handoff contract

`VerifiedDigestHandoff` has an exact path allowlist. It streams a candidate SHA
once while comparing descriptor and path identities before and after the read.
The caller first records the observed digest in the existing durable identity
receipt, then authorizes it only when it equals both the frozen expected digest
and the digest from the verified portable registry. The digest is held only in
this worker's memory.

Before the first `load_campaign_split()` call, install the wrapper on the
already imported pinned `wnn.portability` module. A hit requires the same
canonical path and current `(device, inode, size, mtime_ns, ctime_ns, mode)`
identity. The original verifier still checks its frozen source identity and
compares the returned digest with the portable registry. A miss calls the
original `_content_digest` on the current bytes; it never reuses a stale
digest. Other paths always call the original verifier. `wnn.splits.file_sha256`
is untouched.

The profile worker already uses a runtime shim for `_content_digest`; E-522
would replace the pagecache-only raw-file rehash route with the handoff and
retain its fadvise callback, memory receipts, resource guards, and strict
failure behavior. The fit worker uses the same helper before its portable
split check. Neither current packet nor pinned trainer/archive was modified.

Minimal integration sketch (not applied to any E-520 file):

```python
handoff = VerifiedDigestHandoff((raw["train"], raw["valid"]))
receipts = {}
for split_name, path in raw.items():
    source_name = f"datasets/{split_name}.parquet"
    candidate = handoff.hash_file(path, on_progress=existing_pagecache_adviser)
    observed = {"bytes": candidate.bytes_hashed, "sha256": candidate.sha256}
    persist_identity_check(receipt_path, identity, source_name, EXPECTED[split_name], observed)
    handoff.authorize(
        candidate,
        expected_sha256=EXPECTED[split_name]["sha256"],
        registry_sha256=portable["sources"][source_name]["sha256"],
    )
    receipts[split_name] = candidate

import wnn.portability as portability
handoff.install_portable_verifier(portability)
split = load_campaign_split(split_path, source_root)
```

The real integration must retain footer checks and the current persist-before-
mismatch receipt order. If a path changes after the first hash, authorization
or lookup misses and the pinned verifier hashes current bytes; a changed digest
then fails its registry comparison.

## Frozen recipe boundary

See [FROZEN_RECIPE_DIFFERENCE.md](FROZEN_RECIPE_DIFFERENCE.md). E-522 changes
only duplicate content hashing within a single verified worker. It does not
change the raw-only GRU recipe, fold, target treatment, optimizer, sampler,
checkpoint, output schema, grader, or endpoint.

## Local synthetic checks

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python -m pytest -q \
  research/campaign_2026-10-06/bandit_continuation/throughput_optimization_E522/tests
```

The measured microbenchmark in `E522_SMALL_FILE_IO_MEASUREMENT.json` hashes an
8 MiB synthetic file twice for the baseline and once before two verifier
lookups for the handoff path. It reports logical bytes read and local elapsed
time only. It read 16,777,216 baseline bytes versus 8,388,608 handoff bytes,
with zero verifier fallback calls; local wall time was 0.0122 s versus 0.0065
s. This is not a parquet or Kaggle timing forecast.

Reproduce the small bounded measurement with:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python research/campaign_2026-10-06/bandit_continuation/throughput_optimization_E522/measure_io.py
```

E-522 is code-only. No trainer/source archive, profile/fit packet, running job,
provider state, GPU workload, fit, or grader was modified or executed.

# E520 profile v7: verified digest handoff

This isolated candidate leaves E520 v6 and the frozen source archive unchanged.
It reuses the initial raw train/valid SHA-256 during `load_campaign_split()` only
when the initial complete read matches both fixed expected hashes and portable
registry hashes and the exact canonical path plus `(device, inode, size,
mtime_ns, ctime_ns, mode)` remains unchanged. A miss delegates to the pinned
portable verifier's complete hash. The portable registry comparison and all
E520 split, target, footer, mount and resource checks remain active.

The one integration change is `profile_worker.py`: it hashes raw inputs through
the shared E522 helper while preserving page-aligned `POSIX_FADV_DONTNEED`
advice and memory-stat receipts, authorizes each verified digest before split
loading, installs the handoff on only the pinned
`wnn.portability._content_digest`, and records per-raw-path lookup/hit/miss/
fallback counts. The bundled helper is the v7 copy; the independently reviewed
E522 core helper remains frozen. No digest persists outside this worker.

The v6 run measured one repeat train read of 29,240,219,011 bytes in 748.676424
seconds. Its valid repeat began for 4.448242 seconds but exact partial bytes are
unknown. The full potential duplicate train/valid pair is 34,855,728,139 bytes
by known source sizes. Initial train and valid hashes took 795.473306 and
188.747190 seconds. Thus, approximately 984.22 seconds of the worker's 1,700
second cap may still be used by the initial identity pass alone. The remainder
must cover portable verification, fold checks, environment/device collection
and the actual 20-batch profile; no completion or runtime forecast is claimed.

The worker retains one-GPU serial execution, no fit, no checkpoint, no
prediction export, and the existing RSS/VRAM/headroom/disk caps. Its new 1,700
second internal wall cap is a distinct execution version, bound in this plan,
worker, builder and generated packet metadata. Current campaign quota,
provider runtime and live resources remain controller admission inputs and are
not asserted by this candidate.

The profile receipt also records a bounded `nvidia-smi` sample of every
physical GPU every five seconds around `trainer.main()` (maximum 24 samples,
two-second query timeout), including utilization and memory fields. The query
records all physical rows even though Torch sees one selected GPU. Missing
`nvidia-smi`, malformed output, or `N/A` utilization is stored as unavailable;
the worker never fills unknown values with zero.

Run the bounded local checks with one CPU and a 1 GB memory limit:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python -m pytest -q \
research/campaign_2026-10-06/bandit_continuation/throughput_optimization_E522/e520_profile_v7_candidate_v1/tests
```

For review targets only (no packet build or provider call):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s scripts/limited \
.venv/bin/python \
research/campaign_2026-10-06/bandit_continuation/throughput_optimization_E522/e520_profile_v7_candidate_v1/prepare_profile_packet.py \
--review-targets-only --repo-root .
```

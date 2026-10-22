# Online relationship cache and delivery review (2026-10-22)

This bounded source review covers delivery commit [`6d63b179ed89371b0332270fb6855891b2963c6e`](https://github.com/VladKochetov007/WunderFunder/commit/6d63b179ed89371b0332270fb6855891b2963c6e) in the isolated `agent/online-data-cache-20261022` worktree. It traces cache construction, cache validation, panel handoff and export across `online_relationship_data.py`, `online_relationship_data_writer.py`, `online_relationship_sources.py`, `online_relationship_cache.py`, `online_relationship_cache_validation.py`, `online_relationship_delivery.py`, `online_relationship_exports.py` and `scripts/online_relationships.py`.

No open source blocker remains in the reviewed contract. The builder reads evaluation row groups with `targets=False, mask=False`, leaves those columns zero, and the cache iterator rejects nonzero evaluation target or weight columns. The fitter recomputes exact source-qualified fit/evaluation order and checks the pinned split/subset hashes, sidecar identities, parent checkpoint/provenance, and the consumed FP32 parent mean/std bytes. NPY dimensions and dtype are checked against the headers and payload digests; the separate parent prediction payload is bound to the evaluation cache sidecar and row order. Profile caches live in a separate directory and carry sampled/incomplete markers; the full-panel loader requires complete exact memberships.

The earlier review snapshot exposed missing split/subset and parent-normalization bindings. Delivery closed those checks in the reviewed commit. This was source review only: no Parquet data, cache payloads, labels, tests, fits, scores or runtime profiles were read or produced by this reviewer. Delivery reports Ruff and training-structure checks passed; I did not rerun them. The recipe at this snapshot remains `deferred_entire_panel_after_feature_cost_profile` and lacks the root-approved row-weight amendment, so cache preparation and fitting correctly refuse it until that amendment is written. No panel execution or scientific approval follows from this review.

The bounded NumPy consumer tests in [`tests/test_online_relationship_cache.py`](../../../tests/test_online_relationship_cache.py) exercise delivery commit `6d63b179e` through its actual `load_sequence_cache` implementation. All 5 tests passed: one valid mmap/four-view/source-ref check, bad-magic and truncated-header rejection with matching sidecar digests, and rejection of nonzero evaluation targets or weights. The run read no Parquet data and imported no Torch. Its exact command was:

```sh
TMPDIR=/home/vlad/development/wunder/research_tmp \
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/vlad/development/wunder/research_tmp/online-data-cache-20261022/src:/home/vlad/development/wunder/research_tmp/online-residual-fit-component-20261022/src \
/usr/bin/time -v python -m pytest -p no:cacheprovider \
  --basetemp=/home/vlad/development/wunder/research_tmp/online-cache-test-slot-20261022-r1 \
  -q tests/test_online_relationship_cache.py
```

This first command was run directly, without `scripts/limited` or a transient systemd scope, so the assigned 25% CPU/300 MB job limits did not apply. The process inherited `app-com.microsoft.VSCode-1177949.scope`; its live systemd properties showed `CPUQuotaPerSecUSec=infinity`, `MemoryMax=infinity`, and `MemorySwapMax=infinity` (`cpu.max` was `max 100000`). Process accounting reported 0.92s wall, 86,176 KiB maximum RSS, 0 swaps, and 148% CPU utilization. No before/after cgroup CPU counters were captured, so per-run cgroup CPU usage cannot be reconstructed. This initial run is not verified under the assigned limits. A corrected bounded rerun follows.

The corrected run passed all 5 tests through the actual `scripts/limited` wrapper with the assigned `CPUQuota=25%`, `MemoryMax=300,000,000`, and zero swap. Its exact command was:

```sh
WNN_MEM_MAX=300000000 WNN_CPU_QUOTA=25% WNN_RUNTIME_MAX=120 WNN_SCOPE_UNIT=online-cache-consumer-r2.scope OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/vlad/development/wunder/wnn_connectome_starterpack/src /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited /bin/bash -c 'set +e
/usr/bin/time -v /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python -m pytest -p no:cacheprovider --basetemp=/home/vlad/development/wunder/research_tmp/online-cache-test-slot-20261022-r2 -q tests/test_online_relationship_cache.py
test_status=$?
group=$(cut -d: -f3 /proc/self/cgroup)
printf "TEST_CGROUP=%s\\n" "$group"
printf "CPU_STAT\\n"
cat "/sys/fs/cgroup$group/cpu.stat"
printf "MEMORY_PEAK="
cat "/sys/fs/cgroup$group/memory.peak"
printf "MEMORY_EVENTS\\n"
cat "/sys/fs/cgroup$group/memory.events"
exit "$test_status"'
```

Pytest reported `5 passed in 1.19s`; `/usr/bin/time` measured 1.74s wall, 26% CPU, 85,988 KiB maximum RSS, and zero swaps. The terminal scope was `/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice/online-cache-consumer-r2.scope`. Its `cpu.stat` reported `usage_usec=473244`, `nr_periods=19`, `nr_throttled=18`, and `throttled_usec=1270328`; `memory.peak` was 73,621,504 bytes, and `memory.events` reported zero max, OOM, and kill events. This run is verified under the assigned limits. The full-panel profile-marker consumer check was omitted because `online_relationship_cache_validation.py` imports Torch, as directed for this resource window.

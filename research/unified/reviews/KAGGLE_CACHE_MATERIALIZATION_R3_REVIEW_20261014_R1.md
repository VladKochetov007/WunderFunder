# Kaggle QAT cache materialization R3 review

This is an independent offline review of the R3 recovery for the R2 cache-root failure. It covers the bootstrap change, sealed R3 payload, actual-input preflight fixture, bounded copy tests and cleanup path. No Kaggle action, fit, full-cache copy, prediction or quality evaluation was performed.

## Decision

The reviewed change clears the offline execution-validity gate for one fresh root-admitted R3 profile attempt. It copies only the three manifest-pinned cache files after the existing actual-input preflight and cache handoff pass, verifies bytes while streaming, replaces the staging symlinks with contained regular files, and leaves the downstream cache-consumer checks unchanged. This is not evidence that the Kaggle filesystem has enough usable disk, that a 1.91 GB copy will finish within the deadline, or that any QAT profile or fit will complete.

## Independent checks

- The R3 bundle archive SHA-256 is `6cfc0b87762e9ad91be1c9528ee767cb5c021364f62620d8c5b63e04cf383a6a`. Its archived `scripts/kaggle_qat_kernel_r1.py` has SHA-256 `97c6a5468cc6d20b7768b63328f89592b560b186cdb3c81ef928721da0405d78`, matching the current reviewed template.
- The external kernel bootstrap at `/home/vlad/development/wunder/qk14r3b/profile_upload/kernel/kernel.py` has SHA-256 `5222029435057c0241afd8069550acee9651571cb25d4eef642d8099327f3d3f`. Its bytes equal the current template with exactly the four packet-bound substitutions: bundle slug, archive SHA, external mount map and 1,800-second process timeout.
- I extracted the four sealed R2 job specs and compared them to the four R3 specs. After removing the derived `code_sha256` and the bootstrap descriptor's size/SHA, every other field matched, including the preflight and all three candidate recipes.
- The R3 manifest selects exactly `metadata.json`, `main_homogeneous_train.npy` and `main_homogeneous_valid.npy` under `cache/clean_t3_f0_h390_t3r5_a9_r2`. Their pinned sizes total `1,910,723,660` bytes, and the SHA values match the content-addressed external filenames. These are the three cache entries in the frozen actual-input preflight fixture.
- The materializer requires a passing preflight, exact manifest/receipt size and hash agreement, the expected three-file inventory, sufficient free space, and a filesystem other than `tmpfs` or `ramfs`. It reads and writes in at most 1 MiB blocks, checks the copied length and SHA, syncs output, advises the source/destination file ranges out of cache, and atomically replaces each in-root staging symlink. Candidate execution follows only after this helper returns successfully.
- `release_mounts` removes external links, successfully materialized regular files confined to the registered cache root, and contained `.materializing` partials. The focused tests exercise successful copy/source preservation, post-preflight source mutation, receipt disagreement, expired deadline and cleanup.
- Focused validation passed: `3 passed` for `tests/test_qat_cache_materialization.py` under the project limit wrapper with `TMPDIR=/home/vlad/development/wunder/qk14r3b` on btrfs. A default run failed before copy logic because this host maps `/tmp` to `tmpfs`; the helper correctly rejected that test location. The fixture should therefore be run with an explicit physical temporary directory on this host.

## Limits and handoff

The test fixture derives paths and producer identities from the actual preflight receipt but substitutes tiny deterministic payloads. It verifies bounded control flow and cleanup; it does not copy or rehash the actual 1.91 GB cache arrays. The R3 runtime must still demonstrate adequate physical working disk and complete its copy before the existing worker deadline and finalization reserve. The materialization receipt is written to `/kaggle/working/results/preflight/cache_materialization.json`; the top-level sequence receipt gates on it indirectly through sequential control flow but does not repeat its status or identity.

The earlier R2 preflight took `313.992` seconds to hash 34 inputs totaling `40,075,446,871` bytes, and R2 failed at cache containment before any profile batch or optimizer update. Those measurements explain why this recovery is needed; they do not predict R3 copy time or establish a QAT result. See the [R2 failure record](../../campaign_2026-10-13/qat_renewable_r1/kaggle_f0_s4_r2/README.md) and [R3 packet](../../campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r3/PACKET_R3.json).


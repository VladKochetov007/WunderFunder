# R5 Kaggle resource-handoff R7 correction review

**Verdict: PASS for the R5 worker-name/archive-closure correction.** The exact R7 archive now agrees with the frozen driver for profile, fit, and export spec names and internal worker/output identities. The previous R1 review's release blocker is resolved in these bytes. This is not root admission or actual-device profile approval: the R5 root release remains pending and unauthorized.

## Scope and sealed pins

Reviewed actor commit `c54dbb8b4a201a3fadf3e9b81375e523b1ae9398`, limited to the R7 correction, exact archive/manifest closure, preparation and conditional-release bindings, and whether the R6 resource-handoff code/contract changed.

- `bundle.wnn`: SHA-256 `1cf431474a7aed41a8df6606373c63a4d5816bdf0c77b4deff0c72dfefce2bca`, 245,774 bytes.
- `bundle-manifest.json`: SHA-256 `e46c6b2f72d354c989fc41a9a67078404f205c35308f97b37deeacaeb42b03f8`.
- `kernel.py`: SHA-256 `95b948cd45cd29088a1b1f976d842071adb8db4bfb92acce16d0b03181267320`.
- `teacher_cohort_r5_driver.py`: SHA-256 `e931157aa1c8be25180d4599fdc51cbb847a77f9c588d5e9245ba0a6255554e1`.
- Resource handoff contract SHA-256 `060bfb8b3ad38822ee32c2c148b8ce82b244ade83209b2e9efa0e344d2e6afe5`.
- R5 preseal SHA-256 `9909a92a4e408bf196abeb419aebd94b17b5bc2b97fc76832e0836aaef64e3b9`.
- R5 preparation receipt SHA-256 `d9a681764a14468cc2b1157d60a1b5654680b825f9f0e780f9fb00c90a40d84e`.

I verified all 95 archived files (1,234,237 expanded bytes) against the manifest. The manifest's archive size and SHA match the payload. Its 14 spec paths exactly equal the driver's requested set: two paired T4 profiles, three fits, and nine exports. Every embedded spec `name` equals its filename stem; both profiles' `--name` values equal the driver names; every export `--output-name` and declared prediction artifact uses the intended `_r5` output identity. All 14 spec hashes match the R5 preseal map.

The bootstrap's expected archive SHA matches the archive. Its R5 admission and root-release paths are present in the archive and their content hashes match the bootstrap constants. The preparation receipt matches the archive, manifest, bootstrap, driver, preseal, and resource-contract pins.

## Resource handoff and authority

The driver and resource contract are byte-identical to R6, so the R7 naming correction does not change the memory-handoff or admission behavior reviewed in R1. The contract keeps each fit's 8,000,000,000-byte RAM reservation. Cache advice remains restricted to the exact allowlist, after paired profiles exit, using current-attempt producer receipts and unchanged input-stat snapshots; `POSIX_FADV_DONTNEED` is described as best effort, not proof of eviction. The driver still takes fresh host/cgroup/device snapshots and refuses to start fits unless unchanged `admit_specs` returns exactly two. There is no serial fallback, reservation reduction, cap change, or input mutation.

`ROOT_CONDITIONAL_RELEASE_R5.json` remains `authorized: false`; `CONDITIONAL_FIT_ADMISSION_R5.json` remains `release_authorized: false` and pending. The preparation receipt records no provider action. R4 remains an actual `ERROR` with `ValueError: Worker reservation exceeds host operational budget`; the actual fixture says the cache cause is unproven and no fit workers started. No R5 device profile, fit, provider action, or new mounted-input read was performed for this review.

The earlier name-resolution finding is closed for the exact R7 payload. Actual Kaggle/T4 capacity and the R5 profile gate still require the separate root-owned admission and execution. See [`R5_RESOURCE_HANDOFF_INDEPENDENT_REVIEW_R1.md`](R5_RESOURCE_HANDOFF_INDEPENDENT_REVIEW_R1.md) for the preserved R6 finding and R4 producer evidence.

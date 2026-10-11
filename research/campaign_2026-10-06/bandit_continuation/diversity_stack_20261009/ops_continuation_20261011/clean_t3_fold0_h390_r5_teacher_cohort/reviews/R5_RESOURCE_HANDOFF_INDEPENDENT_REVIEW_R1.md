# R5 Kaggle resource-handoff independent review

**Verdict: BLOCKED for release.** The resource-handoff design is bounded and fail-closed at fit admission, but the sealed worker driver cannot resolve its first profile specs from the exact archived payload. Keep the root release pending and do not dispatch this archive.

## Scope and pins

This review covers actor commit `7ea91f9aef7233c35bb7d4e241202345e5f732d9`, with the R5 packet under `package_r6/`. It is static packet/producer-fixture review only: no provider calls, profile, fit, upload, or input-data rehash.

The independently rehashed package pins are:

- `bundle.wnn`: SHA-256 `8afc11ea361c4b0ab5725bfc0f109b2b181570ec14c94ba9783dab766c3ac3c2`, 245,775 bytes.
- `bundle-manifest.json`: SHA-256 `d7847c5a57448610855bce864e197781b7a3e6f3122aa3aed2dda69bd01cd5f6`.
- `teacher_cohort_r5_driver.py`: SHA-256 `e931157aa1c8be25180d4599fdc51cbb847a77f9c588d5e9245ba0a6255554e1`.
- `R5_RESOURCE_HANDOFF_CONTRACT_R1.json`: SHA-256 `060bfb8b3ad38822ee32c2c148b8ce82b244ade83209b2e9efa0e344d2e6afe5`.
- `R5_PRESEAL_BINDINGS_R1.json`: SHA-256 `0727536611b3c132410ff0f06cc697417f92c972c19f2604ca6111c425c2fb54`.
- `R5_PREPARATION_RECEIPT_R1.json`: SHA-256 `68c68546c5c9481ec5a574615d09dda65707920d854da794b99f2d181305b50f`.

I streamed all 95 regular archive members (1,234,237 expanded bytes) against the manifest. Every path, size, and SHA-256 matched, and the manifest's archive size and digest matched the actual archive.

## Release-blocking finding: driver/spec identity mismatch

The archive manifest contains 14 worker specs: profiles and exports are named `_r3`, while the three fit specs are named `_r5`. In contrast, the driver declares `_r5` for all profile, fit, and export names at lines 22–27. `ensure_pair_admission()` constructs `kaggle_specs/{name}.json` directly from the passed names (line 631), and `main()` passes those `_r5` profile names (line 727). Therefore the first `runner.read_spec()` targets a profile JSON absent from the sealed archive and raises before either new profile starts.

Renaming only the two profile JSON files would not close the mismatch. Each archived profile spec's internal `name` and `--name` argv are still `_r3`; the driver validates `_r5` summary names and reads `_r5` worker directories, logs, and input-identity receipts (lines 91–115, 119–136, and 362–370). Similarly, export spec filenames and internal names are `_r3`, while the driver expects `_r5` export names; their prediction `--output-name` values are `_r5`. The successor must make the filename, spec `name`, training/export argv, runner result paths, driver lists, and validators agree, then regenerate and rebind the archive, manifest, preseal, and release packet. Add a static closure check that every driver-requested spec and expected result name resolves in the final archive.

The current `R5_PREPARATION_RECEIPT_R1.json` correctly says `prepared_pending_independent_review_and_root_release`; `ROOT_CONDITIONAL_RELEASE_R5.json` remains unauthorized/pending. The naming defect means that positive archive hashing and root's later gate cannot make this execution path runnable.

## Resource-handoff assessment

The bounded cache-handoff policy is sound in its stated scope, subject to a corrected runnable packet and fresh runtime admission:

- `R5_RESOURCE_HANDOFF_CONTRACT_R1.json` enumerates exactly eight cache files and the two supplied Parquet files (38,101,850,961 bytes total). It retains the unchanged `kaggle_gpu_runner.admit_specs` behavior and an 8,000,000,000-byte RAM reservation for each fit.
- After both profiles exit, the driver verifies this attempt's profile receipts: exact cache SHA/size identities and exact complete Parquet hash events in the worker logs. It compares pre-profile, pre-advice, and post-advice device/inode/size/mtime/ctime metadata and confines `POSIX_FADV_DONTNEED` to the declared cache and Parquet paths. It explicitly labels the advice best-effort and does not claim that a successful call evicted bytes.
- The driver records host/cgroup/device snapshots and runs fresh admission for the unchanged two-fit specs. It refuses unless exactly two fits are admitted; there is no serial fallback, reduced reservation, cap change, `drop_caches`, or data mutation. This remains a design/code finding, not measured R5 headroom or a successful R5 profile.

The actual R4 fixture is consistent with the stated incident: terminal `ERROR`, `ValueError: Worker reservation exceeds host operational budget`, both paired profiles had completed, no fit worker started, no post-profile host/cgroup inventory was persisted, and the file-cache cause is explicitly unproven. Fixture SHA-256 is `f3770c483ee72c2a0b0fa6f751b2e4072f9d40bcd23d934ac06019230841ae1a`. The copied profile logs are byte-bound to the actual R4 `_r3` artifacts: seed 3 log `1f3d936ca0e878ab6d4cef5c6d9f09b00f5db6a4cfb8b0274b121959e4e695a7`; seed 4 log `b683629c6409ce292570d94c0a4c90aee5935e98cc6e297511521731d167fafc`. Both input-identity receipts have SHA-256 `6593ee2470dbc7e296fa72f063fbb0cc295c3ef1eb467dd87feeaeecc7452203`. Those receipts bind all eight cache identities; each actual R4 worker log contains complete train and valid Parquet SHA events matching the contract. These are prior producer receipts, not an R5 runtime result.

## Checks and limits

Checks performed: archive/manifest full member closure; direct comparison of driver constants, archive `spec_paths`, embedded profile/export spec names and argv; static review of resource handoff and fit-admission gates; comparison against the actual R4 producer fixture/logs. No R5 profile or fit was run. The 38.1 GB mounted inputs were not reread. No claim is made that DONTNEED lowers cgroup memory or that a corrected packet will pass actual admission.

The machine-readable findings and all eight cache pins are in [`R5_RESOURCE_HANDOFF_INDEPENDENT_REVIEW_R1.json`](R5_RESOURCE_HANDOFF_INDEPENDENT_REVIEW_R1.json).

# Amplitude recipe review R2

This receipt independently reviews the sealed [R2 amplitude packet](../../target_amplitude_f0_s3_r1/README.md) and the MAIN controller's operational helper-binding fix. The allocator policy, shared-state source manifest, and absolute P2/P3 resume paths check out. The actual P1 attempt nevertheless failed before training because both fit and export commands still pass the campaign split manifest as a relative path while the worker runs from the source-overlay directory. Treat this as an execution-path failure, not a scientific result; a successor must use an absolute MAIN split path and a fresh attempt ID.

- [Detailed independent review](INDEPENDENT_AMPLITUDE_RECIPE_REVIEW_R2_20261013.md) records the recipe, exact failure, and corrected launch conditions.
- [Machine-readable receipt](INDEPENDENT_AMPLITUDE_RECIPE_REVIEW_R2_20261013.json) binds packet, overlay, controller/helper, and runtime-attempt identities.

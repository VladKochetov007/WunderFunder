# E482 private T4 replication: completed fits, grading pending

The exact private v2 kernel reached provider COMPLETE at 2026-10-05T04:48:23Z. All six seed271828 baseline/sine/cubic × AdamW/Muon fits completed, each with474 updates and61,058 parameters. Root rehashed all six checkpoint bytes and the23-file downloaded inventory. The two workers used distinct physical T4 UUIDs, and each retained its probe GPU through fitting. Both native output/state/input-gradient/parameter-gradient/clipped-step parity gates passed before fitting.

All probe and fit exit codes are zero. The recorded OOM counters stayed zero. Fit process-tree RAM peaks were1,652,441,088 and1,618,296,832 bytes; each worker had a6GiB stop. Fit CUDA allocator peaks were about79MB per worker, with100,663,296 bytes reserved. These allocator figures are not physical VRAM measurements: the physical watchdog stop was3GB. Both GPUs were idle again at completion. The host had32,212,254,720 bytes capacity and25,399,783,424 bytes available at the end.

The sum of the six fit timers is1430.661 seconds (0.3974 device-hours), excluding setup and probes. Provider quota accounting is separate. The original v1 worker-CLI failure remains preserved and contributed no scientific fits.

No test predictions have yet been scored. Independent grader review found an actual admission-receipt path mismatch before its no-test gate; the candidate will be corrected and frozen before grading. All score claims await that gate and the separately admitted independent scorer. Precision flags were serialized after restoration in both the outer receipt and nested native-parity receipt; neither snapshot proves active flags. Pinned context-manager source and actual parity checks are the numerical evidence; see PRECISION_METADATA_REVIEW.md. Muon's BF16 orthogonalization is intentionally unchanged.

This repeats a recipe on different hardware using the same exposed diagnostic panel and initialization, and is not new-market confirmation or a public-score prediction.

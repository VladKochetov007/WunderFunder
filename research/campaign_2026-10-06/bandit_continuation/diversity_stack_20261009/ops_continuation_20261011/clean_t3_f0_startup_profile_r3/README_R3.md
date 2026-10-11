# Clean T3 fold-0 startup profile R3

This is an identified one-worker T4 diagnostic profile packet. It does not contain the full teacher cohort and does not authorize a fit or provider mutation.

R1 bound two concurrent profile workers, each with a 240-second timeout. Both worker logs were empty, each reached about 0.58 GB RSS, both recorded zero GPU allocation, and the runner timed out before any worker finished. Source inspection shows that train_rnn calls load_campaign_split before configure_torch; load_campaign_split performs portable identity verification; and the portable verifier streams a full SHA-256 over the mounted parquet files in 1 MiB reads. The R1 observations fit that pre-GPU hashing explanation, but R1 did not capture a stack trace or hash progress, so the exact stalled frame is unproven.

R3 retains every full-content hash and all original split, fit-membership, and source-identity checks. It profiles one worker to avoid duplicating simultaneous full-dataset reads. It enables unbuffered output, per-file hash byte/time progress, split-load phase events and a repeating 60-second Python stack dump. The worker timeout is 1,800 seconds under the existing 6,000-second kernel-process ceiling.

The command retains the R1 shape: 5 warmup and 25 measured updates of the seed-3 H384 GRU, batch 32/chunk 1000, Muon-all/OneCycle/fp16. It enables `--grad-stats` for pre-clip gradient-norm/finite-step summaries and emits `PROFILE_UPDATE_COUNTS` for optimizer attempts, AMP-skipped updates, completed updates, and completed batches before the profile-only early return. The statistics do not change optimizer steps. They add per-step device reductions plus host synchronization at the summary, so do not treat this profile's timing as an unbiased full-fit throughput estimate.

This profile does not exercise checkpoint save/reload; that remains a separate integrity gate before full-fit artifacts are promoted.

This is one actual T4 profile only. It does not establish two-worker concurrent hashing, cohort memory, full-fit/export timing, or a three-teacher admission case. Those remain root-owned gates after the terminal profile result and fresh provider/quota checks.


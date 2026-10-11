# Local old-Z4 F0 family-student resource profile R1

This packet prepares only the exact F0 H230 family-student one-batch resource profile. It does not authorize or start the profile or the two-epoch scientific fit. Root owns live admission after the exact H230 parent fit/export has produced and bound its model_ep3.pt.

The command is derived from the frozen SEQUENCE_F0_STUDENT_R3.json fit step. The only changes are a 600-second timeout, a unique profile run name, omission of --save-training-state because the frozen trainer rejects continuation saving with a profile early exit, profile-only allocator/loader-memory limits, and --grad-stats --profile-warmup 1 --profile-steps 1. Method and data arguments remain bound to the fit sequence and input/source manifests.

The source sets max_batches = profile_warmup + profile_steps, so it runs two complete training batches: one warmup and one timed. With chunk 2,000 over 20,000 rows, frozen GradStats observes 10 unscaled pre-clip gradient vectors per complete batch, or 20 observations if both batches complete. It reports steps and nonfinite_steps, but returns before EpochUpdateCounter.finish(). These observations do not establish successful GradScaler optimizer-update counts. Timing includes gradient-stat collection and its end-of-epoch host copy/synchronization; this is resource viability evidence, not full-fit throughput.

Inputs are source-qualified F0 membership (10,606 train groups plus 1,405 eligible valid groups; 468 previously inspected fold-evaluation groups excluded), the pinned split and propensity, the own eight-source family-cache metadata, and the seed-3 parent checkpoint. The parent digest is pending its real terminal producer receipt. This fold was previously inspected, so it is robustness evidence rather than a fresh holdout.

The validator uses only the standard library. It hashes the 55-file source manifest and small identity files; it does not import Torch, read training arrays, profile a device, or launch work.

# minGRU Q8 phase profile R1

This bounded successor measures why the native minGRU Q8 callback took about three times as long as the equal-matrix-count GRU Q8 callback in the paired synthetic-weight profile. It adds feature-gated counters only; it does not change the recurrence, weights, quantized representation, precision selection, or default production path.

- [Method and source comparison](METHOD_R1.md) records the matched fast-activation setting, Q8 matvec family, quantization work, likely SIMD differences, and the phase counters to measure.
- [Instrumentation source](../../../../native/src/min_gru.rs) splits Q8 activation quantization and VNNI matvec time, then records normalization, activations, recurrent updates, skip concatenation, the causal raw-feature builder, and inclusive step/body spans. The counters and Python accessors exist only with Cargo feature `profile`.
- [Phase runner](../../../../scripts/profile_min_gru_q8_phases.py) compares the profile build with the sealed uninstrumented output and reset-state hashes before emitting per-phase RDTSC totals for three complete raw112→211 callbacks.
- [Reused fixture runner](../../../../scripts/profile_min_gru_cost_pair.py) is copied byte-for-byte from sealed commit `6ec323bc8286349e240e7f25df2d1976bc2d90fe` (SHA-256 `5c5e27e877dc06596a843f89553e119c1b5efb460ee1a37e206cd2e554a31644`) to preserve the original synthetic weights and source-qualified fitting-sequence fixture construction.
- [Admission status](PENDING_R1.json) preserves the initial wait state; the later [resource gate](RESOURCE_GATE_R1.json) admits one feature-profile build and four full callback runs with a 1CPU/2GB worker bound. Build and measurements remain pending.
- [Sparse staging failure](STAGING_FAILURE_R1.json) preserves the exact command and identifies the missing worktree-local sparse pattern for the new runner; staging then succeeded after updating only this worktree's patterns.

Current status: instrumentation and runner are prepared in isolated branch `cpu-inference/min-gru-q8-phaseprofile-20261015`, based on current MAIN commit `39008b8eeb27d69d97eb9d5431241cac426ec9a2`. The profile build will compile the merged independent-width GRU implementation and native minGRU together. No optimized callback, quality result, CPU-capacity claim, or competition-host claim is present.

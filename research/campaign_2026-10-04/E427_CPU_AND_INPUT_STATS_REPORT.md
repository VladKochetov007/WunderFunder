# E427 implementation and input-statistics evidence

The bounded synthetic CPU worker passed all eight tests, including no-op parent parity, uneven chunk state carry, causal stale-cycle context, future suffix invariance, identical arm capacity, update-gate-only intervention, post-Muon structural projection and shared student/EMA context at full `[32,20000,211]` input shape. Peak RAM was 1,111,437,312 bytes, exit 0 and teardown verified. These CPU tests do not prove production recurrent GPU headroom or native inference readiness.

The separate input-only statistics worker read all 242,220,000 rows of the exact public seed-3 parent's fitting membership: 10,606 TRAIN sequences plus 1,505 VALID sequences. It rejected corrupt TRAIN group 528 and excluded all 368 evaluation VALID sequences. It requested raw features and sequence identifiers only; neither labels nor scored masks were read. Its fallback mean and 64 deterministic endpoint samples bind full-file hashes, exact parent checkpoint/provenance and membership identities. It completed in 582.226 seconds, with cgroup RAM bounded at 2,147,483,648 bytes, no new OOM, exit 0 and verified teardown. Page-cache pressure reached the bound; a larger unbounded in-memory scan was not used.

Raw proof: `runner/attempts/e427-cpu-source-checks-v1.json`, its JUnit XML, `runner/attempts/e427-context-stats-s3-v1.json`, and `checks/e427_context_stats_s3_v1.json`. The live marimo notebook saves source-hashed PNG/SVG endpoint illustrations under `eda/e427_volume_gate/`; these are input-only descriptive examples, not forecasting evidence.

The aligned/stale/no-op full-production-shape GPU smokes and the three fixed one-epoch fits remain pending. No score or deployable improvement is claimed.

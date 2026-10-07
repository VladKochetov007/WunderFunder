# H232 conditional V5 packet assembly

- Packet: `E529_H232_FULLFEATURE_GRU384_B8_AMP4096_CONDITIONAL_PROFILE_FIT_V5`
- Status: assembled; independent V5 review and fresh admission are pending.
- Provider push and conditional fit authorization: **false**.
- Worker SHA256: `557bad7673af3261c792a27d5b7897ff9af0baae9df5e170c5b1b18f9987cb83`
- Embedded ZIP SHA256: `63a1aeeaf0ffb8decc4ba46bc9fb560a447dc9952251781d850a7a645b3098c8`
- Wrapper code SHA256: `8f936379e7680c518a35c2c97d08e515c4f2fa23fb1db31f23d3b11b77b8468d`
- The V5 gate derives width 211 from the pinned archived feature source, preserves the measured 73,120,000-byte loader budget, and checks raw train/valid profile-window lookup/fallback counts per path. All misses must invoke the original content verifier.
- H232 scientific fields and AMP4096 scale are unchanged; V4 remains immutable and its approval does not carry forward.
- Validation: focused gate tests 10/10, Ruff clean, embedded bundle/member hashes verified. No fit or provider operation was run.
- Machine-readable receipt: `research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v5/H232_CONDITIONAL_V5_ASSEMBLY_20261008_R1.json`

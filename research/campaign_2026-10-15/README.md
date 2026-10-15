# October 15 campaign

This folder tracks preparation for an exact-EPYC9224 Vast-A CPU probe. The approved scope is up to$3; the current deliverable prepares data for runtime measurement and does not create an instance or perform a provider operation.

- [Vast-A CPU probe](vast_a_cpu_probe_r1/README.md) records the historical offer work, source-qualified feature-only fixture, frozen [manifest](vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json) and [preparation receipt](vast_a_cpu_probe_r1/PREPARATION_RECEIPT_R1.json). The binary payload stays in ignored physical storage and is identified by committed hashes. It contains complete F0 fitting sequences but no labels, official scoring mask, teacher outputs or cache values; it supports runtime measurement only.

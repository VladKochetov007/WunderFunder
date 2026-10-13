# QAT with renewable local and Kaggle capacity

[Current implementation plan](PLAN.md) records the user's complete QAT/compression experiment sequence, unlimited campaign GPU hours within actual local/provider capacity, matched controls, exposed cross-fold validation and strongest-endpoint delivery. It is an authorized research plan, not a measured quality result.

- [Contract](CONTRACT.json) binds objective, data/metric invariants, resource limits, owners and confirmation requirements.
- [Frontier](FRONTIER.md) distinguishes implementation, actual profiling, fits, native evaluation and package readiness.
- [ADR0036](../../../specs/adr/0036-qat-renewable-capacity.md) records the latest authority and supersedes proposed 4/2/2-hour allocations without relaxing physical RAM, CPU or provider quotas.
- [Historical E419 QAT report](../../campaign_2026-10-04/E419_RECOVERED_QAT_REPORT.md) found no established weight-only Q8 QAT gain over matched fine-tuning/PTQ on exposed F0; it motivates W4, mixed precision, activation semantics and exact export tests.
- [Primary-source components](SOURCES.md) links the three requested RNN/RNN-T/LSQ papers, records proposed transfers and distinguishes published findings from unmeasured local hypotheses.
- [Corrected live Kaggle quota](KAGGLE_QUOTA_ENTRY_R2.json) records 34,439.317 GPU seconds and 72,000 TPU seconds remaining. The [first receipt](KAGGLE_QUOTA_ENTRY_R1.json) has correct typed numbers but an incorrect decision based on the SDK text display dropping a day; R2 explicitly corrects it before any admission.
- [Local admission snapshot](LOCAL_STATUS_ENTRY_COMPACT_R1.json) confirms the shared runtime is open, with no active attempts and unchanged CPU/RAM/no-swap limits. Historical attempts are not active reservations.

Canonical navigation: [inference research](../../unified/inference/README.md); related [operations](../../unified/operations/README.md), [sources](../../unified/sources/README.md), [research index](../../../RESEARCH.md).

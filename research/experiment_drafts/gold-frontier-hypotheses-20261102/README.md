# Gold frontier hypothesis portfolio

Five comparisons toward a stronger Connectome submission. Training code and
commands for A–D and the bounded OOF analysis for E are prepared in this PR.
**No receiver acceptance, numerical parity, GPU profile or scientific fit has
run**. E additionally needs actual qualified predictions and operator-reviewed
lineage receipts. This is not a claim of launch admission or submission readiness.
No running job is touched.

- [HYPOTHESES.md](HYPOTHESES.md): mechanisms, controls, costs, leakage,
  uncertainty, existing related work and explicit falsifiers.
- [recipe.json](recipe.json): prospective IDs and fitting budgets; incomplete
  recipes are deliberately not a launch interface.
- [PR.md](PR.md): review and decision brief.
- [A: filtered residual B300](FILTERED_RESIDUAL_B300.md): six-epoch matched label-only GRU commands.
- [B: PCGrad B300](pcgrad/README.md): single-GPU label-only GRU.
- [C: prefix burn-in B300](burnin/README.md): controlled reset vs prefix replay.
- [D: TCN B300](TCN_B300.md): causal TCN/GRU teacher configs and commands.
- [E: nested OOF](nested_oof/README.md): bounded offline fit and scoring contract.
- [A/B/C evaluation](legacy_grade/README.md): full-sequence export and official scorer.

Draft PR: [https://github.com/VladKochetov007/WunderFunder/pull/58](https://github.com/VladKochetov007/WunderFunder/pull/58).

Base: `research-team` at `7e4074314a62d0907c9608f9220cf72958708a2d`.
Reference evidence: [PR53](https://github.com/VladKochetov007/WunderFunder/pull/53#issuecomment-6545905799);
[PR54 artifacts](https://github.com/VladKochetov007/WunderFunder/pull/54#issuecomment-6555754846).
Existing ideas and continuations: [PR22](https://github.com/VladKochetov007/WunderFunder/pull/22),
[PR55](https://github.com/VladKochetov007/WunderFunder/pull/55),
[PR56](https://github.com/VladKochetov007/WunderFunder/pull/56),
[PR57](https://github.com/VladKochetov007/WunderFunder/pull/57),
[R6 conflict diagnostic](../../campaign_2026-10-19/idea_screens_r6/RESULTS.md), and
[earlier frequency analogy](../../campaign_2026-10-05/review_20261005/MECHANISM_AUDIT.md).
Those older observations do not prove these recipes work. Published full-fit PR54
weights cannot be repurposed as clean evaluation parents.

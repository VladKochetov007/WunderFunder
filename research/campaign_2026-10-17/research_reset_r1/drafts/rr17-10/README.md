# rr17-10: Nonlinear common/contrast model with asymmetric capacity

**Stage:** proposal-only staging packet. No implementation, adapter, trainer, fit, inference run, profile, grade, archive, PR URL or merge SHA is included. Training remains unauthorized. The parent controller will install this packet on its isolated research-team branch; the links here resolve within this staging packet and are adjusted when installed under `research/experiment_drafts/rr17-10/`.

## Packet files

- [PR.md](PR.md) answers all eleven experiment-draft sections, including the matched 30-cell-per-arm future matrix, metric/uncertainty plan, leakage gates, missing bindings and non-training validation command.
- [recipe.json](recipe.json) is the machine-readable proposal. It explicitly records unresolved model/data/membership/compute/ancestry/training-command fields as null with gates; it is not executable.
- [README.md](README.md) is this packet index and status note.
- No implementation/adapters are present; that omission is deliberate until a mechanism is selected, source lineage is bound and implementation is authorized.

## Canonical proposal and provenance

- [Canonical 25-card proposal](../../IDEAS_25.md) — card `H-RR17-10`: **Nonlinear common/contrast model with asymmetric capacity**. The canonical JSON card carries the mechanism, prior overlap, no-fit gate, falsifier, future control, compute path and source list.
- The current campaign contract is no-fit: it permits proposal, public-literature and deterministic analysis work but does not authorize predictive fitting. All source-qualified folds are exposed development data.
- This packet preserves the card’s evidence and counterevidence: A linear rotation with inverse reconstruction is an exact no-op. The targets are correlated, but strata are retrospective, the exact target law has 1,253 counterexamples, and weighted Pearson uses separate target measures; correlation alone does not justify tying them.
- Prior overlap: Reopens the rejected common/contrast coordinate proposal only as an explicit nonlinear capacity constraint; no hard target equality or class gate.
- Sources cited by the card:
- [https://arxiv.org/html/0912.0902](https://arxiv.org/html/0912.0902)
- [https://wundernn.io/connectome/docs/data_overview](https://wundernn.io/connectome/docs/data_overview)

## Current decision

Status is `proposal`; no expected score gain is assigned. The cheapest no-fit gate is: Before fitting, verify parameter ties change the function class, preserve two free outputs, and do not use target-derived strata. Compare parameter/FLOP counts to independent heads. The next decision is to bind exact code/data/membership/ancestry/compute and commands, assign an owner and effect threshold, and obtain explicit future fit authority. Do not infer any result from the documentation check.

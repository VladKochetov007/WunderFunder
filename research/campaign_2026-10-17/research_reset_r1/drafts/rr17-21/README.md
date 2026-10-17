# rr17-21: Bounded group-DRO training objective with pooled WP primary

**Stage:** proposal-only staging packet. No implementation, adapter, trainer, fit, inference run, profile, grade, archive, PR URL or merge SHA is included. Training remains unauthorized. The parent controller will install this packet on its isolated research-team branch; the links here resolve within this staging packet and are adjusted when installed under `research/experiment_drafts/rr17-21/`.

## Packet files

- [PR.md](PR.md) answers all eleven experiment-draft sections, including the matched 30-cell-per-arm future matrix, metric/uncertainty plan, leakage gates, missing bindings and non-training validation command.
- [recipe.json](recipe.json) is the machine-readable proposal. It explicitly records unresolved model/data/membership/compute/ancestry/training-command fields as null with gates; it is not executable.
- [README.md](README.md) is this packet index and status note.
- No implementation/adapters are present; that omission is deliberate until a mechanism is selected, source lineage is bound and implementation is authorized.

## Canonical proposal and provenance

- [Canonical 25-card proposal](../../IDEAS_25.md) — card `H-RR17-21`: **Bounded group-DRO training objective with pooled WP primary**. The canonical JSON card carries the mechanism, prior overlap, no-fit gate, falsifier, future control, compute path and source list.
- The current campaign contract is no-fit: it permits proposal, public-literature and deterministic analysis work but does not authorize predictive fitting. All source-qualified folds are exposed development data.
- This packet preserves the card’s evidence and counterevidence: Group-DRO is a known robust-risk method, but EDA class strata are retrospective and deliberately equal-sampled; they cannot define hidden-test mixture or inference gates. Reweighting can optimize a different estimand from pooled WP.
- Prior overlap: Reopens group robustness only as training loss bookkeeping; not a new class-conditional serving model or an assumption of population shift.
- Sources cited by the card:
- [https://arxiv.org/abs/2212.13669](https://arxiv.org/abs/2212.13669)
- [https://wundernn.io/connectome/docs/data_overview](https://wundernn.io/connectome/docs/data_overview)

## Current decision

Status is `proposal`; no expected score gain is assigned. The cheapest no-fit gate is: On training rows only, count group support and target-weight mass; toy-check how the robust objective can alter pooled weighting. Never use validation labels to choose groups or weights. The next decision is to bind exact code/data/membership/ancestry/compute and commands, assign an owner and effect threshold, and obtain explicit future fit authority. Do not infer any result from the documentation check.

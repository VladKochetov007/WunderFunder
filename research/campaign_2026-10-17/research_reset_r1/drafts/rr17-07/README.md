# rr17-07: Distribution- and dependence-preserving distillation versus mean/family targets

**Stage:** proposal-only staging packet. No implementation, adapter, trainer, fit, inference run, profile, grade, archive, PR URL or merge SHA is included. Training remains unauthorized. The parent controller will install this packet on its isolated research-team branch; the links here resolve within this staging packet and are adjusted when installed under `research/experiment_drafts/rr17-07/`.

## Packet files

- [PR.md](PR.md) answers all eleven experiment-draft sections, including the matched 30-cell-per-arm future matrix, metric/uncertainty plan, leakage gates, missing bindings and non-training validation command.
- [recipe.json](recipe.json) is the machine-readable proposal. It explicitly records unresolved model/data/membership/compute/ancestry/training-command fields as null with gates; it is not executable.
- [README.md](README.md) is this packet index and status note.
- No implementation/adapters are present; that omission is deliberate until a mechanism is selected, source lineage is bound and implementation is authorized.

## Canonical proposal and provenance

- [Canonical 25-card proposal](../../IDEAS_25.md) — card `H-RR17-07`: **Distribution- and dependence-preserving distillation versus mean/family targets**. The canonical JSON card carries the mechanism, prior overlap, no-fit gate, falsifier, future control, compute path and source list.
- The current campaign contract is no-fit: it permits proposal, public-literature and deterministic analysis work but does not authorize predictive fitting. All source-qualified folds are exposed development data.
- This packet preserves the card’s evidence and counterevidence: Soft targets can preserve output structure in principle, but no project result validates this transfer. E468 joint codebook was negative against a losing anchor, and histogram MSE/CE were negative; those are strong warnings against renaming a probability head.
- Prior overlap: Continues the R2 distribution-preserving KD synthesis. Distinct from hidden-state hints only if it carries conditional output distribution/dependence, not hidden activations.
- Sources cited by the card:
- [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
- [https://arxiv.org/abs/2608.13391](https://arxiv.org/abs/2608.13391)

## Current decision

Status is `proposal`; no expected score gain is assigned. The cheapest no-fit gate is: Check that an existing causal teacher has stable out-of-fold quantile/dependence outputs beyond its means. If no such frozen artifact exists, do not generate a teacher now. The next decision is to bind exact code/data/membership/ancestry/compute and commands, assign an owner and effect threshold, and obtain explicit future fit authority. Do not infer any result from the documentation check.

# rr17-04: Teacher hidden-state hints versus output-only distillation

**Stage:** proposal-only staging packet. No implementation, adapter, trainer, fit, inference run, profile, grade, archive, PR URL or merge SHA is included. Training remains unauthorized. The parent controller will install this packet on its isolated research-team branch; the links here resolve within this staging packet and are adjusted when installed under `research/experiment_drafts/rr17-04/`.

## Packet files

- [PR.md](PR.md) answers all eleven experiment-draft sections, including the matched 30-cell-per-arm future matrix, metric/uncertainty plan, leakage gates, missing bindings and non-training validation command.
- [recipe.json](recipe.json) is the machine-readable proposal. It explicitly records unresolved model/data/membership/compute/ancestry/training-command fields as null with gates; it is not executable.
- [README.md](README.md) is this packet index and status note.
- No implementation/adapters are present; that omission is deliberate until a mechanism is selected, source lineage is bound and implementation is authorized.

## Canonical proposal and provenance

- [Canonical 25-card proposal](../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md) — card `H-RR17-04`: **Teacher hidden-state hints versus output-only distillation**. The canonical JSON card carries the mechanism, prior overlap, no-fit gate, falsifier, future control, compute path and source list.
- The current campaign contract is no-fit: it permits proposal, public-literature and deterministic analysis work but does not authorize predictive fitting. All source-qualified folds are exposed development data.
- This packet preserves the card’s evidence and counterevidence: FitNets introduced intermediate hint matching, but on vision tasks. Existing local student work mostly tests output targets; old moment/family auxiliaries are not hidden-state matching. No evidence shows hidden-state transfer helps this stream, and an incompatible state may overconstrain the student.
- Prior overlap: Distinct from output KD in card 02 and from label-derived moment/family heads; adjacent to existing representation distillation proposals.
- Sources cited by the card:
- [https://arxiv.org/abs/1412.6550](https://arxiv.org/abs/1412.6550)
- [https://arxiv.org/abs/2608.13391](https://arxiv.org/abs/2608.13391)

## Current decision

Status is `proposal`; no expected score gain is assigned. The cheapest no-fit gate is: Use synthetic causal sequences to verify row alignment, reset behavior, projection shape, and future-suffix invariance. Audit that teacher state at t depends only on inputs through t. The next decision is to bind exact code/data/membership/ancestry/compute and commands, assign an owner and effect threshold, and obtain explicit future fit authority. Do not infer any result from the documentation check.

## Installed branch status

Installed on `experiment/rr17-04` from research-team `ed40bea53b66f65a437f9ea3658a982ebdbe5251`. This remains a proposal with no executable training adapter. The original staging description above is preserved as preparation history; actual publication and commit identity are recorded in the campaign publication receipt. No PR exists until an authenticated API session creates it.

# rr17-02: Large causal teacher with a same-context compact student

**Stage:** proposal-only staging packet. No implementation, adapter, trainer, fit, inference run, profile, grade, archive, PR URL or merge SHA is included. Training remains unauthorized. The parent controller will install this packet on its isolated research-team branch; the links here resolve within this staging packet and are adjusted when installed under `research/experiment_drafts/rr17-02/`.

## Packet files

- [PR.md](PR.md) answers all eleven experiment-draft sections, including the matched 30-cell-per-arm future matrix, metric/uncertainty plan, leakage gates, missing bindings and non-training validation command.
- [recipe.json](recipe.json) is the machine-readable proposal. It explicitly records unresolved model/data/membership/compute/ancestry/training-command fields as null with gates; it is not executable.
- [README.md](README.md) is this packet index and status note.
- No implementation/adapters are present; that omission is deliberate until a mechanism is selected, source lineage is bound and implementation is authorized.

## Canonical proposal and provenance

- [Canonical 25-card proposal](../../IDEAS_25.md) — card `H-RR17-02`: **Large causal teacher with a same-context compact student**. The canonical JSON card carries the mechanism, prior overlap, no-fit gate, falsifier, future control, compute path and source list.
- The current campaign contract is no-fit: it permits proposal, public-literature and deterministic analysis work but does not authorize predictive fitting. All source-qualified folds are exposed development data.
- This packet preserves the card’s evidence and counterevidence: This is the existing PR1/PR3 route, supported by recipe-specific local KD positives. Those panels are exposed and do not identify a universally better teacher or student transfer; the newer warm-parent result is one trajectory.
- Prior overlap: Explicit continuation, not new: existing causal Transformer/GRU teachers, heterogeneous GRU KD, and teacher-cohort proposals.
- Sources cited by the card:
- [https://arxiv.org/abs/2608.13391](https://arxiv.org/abs/2608.13391)
- [https://github.com/nv-tlabs/cmd](https://github.com/nv-tlabs/cmd)
- [https://wundernn.io/connectome/docs/data_overview](https://wundernn.io/connectome/docs/data_overview)

## Current decision

Status is `proposal`; no expected score gain is assigned. The cheapest no-fit gate is: Audit teacher/student preprocessing hashes, context masks, free-running state, and reset parity. If contexts differ or the source cache is incomplete, resolve provenance before considering a comparison. The next decision is to bind exact code/data/membership/ancestry/compute and commands, assign an owner and effect threshold, and obtain explicit future fit authority. Do not infer any result from the documentation check.

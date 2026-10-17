# rr17-20: Cross-fitted scalar calibration under the weighted metric

**Stage:** proposal-only staging packet. No implementation, adapter, trainer, fit, inference run, profile, grade, archive, PR URL or merge SHA is included. Training remains unauthorized. The parent controller will install this packet on its isolated research-team branch; the links here resolve within this staging packet and are adjusted when installed under `research/experiment_drafts/rr17-20/`.

## Packet files

- [PR.md](PR.md) answers all eleven experiment-draft sections, including the matched 30-cell-per-arm future matrix, metric/uncertainty plan, leakage gates, missing bindings and non-training validation command.
- [recipe.json](recipe.json) is the machine-readable proposal. It explicitly records unresolved model/data/membership/compute/ancestry/training-command fields as null with gates; it is not executable.
- [README.md](README.md) is this packet index and status note.
- No implementation/adapters are present; that omission is deliberate until a mechanism is selected, source lineage is bound and implementation is authorized.

## Canonical proposal and provenance

- [Canonical 25-card proposal](../../IDEAS_25.md) — card `H-RR17-20`: **Cross-fitted scalar calibration under the weighted metric**. The canonical JSON card carries the mechanism, prior overlap, no-fit gate, falsifier, future control, compute path and source list.
- The current campaign contract is no-fit: it permits proposal, public-literature and deterministic analysis work but does not authorize predictive fitting. All source-qualified folds are exposed development data.
- This packet preserves the card’s evidence and counterevidence: The weighted-mean geometry is exact for the official score, and positive affine transforms do nothing before clipping. Histogram MSE/CE were negative; this proposal is a low-dimensional scalar post-map, not another codebook head, and has no measured gain.
- Prior overlap: New narrow candidate from R3; overlaps histogram calibration enough that the exact old arm and held-out component protocol must be checked first.
- Sources cited by the card:
- [https://arxiv.org/html/0912.0902](https://arxiv.org/html/0912.0902)
- [https://wundernn.io/connectome/docs/data_overview](https://wundernn.io/connectome/docs/data_overview)

## Current decision

Status is `proposal`; no expected score gain is assigned. The cheapest no-fit gate is: Inspect existing frozen-checkpoint outputs if available; bounded inference of a frozen checkpoint is allowed under the contract if needed. Plot only predeclared weighted bin means, support, and component stability; do not fit a calibration map now or tune on grade components. The next decision is to bind exact code/data/membership/ancestry/compute and commands, assign an owner and effect threshold, and obtain explicit future fit authority. Do not infer any result from the documentation check.

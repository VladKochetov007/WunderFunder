# rr17-05: Asynchronous i1 lag reliability gate with a no-i1 expert

**Stage:** proposal-only staging packet. No implementation, adapter, trainer, fit, inference run, profile, grade, archive, PR URL or merge SHA is included. Training remains unauthorized. The parent controller will install this packet on its isolated research-team branch; the links here resolve within this staging packet and are adjusted when installed under `research/experiment_drafts/rr17-05/`.

## Packet files

- [PR.md](PR.md) answers all eleven experiment-draft sections, including the matched 30-cell-per-arm future matrix, metric/uncertainty plan, leakage gates, missing bindings and non-training validation command.
- [recipe.json](recipe.json) is the machine-readable proposal. It explicitly records unresolved model/data/membership/compute/ancestry/training-command fields as null with gates; it is not executable.
- [README.md](README.md) is this packet index and status note.
- No implementation/adapters are present; that omission is deliberate until a mechanism is selected, source lineage is bound and implementation is authorized.

## Canonical proposal and provenance

- [Canonical 25-card proposal](../../IDEAS_25.md) — card `H-RR17-05`: **Asynchronous i1 lag reliability gate with a no-i1 expert**. The canonical JSON card carries the mechanism, prior overlap, no-fit gate, falsifier, future control, compute path and source list.
- The current campaign contract is no-fit: it permits proposal, public-literature and deterministic analysis work but does not authorize predictive fitting. All source-qualified folds are exposed development data.
- This packet preserves the card’s evidence and counterevidence: Frozen CPU replay on strict425 groups [3, 6, 14, 15] (11,522 scored rows) moved WP from 0.06289742 with ZERO_COLS=None to 0.72994782 after settings restoration; manual/config inputs and predictions matched bitwise. This isolates a serious preprocessing defect on that slice, not i1 benefit or general model quality. Sampled i1 drift is confounded; lag effects vary by stratum.
- Prior overlap: Reopens i1-flow/cross-impact after a confirmed restore defect; overlaps prior i1 features and cross-stream lags. Replay validates restoration only.
- Sources cited by the card:
- [https://arxiv.org/abs/2112.13213](https://arxiv.org/abs/2112.13213)
- [https://www.ms.u-tokyo.ac.jp/~nakahiro/mypapers_for_personal_use/hayyos03.pdf](https://www.ms.u-tokyo.ac.jp/~nakahiro/mypapers_for_personal_use/hayyos03.pdf)
- [https://wundernn.io/connectome/docs/data_overview](https://wundernn.io/connectome/docs/data_overview)

## Current decision

Status is `proposal`; no expected score gain is assigned. The cheapest no-fit gate is: First reproduce bitwise manual/config parity. For the lag hypothesis, use predeclared input-only lag and phase-shift controls across sequence halves; never tune on labels or retrospective classes. The next decision is to bind exact code/data/membership/ancestry/compute and commands, assign an owner and effect threshold, and obtain explicit future fit authority. Do not infer any result from the documentation check.

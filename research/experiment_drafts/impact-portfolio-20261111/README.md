# Next-impact portfolio: ten research screens

Published as [draft PR81](https://github.com/VladKochetov007/WunderFunder/pull/81) targeting `research-team`.

**Prepared designs, not runnable experiments or measured improvements.** This
portfolio reviews all 21 recent PRs **#60–80**, checks older overlaps, and proposes
ten separately selectable comparisons: seven on the current Transformer recipe
and three on K16 GRU. It carries the team's validation, streaming, metrics and
logging contract forward. Hardware optimization remains with the receiver.

## What changed our priorities

- [PR79](https://github.com/VladKochetov007/WunderFunder/pull/79#issuecomment-6778521007):
  K16 GRU beats K1 by +0.00115142/+0.00103319 WP at fixed E6. This is the clearest
  recent positive paired screen, not a public score or comparison to every model.
- [PR74](https://github.com/VladKochetov007/WunderFunder/pull/74#issuecomment-6781817312):
  the Transformer K16 effect is mixed (+0.000258/−0.000132 vs K1). Do not assume
  the GRU result transfers to another backbone.
- [PR72](https://github.com/VladKochetov007/WunderFunder/pull/72#issuecomment-6731395515):
  Muon/chunk256 0.680881/0.682194 beats AdamW/chunk256 0.670532/0.672148;
  Muon/chunk512 0.680625/0.681024 offers no gain. Preserve the useful baseline.
- [PR70](https://github.com/VladKochetov007/WunderFunder/pull/70#issuecomment-6716525560):
  label-objective mixtures outperform the proposed correlation objective. PR71's
  dropout/event/SWA results also fail to beat TL3. These are retained negatives.
- [PR78](https://github.com/VladKochetov007/WunderFunder/pull/78#issuecomment-6775969810)
  input pretraining is mixed (+0.003053/−0.000748); PR77 specialists are mixed or
  negative. Follow-ups must change the mechanism, not just rename those arms.
- At the 05:35 UTC review cutoff, PR75/80's latest 05:30 wrapper events still
  mirror producer logs from about02:10. Current execution status is unknown;
  there are no accepted final comparisons yet. PR80 already covers K64, width512
  and dropout0.1: no duplicate launches are proposed here.

All numbers are team-reported, repeatedly exposed development results; no scoring
or weight-byte verification was performed here. Full status, sources and limitations
are in [the PR audit](PR_REVIEW.md).

## Ten comparisons

| Card | Backbone | Question | Priority / history |
|---|---|---|---|
| [01 Numerical PLE](cards/01-numerical-ple.md) | Transformer | Do continuous nonlinear scalar embeddings help? | 1; continuation of level encoding |
| [02 Same-row group tokens](cards/02-same-row-group-tokens.md) | Transformer | Does nonlinear mixing of schema groups help before temporal attention? | 2; modern successor to PR8, not a new duplicate PR |
| [03 Target-private adapters](cards/03-target-private-adapters.md) | Transformer | Can both targets share time modeling but use different nonlinear readouts? | 3; differs from independently fitted specialists |
| [04 Intermediate supervision](cards/04-deep-supervision.md) | Transformer | Does direct supervision at depths6/12/18 help the final head? | 4 |
| [05 Shared-depth blocks](cards/05-tied-depth.md) | Transformer | Does sharing12 parameter sets over24 logical layers improve generalization? | 9; separate24 KV caches |
| [06 Pairwise ordering](cards/06-pairwise-ranking.md) | Transformer | Does a small ranking auxiliary improve official WP? | 5; ranking is not Pearson |
| [07 Future-label heads](cards/07-future-label-heads.md) | Transformer | Do row-offset targets1/16/64 improve the current prediction? | 6; old backlog continuation, no future inputs |
| [08 State perturbation](cards/08-state-perturbation.md) | K16 GRU | Can less brittle state improve clean prediction? | 8; training-only noisy branch |
| [09 Variance-head auxiliary](cards/09-nll-transfer.md) | K16 GRU | Does the earlier NLL idea help this stronger recipe? | 7; from scratch, no checkpoint transfer |
| [10 Clip-event head](cards/10-clip-event-head.md) | K16 GRU | Does classifying signed tail events improve the unchanged regression head? | 10; train-only prevalence gate first |

Start with **01**, then **02/03** according to receiver capacity. This ranking is
an information-value judgment, not a measured quality ordering. Card02 should
continue the existing [PR8](https://github.com/VladKochetov007/WunderFunder/pull/8)
lineage with an explicit modern recipe amendment. No ten-way joint launch is
requested. Each selected card needs a mechanism-specific implementation PR and
receiver admission; avoid another draft of an already identical experiment.

Each screen inventories four fresh fits (two controls + two treatments), **40 if
all ten were selected**. No teacher fitting is hidden, no controls are assumed
reusable, and no fits are currently selected or dispatched. Profiles and artifact
export are additional engineering work. Same epochs do not mean equal FLOPs.

## Files and use

- [CONTRACT.md](CONTRACT.md): frozen scientific baseline, data/mask/state,
  precision, fixed endpoints, fit accounting, uncertainty and receiver artifacts.
- [recipe.json](recipe.json): machine-readable **proposal** inventory, per-card
  parameters, exact historical config origins, empty execution/result fields.
- [cards/](cards/): mechanisms, controls, falsifiers, leakage checks, missing adapters
  and deployment implications. No card provides a fake trainer command.
- [PR_REVIEW.md](PR_REVIEW.md): all recent PRs, positive/negative/partial results,
  stale telemetry and primary source links.
- [SOURCES.md](SOURCES.md): primary papers and limits of transfer from them.
- [REVIEW.md](REVIEW.md): independent criticism and concrete design corrections.
- [STATIC_REVIEW.json](STATIC_REVIEW.json): JSON, link and source-hash inspection;
  no runtime or scientific acceptance.
- [sources/](sources/): two byte-preserved reference configs and PR-head metadata;
  these configs still contain historical seed/device settings and must not be run
  as new experiments.
- [PR.md](PR.md): eleven-section publication contract.

Local work was limited to reading sources, editing docs/JSON and static metadata
inspection. No model imports, tests, inference, training, profiling or existing-job
changes. PR76's existing CPU ZIP remains separate and not receiver-validated here.

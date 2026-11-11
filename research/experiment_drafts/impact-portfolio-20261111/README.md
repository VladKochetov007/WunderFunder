# Next-impact portfolio: six implemented screens, four proposals

Published as [draft PR81](https://github.com/VladKochetov007/WunderFunder/pull/81) targeting `research-team`.

**Current status:** cards 01–06 have implementation in this packet and await
receiver acceptance. Cards 07–10 remain proposals. The first six cards have
24 fresh fits planned in total (four per card: two seeds, control and treatment),
with zero fits executed and no old controls reused. No profile, grade, or result
is available. The packet does not automatically launch jobs.

All six implementations use the exact plain Transformer runtime from PR75,
commit `8626000a4b965763a70011a0291a891aa6535d7e`, staged as a separate clean,
read-only worktree on the receiving host. New experiment adapters live under
`runtime/`; the repository's existing `src/`, `scripts/`, training configs,
optimizer, training loop, validation, and checkpoint code are unchanged.
Runtime admission and source identity are checked on the receiver. Static packet
preparation is not receiver acceptance or scientific evidence.

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
- The [PR60–80 review](PR_REVIEW.md) records that the then-current PR75/80
  wrapper events mirrored older producer logs. That cutoff did not establish
  execution status or accepted final comparisons. PR80 already covers K64,
  width512 and dropout0.1; this portfolio does not duplicate those launches.

All cited figures are team-reported, repeatedly exposed development results;
they are not fresh measurements from this packet. No scoring or weight-byte
verification was performed here. See the [PR audit](PR_REVIEW.md) for sources
and limitations.

## Ten comparisons

| Card | Backbone | Question | Current status / history |
|---|---|---|---|
| [01 Numerical PLE](cards/01-numerical-ple.md) | Transformer | Do continuous nonlinear scalar embeddings help? | Implemented; receiver acceptance pending; continuation of level encoding |
| [02 Same-row group tokens](cards/02-same-row-group-tokens.md) | Transformer | Does nonlinear mixing of schema groups help before temporal attention? | Implemented; receiver acceptance pending; PR8 continuation in this same PR |
| [03 Target-private adapters](cards/03-target-private-adapters.md) | Transformer | Can both targets share time modeling but use different nonlinear readouts? | Implemented; receiver acceptance pending; differs from independent specialists |
| [04 Intermediate supervision](cards/04-deep-supervision.md) | Transformer | Does direct supervision at depths6/12/18 help the final head? | Implemented; receiver acceptance pending |
| [05 Shared-depth blocks](cards/05-tied-depth.md) | Transformer | Does sharing12 parameter sets over24 logical layers improve generalization? | Implemented; receiver acceptance pending |
| [06 Pairwise ordering](cards/06-pairwise-ranking.md) | Transformer | Does a small ranking auxiliary improve official WP? | Implemented; receiver acceptance pending |
| [07 Future-label heads](cards/07-future-label-heads.md) | Transformer | Do row-offset targets1/16/64 improve the current prediction? | Proposal only; labels are not future inputs |
| [08 State perturbation](cards/08-state-perturbation.md) | K16 GRU | Can less brittle state improve clean prediction? | Proposal only; training-only noisy branch |
| [09 Variance-head auxiliary](cards/09-nll-transfer.md) | K16 GRU | Does the earlier NLL idea help this stronger recipe? | Proposal only; from scratch, no checkpoint transfer |
| [10 Clip-event head](cards/10-clip-event-head.md) | K16 GRU | Does classifying signed tail events help the unchanged regression head? | Proposal only; train-only prevalence gate first |

The six implemented cards retain their separate controls and treatments. The
other four are not part of the current 24-fit plan. If every card were later
selected, the complete inventory would total 40 fits; that is not a launch
request or an executed queue. Card02's earlier note proposing a separate PR is
superseded: its implementation is included in PR81 alongside cards01 and03.

[Cards04–06 implementation amendment](IMPLEMENTATION_04_06.md) documents the new objective driver, copied/tied-depth control, and twelve additional fits. Prior first-three-card source receipts remain bound to their original commit.

## Frozen implementation and preprocessing

The common Transformer is the pinned PR75 plain recipe: H768, 24 layers,
12 heads, FF multiplier4, context1024, gradient horizon256, four epochs and
fixed E4 primary endpoint. Optimizer, schedule, BF16/BF16x9 behavior, streaming,
validation and checkpoint semantics remain those of the canonical runtime.
The runtime is read from the separate PR75 worktree; this packet does not copy
or patch the existing training pipeline.

Card01's PLE knot estimator uses 64 evenly spaced row positions from each of all
10,325 fitting sequences (660,800 rows total), then computes 16 linear empirical
quantiles per normalized feature. This is a deterministic, label-independent
sample of every train sequence, not an exact quantile over every row. Canonical
normalization still uses all fit rows. Valid and test data do not contribute to
knots. The receiving host creates one shared preprocessing artifact for all
arms.

## Receiver commands and evidence

- [Implementation record](IMPLEMENTATION.md): exact adapter design, runtime
  boundary, preprocessing estimator, checks and evidence limits.
- [B300 command guide](B300_COMMANDS.md): prepare, receiver checks, per-cell
  profile and explicit approval, fit, endpoint export/score and comparison.
  Commands are prepared, not executed; use a provisioned Linux host and actual
  owner, data, free B300s and resource bounds.
- [Runtime launcher](runtime/launch.py), [worker](runtime/worker.py),
  [canonical-runtime bindings](runtime/impact81/bindings.py),
  [PLE and group adapters](runtime/impact81/representations.py), and
  [target adapter](runtime/impact81/target_head.py).
- [MIDAS handoff instructions](midas/README.md),
  [metadata-only route preview](midas/route-preview.json), and
  [fit-evidence exporter](midas_evidence.py). The route preview is not a worker
  claim or execution receipt; the exporter packages evidence from completed
  runs and does not copy model weights.

The receiver must accept the pinned runtime, preprocessing, causal/reset/chunk
behavior, reconstruction, and new-parameter gradients before any profile.
Every full-size cell needs its own 50-update profile and operator-approved
receipt before fitting. Profiles are engineering checks, not quality results.
The MIDAS registration/route is metadata only; it does not launch remote work.

## Files and limits

- [CONTRACT.md](CONTRACT.md): shared data, metric, source, recipe, fit and
  acceptance contract.
- [recipe.json](recipe.json): mixed implementation/proposal inventory, exact
  pins, per-card design and separate planned/executed counts.
- [cards/](cards/): mechanism, control, treatment, falsifier, leakage and
  card-specific diagnostics.
- [PR_REVIEW.md](PR_REVIEW.md), [SOURCES.md](SOURCES.md), and [REVIEW.md](REVIEW.md):
  prior results, literature and independent design criticism.
- [Current static implementation review](STATIC_IMPLEMENTATION_REVIEW_04_06.json): source/config/command checks for cards04–06, not runtime or numerical acceptance. Earlier static receipts retain their original source scope.
- [sources/](sources/): byte-preserved historical configs. They are not
  launch configs; per-cell descriptors bind seed-only copies for this screen.
- [PR.md](PR.md): the English 11-section publication body.
- [research hypotheses](../../hypotheses.jsonl) and
  [experiment ledger](../../experiments.jsonl): append-only status history.

No local model imports, construction, tests, inference, training, profiling or
job changes were performed. Receiver acceptance is pending, all 24 planned fits
remain unrun, and no quality or leaderboard claim follows from this packet.

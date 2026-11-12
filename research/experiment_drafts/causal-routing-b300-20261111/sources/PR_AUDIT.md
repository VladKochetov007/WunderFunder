# Live PR audit for Signal Routes

**Checked:** 2026-11-12 00:17:58 UTC using read-only `gh` calls. All five PRs are
open. The five PRs had no issue comments and no submitted reviews at the cutoff.
Results below are claims in the linked PRs, not new measurements by this audit.

| PR / head | Status and reported outcome | Collision with signed-half attention / dense per-token FFN routing |
|---|---|---|
| [#82](https://github.com/VladKochetov007/WunderFunder/pull/82) `c0dc2e7b85847d45e088e6090cd1bffad5fdf549` | Proposal only: ten L4 GRU screens, 20 new fits plus two train-prediction exports; PR states no fits/profiles. Prior evidence motivates the menu: cold Muon, since-event weighting, next-change/structure heads, long gradient horizon, EMA, regime inputs, volume augmentation, class-routed readout and distillation. | Closest adjacent idea is card09's *causal class-routed output head* and card07's regime channels. Neither routes each token through Transformer FFN experts or splits attention into signed halves. Card04 is auxiliary supervision. Keep the new mechanisms at internal attention/FFN computation and document this distinction. |
| [#83](https://github.com/VladKochetov007/WunderFunder/pull/83) `9f01bb0eb9f104f7788ec02aa11d94d303d390de` | Results-only PR; no new fits in this PR. It reports TabM-GRU k4 −0.0012 vs single Muon; NorMuon no gain; view-mask p=.15 standalone −0.0014 and pair +0.00025; epoch 3–5 weight averaging +0.00004/+0.00002. The reported zero-fit blend law is pair gain ≈.168·(1−ρ). | No architecture collision. This is ensemble/value and model-selection evidence, not internal attention or per-token FFN routing. |
| [#84](https://github.com/VladKochetov007/WunderFunder/pull/84) `47d239eb847cac5afd308ffd63aad43a8157fe58` | Results-only PR; no new fits in this PR. Reported cold GRU Muon gains: +.0038/+ .0043 vs AdamW 1e−3 and +.0026/+ .0035 vs AdamW 5e−4 across two exposed folds. Warm-stage transfer is only +.00026 [−.00041,+.00101]. It also flags cuDNN GRU recurrence precision. | No attention/FFN-routing overlap. It is an optimizer result; the Signal Routes recipe retains its pinned Transformer optimizer and BF16x9 contract. |
| [#85](https://github.com/VladKochetov007/WunderFunder/pull/85) `38db05ea37e7f790cd05f6d221d8f88173db83bb` | Teacher-track continuation proposal, stopped earlier by user. PR reports reproduced PR49 teachers (fbc1k E3 .69142), D1 train-only distilled child .68997 at E5 (+.00326 over max(parent .68671, labels-only .67562), but below teacher), and TL3C blend public .66355 vs valid .692336. D2 rerun/seed confirmation remain pending; this PR launched no fits. | Distillation and cached teacher targets, not attention/FFN routing. Avoid adding teacher dependencies to the routing screen. |
| [#86](https://github.com/VladKochetov007/WunderFunder/pull/86) `742aaacc9a3d4094d93d2f897a6cfe4a42f5f9a5` | Proposal only; zero-fit clustering phase and four-fit leave-regime-out follow-up have not run. PR reports prior adversarial validation AUC .984 and a proposal to compare two recipes on held-out regime clusters. | Validation-split design only; no model mechanism overlap. Preserve the packet's official split and describe this as a same-split mechanism screen, not regime-robust confirmation. |

## Novelty decision

None of #82–86 proposes signed-half attention or dense per-token FFN routing. The
nearest wording is #82's class-conditioned *readout*; it changes output selection
by causal class, while this screen changes internal Transformer computation on
each token. #82's regime channels and next-change heads are also distinct input
and supervision paths. PR83–86 concern ensembles, GRU optimization/distillation,
or evaluation design. The design remains nonduplicate at the mechanism level;
the adjacent routing language should be called out in the experiment cards.

The audit is based on PR bodies, changed-file lists, issue comments, and submitted
reviews only. It does not independently validate the historical result artifacts.

## Earlier audit during preparation, 11 November

The following entries preserve the preceding read-only review. They are not new
measurements and are not assertions that a job is currently running.

| PR | Reported evidence and decision for this packet |
|---|---|
| [#70](https://github.com/VladKochetov007/WunderFunder/pull/70#issuecomment-6716525560) | Four fixed-E4 fits reported. Fixed TL3 mixtures with label WMSE: .692531/.692454; correlation objective: .690990/.690883. Source/checkpoint/grade byte audit remained pending. Do not repeat the correlation/blend objective. |
| [#71](https://github.com/VladKochetov007/WunderFunder/pull/71#issuecomment-6716252348) | Six E6 fits and 24 SWA/E6 grades reported; fixed mixtures approximately .692104–.692304, below TL3 .692377. Dropout, event211 and SWA are already covered. |
| [#72](https://github.com/VladKochetov007/WunderFunder/pull/72#issuecomment-6731395515) | Muon/chunk256 .680881/.682194; AdamW/chunk256 .670532/.672148; Muon/chunk512 .680625/.681024. Producer grades only; checkpoint byte audit pending. AdamW also changed decay coverage. No optimizer/horizon retest here. |
| [#73](https://github.com/VladKochetov007/WunderFunder/pull/73#issuecomment-6727640776) | TL3 residual correlations .98954–.99664 are descriptive and contain shared-target noise; position patterns do not establish reset boundaries. No claim of proven long-memory mechanism. |
| [#74](https://github.com/VladKochetov007/WunderFunder/pull/74#issuecomment-6781817312) | K16−K1 paired WP deltas +.000258/−.000132; K16−plain +.000482/−.000211. Mixed Transformer stem evidence. No further stem kernel screen. |
| [#75](https://github.com/VladKochetov007/WunderFunder/pull/75#issuecomment-6788522993) | Latest final status was `unverified_at_deadline`; cached training progress did not establish completed QK-normalization scores. Pinning its plain runtime does not claim a successful QK result. |
| [#77](https://github.com/VladKochetov007/WunderFunder/pull/77#issuecomment-6775591484) | Shared GRUs .680506/.682647. Target specialists were mostly negative or mixed; joint specialists .680591/.681499. No target-specialist trunk training here. |
| [#78](https://github.com/VladKochetov007/WunderFunder/pull/78#issuecomment-6775969810) | Train-only dynamics pretraining changed WP by +.003053/−.000748 across two seeds. Mixed evidence; do not repeat input-dynamics SSL. |
| [#79](https://github.com/VladKochetov007/WunderFunder/pull/79#issuecomment-6778521007) | K16−K1 +.00115142/+.00103319, K16−plain +.00114359/+.00152650: consistent reported GRU improvement. Its K16/K1 intervention is already covered. |
| [#80](https://github.com/VladKochetov007/WunderFunder/pull/80#issuecomment-6788525028) | K64, width512 and dropout had no verified final summary by the reported deadline. Preserve that uncertainty; these are occupied experiments, not failed mechanisms. |
| [#81](https://github.com/VladKochetov007/WunderFunder/pull/81) | At `650c21e19cdaef4fd9605311098eab45f5316e40`, ten implemented mechanisms and 40 planned fits, zero executed. Numerical embeddings, group-token mixing, private target heads, deep supervision, tied depth, ranking, future-label heads, state perturbation, NLL and clipping-event heads are excluded from this new packet. |

The [#55 portfolio](https://github.com/VladKochetov007/WunderFunder/pull/55) already
covers temporal/disagreement KD, gated output heads, regime teacher mixtures,
multiscale Transformers, DeltaNet/Mamba teachers, scale encodings and group-robust
objectives. [#58](https://github.com/VladKochetov007/WunderFunder/pull/58) covers
frequency residual loss, target PCGrad, burn-in/TBPTT, TCN and nested-OOF mixtures.
Earlier H-082 and H-106/E-425 prefix-specialist routing did not establish an
advantage over uniform/hash routes. SR-F retains that counterevidence and changes
the intervention to token-local routing inside the shared FFN, not a relabeled
prefix-specialist readout. PR64 recency bias, PR65 SDPA output gating and PR75 QK
normalization are adjacent to SR-A but do not test signed map interaction.

The immutable routing lineage is in the
[H-082/H-106 ledger snapshot](https://github.com/VladKochetov007/WunderFunder/blob/650c21e19cdaef4fd9605311098eab45f5316e40/research/unified/archive/2026-10-12/ledger-snapshot/hypotheses.asof-20261012.jsonl#L81)
and [E-424/E-425 snapshot](https://github.com/VladKochetov007/WunderFunder/blob/650c21e19cdaef4fd9605311098eab45f5316e40/research/unified/archive/2026-10-12/ledger-snapshot/experiments.asof-20261012.jsonl#L424).
E-424 reports learned-route deltas −0.0000543 against uniform and −0.000198 against
hash routing. E-425 is marked completed with no supported gain; its three
conditional paired intervals include zero. Those outcomes are preserved rather
than counted as evidence for the new FFN mechanism.

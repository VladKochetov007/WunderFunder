# Sources, scope and transfer limits

All sources below are primary papers or supplied competition/project documents. They support a method definition or a constraint, not a measured Wunder gain. Search snippets from secondary aggregators were not used for technical conclusions. There was no model/data execution. The repository arxiv-search skill path is a missing worktree gitlink; its tracked SKILL.md was read via Git and primary web search used without installing anything.

## Ground truth for this discussion

- [Supplied data/API/metric contract](../../../docs/data_overview.md):112 anonymous channels, two future indicators,20,000-row sequences,99 warm-up rows, stable channel identities but no documented depth ordering. seq_ix, hidden scoring membership and future labels are not online features. [Canonical scorer](../../../src/wnn/gmetric.py) supplies the unchanged pooled clipped weighted Pearson.
- [Prior25 cards](../../campaign_2026-10-17/research_reset_r1/IDEAS_25.json) and [R2 measured critique](../../campaign_2026-10-17/research_reset_r2/REPORT_RU.md) are used only to avoid rediscovery and invalid inferences. We did not restore checkpoints, rerun old grades or inspect large arrays. Pooled within/between decomposition, generic prefix normalization, covariance orientation, equivariant family encoders, generic i1 lag gates and simple output smoothing are not new ideas.

## Primary literature consulted

| Source | What was checked | Relevance and limit |
|---|---|---|
| [Yu et al., Gradient Surgery, NeurIPS2020](https://papers.nips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) | Full text through methods/theory, including Algorithm1 and its conditions. | Changing conflicting gradient directions is distinct from equal scalar loss weights. Negative cosine alone does not establish harm, and neither theory nor results guarantee a Wunder gain. |
| [Katharopoulos & Fleuret, ICML2018](https://proceedings.mlr.press/v80/katharopoulos18a.html) | Official paper page/abstract; proposer inspected method description. | Importance sampling seeks lower stochastic-gradient variance with correction, not a new target distribution. Recurrent prefix processing and sampler overhead may remove the benefit. |
| [Kapturowski et al., R2D2, ICLR2019](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf) | Indexed primary abstract on representational drift/state staleness. Direct PDF retrieval returned browser verification. | Motivates a state-consistency question; replay-based distributed RL is different from supervised chunk training. No RL performance figure is transferred. |
| [Zaheer et al., Deep Sets](https://arxiv.org/abs/1703.06114) | Proposers inspected set-function description. | Supports set operators, not exchangeability of stable anonymous IDs. Generic equivariance is a duplicate of the old portfolio and is excluded from the new25. |
| [Aitchison,1982](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1982.tb01195.x) | Publisher citation/overview and the scale-invariance of the log-ratio construction. | A possible volume-shape chart only if the relevant features have a valid positive-composition interpretation; zeros/negatives and unknown encodings can defeat it. |
| [Huth & Abergel, high-frequency lead/lag](https://arxiv.org/abs/1111.7103) | Primary abstract. | Other markets exhibit asymmetric variable delays. It establishes neither the direction nor existence of a useful delay in the provided anonymous pair. Generic lag gating is already an old idea. |
| [Miao et al., proxy identification](https://arxiv.org/abs/1609.08816) | Primary abstract and explicit independent-proxy/rank assumptions. | Predictive residualization may be proposed; causal identification is not licensed by merely calling i1 a proxy. |
| [Gneiting & Raftery,2007](https://doi.org/10.1198/016214506000001437) | Abstract/official overview; author PDF retrieval unavailable. | Background on elicited forecast functionals. The weighted-Pearson/MSE proof in ROOT_NOTES is our own task-specific algebra, not a quoted theorem from this paper. |
| [Orthonormal-basis system identification](https://doi.org/10.1016/S0967-0661(02)00113-2) | Proposer cited a system-identification description; root could not retrieve publisher/registry text. | Supporting reading for a Laguerre proposal, not verified empirical support. Retain only the explicitly defined filter-basis hypothesis; do not rely on an unverified application claim. |

## Examined analogies not adopted as ready methods

- [UQ-LOB,2026](https://arxiv.org/abs/2609.31491): primary abstract explicitly conditions on recently realized outcomes and reports selective forecasting. Our callback receives no true outcomes and must predict on every required row. That complete mechanism cannot simply be imported; a label-free uncertainty head would be a different proposal.
- [Tancik et al., Fourier features](https://proceedings.nips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html): inspected primary abstract/introduction. Low-dimensional coordinate fitting does not establish an anonymous financial-feature spectral bottleneck; no automatic Fourier candidate is promoted.
- [Lopez-Paz et al., generalized distillation](https://arxiv.org/abs/1511.03643): primary abstract. Training-only privileged information is a distinct setting, but another KD recipe is not the focus of this session and was not made a selected direction.

New secondary-round literature, if material to a final card, is recorded in its proposal file and must be interpreted at that stated inspection level. None of these papers identifies the methods used by competition leaders.

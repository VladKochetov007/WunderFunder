# Primary sources and transfer limits

Read on 11 November2026 through primary abstract/publication pages. These support
mechanism families; their published benchmark gains are not priors on WunderFunder
WP. No external checkpoint or dataset is admitted by citing a paper.

| Source | Supported idea | Proposed adaptation / limit |
|---|---|---|
| [Gorishniy et al., numerical embeddings](https://arxiv.org/abs/2203.05556) | Piecewise-linear and periodic scalar embeddings can be useful in tabular networks. | Card01 tests continuous train-quantile interpolation on streaming features. Added capacity remains a competing explanation. |
| [Lee et al., Deeply-Supervised Nets](https://arxiv.org/abs/1409.5185) | Companion objectives at intermediate representations. | Card04 uses task regression heads on a causal Transformer; source evidence is from vision. |
| [Lan et al., ALBERT](https://arxiv.org/abs/1909.11942) | Parameter reduction and sharing in deep language models. | Card05 ties block parameters but keeps24 separate streaming caches. No BERT weights, bidirectional mask or NLP score transfer. |
| [Burges et al., RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) | A probabilistic pairwise ranking objective learned by gradient descent. | Card06 is a small auxiliary alongside WMSE; pair concordance is not the competition metric. |
| [Erichson et al., Lipschitz RNN](https://arxiv.org/abs/2006.12070) | Recurrent dynamics and stability can motivate architecture/robustness analysis. | Card08 uses a different empirical perturbation penalty; it inherits none of the paper's stability theorem. |
| [Kendall & Gal, uncertainty](https://arxiv.org/abs/1703.04977) | Input-dependent aleatoric modeling can alter regression objectives. | Card09 is a clipped-label surrogate, not calibrated uncertainty or Bayesian inference. |

Cards02/03/07/10 are concrete team-motivated hypotheses, not attributed to a paper
that did not test them. The repository's historical
[NLL panel](https://github.com/VladKochetov007/WunderFunder/blob/7e4074314a62d0907c9608f9220cf72958708a2d/research/campaign_2026-10-19/full_gru_tweaks_r1/FIRST_CELL_COMPARISON_20261021.md)
reported NLL−Control+0.000738421, conditional unadjusted95% component interval
[-0.000292725,+0.001869540], one exposedF0/seed1001/fixedE5. It used a shared parent
and a different recipe/target lineage. Card09 transfers the **idea**, trains both
arms from scratch and requires no historical parent. It is neither replication
nor proof that NLL helps K16.

Duplicate review covered PR48 and PR55–80 plus older indexed mechanisms. In
particular, [PR8](https://github.com/VladKochetov007/WunderFunder/pull/8) already
proposed grouped encoding. Card02 modernizes that lineage and specifies nonlinear
same-row mixing; it must not be advertised as an untouched idea. Strict-prefix
normalization, PCGrad, direct raw-head bypass, residual distillation, hidden-state
decorrelation and future-input pretraining were excluded from this list because
close proposals/experiments already exist. Novelty is not required for competition
work; clear lineage and a useful unmeasured contrast are required.

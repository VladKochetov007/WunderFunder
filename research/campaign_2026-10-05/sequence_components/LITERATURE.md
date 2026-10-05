# Recurrent models, attention memory, and spline components: 2020–2026

This is a targeted reading list, not an exhaustive list of ICML recurrent-model
papers. Venue labels are checked against primary proceedings or the author's
arXiv page. Long-context language or reinforcement-learning gains are motivation
for experiments here, not evidence of market prediction gain.

| Year/venue | Paper | Useful detail for this campaign |
|---|---|---|
| ICML2020 | [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html) | Kernelized causal attention admits recurrent accumulators. Test a small associative-memory branch without an entire Transformer. |
| ICML2021 | [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) | Additive outer-product storage can collide. Delta-rule overwriting is a distinct candidate, not ruled out by an additive-memory null. |
| ICLR2022, related | [Efficiently Modeling Long Sequences with Structured State Spaces](https://openreview.net/forum?id=uYLFoz1vlAC) | S4 links structured dynamics and efficient sequence processing. It is not an ICML2022 paper. |
| ICML2023 | [Resurrecting Recurrent Neural Networks for Long Sequences](https://proceedings.mlr.press/v202/orvieto23a.html), [arXiv](https://arxiv.org/abs/2303.06349) | LRU's linear/diagonal recurrence, stable parameterization and normalization motivate controlled state design. A real positive-decay filter bank is a simpler first component, not the full complex-valued LRU reproduction. |
| ICML2024 | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://proceedings.mlr.press/v235/dao24a.html) | Mamba-2 connects structured recurrence and attention. This motivates hybrids and implementation-specific cost checks rather than an architecture-name contest. |
| ICML2025 | [Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism](https://proceedings.mlr.press/v267/bick25a.html) | Small selective retrieval roles may complement recurrent compression; applicability to these targets is unmeasured. |
| ICML2025 | [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://proceedings.mlr.press/v267/sun25h.html) | Hidden state can itself be an online learned model. Use causal input-only self-supervision if testing this later; unavailable true test targets cannot update it. |
| ICML2025 | [Understanding and Improving Length Generalization in Recurrent Models](https://proceedings.mlr.press/v267/buitrago25a.html) | Long rollout tests must cover state distribution and stability, not just allocation success. |
| ICML2026 | [Why Linear Recurrent Memory Works in Partially Observable Reinforcement Learning](https://proceedings.mlr.press/v306/zhao26j.html) | Linear memory has a belief-representation justification for particular deterministic/nearly deterministic HMMs. Those assumptions have not been established for this dataset. |
| ICML2026 | [Q-Delta: Beyond Key–Value Associative State Evolution](https://proceedings.mlr.press/v306/park26f.html) | Query-aware corrective memory is another explicit update-rule intervention. Treat it separately from a small positive-kernel readout. |
| ICML2026 | [Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences](https://proceedings.mlr.press/v306/tumma26a.html) | Diagonal preconditioning modifies online regression memory updates. Its language-model evidence does not supply a public-score expectation here. |
| ICLR2025, related | [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756) | Learned univariate spline edge functions suggest a small nonlinear decoder or gate. The representation theorem does not promise generalization or economical streaming inference. |

The arXiv skill helper was used and its search outputs are preserved under
`../checks/ARXIV_*.txt`. Its broad relevance ranking returned unrelated or
secondary title matches for the requested titles. The exact primary arXiv and
proceedings pages above were therefore verified directly; no top-ranked search
hit was substituted for a requested paper.

The immediate experiments are intentionally small and specific: linear memory
with a fixed versus learned spectrum; spline versus nearly equal-capacity MLP
heads; recurrent attention versus static branches of nearly equal size; and
both readouts for the combined memory. Keep nonlinear input embeddings outside
the linear state dynamics. Preserve per-row updates even when observable price
codes stay constant. A later spline write gate, event-ring retrieval,
oscillatory LRU or delta-rule memory is a new intervention with its own control.

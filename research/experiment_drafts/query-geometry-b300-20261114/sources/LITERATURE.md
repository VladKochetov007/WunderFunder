# Geometry reference and claim boundary

Checked 2026-11-14: Su et al., [RoFormer: Enhanced Transformer with Rotary Position
Embedding](https://arxiv.org/abs/2104.09864v5). RoPE represents position through
rotations and creates relative-position dependence in attention. This explains
the native adjacent-pair representation on which the new query operation acts.

The paper does **not** supply this stochastic query regularizer or evidence of
market-data gains. We do not claim to reproduce its experiments, position scaling
or an invariant transformation of targets. No third-party code is copied.
Our paired expectations follow directly from E[s]=0, s²=1, q·Jq=0 and ||Jq||=||q||.
BF16 identities are approximate; a server-only oracle checks the implementation.

This design responds to the team's concrete backlog, not a novelty claim about
all perturbation literature. The pinned [PR audit](PR_AUDIT.md) records its
distinctness from PR99's masking/staleness, noisy GRU state, raw volume scales,
MixStyle and SAM. Neither a paper analogy nor absence of an exact prior PR is
evidence that the new model improves quality.

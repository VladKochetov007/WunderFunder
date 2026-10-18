---
kind: source_review
status: sources_checked_proposal_only
created_utc: 2026-10-18T01:57:09Z
updated_utc: 2026-10-18T01:57:09Z
hypothesis_ids: [H-2026101801]
experiment_ids: [E-2026101801]
evidence_paths:
  - research/unified/sources/competition-writeups/discord_tricks.txt
  - research/experiment_drafts/causal-predictive-state-gru-b300/PR.md
---

# Causal predictive-state sources

This source assessment motivates one small auxiliary objective for a causal
GRU teacher. It records external claims as hypotheses, not competition results.

## Primary literature

- [Kolbeinsson et al., *Self-supervision of wearable sensors time-series data for influenza detection* (arXiv:2112.13755)](https://arxiv.org/abs/2112.13755) studies next-day time-series prediction for wearable-sensor representations. Its abstract reports that next-day resting-heart-rate or time-in-bed objectives yielded useful downstream influenza-like-illness representations in that application. This is direct precedent for testing future-input prediction as representation learning; the data, target and objective differ from our synthetic market-state sequences, so it gives no expected score gain here.
- [Venkatraman et al., *Predictive-State Decoders: Encoding the Future into Recurrent Networks* (arXiv:1709.08520)](https://arxiv.org/abs/1709.08520) adds auxiliary future-observation prediction to recurrent representations and reports improvements in its probabilistic filtering, imitation-learning and reinforcement-learning settings. The local experiment borrows the training pattern only; its empirical results do not transfer directly.
- [Der et al., *A Systematic Evaluation of Generated Time Series and Their Effects in Self-Supervised Pretraining* (arXiv:2408.07869)](https://arxiv.org/abs/2408.07869) reports that many self-supervised time-series models in their studied classification setup were outperformed by simple supervised models, and evaluates generated-data augmentation. This is a useful counterweight: predictive reconstruction can consume compute without helping the task. It supports measuring task WP and preserving a strict matched supervised control.

The papers were rechecked from their arXiv records on 2026-10-18. No weights,
datasets or numerical results were imported.

## Competition anecdote supplied by the user

The imported [competition Discord notes](competition-writeups/discord_tricks.txt),
around lines 246-249, attribute a substantial score gain to a lightweight GRU
trained to predict the next sequence step, whose final hidden state was then
used as Transformer input. The contributor and complete training/evaluation
recipe are not independently verified, and the cited description says it
pretrained on the complete dataset including validation. We therefore treat it
as an idea source only: the present recipe excludes each held-out outer
component from all fitting and normalization, and tests an auxiliary objective
on the task GRU rather than copying the reported pretraining procedure.

## Local interpretation

The fit-time callback can observe state rows but receives no future competition
labels or scoring mask. Label-supervised online updates at inference are not
available. A GRU's recurrent hidden state is updated from each current input,
so causal state learning remains possible. The proposal predicts the next row
of the canonical 211-dimensional model input from each fitting row, normalized
with fitting-only statistics and robust SmoothL1 loss; it then discards this
head during inference. This is a test of representation shaping, not online
parameter adaptation. The input-only test-time adaptation variant remains a
separate hypothesis requiring its own bounded, deterministic serving study.

Related records: [H-2026101801](../ideas/CAUSAL_PREDICTIVE_STATE_GRU_B300_20261018_R1.md),
[E-2026101801](../../experiment_drafts/causal-predictive-state-gru-b300/recipe.json),
and the [draft packet](../../experiment_drafts/causal-predictive-state-gru-b300/README.md).

# E-518 independent evidence review

- Reviewer: `/root/complementarity_card_critic`, native `gpt-6-luna max`.
- Scope: source identity, saved predictions, exact official endpoints, registered
  bootstrap and pass decision. No neural fit; no changes to project files.
- Decision: **artifact integrity PASS; registered scientific screen FAIL**.

The reviewer verified the frozen input identity and population (432 groups,
123 chains, 41 chains per fold; 150/130/152 groups per fold). They independently
fit the registered exact target-0 WP grid without using
`metric_aligned_gate.py` to generate predictions, reconstructed both saved OOF
arrays exactly, and recomputed all fold and pooled official endpoints plus the
2,000 paired complete-chain bootstrap replicates. Source, partition, row-group,
runner, notebook, preregistration, array/file, plot and result hashes matched
the recorded values.

| Quantity | Independent value |
|---|---:|
| Anchor pooled WP | 0.6832264634983685 |
| Gated pooled WP | 0.6832229114911053 |
| Global-control pooled WP | 0.6832256787827535 |
| Gated minus anchor | −0.0000035520072632 |
| Gated minus global control | −0.0000027672916483 |
| Gated minus anchor 95% interval | [−0.0001008220, +0.0000791786] |
| Gated minus global 95% interval | [−0.0000835172, +0.0000707385] |

Fold gate/global betas were `(0,0)`, `(0.01,0.03)`, and `(0.02,0.01)`. Only the
positive-gate-beta-count condition passed. Endpoints, intervals and pass logic
match the stored result. E-518 therefore fails its preregistered gate even
though two folds selected a small positive gate beta.

All source groups had been exposed in E-514/E-517; there is no public-transfer
inference. The reviewer disclosed that execution used the Marimo code-mode
scratchpad to call the registered runner after the run-button setter failed to
schedule its dependent action cell. The attempt receipt records this path. The
independent audit did not execute that path or rely on it to generate the saved
predictions.

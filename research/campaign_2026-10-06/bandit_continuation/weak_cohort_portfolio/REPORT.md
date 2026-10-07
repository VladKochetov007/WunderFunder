# E-513 result: weak-model portfolio as a residual ensemble

> **Post-score correction (2026-10-07):** The reported score replay is descriptive only. The preregistered OOF gate failed because propensity fitting/checkpoint-selection ancestry overlaps the evaluated folds; the first bootstrap seed also deviated from its preregistered rule. Do not interpret this report as clean held-out evidence. Full disposition: [post-score provenance correction](POST_SCORE_PROVENANCE_CORRECTION_20261007.md) and [independent review](INDEPENDENT_REVIEW_MEMO_20261007.md).

Status: cross_fold_scoring_and_chain_bootstrap_complete.

This is a retrospective, exposed-fold prediction analysis. It does not establish a fresh holdout result, a public-score gain, or training-seed robustness.

## Integrity gate

- Frozen source hashes matched: 40/40.
- Full-fold run-card score checks passed: True.
- Pure-chain groups: fold 0 = 432; fold 1 = 399.
- The legacy arrays have no row-order sidecars; alignment is reconstructed from the shared target-cache row order and score reproduction.

## Cross-fold comparisons

| Calibration → evaluation | Cohort lambda | Cohort WP | Δ vs anchor | Paired-chain 95% interval | Δ vs single control | Paired-chain 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| 0 → 1 | 0.40 | 0.675570 | +0.000428 | [-0.000249, +0.001146] | -0.000031 | [-0.000433, +0.000368] |
| 1 → 0 | 0.35 | 0.677387 | +0.000533 | [-0.000598, +0.001531] | +0.000213 | [-0.000341, +0.000859] |

Both cohort point deltas versus the anchor are positive, but both intervals include zero. The cohort does not beat the single-member control consistently: its point delta is slightly negative in one direction and positive in the other, with both intervals including zero.

The selected cohort strengths, 0.35 and 0.40, sit at the top of the frozen grid. This is a useful interaction signal, not permission to extend the grid on these already-used folds.

## Standalone strength and averaging

| Pure fold | Anchor | Basehead | TD target | Prop-v2 | Equal weak mean | Mean member standalone WP |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.676854 | 0.662881 | 0.673658 | 0.675231 | 0.675890 | 0.670590 |
| 1 | 0.675143 | 0.663134 | 0.672033 | 0.671667 | 0.673934 | 0.668945 |

The fixed weak mean is stronger than the arithmetic mean of member standalone scores on both folds. That supports studying portfolio averaging as a mechanism, while the anchor comparison remains uncertain.

## Bandit-tree decision

Keep H-201 at insufficient evidence. Stop allocation to this exact fixed cohort and blend on these exposed folds. Retain the broader weak-model portfolio family for a materially different candidate cohort, teacher target, or representation contribution. The next confirmation must be separately preregistered and use evaluation units not selected from these same results.

E-513 does not change the current best public score and is not a submission candidate. Its paired bootstrap is conditional on fixed predictions and does not estimate training-seed variation or public transfer.

The bootstrap helper source was not included in the original 40-file pre-score manifest. The official point scorer was hash-bound and passed run-card reproduction; the exact bootstrap and notebook source hashes are captured in POST_SCORE_CODE_RECEIPT_E513.json and should be replay-checked before promotion.

## Figures

- [Standalone strength](plots/standalone_strength.png)
- [Cross-fold blend curves](plots/crossfold_lambda.png)
- [Weak residual correlation](plots/weak_residual_correlation.png)

Notebook: [weak_cohort_eda.py](weak_cohort_eda.py).

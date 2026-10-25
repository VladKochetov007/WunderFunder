# B300 diagnostic validation audit

This is a static source-and-receipt review of the earlier B300 epoch diagnostics. I did not rerun evaluation, import project modules, run tests, or execute inference. The local evidence bundle is `/Users/alfa/Documents/WunderFunder_audits/pr32-diagnostics-20261025/`; its evaluator pins source commit `f1beeb4c210a9da01735e03b1ff6e27be878b873` in `operator/evaluate.py`. The bundle contains result and independent-check receipts, but no prediction, target, or mask `.npy` arrays, so I could not independently recompute metrics from the arrays.

## Epoch labels and reported values

The apparent disagreement is a zero-based versus one-based epoch label: `checkpoint_epoch` is zero-based, while `completed_epochs` is the number of epochs completed. The exact receipts agree on `official_metric_id=global_wp_clip2_v1`; within each fold, they use the same heldout selected-row count. All listed checkpoints say `checkpoint_is_final=false` and `selection_policy="diagnostic only; original fixed epoch30 endpoint unchanged"`.

The underlying identities also stay fixed within each fold: F0 has split SHA `15851c8055806566ff326793073cf24706b1c684c5bccd623302c999d31be041` and heldout mask SHA `b3c67d6644094842581374fe1c71b92a3c05a87b4f2f25bec4b51a2b7e73ea1a`; F2 has split SHA `9fff6617d7f701735c2edee57945d33d6e3cce3573197cd7980b927b8b504fa4` and heldout mask SHA `3946ce2b01ee1d50e7813ed9a38a07e1be7b5d1041974c5f1d361bf2d56a5b70`.

| Fold | `completed_epochs` | `checkpoint_epoch` | Heldout official WP | Selected rows / sequences | Receipt relative to `validation/evaluations/` |
|---|---:|---:|---:|---:|---|
| F0 | 16 | 15 | 0.3848928049 | 808,863 / 468 | `t1792_f0/inode-15068161085-mtime-1792869037876840682/RESULT.json` |
| F0 | 17 | 16 | 0.3350959717 | 808,863 / 468 | `t1792_f0/inode-15068102683-mtime-1792873385119926767/RESULT.json` |
| F0 | 21 | 20 | 0.2671175970 | 808,863 / 468 | `t1792_f0/inode-15068102693-mtime-1792890766663005350/RESULT.json` |
| F2 | 15 | 14 | 0.4140200077 | 903,071 / 443 | `t1792_f2/inode-15588523038-mtime-1792864853336902598/RESULT.json` |
| F2 | 16 | 15 | 0.3863164643 | 903,071 / 443 | `t1792_f2/inode-15588529602-mtime-1792869201368226659/RESULT.json` |
| F2 | 21 | 20 | 0.2831396272 | 903,071 / 443 | `t1792_f2/inode-15589793300-mtime-1792890927811454345/RESULT.json` |

For example, “F0 epoch 16: 0.3849” means `completed_epochs=16`, whose receipt has `checkpoint_epoch=15`; it is not the receipt with `checkpoint_epoch=16` (`completed_epochs=17`, WP 0.3351). The same convention explains F2: completed epoch 15 is WP 0.4140, while completed epoch 16 is WP 0.3863. These are not metric or mask discrepancies.

For each listed receipt, a corresponding `INDEPENDENT_METRIC_CHECK.json` in the same directory records `status="verified"` and recomputation values matching its result to rounding precision. This is evidence that the captured verifier completed, not a fresh recomputation by this audit; the arrays needed to repeat that check are absent from the local bundle.

## What the source supports

At the pinned source, `sequence_data.py` builds fitting membership from the split's train and valid fit groups, reserves its valid evaluation groups, and computes input normalization from fitting members only. The normalization reader requests features without targets or masks. `sequence_score.py` requires exact coverage of the split's evaluation groups and scores each heldout sequence with its scored mask after clearing warmup rows 0–98. `gmetric.py` identifies the official metric as `global_wp_clip2_v1`, clips selected predictions and targets to `[-2, 2]`, weights by absolute clipped targets, and pools selected rows globally.

The diagnostic evaluator follows the same warmup boundary, but distinguishes masks by phase: its train probe computes WP on all rows after warmup, while heldout WP uses the heldout scored mask. Heldout `weighted_mse_all_after_warmup` also uses all post-warmup rows, and `weighted_mse_scored` is recorded separately. Train-probe WP and heldout official WP therefore are not an apples-to-apples train/validation comparison; likewise, the two heldout WMSE fields answer different mask questions.

## Finding and limit

I found no static evidence in the audited path that fitting rows enter heldout scoring or that these receipts use a different official metric. Repeatedly measuring the same outer heldout fold across epochs does expose that fold to checkpoint-selection decisions: the receipts correctly label the series diagnostic-only and leave the fixed epoch-30 endpoint unchanged. The curves should not be used to choose a new checkpoint and then described as an untouched validation result. This review establishes source/receipt consistency only; because the prediction arrays are not in the local bundle and no evaluation was rerun, it is not an independent reproduction or proof that every original execution was error-free.

Key source paths at the pinned commit: `src/wnn/training/sequence_data.py`, `src/wnn/training/sequence_score.py`, and `src/wnn/gmetric.py`. Diagnostic implementation and verifier copies are in the bundle at `operator/evaluate.py`, `operator/metrics.py`, and `operator/verify_metrics.py`.

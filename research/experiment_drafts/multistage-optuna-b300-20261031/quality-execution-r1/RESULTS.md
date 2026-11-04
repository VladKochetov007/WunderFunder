# Completed batch/TBPTT screen and TL3 mixtures

Five fresh fixed-epoch4 fits, exact reused A/1301, all five Source6 exports and
all17 final rows completed. No cells or composites are missing. Delivery finished
at 2026-10-31 16:04:40 UTC (82.1 minutes after the first profile).
Foreign NIER/MPS were preserved; all owned GPU-producing commands exited.

| Comparison | Full official-valid WP | Delta vs TL3 |
|---|---:|---:|
| TL3_replayed | 0.692377425 | +0.000000000 |
| TL3_plus_previous_LSTM_0.10 | 0.692502063 | +0.000124638 |
| A1301 | 0.682425941 | -0.009951484 |
| A1302 | 0.682175669 | -0.010201756 |
| B1301 | 0.681557570 | -0.010819855 |
| B1302 | 0.681648999 | -0.010728426 |
| C1301 | 0.682238823 | -0.010138601 |
| C1302 | 0.680794173 | -0.011583252 |
| A_seed_mean | 0.683351603 | -0.009025822 |
| TL3_plus_A_seed_mean_0.05 | 0.692347330 | -0.000030095 |
| TL3_plus_A_seed_mean_0.10 | 0.692266369 | -0.000111056 |
| B_seed_mean | 0.683193608 | -0.009183817 |
| TL3_plus_B_seed_mean_0.05 | 0.692335076 | -0.000042349 |
| TL3_plus_B_seed_mean_0.10 | 0.692241407 | -0.000136018 |
| C_seed_mean | 0.683245345 | -0.009132080 |
| TL3_plus_C_seed_mean_0.05 | 0.692357940 | -0.000019485 |
| TL3_plus_C_seed_mean_0.10 | 0.692285358 | -0.000092067 |

The three seed-mean scores above are scores of averaged predictions, not means
of the individual WP values. Coefficients are fixed; no primary-valid fitting.

## What this screen establishes

- B versus A is lower at both matched seeds: -0.000868371 and -0.000526670.
  At four epochs the joint smaller-batch/more-update recipe does not improve
  this GRU. It costs about69minutes per fit versus24 for A on the shared node.
- C versus B is mixed: +0.000681254 and -0.000854826. About52-54minutes per fit
  does not establish a useful longer-TBPTT effect with only two seeds.
- All six fixed TL3 additions of the new equal-seed means are negative. None
  supports replacing TL3 or adding this new GRU pool at the tested weights.
- The separate cheap prior-model panel has best observed90%TL3 +10%previous
  LSTM at0.692502063, delta+0.000124638. This is a previously completed8E LSTM,
  not one of the five new GRUs. Its contribution is plausible but unconfirmed.
  Its posthoc conditional95% interval crosses zero; the complete descriptive
  [sensitivity panel](sensitivity/README.md) includes all six tested mixtures.
- Historical90%TL3 +10%GALAM was0.692496447 and was not rescored in this screen.
  The roughly0.0000056 numerical difference does not establish superiority over
  that exposed historical candidate.

## Bounds and evidence

Train10,325/valid0 ancestry; full1,873 valid sequences and3,528,273 scored rows.
No random primary folds, shuffled seq_ix chronology, learned primary weights,
changed epochs or additional scientific fits. All endpoints use strictFP32.
The final comparison panel was frozen after early curves, cheap mixtures and
A/1302 native epoch4 were known; see the additive [exposure clarification](inputs/panel-exposure-addendum-r1.json).
Valid remains development-exposed. No significance or public/private forecast.

The scorer compares canonical global WP with separately implemented raw-label
NumPy arithmetic within1e-10 on every final row. Independent reviewer checks
source/config/checkpoint/split bindings and interpretation. Prediction arrays
remain on the node; the local review cannot independently rehash those arrays.
Full Parquet SHA is not recomputed by this final scorer; prior full-file transport
verification, footer/selected-label hashes and mask/order identities are retained.

[Final report](final/REPORT.json), [CSV](final/COMPARISON.csv),
[train/valid curves](final/train_valid_curves.png),
[process and delivery audit](audit/ROOT_TERMINAL_SUMMARY.json),
[root archive verification](audit/LOCAL_ARCHIVE_VERIFICATION.json).

[Results posted in PR53](https://github.com/VladKochetov007/WunderFunder/pull/53#issuecomment-6545810833).
[Five configured-final model weights](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr53-quality-horizon-checkpoints-20261031-r1)
include native child/terminal/epoch metadata and hashes. All three downloaded
GitHub assets and all63 internal files were independently verified by root;
each model SHA matches its actual Source6 score record. Exact reused A/1301
remains in the previous release. No partial or validation-selected model is
silently substituted for these five configured-final endpoints.

Dashboard: http://127.0.0.1:18777/ ; five qh-* views retain the actual four-epoch
trajectories. Train diagnostic uses16 fitting sequences/all-after-warmup rows;
valid uses1,873 sequences/is_scored. Their absolute gap is not an overfit test.

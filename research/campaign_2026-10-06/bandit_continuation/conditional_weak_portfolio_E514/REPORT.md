# E-514 result — weak portfolio and conditional-signal screen

Status: completed exploratory chain cross-fit on previously inspected fold-0 groups.
This is not fresh confirmation, a public-score forecast, or a submission decision.

Source manifest SHA-256: `8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd`.
Pure fold-0 sample: 432 groups across 123 connected chains; each chain was held out once.

| Strategy | OOF global WP | Delta vs C1 pair | Chain-bootstrap 95% interval vs anchor |
|---|---:|---:|---:|
| weak equal | 0.683226 | +0.000000 | [+0.000000, +0.000000] |
| least correlated | 0.683226 | +0.000000 | [+0.000000, +0.000000] |
| conditional signal | 0.683226 | +0.000000 | [+0.000000, +0.000000] |
| best single | 0.683226 | +0.000000 | [+0.000000, +0.000000] |

Member selection and lambda were fit on the other two outer chain folds. The equal weak mean is `ab12`, `ab48`, and `baseline×0.25` averaged at the prediction level. The official metric is used for every calibration and evaluation comparison.

The paired bootstrap is conditional on fold-specific member/weight choices. It does not account for candidate selection, model training, or public-transfer uncertainty. The calibration groups and the candidate pool have prior exposure.

The correlation and standalone-score plots are descriptive summaries of the exposed fold. They do not choose a post-hoc winner.

Figures: [prediction correlation](plots/prediction_correlation.png), [OOF deltas](plots/oof_delta_vs_anchor.png), [member selection](plots/member_selection_frequency.png).

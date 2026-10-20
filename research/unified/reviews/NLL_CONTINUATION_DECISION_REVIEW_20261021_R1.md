---
kind: review
status: complete
created_utc: 2026-10-20T22:22:00Z
updated_utc: 2026-10-20T22:22:00Z
hypothesis_ids: [H-NLL-STRONG-GRU-20261020-R1]
experiment_ids: [E-OVERNIGHT-NLL-ADAPTATION-R1]
evidence_paths:
  - research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/score_plan_r1.json
  - research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/FOLD_ASSET_CATALOG_R1.md
---

# Independent review of the NLL continuation decision

## Question or claim

Which completed evidence should determine a second-fold fit, final all-data
training, or submission packaging after the running WMSE/NLL comparison?
Independent reviewer: native `overnight_evidence_audit`, gpt-6-luna/max,
metadata and source inspection only. The controller condensed the returned
review here; no new scoring, model execution or training was performed.

## Mechanism and method

The frozen screen has seven endpoints on 468 valid-source groups and
141 components. The other 1,405 valid groups are not scored. It uses
`global_wp_clip2_v1` and 2,000 paired component-bootstrap replicates, with fixed
checkpoints and blend coefficients. Intervals condition on fitted predictions;
F0 is exposed and this screen does not measure fitting-seed uncertainty.

## Evidence and interpretation

The control has completed fitting; NLL remains live and no grade was available
to this reviewer. A useful gain against a predeclared matched reference would
support the conditional F1 WMSE/NLL pair before production training. This adds
fold robustness evidence. It does not create a fresh holdout or disentangle
fitting seed from fold. Weak or uncertain effects require allocation judgment;
an interval crossing zero is not an automatic rejection.

TL3 is absent from the seven endpoints. Its historical F0 diagnostic is
0.7018181213 on the same panel, but exact teacher and initializer membership
are unknown and package metadata identifies all-data parents. It cannot serve
as a clean comparator. The seven-endpoint grade supplies no paired TL3 interval,
and no local delta should be added to its public score.

## Limits and validity

The available all-data asset
`bd26_h230_all_data_student_gru256_seed3_history_swap_v2@ep1` is a historical
AMP H256×2 WMSE model with 31 output rows, fitted on all supplied data. Its
cache combines clean-fold sources and an all-data history parent, explicitly
without a held-out claim. It is neither a clean comparator nor an exact
production counterpart to this 25-output strict-FP32 adaptation. A production
recipe must declare these differences or construct compatible targets.

## Decision and next step

Complete the frozen exports and grade first. If they warrant further training,
freeze the second-fold recipe and reserve its complete execution cost. Select
the production recipe and checkpoint rule from available fold evidence.

Before releasing the selected ZIP, verify checkpoint/provenance and exports,
build the required AVX2 library, and run ABI, Python3.11 callback/reset and
source-qualified official-metric replay on the actual ZIP. Quantized quality
and runtime must be measured on that endpoint. No public forecast is required
to package a locally supported candidate. This review admits no new jobs.

## Source records

- [Frozen NLL screen](../../campaign_2026-10-20/submission_delivery_r1/nll_adaptation/README.md): source and recipe bindings, fitting progress and seven endpoints; predictive grades pending at review.
- [Fold/production asset catalog](../../campaign_2026-10-20/submission_delivery_r1/nll_adaptation/FOLD_ASSET_CATALOG_R1.md): metadata-only ancestry, shape and cache inventory.
- [Native command manifest](../../campaign_2026-10-20/submission_delivery_r1/native_preparation/COMMAND_MANIFEST.md): prepared package and replay commands, with outstanding execution gates.
- [Historical TL3 diagnostic](../../campaign_2026-10-18/online_features_r1/tl3_diagnostics_r1/README.md): exposed evaluation with unknown exact ancestry.

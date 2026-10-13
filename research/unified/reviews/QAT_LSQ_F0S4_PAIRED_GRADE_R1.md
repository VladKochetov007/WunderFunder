# 10% LSQ-inspired learned-scale QAT, seed-4 matched deployment grade R1

## Finding

The matched seed-4 F0 pair gives a positive learned-minus-control full-fold WP
delta of `+0.00049264`: learned row steps score `0.68212533`, versus
`0.68163269` for nonlearned per-update absmax row steps. The paired 10,000-draw
component-bootstrap interval is `[-0.00016179, +0.00124463]`; it crosses zero.
This is a second positive, uncertain F0 result, not a quality promotion or
evidence of public transfer.

| Pair (learned minus nonlearned per-update absmax) | Control WP | Learned WP | Delta WP | Descriptive 95% paired interval |
|---|---:|---:|---:|---:|
| Seed 3 | 0.68129828 | 0.68149242 | +0.00019414 | [-0.00015310, +0.00053581] |
| Seed 4 | 0.68163269 | 0.68212533 | +0.00049264 | [-0.00016179, +0.00124463] |

The two deltas are useful paired seed/parent context (`n=2`), not a pooled
training-seed estimate: each seed has its own CP5 parent and fit RNG, while
both pairs intentionally reuse the same 315-component fitting membership and
the same already-exposed F0 evaluation panel. The S4 manifest has experiment
ID `E-QAT-LSQ-10PCT-F0S4-20261013-R1`; reusing S3 membership was deliberate,
not accidental ancestry. Bootstrap intervals condition on fixed predictions
and do not estimate training variation.

## Match and endpoint audit

Both S4 arms used frozen training source `6dde7d9d268475102849fda43b31b8e42162b3c8`
and the seed-4 clean-T3 F0 CP5 initializer (checkpoint SHA256
`e6be82ff799f4ea2b62e770e9c340a8cfd8f98780d1c7dec4a76eab5a853d35c`). Their
recorded fit arguments differ only in run name and `qat_learn_scales` (`false`
for control, `true` for treatment). Both completed 1,440 optimizer updates in
72 batches, with no skipped updates or EMA updates. They share the frozen
10%-subset manifest (`bfa06787…1d40a`) and selected-member digest
(`254723b3…b1c30`), F0 split, 1,048/104 ordered train/validation sequence
membership, 211-feature normalization, sampler/sample-order state, propensity
source and homogeneous strong-teacher cache metadata. The teacher-cache
coefficients are 0.1 each for the same three epoch-2 strong teachers; the
cache metadata SHA256 is `1e4c5c49…e3dc8`. The detailed identities and exact
hashes are in the [fit and representation audit](receipts/QAT_LSQ_F0S4_MATCHED_GRADE_AUDIT_R1.json)
and frozen S4 [execution recipe](../../campaign_2026-10-13/qat_renewable_r1/lsq_f0_s4_r1/EXECUTION_R1.json).

The independent grade checks all 468 ordered F0 groups, 141 source-qualified
components and 808,863 selected rows. The official metric is
`global_wp_clip2_v1`; its unchanged native-Q8 anchor manifest is
`8894af70…19556`. Both result scores were recomputed from their stored
component sums. The separate audit verified 11 deployment source files, all
four archive files per endpoint, all 12 encoded code/scale fields against
checkpoint metadata, and all 16 effective tensors. No Torch or prediction
arrays were loaded for grading. The learned export labels its scale treatment
LSQ-inspired, not a full canonical LSQ reproduction; the control recomputes
detached absmax row steps after each update and is not a fixed-initial-scale
control.

The actual [fixed result](../../../artifacts/qat_renewable_r1/lsq_s4_10pct_fixed_deployment_r1/recurrent_w4/result.json),
[learned result](../../../artifacts/qat_renewable_r1/lsq_s4_10pct_learned_deployment_r1/recurrent_w4/result.json),
and their [fixed manifest](../../../artifacts/qat_renewable_r1/lsq_s4_10pct_fixed_deployment_r1/manifest.json)
and [learned manifest](../../../artifacts/qat_renewable_r1/lsq_s4_10pct_learned_deployment_r1/manifest.json)
are the scored endpoints. The corresponding [fixed fit provenance](../../../artifacts/runs/qat_lsq_f0_s4_10pct_fixed_r1/provenance.json)
and [learned fit provenance](../../../artifacts/runs/qat_lsq_f0_s4_10pct_learned_r1/provenance.json)
bind the training executions. The [S4 profile-recipe review](QAT_LSQ_F0S4_PROFILE_RECIPE_REVIEW_R1.md)
records the hash of the frozen execution recipe and seed-4 input-identity
packet.

The [frozen grade receipt](receipts/QAT_LSQ_F0S4_PAIRED_GRADE_R1.json) and
[fit/export audit receipt](receipts/QAT_LSQ_F0S4_MATCHED_GRADE_AUDIT_R1.json)
preserve endpoint hashes and checks. The seed-3 counterpart is the
[earlier paired grade](QAT_LSQ_10PCT_PAIRED_GRADE_R1.md).

## Limits

F0 is an exposed development fold, not pristine validation. Both experiments
reuse the same source-qualified 10% fitting members and teacher/cache lineage;
the changed parent and seed do not remove exposure. The paired component
bootstrap resamples 141 components with seed `20261013` for 10,000 draws, but
the stored archive has six legacy raw sums rather than centered moments, so
the interval is a descriptive approximation to the exact centered-moment
metric bootstrap. Two positive point deltas justify retaining the mechanism
for portfolio consideration, but do not establish robust benefit, a
cross-fold result, final-package superiority, or a public score.

Experiment `E-QAT-LSQ-10PCT-F0S4-20261013-R1`; hypothesis
`H-QAT-LSQ-20261013`.

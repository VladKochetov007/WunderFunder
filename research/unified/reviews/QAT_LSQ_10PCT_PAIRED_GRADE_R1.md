# 10% LSQ-inspired learned-scale QAT deployment grade R1

## Finding

On the matched 10% seed-3 F0 fits, learning all selected W4 and Q8 row steps
raises full-fold WP by `0.00019414` against the **nonlearned per-update
absmax-row-scale** control. The paired 10,000-replicate component-bootstrap
interval is `[-0.00015310, +0.00053581]`, so the estimate remains uncertain.
The t0 delta is `+0.00045379`; t1 is `-0.00006552`. This is a small positive
point estimate for an LSQ-inspired scale-gradient treatment, not a quality
promotion or reason to kill the mechanism.

The independent grader consumed only the small `component_sums.npz` files and
JSON manifests/results. It did not read `predictions.npy`, load Torch or run a
model. The [frozen grader receipt](receipts/QAT_LSQ_10PCT_PAIRED_GRADE_R1.json)
and [provenance/export audit](receipts/QAT_LSQ_10PCT_MATCHED_GRADE_AUDIT_R1.json)
preserve the paired draws, input hashes, common panel and fit matching checks.

| Contrast, candidate minus reference | Reference WP | Candidate WP | Delta WP | 95% paired interval |
|---|---:|---:|---:|---:|
| **Primary:** learned QAT minus matched nonlearned-per-update-absmax QAT | 0.68129828 | 0.68149242 | +0.00019414 | [-0.00015310, +0.00053581] |
| Context: learned QAT minus 10% FP32 continuation, then W4 PTQ | 0.68141321 | 0.68149242 | +0.00007921 | [-0.00072317, +0.00084536] |
| Context: learned QAT minus original 10% FP32 continuation/native-Q8 endpoint | 0.68181850 | 0.68149242 | -0.00032608 | [-0.00071591, +0.00004432] |

Only the first row isolates learned versus nonlearned scale updates under the
same fit source, initializer and sample order. The two floating-endpoint rows
are deployment context, not additional matched LSQ controls. The full-fold
component totals exactly reproduce each stored WP. All comparisons use the
same exposed F0 panel: 468 ordered groups, 141 connected components and
808,863 selected rows, with the unchanged R2 native-Q8 endpoint as the mask and
target-statistic anchor. The frozen grade receipt binds baseline manifest
`8894af70…19556` and official metric source SHA256
`2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa`.

## Match and export checks

Both arms use source revision `6dde7d9d268475102849fda43b31b8e42162b3c8`
and the same frozen trainer/source identity. The shared parent is the clean-T3
F0 seed-3 CP5 checkpoint, SHA256
`e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`; both
actual initial checkpoint files were streamed and matched their hashes.
Provenance comparison found the full CLI arguments identical apart from run
name and `qat_learn_scales` (`false` in control, `true` in treatment). Both
completed 1,440 optimizer updates in 72 batches, with zero skipped updates,
raw epoch-0 selection, no EMA/SWA, FP32 AdamW (`lr=1e-4`, `wd=0.01`) and
OneCycle.

The fit pools have the same ordered membership hashes (1,048 train and 104
validation sequences), the same 10% subset manifest and members, F0 split
manifest, 211-element normalization, and sample-order RNG digest
`023e9eb3…de4bc9`. Both use the same homogeneous H390 teacher-cache metadata
(`1e4c5c49…e3dc8`) with strong-family teacher coefficients 0.1/0.1/0.1, the
same verified cache files and propensity checkpoint (`c20260929_prop_f0`,
power 2, epsilon 0.03; checkpoint SHA256
`b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7`). The
full split and subset manifest byte hashes are in the machine audit.

The learned export manifest identifies
`lsq_inspired_per_output_row_step_sizes_v1` and explicitly says it is not full
canonical LSQ: latent weights keep the project's identity STE. Its recurrent
W4 preset learns row steps for all four GRU matrices and both Q8 blocks of
`head.0.weight`. The fixed control recomputes detached absmax steps after
weight updates; it does not hold the initial numeric scales constant. Without
loading model tensors, I checked all 12 encoded code/scale payload hashes in
each archive against its manifest. All matched. The six learned row-scale
arrays differ from the control at every row. The fixed archive also exactly
reproduces the prior uniform-W4 artifact hash, which supports control
reproducibility but is not an independent training replication.

## Limits and interpretation

The bootstrap resamples the 141 connected components with seed `20261013` for
10,000 paired replicates. It applies the official metric to the stored six raw
sums, so its intervals approximate the centered-moment metric bootstrap. F0
has prior exposure and this is one fitting seed on a 10% adaptation subset; it
does not establish cross-seed/fold robustness, pristine-CV performance, public
transfer or a stronger final package. The single fit logs report 152.9 seconds
for fixed and 164.9 seconds for learned, a small environment-specific cost
observation rather than a throughput benchmark.

I recommend one small matched confirmation on another source-qualified fold
and fitting seed after the current root-owned GPU cohort releases capacity.
The [allocation note](QAT_LSQ_SCALE_CONFIRMATION_ALLOCATION_R1.md) is separate
from this grade and grants no compute admission.

Related records: [LSQ recipe/code review](QAT_LSQ_ROW_SCALE_RECIPE_REVIEW_R1.md),
[paired fit completion](../../campaign_2026-10-13/qat_renewable_r1/LSQ_PAIRED_FIT_COMPLETION_R1.json),
[profile acceptance](../../campaign_2026-10-13/qat_renewable_r1/LSQ_PROFILE_ACCEPTANCE_R1.json),
[fixed archive reproduction](../../campaign_2026-10-13/qat_renewable_r1/LSQ_FIXED_ARCHIVE_REPRODUCTION_R1.json),
and [exact-export recovery](../../campaign_2026-10-13/qat_renewable_r1/LSQ_EXPORT_RECOVERY_R2.json).
Experiment `E-QAT-LSQ-10PCT-F0S3-20261013-R1`, hypothesis
`H-QAT-LSQ-20261013`.

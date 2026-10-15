# F0 student-frontier H192×2 grade review R1

This read-only review checks the terminal H192×2 fixed-epoch-0 grade and its fit/grade identity against the matched H256×2 seed-3 control. It reads only small JSON provenance and receipts; it does not load checkpoints or prediction arrays, run a numerical regrade, or call a provider.

## Identity and matched recipe

The [H192×2 grade report](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h192_l2_r2_fold0.json) has SHA-256 `7b24fccc5311202bc5bf861d08c1d9045acc0576f918f71aefaf12a1273f50be`; the [grade identity validation](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h192_l2_r2_fold0_validation.json) has SHA-256 `ba776bc3996452e45d8af2e873567e184dc3d3c60fe3e980dba914913aae7538` and reports pass. Both bind the same F0 split hash, valid-target hash, 468 evaluation groups, official metric `global_wp_clip2_v1`, and H256×2 reference.

I compared the H192 and H256 run provenance and fit-validation JSON. Serialized training arguments match after removing run name, hidden width and layer count. Both runs use seed 3, one FP32 epoch, fixed checkpoint epoch 0, B8/C2000, homogeneous teacher targets with alpha `.9`, the same clean-F0 cache and source contract, propensity source, split and sample-order digests, and the same H256×2 epoch-5 normalization buffers. Both record 11,936 sequence exposures, four dropped tail sequences, 14,920 attempted and completed updates, zero AMP skips, zero recorded nonfinite-gradient steps, and 16 checked finite terminal tensors. The H192 fit receipt SHA-256 is `fbbff7fa0ba80112f73a244863417a030901aa2b519ae9c3a4737ce7a4e06018`; the matched H256 fit receipt is `1042e31a73456edd88f40a8bd03b11c91ff346907ace16c50abc10b58fecdf43`.

The fit receipts explicitly do not serialize per-step loss finiteness and perform no quality evaluation. The grade report binds the candidate checkpoint and full-valid prediction hash, and the comparison uses the same 808,863 selected rows and row-group identity as its reference. This is artifact/receipt identity review, not an independent score recomputation.

## Result and limits

H192×2 reports WP **0.6686534**; the matched H256×2 control reports **0.6697776**. The paired component-bootstrap difference is **−0.0011243**, with 95% interval **[−0.0023692, −0.0000700]**, 2,000 resamples over 141 components and 468 sequences. The interval is below zero for this reported exposed F0 panel, indicating H192×2 scored lower than the matched H256×2 control here. It does not estimate variation across training seeds, establish that width 192 is universally worse, or turn this repeated development panel into a fresh holdout. There is no public forecast or public score claim.

**Verdict:** the H192×2 result is valid as a matched, exposed fold-0 development comparison and is numerically below its H256×2 control on this panel. Preserve it as evidence; do not promote it over the control or reject the architecture family from this single seed/fold result. No independent numerical regrade was performed.

The initializer resource profile remains at frozen commit `c85bbc5251372d1dd7701db047a75e3d3147f86a`; its separate [review](INITIALIZER_B1_RESOURCE_PROFILE_REVIEW_20261014_R1.md) records the three launch blockers. No successor packet was present during this access review.

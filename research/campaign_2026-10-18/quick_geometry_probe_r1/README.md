# Quick causal geometry submission probe

Experiment **E-GEOMETRY-QUICK-PROBE-20261018-R1**. The latest user explicitly supersedes the multi-day local R4 allocation with a few local probes, one newly trained exploratory submission ZIP within two hours, and a draft PR for teammate 8×B300 confirmation.

## What runs locally

One paired short continuation of `warm_mechanism_f0_s3_c_r3@ep1`, a strong H256×2 GRU with 211 raw/event inputs and recorded exposed-F0 WP 0.6812265132. Append 60 causal geometry/exact-level channels with zero input weights, mean zero and scale one; the initial function is preserved apart from floating-point kernel rounding. Compare active features with the same-width zero-suffix control. Each arm uses the same seed 3, eligible component-excluded F0 fitting membership and 4,096 complete-sequence draws with replacement, one epoch, AdamW max LR 3e-5, OneCycle, weight decay .01, gradient clip 1, FP32/TF32 off, fixed epoch 0 and no outer-score checkpoint selection. Parent auxiliary construction is inherited. This is a warm-continuation screen, not a new from-scratch comparison or evidence about teacher/student transfer.

The provisional delivery is a single candidate using the feature arm. Its grade is compared to the matched control and parent; exposed F0 and one fitting seed do not establish significance or robust transfer. A negative or uncertain result is retained as evidence about this short recipe. The server owns wider seed/fold comparisons.

## Files and current status

- [recipe.json](recipe.json) freezes actual commands, fitting unit, scope, budget and interpretation.
- [run_pair.py](run_pair.py) executes the two serial fits, then full-F0 held-out prediction exports and official grades, preserving phase logs and return codes.
- [profile_command.json](profile_command.json) and [profile_r1_cli_failure.log](profile_r1_cli_failure.log) preserve a prefit CLI failure: profiling cannot save continuation state; no updates occurred. [Corrected profile command](profile_command_r2.json) removes that incompatible option.
- [profile_r2.log](profile_r2.log) records 60 finite FP32 optimizer updates, median batch wall 920.28 ms for B8/full20k rows, cgroup peak 3,328,434,176 bytes and CUDA allocated/reserved peaks 662,342,656/1,109,393,408 bytes. This short shape profile is not a quality result or guaranteed long-run timing.
- [runtime/](runtime/) contains controller attempt receipts. Initial 5 GB admission was rejected because the projected parent commitment exceeded the 17 GB target; no fit started. R2 reserves 4.5 GB, including over 1.16 GB above measured profile peak, one CPU core and no swap, under the unchanged aggregate two-core parent.
- `progress.jsonl` and `{feature,control}.{train,predict,score}.log` preserve serial execution receipts, training losses/LR/gradient norms, full prediction export logs and official scores as they land. Checkpoints and resume states stay in `artifacts/runs/geometry_quick_f0_s3_{feature,control}/`; predictions and sidecars are in `cache/`. These payload locations are not committed model weights.

## Previous local allocation closed

[The frozen R4 matrix](../../campaign_2026-10-17/geometry_from_scratch_r4/README.md) was stopped at the user's request after roughly 100 batches of the first control and before an epoch checkpoint or quality score. Its logs and controller receipts are preserved, and its admission barrier is closed. It is cancellation for resource scope, not negative scientific evidence.

## Submission and server follow-up

Serving must reconstruct the same 60 causal features and 99 event features, update during warmup, reset between sequences, pass checkpoint/native parity, and fit the ZIP/runtime envelope. Native CPU-policy-D libraries are reused only after checking their actual exported API. Packaging results, measured callback throughput and ZIP hash will be appended here. No public score is predicted from a local delta.

A separate draft PR targets `research-team` with strict FP32 B300 recipes, 30 paired seed/fold cells per scientific arm, component/seed uncertainty, ablations and evidence/weight archive instructions. Its actual URL will be recorded when published.

[October 18 campaign index](../README.md) links this scoped work.

[Independent route review](../../unified/reviews/QUICK_GEOMETRY_ROUTE_REVIEW_20261018_R1.md) verifies the paired design and identifies serving risks before results; it does not approve the eventual artifact.

[User-supplied public reference](public_reference_user_20261018.json) binds the pasted portal names/IDs: tl3 0.66832 in 39m12s, tl4 0.66830 in 50m7s, and CPU-D xteam 0.66808 in 47m27s. It reports 0/5 submissions today and 97 total; timezone and current quota are not independently verified. The new single-GRU probe is not yet expected to beat the mature three-model tl3 ensemble. A fixed contribution to that endpoint is eligible only if actual packaging/runtime and honest evidence support the exploratory release.

## Delivery deadline and first actual endpoint

[Scope/fit receipt](scope_receipt.json) records the hard delivery deadline: **02:50 Kyiv on October18 /23:50UTC October17**. The feature arm completed all5,120optimizer updates (512batches), with finite gradients, in735.1training seconds. Its checkpoint exists, but prediction/grade remain pending until the serial matched control finishes. The full fit averaged1.436seconds/batch including its cold work; the five-batch0.920-second profile underestimated full-fit wall cost. Do not treat short-profile extrapolations as measured end-to-end costs.

[compare_pair.py](compare_pair.py) reuses official stable centered component statistics and paired component bootstrap, aligns producer row-group metadata, and checks shared initializer, parameter, fitting-membership, prior and normalization identities. Its eventual comparison JSON distinguishes feature/control from each child/parent continuation and conditions uncertainty on the fixed fits and exposed F0.

[geometry_prior.json](geometry_prior.json) is the checkpoint/provenance-derived fitted prior for selected raw columns0,3,52,55 and all60 feature names. It contains constants and64source-qualified fitting groups, not raw competition rows. It is required in the submission ZIP.

Hypothesis **H-GEOMETRY-WARM-ZERO-20261018-R1**, child of **H-DISCRETE-OCCUPANCY-20261017-R1**, links this exploratory warm continuation to the earlier discrete/prefix investigation. The mechanism concerns optimization and persistent causal context, with no numerical prior or assumed additive public gain.

## Completed paired result

Both fits, prediction exports and official grades completed at 22:02 UTC on 17 October. Each arm made 5,120 finite optimizer updates, with no nonfinite update or OOM. The feature fit took 735 seconds and the control 718 seconds. The 4.05 GB worker peak stayed within its 4.5 GB reservation.

| Fixed endpoint | Official F0 WP |
|---|---:|
| Original parent | 0.6812265132 |
| Zero-feature continuation | 0.6804784081 |
| Active-feature continuation | 0.6804165153 |

[comparison.json](comparison.json) records identity checks and 2,000 paired component-bootstrap replicates. The active-minus-zero difference was −0.00006189, conditional 95% interval [−0.00012172, −0.00000319]. Both short continuations were below their parent. These are fixed-fit intervals over 141 components / 468 exposed-F0 sequences, not fitting-seed or public-transfer uncertainty. The tested continuation gives no evidence of a gain; it does not settle the feature family.

[quick_probe_results.png](quick_probe_results.png) shows actual training curves and all three conditional score contrasts; [plot_results.py](plot_results.py) reproduces it. The separately identified [frozen-backbone adapter probe](../geometry_adapter_probe_r2/README.md) tests whether preserving the parent avoids continuation drift. No additional local matrix is planned.

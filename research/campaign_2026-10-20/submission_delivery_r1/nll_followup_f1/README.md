# Conditional NLL follow-up: fold 1

**Status: `NOT_ADMITTED`.** This directory freezes a possible two-fit follow-up after the active fold-0 WMSE/NLL pair is independently graded. It does not launch or authorize work. The exact command arrays, source pins, metadata hashes, admission gate, and production-path feasibility notes are in [F1_FOLLOWUP_PLAN_R1.json](F1_FOLLOWUP_PLAN_R1.json).

## Proposed comparison

The prospective panel is one matched WMSE fit and one matched Gaussian NLL fit on fold 1, both from `oldz4_h230_parent_f1_s3_r4@ep3`, fit seed 4, strict FP32, two epochs, fixed zero-based CP1, batch 16, chunk 2000, clip 1, inherited preprocessing, alpha 0.9, and cache scale 1.0. It reuses the recorded eight-source/four-family fold-1 cache and fold-1 propensity. Both arms keep `--loss wmse`; only `--mean-variance-objective` differs. The family head stays off, so this parent is 23 rows and the NLL child is 25 rows after adding the two logvariance rows. The `h230` run/cache names do not describe hidden width; provenance records H256×2.

This is 2 total new fits (1 per arm), no teacher fits, and no exact WMSE control reused from F0. The plan estimates 2.5 hours for both fits plus 45 minutes for exports. F1's 457 evaluation groups across 149 components are previously exposed. Its score is robustness evidence, not a fresh-holdout estimate.

Do not read a F0-to-F1 score difference as an objective effect: F0 uses a different parent family, seed, fold and three-teacher homogeneous cache. Only the within-F1 WMSE/NLL pair is matched. F0's current clipping diagnostics are recorded as optimization diagnostics in the machine-readable plan; they do not change the frozen clip value and are not quality results.

## All-data production route

The most concrete reusable production starting point in the catalog is `bd26_h230_all_data_student_gru256_seed3_history_swap_v2@ep1` with `cache/h230_all_data_history_swap_v2`; provenance records full train/valid membership and an available parent history checkpoint. Its 31-row family-enabled head can become 33 rows by appending two logvariance rows, but this is a distinct production adaptation from the 25-row F0/F1 student.

The recorded all-data history targets and family targets offer the lowest-cost reuse route with zero new teacher fits or exports if their actual bytes and hashes verify. This route retains the existing mixed clean-fold/all-data history ancestry. A closer target-source route can reuse the eight existing fold-1 teacher checkpoints and their frozen coefficients: the current cache covers 10,533 train and 1,416 valid fit groups, so inference must fill 73 excluded train groups and 457 evaluation valid groups before rebuilding full-coverage main and family targets. That requires no teacher fits, but the eight-teacher inference and cache-build cost is unmeasured. Both routes need a reviewed, explicit all-data membership/cache validator: the pinned NLL trainer rejects variant `history`, while its fold cache validator requires clean-fold membership and evaluation-group exclusion. No production code or runnable production command is included here.

## Sources and limits

- [Current NLL adaptation packet](../nll_adaptation/README.md) defines the active F0 experiment and its frozen trainer source.
- [Fold asset catalog](../nll_adaptation/FOLD_ASSET_CATALOG_R1.md) records the fold-1 parent, cache and propensity ancestry and the all-data candidate assets.
- The trainer source is pinned to commit `68ff682977b822f38351aac0f621146f5f13b440` in `/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020`; no trainer source was changed here.
- This packet is recipe work only. No fitting, export, array loading, score, or grader change occurred.

# Clean T3 fold-0 teacher cohort R2

Three fresh H384 GRU teachers use Muon-all, fold-0 own propensity power 2, seeds 3/4/5, 3 epochs and fixed epoch-2 checkpoints. The source snapshot is byte-identical to reviewed R1 `fb32325fa1010615c12a45e67e81edfd18141e288c620a5c73664d45beb397ea`; profile archive `0ba1b1f7b99dc5dabd892c7d99aa030733465aebc73bb652455c9bc527e0d65f`; fold-0 split `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`; own propensity checkpoint `b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7`.

The 10,535 train-fit and 1,405 valid-fit groups exclude all 468 fold-0 valid evaluation groups. Six fit-membership cache exports are separate from three all-1,873-row full-valid feature-only scorer exports.

Private dataset `vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r2` creates title-derived kernel ID `vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r2-runner`. Its embedded admission gate is pending and fails before external mounts or GPU worker launch. No upload or execution was performed by this builder.

The 14,400-second wall and 28,800 two-device GPU-second figures are ceilings, not timing estimates. Actual profile results, fit/export ETA and full-cohort feasibility remain unknown; admission is false until root’s live gates pass.

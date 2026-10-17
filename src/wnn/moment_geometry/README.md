# Causal sequence moment investigation

Reusable research code for the [registered mean/std geometry investigation](../../../research/campaign_2026-10-17/moment_geometry_r1/README.md). Full-sequence descriptive statistics are never prediction inputs.

- [scan.py](scan.py): bounded one-reader raw-feature prefix moments.
- [analysis.py](analysis.py): train-only exploratory arc consensus and descriptive prefix errors.
- [descriptive.py](descriptive.py): all112contact sheets, six selected arcs, raw-support and prefix-path figures.
- [features.py](features.py): reset-local expanding moments, signed residual, frequent-anchor occupancy and normalized current channels; fit constants use fitting membership only.
- [staging.py](staging.py): whole-component membership and disk arrays, reusing existing event_features.
- [provenance.py](provenance.py): approved physical storage, trusted footer/index/cache binding and actual producer checks.
- [backend.py](backend.py): prepared-array adapter to the canonical TrainingRun, model factory, optimizer/scheduler, forward/backward/update, GradStats and checkpoint components.
- [runner.py](runner.py): bounded FP32 CUDA pilot reusing existing RNNModel and weighted_mse_loss; fixed final epoch, paired seeds, matched zero-channel controls.
- [grading.py](grading.py): streaming official metric and explicitly conditional approximate legacy-sum component resampling.

The pilot does not replace the shared production trainer. It isolates a small representation comparison, omitting teacher caches, EMA, continuation and serving machinery intentionally. The [modular trainer](../training/README.md) remains the reusable production path.

- [discrete.py](discrete.py) and [discrete_staging.py](discrete_staging.py): fitting-only top-three-level dictionary plus unknown class, row/event occupancy and dwell; shared physical array staging for the adaptiveR2 comparison.
- [recipe.py](recipe.py) and [run_metadata.py](run_metadata.py): common pilot budgets/channel schema and reproducible per-endpoint metadata; avoid duplicating these in model variants.

- [results.py](results.py): complete outcome/progress collection, fixed-half prediction mixtures and paired-seed/fold/training visualizations. Its output is analysis evidence, not an optimizer or leaderboard-selected blend.
- [online.py](online.py): reset-local causal FP64 moment/count accumulators cast to a fixed 60-channel FP32 feature suffix, fitting-only arc/atom/dictionary priors, and exact indexed 20/40 block controls used by the [B300 experiment draft](../../../research/experiment_drafts/causal-geometry-discrete-b300/README.md). B300 profiles, the 120-fit matrix and native serving checks remain pending.

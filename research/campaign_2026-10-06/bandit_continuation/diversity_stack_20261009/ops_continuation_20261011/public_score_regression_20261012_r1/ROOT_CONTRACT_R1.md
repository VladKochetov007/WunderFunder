# Public-score regression and parallel submission research

Root contract recorded 2026-10-12T00:32:38Z. The user explicitly requests regression of expected public score using the supplied upload history, investigation of both public calibration and out-of-fold stacking, and preservation of inference optimization and submission-quality research. This is bounded analysis within existing authority. ADR0030 still selects submissions from research and valid local evidence; calibration assesses transfer and does not change the official metric or tune blend weights against public labels.

## Input and ownership

The supplied portal excerpt has 42 visible events: 38 completed scores, three timeouts and one check error. The portal's 45-total counter leaves three events unavailable. Preserve literal IDs, filenames, displayed times, outcomes, scores and resource observations; their timezone and score-availability times are unverified. The scored 61m22s event remains a success. Censored events have no score target and are never assigned zero.

Root owns methodology, admission and integration. Luna/max mapper/analyst owns a source-qualified table and bounded computational comparison in an isolated worktree. Root independently reviews the calculations and artifact claims before promotion. All reconstructed features and backtests are retrospective, exposed discovery; preserving a time ordering does not turn reconstructed features into a historical prospective forecast.

Match each upload to receipts, shipped members, fixed weights, precision/backend, fitting membership and teacher/initializer ancestry. Distinguish exact-weight held-out measurements, recipe-sibling proxies, in-sample features and missing features. An available checkpoint or named validation file does not establish held-out treatment. Compare protocols separately unless a declared normalization has defensible coverage. Deduplicate repeated weights and retain distinct seeds as training variation. Blends can connect ancestor clusters; report sensitivity to conservative family grouping.

## Bounded regression comparison

The target is public score conditional on successful serving. Portal runtime, memory, terminal outcome and public score cannot be predictor features available before upload. Model feasibility is evaluated separately using the existing native serving receipts. No numerical ZIP11 forecast may use the mismatched fold-1 H390 prototype as an exact local score.

Start with source coverage. Compare a training-only mean and a recent-five-distinct-artifact mean with a small, fixed-shrinkage regression. Use only source-qualified recipe metadata for the broad history. Evaluate local-score calibration separately on each comparable audited protocol subset; missing local scores do not acquire held-out status through imputation. Family effects may be partially pooled, with unseen families falling back to the intercept and explicit extrapolation uncertainty. Keep the feature set small, freeze it before fitting, and record all fitted variants. No HPO or penalty sweep is authorized by this analysis.

For a standardized design, ridge minimizes `sum_i w_i (y_i - a - x_i b)^2 + lambda ||b||^2`; the intercept is unpenalized, preprocessing and means use training rows only, and the fixed penalty is recorded. A richer model must beat its same-row baseline outside its fitted rows before being used operationally. Report MAE/RMSE and row/group counts, with both conservative family/ancestry exclusions and forward chronological blocks. Keep simultaneous displayed-time events together. A split with insufficient training support is reported rather than repaired using its evaluation rows. Treat selecting the best validation protocol/model as further exposure, and freeze any useful predictor for prospective next uploads.

More subdivisions of existing held-out predictions can provide component-distribution features and sensitivity checks; they do not create new independent uploads, new training folds, or fresh confirmation. Keep complete source-qualified components and fixed masks. Full new-fold models require exclusion of all fitted teacher/initializer/propensity ancestors and separate resource admission. No extra GPU cohort is admitted while the existing Kaggle hold remains unresolved.

OOF stacking remains a separate endpoint experiment: freeze a small pool, fit simple nonnegative simplex weights on complete calibration components, and evaluate on excluded components against fixed equal/family weights using final-blend clipping and original label-magnitude weights. The existing unbounded legacy blend optimizer is not an admitted implementation. A new stacker needs an audited prediction/ancestry pool and a bounded consumer; no model refit is implied by repartitioning existing predictions.

## Parallel work and resource boundaries

Local clean-fold old-pipeline parent/student/HFT work continues through its reviewed dependency gates. Corrected Kaggle R6 version2 remains with its existing cloud operator and terminal collector. Vast B has no owned instance row and zero credit; no paid action is authorized.

The independent exact activation/max-reduction actor tests H-CPU-MAX-01 against the unchanged compact Q8 endpoint. PGO and projection-precompute results remain measured non-promotions; Q4-grid/Q8 loss and FP16-storage evidence retain their panel scopes. An exact shared-backbone student/HFT consumer is a separate promising quality-per-CPU arm, currently unimplemented. Native changes remain isolated from live trainer reservations and production ZIPs until parity, paired timing and independent review support them.

Aggregate caps remain five CPU cores, 18 decimal GB working/20 GB hard RAM, no swap, five decimal GB observed VRAM and one GPU worker. Mapper support uses the MAIN `scripts/limited` wrapper with 25%CPU, one thread, 500MB and 120s per scan/model run. Native compilation/panels use 100%CPU, one thread, 1.2GB, 120s and one Cargo build job. Check live aggregate before each step; no large CPU fits, full parquet/global-cache read, provider mutation, extra spending, source rewrite or competing controller.

## Deliverables and sources

Deliver the complete literal roster, row-level artifact/feature map, missingness and exposure counts, reproducible bounded model results, clear applicability to current candidates, and concrete next data dependencies. Preserve all outcomes and failed computations. Update research notes/index and append-only records; do not overwrite frozen ZIPs or historical receipts.

Primary methodological sources read by root: [scikit-learn grouped and chronological cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html) and [Cawley and Talbot, model-selection overfitting and evaluation bias](https://www.jmlr.org/papers/v11/cawley10a.html). These support the evaluation design, not a public-score claim. Existing official metric, source/provenance tools and empirical-validation guidance remain authoritative for project calculations.

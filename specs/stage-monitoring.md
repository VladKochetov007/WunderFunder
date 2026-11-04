# Monitoring every stage of the solution

Standing user instruction,31October2026: save graphs at every stage to track
useful training moments. This is a future execution requirement, not evidence
that the proposed four-stage Optuna/B300 pipeline is implemented or running.
Existing frozen recipes, results and chronology remain unchanged; no fit restarts.

## Persistent outputs and live view

Every experiment/trial/stage retains its producer-native machine-readable events,
epoch evaluations, configuration, source/split/parent identities and checkpoint
index. The local dashboard must select and compare stage, trial, config and seed.
Export actual-data PNG and SVG plots after each completed evaluation and at stage
termination; preserve the final snapshot and source metrics outside the live UI.
A plot export must not trigger an extra model evaluation or fit.

Axes include optimizer updates/epochs and elapsed wall time. Stage changes and
scheduler resets have explicit boundaries; never join different objectives or
parent models into an unexplained continuous curve. Missing/pruned/failed results
stay explicitly missing rather than becoming zero scores.

## Stage-specific plots

| Stage | Required plots |
|---|---|
|1. Base regressors|Train-eval and control/full-valid WMSE on one chart; train-eval and control/full-valid WP on a separate chart; t0/t1; online composite objective separately; LR by optimizer group, preclip gradient norm and clipping/nonfinite counts.|
|2. Fine-tuning|The same metrics, with parent endpoint/reference, fine-tuning boundary and LR reset. Show old and new loss components separately; raw/EMA/SWA checkpoints have explicit roles when evaluated.|
|3. Distillation|Separate label loss, teacher-target loss and EMA/auxiliary terms; student WP/t0/t1 and fixed teacher/control references; actual teacher-pool coefficients and distillation alpha over training when changed.|
|4. Ensemble/Optuna|Inner-train/OOF objective by trial and best-so-far; candidate status and cost; component/family weights; frozen-finalist official-valid WP and delta to TL3. This stage has trials, not invented training epochs.|

Every fitted stage also retains sampled GPU/CPU utilization, VRAM/RAM, data-wait
and processed-row/update throughput, plus validation/checkpoint/export durations.
Resource samples do not establish peak FLOP utilization. Subsample dense telemetry
with an explicit policy; retain every actual epoch evaluation and stage boundary.

## Comparable curves and endpoint selection

Log each curve's metric, reduction, target, fitting/evaluation membership, row
mask, selected-row count and model role. Our fixed train-eval diagnostic uses
all-after-warmup rows, while official valid uses is_scored. Label the differing
masks prominently: their absolute gap is not a matched-distribution overfit test.
Composite training objective is not validation WMSE. Official train has no
is_scored labels; an inner-train proxy is not the official masked endpoint.

Mark scheduled/final, eligible inner-control-selected and best-observed endpoints
separately, with epoch/update/time, objective and checkpoint SHA256. Minimum WMSE
and maximum WP can be different moments. Preserve original checkpoint bytes and
resumable optimizer/scheduler/RNG state where the recipe requires it.

Monitoring primary valid does not authorize fitting blend weights or selecting
teacher/initializer ancestry from it. Optuna, adaptive stopping and learned weights
use a declared fitting-only control/OOF policy; any genuinely chronological inner
split requires actual order evidence, not seq_ix. Keep OOF teachers, initializers
and normalizers outside their held-out members. Primary-valid best-observed markers
are diagnostic and development-exposed; they never silently replace frozen primary
endpoints or become future ancestors. See [ADR0044](adr/0044-chronological-primary-validation.md).

Each graph links its machine-readable source and checkpoint/recipe identities.
Graph exports, checkpoint manifests and terminal metrics must be collected before
an owned worker is released. Earlier inspected data remains exposed.


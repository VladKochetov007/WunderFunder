# Sequence moment geometry investigation

Scoped local research resumed by the user on October17, directly on main. [Contract](contract.json) freezes resource limits, descriptive scan, causal invariants and exploratory model-selection budget before data outcomes. Parent pause remains for unrelated campaigns. The provided screenshot has unspecified feature/moment-window provenance, so initial work searches all112 inputs. No Kaggle allocation is changed.

Results and reproduction commands will be added with their actual receipts. No gain is claimed at registration.

## First execution failure

The initial scan exited137 before reading sampled groups: its1.4GB child cgroup hit OOM while redundant ParquetFile handles expanded metadata (train footer60,437,285bytes; valid10,686,926bytes). This is an execution failure, no negative scientific evidence. The correction passes one existing reader into the streaming scan, closes it between files and reserves2.8GB for that process within the unchanged5 GB aggregate task slice. Marimo has a separate0.9GB child cap. No data or other jobs were modified.

## Registered inputs and implementation

- [Pilot plan](pilot_plan.json) freezes three arms, three paired fitting seeds, two exposed evaluation slices, fixed final checkpoint and exact optimization settings before target scores.
- [Pilot membership](pilot_membership.json) lists all43 fitting and46 evaluation sequences with source-qualified groups/components. The common fitting pool is not two independent fold fits.
- [Interpretation and method](../../unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) explains bounds, point-mass mixture arcs, all112 visual observations, early-prefix errors, source identity and limitations.
- [Marimo notebook](../../../notebooks/sequence_moment_geometry_20261017.py) provides live feature selection, raw train/valid scatter and distributions, contact sheets, prefix errors and raw traces.
- [Implementation package](../../../src/wnn/moment_geometry/) reuses the existing event features, RNNModel, weighted_mse_loss, component graph and official metric. `scan.py` reads one Parquet reader at a time; `analysis.py` fits descriptive curves; `descriptive.py` reproduces plots/support examples; `features.py` implements causal channels; `staging.py` stages source-qualified arrays; `provenance.py` validates physical scratch and trusted source identities.

## EDA artifacts (input-only, exploratory)

All statistics include warmup/unscored rows, ddof0, sample seed20261017,128 complete sequences per source. No target score is represented here.

- [Moment arrays](scan_r1/moments.npz): per-column prefix means, variances and extrema plus exact sampled groups/sequence IDs. This compact statistics artifact contains no complete raw data payload.
- [Source identities and112-column descriptive fit ranking](scan_r1/scan.json): all results, including poor single-circle fits, not just visually selected successes.
- [Six selected arcs](scan_r1/candidate_arcs.png) and [consensus-fit coefficients/fractions](scan_r1/robust_arcs.json): an adaptive train-only consensus over a variance tolerance0.002; not a predictive validation score or an exact support claim.
- All112 raw-feature mean/std contact sheets [0–27](scan_r1/all_columns_1.png), [28–55](scan_r1/all_columns_2.png), [56–83](scan_r1/all_columns_3.png), [84–111](scan_r1/all_columns_4.png): visually inspected. Price branches recur between instruments; volumes are more clustered; dv* has mean/std coupling; a7 has two branches. Visual overlap is not a distribution-shift test.
- [Actual-support receipt and prefix errors](scan_r1/support_probe.json): six adaptively selected validation examples, their distinct-value counts/frequent values and prefix estimates. Near-arc data are not strictly two-valued.
- [Compact illustrative raw traces](scan_r1/example_traces.npz), [rendered traces](scan_r1/support_traces.png) and [prefix paths](scan_r1/prefix_paths.png): frequent flat background levels and changing activity are visible; early positions can differ substantially from final moments. These illustrations do not estimate regime prevalence.

## Reproduction

Use the existing uv environment and the aggregate5 GB `wnn-campaign-moment.slice` inside the two-core campaign parent. Serialize metadata/staging and fitting, start with one thread, and use physical scratch. Parquet metadata can occupy far more RAM than its60MB footer. The successful scan reserves2.8GB for one reader; raw staging reserves3.2GB. These reservations are within, not additions to, the task5 GB aggregate cap.

```bash
uv run --no-sync python -m wnn.moment_geometry.scan --output PATH_TO_NEW_SCAN
MPLBACKEND=Agg uv run --no-sync python -m wnn.moment_geometry.descriptive --output PATH_TO_COMPLETED_SCAN --support
uv run --no-sync python -m wnn.moment_geometry.staging --output PHYSICAL_SCRATCH --phase membership
uv run --no-sync python -m wnn.moment_geometry.staging --output PHYSICAL_SCRATCH --phase raw
uv run --no-sync python -m wnn.moment_geometry.staging --output PHYSICAL_SCRATCH --phase features
```

The recorded pilot uses the already-frozen membership rather than resampling. Only completed task-owned NPY cache pages were advised DONTNEED before fitting; source arrays were not deleted or modified. Follow the contract and actual resource receipts, not these unwrapped example commands, when admitting a job.

## Input and execution receipts

- [User-provided reference image](reference_screenshot.png) is the original motivating plot; exact feature/window provenance remains unknown, so the i0_p0 match is visual rather than proven.
- [Fitting-only preprocessing constants](preprocessing.json) record arc coefficients, frequent anchors, pooled scalers and exact causal-prefix/suffix invariance; no evaluation rows fit these constants.
- [Source binding and producer checks](source_binding.json) link the trusted original footer/index/cache identities and all89selected seq_ix mappings; an independent source pass checks identifiers, raw p0, both targets and prediction/scored masks.

- [Staged artifact hashes](staged_artifacts.json) pin every disk array and preparation receipt before fitting. Large arrays stay in approved physical scratch. `backend.py` adapts prepared arrays to the canonical TrainingRun/model/optimizer/scheduler/forward/backward/update/checkpoint components; the numerical engine is shared.

## Independent prefit recipe review

Native Luna/max first reviewed the question independently and then the implementation without outcomes. Source/storage pins were repaired; a duplicated numerical pilot loop was replaced with a canonical-component adapter. The reviewer caught the canonical CLI's default mixed loss; the recipe now explicitly sets `--loss wmse`. Final static review finds no concrete prefit blocker:43sequences, batch4,512rows,2epochs give880updates; zero-warmup single-cycle SGDR matches the frozen cosine endpoints; causal recurrent state and post-normalization zero controls match. This is recipe review, not empirical approval.

## Adaptive discrete follow-up

[Discrete pilot R2 plan](discrete_plan_r2.json) was registered after all9 R1 scores and before R2 outcomes. It freezes a fitting-only three-level dictionary plus explicit unknown category, row/event occupancy and dwell. Four matched-capacity arms isolate dictionary-only, moment-only and their interaction at H32×1 and H128×2,3paired seeds and the same2epoch budget. The larger model is a capacity probe, not a validated teacher; no KD experiment is claimed. This is an additional exploratory stage, with reused evaluation exposure and no early-dominance rejection gate.

## First pilot result

- [All9endpoint results](pilot_results_r1.json): three paired fitting seeds, official pooled/t0/t1 scores and both exposed validation slices, exact shared-source identities and initial-state hashes. Control meanWP0.29930670; moment/occupancy0.30019002 (+0.00088332); additionally scaled0.29963501 (+0.00032832). The moment effect is negative on fold0 (−0.00788477) and positive on fold1 (+0.01306417). This tiny-model result is heterogeneous, not a production or leaderboard gain.
- [Training progress](training_progress_r1.jsonl) preserves198records with arm/seed, loss, preclip gradient norm, LR, attempted step/epoch, elapsed time and CUDA peak. Every endpoint completed880updates; peak allocatedCUDA≈167MB. Finite successful pilot execution does not retroactively validate every production trainer branch.
- [Checkpoint/prediction/log manifest](pilot_artifacts_r1.json) pins original payload sizes/hashes and approved physical paths. Weights and predictions remain outside Git, ready for a separate artifact handoff.
- [Frozen source snapshot](frozen_source.json) pins all133Python/dependency/entrypoint source files and main commit; these sources were preserved while the job ran.
- [Fit admission](fit_admission.json) records observedGPUheadroom, the1.16GiBallocator cap,3.5GBchild RAM cap and unchanged5GBaggregate task limit. The otherprojectGPUprocess was untouched.
- [Static HTML EDA](eda.html) exports the marimo plots for offline viewing; the live notebook supports feature selection.
- [Independent numerical grade R1](independent_grade_r1.json) recomputes all9pooled/fold scores with the frozen official accumulator within1e-12, checks source and matched initialization hashes,880update clocks and loss flags. Pooled moment deltas across seeds are+0.00044474,+0.00067988,+0.00152535; sampleSD0.00056831. All conditional2,000-draw component intervals cross zero; they do not include fitting/selection/transfer uncertainty and do not close this direction.

### R2 implementation review

An independent read-only builder review confirms all four channel masks and fit-only dictionary/scaler paths. R1 defaults intentionally stay compatible; everyR2 command explicitly supplies `--arms control scaled dictionary combined`. Endpoint metadata now hashes every geometry source module and every staged JSON receipt (including dictionary/scaler and array manifest). The event-frequency channel means frequency of the **training-most-frequent level** among exact raw changes, and dwell means time since exact raw change even within unknown; these definitions are causal and are kept fixed.

- [R2 fitted dictionary/scaler](dictionary_preprocessing_r2.json) and [R2 staged array hashes](staged_artifacts_r2.json) bind the new40-channel suffix. Prefix-only recomputation and future-suffix mutation agree exactly. Base labels/masks/component IDs are immutable hardlinks to the verifiedR1 data.
- [FrozenR2 source](frozen_source_r2.json) binds the main revision and every copied Python/dependency source used by both capacity fits; subsequent main/research-team merges cannot rewrite the running source.
- [Resource receipt](resource_receipt.json) records actual cgroup limits, charged peak and exactly one initial metadata-reader OOM. Subsequent preparation and first pilots completed without new OOM events; the receipt is a time-bounded snapshot, not a promise about future commands.

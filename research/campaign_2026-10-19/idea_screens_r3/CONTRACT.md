# R3: component coverage, feature blocks and nested fit size

Objective_mode: competition research through bounded discovery. Verification tierC for allocation clues, with deterministic source/metric checks at tierA. The user authorized autonomous EDA/simple models. This is not a full recipe/CP30 comparison or submission. R1/R2 remain immutable completed results.

## Questions fixed before new sample labels or scores

1. Does the52-channel subset advantage persist with more component coverage, when i1 and a0:a7 are separate factorial blocks?
2. Does that contrast change with a nested32-versus64-component fitting set?
3. Do the separately fixed signed/absolute EMA diagnostics from R2 persist on a larger, differently sampled panel?

Metadata-only preflight observed2777 fitting components with eligible train groups and141 evaluation components,468 evaluation groups; [raw metadata](PANEL_METADATA.json). No arrays, labels or masks were read for selection.

## Sample, frozen before arrays

Canonical F0 split SHA c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7. First assert complete source-qualified membership and global fit/eval component disjointness, with no duplicate/missing metadata groups.

Select64 distinct fitting components uniformly by deterministic shuffle(seed19021) from components containing canonical eligible **train** groups. Within each selected component choose one complete eligible train sequence uniformly(seed19022; sorted eligible groups). First32 selected fitting components define the nested low-fit set. No supplied-valid fitting rows in R3. Independently select32 distinct evaluation components by shuffle(seed19023), then one complete eligible valid sequence per component uniformly(seed19024). Read96 complete20000-row sequences. Preserve every selected group, including any zero-scored group; never replace using masks/labels. Do not require every eligible group of a selected component to be loaded: all other groups are unused, and full canonical component separation protects the fit/eval boundary.

This component-first, one-group-per-component sample differs explicitly from R1's greedy complete-membership sample. Record component and conditional group inclusion probabilities; report the unmodified official metric on the realized panel, not an unbiased full-fold WP estimate. It is still exposed F0, not a new pristine holdout. Changed fit source/composition prohibits direct score subtraction from R1/R2. Materialize membership JSON before array reads, verify current size/mtime/footer and registered prior full payload identity without another35GB hash scan, and hash every selected array file. Original evaluation masks and fit warmup masks unchanged. Arrays stay only on B.

## Nine fixed analytic cases

Full64 fitting components: seven views i0(0:52),i0+tail(0:52 and104:112),books104(0:104),raw112,raw+signed EMA,raw+absolute-innovation EMA,raw+both. Same R1 causal half-life32/innovation definitions, reset each full sequence, transformations before stride40. Low32 nested components: only raw112 and i0+tail60, on the identical evaluation panel. No lags, horizons or pole search.

Use unchanged R2 weighted-ridge fit/predict numerical helpers; clip labels±2 and train with abs(clipped label) weights. Alpha grid[1,100,10000], shared by two targets, selected only by three deterministic fitting-component folds(seed19025), fold-train-only normalization, full original held-fit masks, pooled canonical WP, exact ties choose larger alpha. Refit selected alpha on all rows of the applicable fitting budget; retain predeclared alpha1 baseline, with no outer selection. Record full inner table AND per-fold additive raw moments, both targets, component moments, drop-one pooled sensitivity, coefficient norms, all counts/failures/cost. Save compact model coefficients/normalizers as non-pickle NPZ on B with SHA/size; do not copy model arrays to Mac.

Primary descriptive contrasts within the full budget: books104−i0 and raw112−(i0+tail) for i1 inclusion; (i0+tail)−i0 and raw112−books104 for tail inclusion. Same selection protocol, not equal coefficient counts. Nested size contrast includes new components/composition and is not an isolated sample-size causal effect or training-seed replication. EMA results are sensitivity checks, not a new family test. Never select alpha1 because its outer metric is higher.

## Bounded execution

No GPU, inference, quantization, historical restoration, paid provider tests or installations. Separate sample materialization and probe units, sequential, each0.5CPU/1.5GB/no-swap/600s, one BLAS thread, CUDA hidden, physical B/research_tmp. Fresh all-populated-child/F3 ownership/source/collision/disk/headroom gates before each launch; sample requires4GB free. Existing F3 controller.1CPU/512MB plus fit1CPU/8GB preserved under parent2CPU/18GB high/20GB max/zero swap and17GB committed target. Independent source/package import smoke and tiny leakage/metric tests before remote execution. Preserve partial outputs; no cap expansion, outcome exclusions, panel replacement or blind retries. Capture raw bytes/status before stopping only exact own exited units. Stop R3 when sample and nine cases finish or a concrete validity/resource failure prevents them. A nonlinear baseline is a later separately fixed job, not included in this envelope.

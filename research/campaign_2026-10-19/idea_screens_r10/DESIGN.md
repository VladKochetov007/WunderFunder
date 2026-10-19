# R10 draft: H-NF19-08 recent joint-support novelty

**Status:** design frozen before implementation after root review SHA-256 `129b30e6dc41e0af08803d333bc64cf429922692efca24c04e7a1971d2ce002d`. No panel arrays, labels, models, or remote resources were accessed while preparing this design.

## Question and falsifier

For one fixed pair of canonical raw channels, how often is the current pair outside the convex hull of its recent 256-row history while each coordinate remains inside its own recent min/max interval? The exact comparison is a same-window, two-coordinate axis-aligned box versus the convex hull of those 256 points. This screens only the H-NF19-08 geometry premise. It does not test prediction, label association, transfer, or value outside the accepted exposed F0 panel.

The fixed coordinates are raw columns `(0, 52)`, the first entries in the canonical `bid_px` slices for `I0=Instrument(0)` and `I1=Instrument(52)`. They are feature identifiers; this design makes no physical-unit, shared-unit, or market-meaning claim. The coordinate binding is anchored to `src/wnn/data.py` SHA-256 `a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00`.

At each eligible time `t`, define the history as rows `[t-256, t)` and the query as row `t`. Partition each query into exactly one of:

1. `box_outside`: outside either coordinate's past-window min/max interval.
2. `box_inside_hull_outside`: within both coordinate intervals, but outside the two-dimensional convex hull of the same past window.
3. `hull_inside`: inside or on that hull.

The second class is the joint-only novelty of interest. Geometry uses per-window min/range normalization independently by coordinate, so the box and hull share the same window and their classifications are invariant to independent affine rescaling/translation (including reflection). Degenerate windows are explicit: a point hull contains only that point; a line hull contains only the segment. Hull membership uses a fixed, dimensionless `1e-12` tolerance after normalization for numerical boundary handling. It is frozen before any panel reads and is neither fit nor tuned. Nonfinite selected coordinates fail the run; rows are not dropped. No outlier trimming, density estimator, threshold search, or adaptive support rule is used.

## Fixed panel and queries

Use only the 64 source-qualified fitting components, one complete train sequence each, from the accepted R3 sample. Validate the pinned `SAMPLE.json` and `MEMBERSHIP.json` identities and the full 64-fit/32-eval metadata relation before any NPY is opened. Keep the R3 fit component order; report its first and last fixed 32-component halves separately and give every component equal weight in component-level summaries. No evaluation NPY is opened or mapped, and no target values are read. Hash and map only the 64 fit `x` and 64 fit `scored` arrays for selection/integrity: 128 fit-side NPYs total.

R3 applies its training stride as `np.arange(0, len(x), 40)` and then selects rows marked by the original scored mask (`analysis_fit.py`, SHA-256 `e5d353e8f71679f1bbb36b0ebecf3cd438291efc8882ffc6f4fae501f6b7d86e`; `probe_helpers.py`, SHA-256 `f2f222e98b7aad087890d6f211ef8a0aa95b39966cdb0cd109118ea2d4795d68`). This screen preserves that original row alignment and additionally requires `t >= 256`. On the accepted fit masks this yields `t = 280, 320, ..., 19960`: 493 queries per sequence, 31,552 total, and 15,776 in each fixed half. The implementation must assert actual mask-selected counts rather than silently assume them. Every query's history is read only from earlier rows; its current row is excluded.

The panel identity is R3 sample SHA-256 `b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64`, membership SHA-256 `e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217`, and split SHA-256 `c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7`. The R3 contract is pinned at SHA-256 `0bc7876fb88acd3b7f0f6d4f4fe953f5b371867c57609b3f0a1ef26e11397c19`.

## Reporting

Primary outputs are exhaustive class counts/rates per component, pooled query-weighted counts, and equal-component summaries for the full 64 and the two fixed 32-component halves. Record each component's actual query denominator. Also report fixed descriptive hull vertex-count and hull-to-box area-ratio summaries, with undefined ratios for zero-area boxes; these describe geometry only and do not alter class membership. Do not treat the 31,552 overlapping windows as independent replications or report a p-value/interval from rows.

No target arrays, clipped-target moments, official scorer, model, or fit are needed. If the joint-only class is absent, this exact pair/window adds no recent-support cases beyond the box in the exposed panel. If it occurs only in a few components, the finding is concentrated in those sampled components. Presence across both fixed halves would show only that this geometric condition appears in more of the exposed panel; it would motivate, not answer, a later target-usefulness test. No prevalence cutoff or prediction claim is prespecified.

## Minimal implementation basis

Keep the packet to at most three focused runtime modules: `geometry.py` (pure fixed 2D hull/box classification and scalar summaries), `panel_io.py` (R3 metadata validation, source-bound fit-only NPY hashing/header checks/read-only mapping), and `run_probe.py` (fixed CLI, progress and create-only receipts), plus the package initializer. Implement only the small schema/role/order/disjointness checks needed for the accepted R3 v2 metadata; do not import the R1 or R6 readers. The metadata fields and array naming/shape rules are anchored to R3 `sample_panel_v2.py` SHA-256 `89adf7df87a798ff852d96afaaa2289fb0f81cb9f8eb7c2dfd80458f8f6abfc6` and `panel_membership_v2.py` SHA-256 `78158ad1801b1c9a8314a78aeef5098cd5813528848269e434124fc9bf7ef071`. Read/hash only fit `x` and `scored` NPYs; do not open the `y` paths or any evaluation NPY. Use only stdlib and NumPy at runtime; do not import canonical `wnn`, PyArrow, Torch, or any model/scoring package. Pin `data.py` as semantic source evidence, not an imported dependency.

The intended B command remains a single CPU-only bounded diagnostic on physical `research_tmp`: 0.5 CPU, 1.4 GB high / 1.5 GB hard memory, zero swap, 600 seconds. Process one sequence at a time; retain only a 256-point window, hull scratch, and scalar/component counters. The expected selected x payload is about 573.4 MB before NPY headers, plus about 1.3 MB of fit masks; hashing reads those payloads once, while the working geometry is only kilobytes per query. The fixed work is 31,552 hull builds over 256 points: approximately 64.6 million comparison-scale sort operations plus at most 8.1 million edge-membership checks. This is a structural cost estimate, not a measured runtime; admission should fail closed if source, memory, or time headroom is not available.

Before any bounded run, add tiny synthetic tests for the three exhaustive categories; point/line degeneracies and tolerance boundaries; strict past-window exclusion/future-suffix invariance; independent per-axis affine invariance; and fit-only metadata/array access (sentinels must reject y/eval loads). Run full B-shaped import/read sentinels with `np.load` denied and confirm no `wnn`, PyArrow, Torch, or array read occurs during smoke import. Root owns code-freeze review, admission, execution, capture, and publication.

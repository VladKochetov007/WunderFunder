# R10 measured result: joint support adds sparse geometric cases

The fixed two-coordinate hull and its matched marginal box give different classifications on274of31,552 queries (0.8684%). These joint-only cases occur in52of64 fitting components,26in each fixed32-component half. The difference is therefore not confined to one component or one half. This establishes a geometric distinction for the tested pair/window; no labels or model were used, so it establishes no predictive benefit.

| Strict-past support class | Queries | Rate |
|---|---:|---:|
| Outside the past axis-aligned box |299|0.9476%|
| Inside the box, outside the hull |274|0.8684%|
| Inside/on the hull |30,979|98.1840%|

The first fixed half has170joint-only cases in15,776queries (1.0776%); the second has104 (0.6592%). Individual component rates range0–3.4483%. The52nonzero component counts and two halves describe coverage, not independent replications or a statistical stability claim. Windows overlap and the entire F0 research panel is exposed.

The comparison uses raw columns0/52, the first components of the canonical i0/i1 bid-price slices, without a shared physical-unit claim. Every query has the same strict256-row history `[t−256,t)`, and both methods use that history. Queries retain the original fitting stride40 and mask, with `t≥256`:493per sequence,31,552total. There was one fixed coordinate/window/tolerance choice and no retuning. Per-axis min/range normalization is label-blind; the deterministic float64 hull uses a frozen1e-12 cross-product tolerance, including near-collinear turn removal. It is not an exact-real arithmetic claim.

Of the histories,1,308produce point hulls,7,245line hulls and22,999area hulls. Degeneracies and the closed boundary are included in the partition. Input-outlier effects, full-dimensional support, conditional target information and usefulness beyond range/volatility features remain untested. A sparse event can still matter for targets; this target-free screen cannot reject or promote a model on event frequency alone. H-NF19-08 is PARTIAL, with the predictive mechanism still open.

The single CPU invocation `8aa5dbfc71e946188eaac5611a4a8e74` exited0at03:51:26UTC19October. It hashed/mapped only128fit x/mask NPYs from the unchanged R3 panel; target and evaluation arrays, models, training and scorer calls were absent. [Source/result receipts](records/README.md), [frozen design](DESIGN.md), [implementation review](IMPLEMENTATION_REVIEW.json), [compact result audit](RESULT_AUDIT.json) and [resource reconciliation](RESOURCE_RECONCILIATION.json) bind the work. No numerical geometry or data replay is performed by the compact audit.

The helper took30.7740seconds; systemd monotonic wall31.24234seconds and CPU15.644817seconds are separate observations. Parent/local memory-event deltas at capture were0. Raw receipts were byte-validated locally before the exact unit was stopped at03:55:03UTC; it became not-found/inactive/dead. A first read-only capture failed on an omitted empty `ControlGroup` property; reviewed V2 requested `systemctl show --all`. This changed capture only, with no compute rerun. All actual input arrays remain on Beleriand; only source, metadata and scalar summaries were collected.

# Prior evidence index and limits

This folder contains only small public research artifacts needed to interpret the proposal. It contains no dataset payloads, predictions, checkpoints, or credentials.

## Existing geometry evidence

- [Registered investigation and R1 figures](../../../campaign_2026-10-17/moment_geometry_r1/README.md): 128 train and 128 valid sequences were used for raw input scans; all112 columns were visually inspected. The note links the [EDA HTML](../../../campaign_2026-10-17/moment_geometry_r1/eda.html), [candidate arc plot](../../../campaign_2026-10-17/moment_geometry_r1/scan_r1/candidate_arcs.png), [raw support traces](../../../campaign_2026-10-17/moment_geometry_r1/scan_r1/support_traces.png), [prefix paths](../../../campaign_2026-10-17/moment_geometry_r1/scan_r1/prefix_paths.png), [all-column sheets 1–2](../../../campaign_2026-10-17/moment_geometry_r1/scan_r1/all_columns_1.png) and [3–4](../../../campaign_2026-10-17/moment_geometry_r1/scan_r1/all_columns_3.png). They are descriptive and sampled; the screenshot's exact source mapping was not verified.
- [Unified R1 validation note](../../../unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) explains the anonymous encodings, causal prefix reconstruction and small-panel limits. The selected raw channels are (0, 3, 52, 55). The observed mean/variance relation has roughly46% inliers under the robust arc fit. Bounded or discrete support mixtures can generate arc-like moments; this does not prove a universal circle, periodicity, market regime or physical interpretation.
- [R1 and R2 discrete pilot source](../../../campaign_2026-10-17/moment_geometry_r1/README.md) records the tiny paired studies: the moment pilot mean delta was+0.0008833 with opposing exposed-fold effects and conditional component intervals crossing zero. H32×1 and H128×2 four-arm screens used43 fitting and46 exposed evaluation sequences; direction and feature combination effects varied with model capacity. Fixed half-mixtures suggested conditional component value, not a standalone or distillation result.

## Copied R3 results

- [R3 summary](summary_r3.json), copied from the frozen [campaign result](../../../campaign_2026-10-17/geometry_continuation_r3/README.md), SHA256 62344e019256930c35044383ebc2b00295471abb89368432ef43f70ed92d9c54. It reports fresh H128×2, 8 epochs, three paired seeds,43 fitting/46 exposed evaluation sequences across16 components. Mean scores were control0.376941, moments0.357199 (−0.019742 vs control), dictionary0.377442 (+0.000501), combined0.359690 (−0.017251). A fixed control+dictionary half-mixture was+0.007337 over the matched control across these same three seeds. This is a tiny, exposed, schedule-specific screen; it is not a significance test, teacher/student transfer result, or production claim.
- [R3 paired deltas figure](paired_deltas_r3.png), SHA256 f0b1ecff2529e8c0c2052b56574d3caf58de19f7c17df60a5ed50e998faf7f71, and [R3 loss figure](loss_r3.png), SHA256 d58e7d67c09e5f3ca0c7c695f2cb6fd18408a099c32816432e5b042584591eae. The panels are small-sample visual summaries, not independent replications.

## Separate local directional probe

[root's frozen short-probe recipe](root_quick_probe_recipe.json), copied with SHA256 1e7069f3a6889b2a9db8fb6f1033a7f4b22a37e3dcb5ca663fa77b3e55ae6244, describes one exposed F0 seed3 comparison continuing a warm H256×2 parent for one short epoch with60 appended channels versus the same-width zero control. It is root-owned and outside this B300 matrix. It is a direction check only: one seed, one fold, warm-start lineage, and no contribution to the 30 crossed identities. The frozen recipe predates this packet; consult root's terminal receipt before quoting any result.

The explicit unresolved alternative is that appending60 channels changes optimization dynamics or interacts with parameterization and seed noise; a feature-regime effect may not be stable at the larger endpoint. Equal-width indexed zero masks isolate feature block visibility while preserving the input dimension and parameter count, but the four arms still cannot identify physical semantics or guarantee transfer to larger teachers and students.

## Literature search

The repository arXiv helper was run with the query “causal online moments discrete state occupancy time series neural” and a five-result cap. Results covered broad clustering, causal discovery and online causal-model topics; no directly applicable primary method was used to motivate this implementation. The mechanism and predictions here come from the repository's own EDA and remain hypotheses.

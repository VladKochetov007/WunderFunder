# Input-only mean/std EDA receipts

[Parent investigation](../README.md) gives sampling, interpretation and reproduction commands. Seed20261017 selects128complete train and128valid sequences, all112raw inputs; variance uses ddof0. Allvalidation is descriptive and exposed/unknown. No target score is encoded here.

- [moments.npz](moments.npz): compact per-prefix means/variances/extrema with exact source groups and sequence IDs.
- [scan.json](scan.json): input identities, all112descriptive train-fitted single-circle rankings and measured scan duration.
- All-column contact sheets [1](all_columns_1.png), [2](all_columns_2.png), [3](all_columns_3.png), [4](all_columns_4.png): complete112-feature mean/std visual inspection, with train/valid colored separately.
- [candidate_arcs.png](candidate_arcs.png) and [robust_arcs.json](robust_arcs.json): six adaptively selected examples with train-only consensus fits; tolerance0.002variance units, not exact support bounds or predictive evidence.
- [support_probe.json](support_probe.json): near-arc/cloud examples' actual support frequencies and prefix errors. The exact two-point-support explanation fails for inspected near-arc examples.
- [example_traces.npz](example_traces.npz), [support_traces.png](support_traces.png), [prefix_paths.png](prefix_paths.png): compact illustrative raw i0_p0 traces, rendered paths and movement of prefix moments. Selected examples are not prevalence estimates.

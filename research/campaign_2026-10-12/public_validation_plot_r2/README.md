# Local validation versus public outcomes: R2

This visual refinement of R1 plots the same 11 rows in `public_score_regression_r1/candidate_features_r1.csv` that have both a numeric local metric and a scored public outcome. It joins each row to `endpoint_coverage_r1.csv`, checks the two scores and protocol labels, and writes the exact plotted values to `PUBLIC_LOCAL_PLOT_DATA_R2.csv`. Thin neutral-gray leader lines connect each scatter label to its point; the inputs and calculations are unchanged from R1.

Run from the repository root with the existing environment:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=681574400 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-12/public_validation_plot_r2/plot_validation_vs_public_r2.py
```

`PUBLIC_VS_LOCAL_BY_PROTOCOL_R2.png` and `.svg` show four separate validation-protocol facets and a paired `public − local` gap panel. Figure labels use compact model descriptions; the portal IDs remain in the plotted-data CSV for traceability. Colors distinguish exact F1 receipts, a package-reported F1 value, source-reported legacy 368-group values, and conditional EMA6 reconstructions. The two reconstructions use hollow diamonds. The dashed diagonal denotes equal local and public values; no regression or calibration line is fitted.

The latest C result is annotated as public-only (`0.6671`, reported by the user). It has no matched local cross-validation value in these inputs and is not plotted as a paired point.

The 11 rows are not 11 independent model replications. The fold1 receipts use exposed validation data and share ancestry; their local values are float checkpoint/pair predictions rather than exact native Q8 scores. The package-reported student row does not establish checkpoint digest or precision conversion. The seven legacy rows use a different 368-group protocol: five are source-reported with ancestry and package equivalence not receipt-verified, and two reconstruct EMA6 member0 rather than the exact served ensemble. These protocol differences explain the facets and prohibit interpreting one pooled regression or the plotted gaps as a public-score predictor.

`PLOT_RECEIPT_R2.json` records the input/output hashes, protocol counts, gap ranges and limitations. This is a descriptive comparison only; it does not change the candidate ranking or estimate public transfer.

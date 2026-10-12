# Validation and public transfer

This domain distinguishes clean held-out evidence, exposed validation, all-data fitting, and public leaderboard observations.

## Interactive results dashboard

Open [the updated results dashboard](RESULTS_DASHBOARD_20261012.html) for the chronological public climb, local/public comparisons by protocol, current A/B/C runtime plot, measured findings, and searchable experiment and hypothesis ledgers. It includes confidence intervals only where the research records contain them and labels public scores as observed portal outcomes. The self-contained HTML works offline; rebuild it with:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python research/unified/validation/build_results_dashboard.py
```

## Read these sources

- [Data, metric, and validation protocol](../archive/2026-10-12/pages/01-data-and-validation.md)
- [Public calibration and plots](../archive/2026-10-12/pages/05-public-transfer.md)
- [Protocol-separated public/local plot](../../campaign_2026-10-12/public_validation_plot_r2/README.md)
- [Runtime versus public score plot](../../campaign_2026-10-12/public_runtime_quality_r1/README_R1.md)
- [Current competition state](../../../specs/current-state.md)

Keep fold, fitting membership, ancestry, selection exposure, and score uncertainty attached to every point. Public results are transfer observations; they are not a replacement for reliable cross-validation or a basis for adding local deltas to public scores.

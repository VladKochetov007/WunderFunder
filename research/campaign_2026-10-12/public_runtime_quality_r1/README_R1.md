# Public runtime and score comparison (R1)

This figure plots exactly three user-transcribed public portal observations for native-Q8 submissions evaluated on the same competition CPU environment. Runtime is the reported portal execution time; score is the corresponding public score. The rows, portal IDs, ZIP hashes and source basis are preserved in `PUBLIC_RUNTIME_QUALITY_R1.csv`.

| Candidate | Blend | Portal runtime | Public score |
|---|---|---:|---:|
| A | H390 s3/s4 equal (50/50) | 24:31 | 0.6656 |
| B | H390 s3 + Z4 equal (50/50) | 25:56 | 0.6668 |
| C | H390 s3/s4/Z4 (25/25/50) | 36:36 | 0.6671 |

The observed ordering is descriptive: A→B adds 85 seconds and 0.0012 score; B→C adds 640 seconds and 0.0003; A→C adds 725 seconds and 0.0015. These are different blend compositions, so the points cannot isolate an effect of runtime or model count. The vertical axis is zoomed to make the small score differences legible. There is no fitted line, correlation coefficient, confidence interval, or forecast.

The local callback profiles are a separate measurement and are not plotted as portal results. In particular, E's local runtime heuristic is not an observed public point and is excluded. Portal wall time and local callback-only time use different measurement boundaries.

Recreate the figure with the repository Python environment, which already has matplotlib installed (no package installation):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-12/public_runtime_quality_r1/plot_public_runtime_quality_r1.py \
  --csv research/campaign_2026-10-12/public_runtime_quality_r1/PUBLIC_RUNTIME_QUALITY_R1.csv \
  --output-dir research/campaign_2026-10-12/public_runtime_quality_r1
```

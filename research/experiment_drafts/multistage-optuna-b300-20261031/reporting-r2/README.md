# Multistage GitHub reporting R2

This is a new, self-contained source capsule for a later publisher launch. The
deployed reporting-r1 capsule and its published snapshot remain unchanged. This
directory includes the reporting source tree and `SOURCE_MANIFEST.json` binds
every bundled Python file by byte count and SHA-256.

R2 consumes the monitor's `multistage_plot_request.json` as the registered stage
label. It verifies the label against the dashboard run ID, family, canonical
config digest, canonical native metadata digest, source digest, and both split
digests. This lets a Stage3 label-only control retain its Stage3 role with
`kd_weight=0` and no initializer. A label-only view without a valid binding is
omitted until the monitor supplies one; the publisher never defaults it to
Stage1. Stage2/3 fits with explicit initializer/teacher ancestry can still use
that ancestry label.

The snapshot retains each dashboard view. PR progress text distinguishes view
count from completed fit count. A terminal fit key binds canonical run metadata
to the exact native event-journal SHA; only the last terminal status is counted
for that identity, so mirrored endpoint views do not inflate completed fits.

The publisher entry point is `scripts/multistage_github_publish.py`. Use the
existing fixed-origin, private token-file, results-branch, and deadline
contract, with a fresh state-file path for R2. Root owns deployment and external
execution.

The local comparison utility is `../../../../scripts/multistage_report_compare.py`
from this capsule directory. It accepts repeated `--primary-report REPORT.json`
and `--inner-objective objective.json` inputs and writes separate CSV, JSON,
PNG, and SVG outputs. Primary reports are report-only exposed full-valid results
compared to the explicitly unmeasured canonical TL3 reference. Inner objectives
remain in a separate train-only proxy table and plot, with no TL3 line or
cross-scope ranking.

Focused source checks:

```powershell
python scripts/multistage_github_test.py
python -m unittest scripts.multistage_report_compare_test
```

The fixtures validate protocol and output behavior; they are not campaign
results. No external API request, data download, model fit, or GPU use is part
of these checks.

# Public results: seed 4 reaches 0.6660

The user supplied the following portal results. Archive hashes match the local
ZIPs; scores and runtime are user-reported portal evidence, not an API lookup.
Times are displayed as Kyiv clock values for the requested timeline; the portal
timezone was not explicitly confirmed.

| Package | Portal time, Oct 4 | Public WP | Runtime | Memory | Submission |
| --- | --- | ---: | --- | ---: | --- |
| Seed-3 combo, equal parent/head | 14:58 | 0.6655 | 37m47s | 625 MB | `10tq248b` |
| Seed-3 parent endpoint | 23:47 | 0.6641 | 33m57s | 624 MB | `0cawau9i` |
| Seed-3 fine-tuned head endpoint | 23:47 | 0.6654 | 34m01s | 624 MB | `0np7c273` |
| **Seed-4 combo, equal parent/head** | **23:48** | **0.6660** | **37m07s** | **626 MB** | **`2e5j6bnt`** |

**The numerical target is reached at the portal's four-decimal precision.**
Seed 4 improves by 0.0005 over seed 3 and 0.0019 over the old 0.6641 D256
incumbent. The improvement from the rescaled 0.6503 starting point is 0.0157.
The previously cited leader was 0.6677, a historical reference rather than a
current leaderboard check. Research remains paused; this report starts no work.

## What the endpoint comparison actually establishes

The final head procedure matters for this deployed recipe: the head endpoint
gains **0.0013** over the fixed parent. The fixed equal parent/head mixture
scores **0.0001** above the head alone, despite the weaker parent endpoint.
Correlation of a mixture is nonlinear; endpoint scores do not average to its
score. The 0.0001 difference is too small to establish a general preference for
these blend weights or justify tuning them on the public leaderboard.

The parent ties the *complete* old D256 package's 0.6641 public point. Those
packages differ in finishing procedures, so this does **not** establish that
Muon, chunk1000 or bootstrap individually have no public benefit. It confirms
the review's warning: forecast and compare complete submitted predictors.
Two fixed full-data combo seeds now outperform the prior public incumbent;
that supports recipe transfer more directly than extrapolating local deltas.
It still does not isolate the package's optimization or credit mechanisms.

The smaller head/parent endpoints ran about 3m46s /3m50s faster than the equal
mixture. They offer measured score/runtime alternatives. Seed 4 itself fits
the runtime limit with substantial margin; no speculative engine speedup is
needed to deploy the current best.

## Validation is useful, but its level is not a public forecast

All local numbers below are previously saved FP32/cache reports. They are not
new fits or rescored predictions. The complete supplied validation file has
1,873 sequences, including 1,505 used for final fitting. The excluded 368 were
repeatedly explored. Only the seed-3 package has a separately measured actual
native callback score over the complete file: 0.700503766.

| Fixed predictor | Full supplied validation | Excluded 368 | Public |
| --- | ---: | ---: | ---: |
| Previous D256 | 0.700792587 | 0.697777394 | 0.6641 |
| Combo seed 3 | 0.700503744 | 0.699822159 | 0.6655 |
| **Combo seed 4** | **0.700771292** | **0.701694557** | **0.6660** |
| Fixed equal seed-3/seed-4 bag | 0.701106873 | 0.701420130 | Unsubmitted |

The full-file score ranks old D256 slightly above seed 4, while public ranks
seed 4 higher by 0.0019. That fitted aggregate would have discouraged a useful
submission. The excluded panel gets this direction right, but it does not
calibrate the magnitude: seed 4 minus seed 3 is **+0.001872 locally versus
+0.0005 publicly**. Its paired component interval includes zero, conditional
on the fitted models; the public result does not turn that interval into a
training-seed uncertainty estimate. Neither a universal transfer slope nor a
narrow public confidence interval follows from these points.

The fixed seed bag has no public score and its excluded-panel point is below
seed 4. Do not add seed gains or forecast a bag improvement from the full-file
score alone. The complete-recipe CV result remains **0.693658483**, matched
control **0.691244104**, from the existing seed-3 fold models. There is no new
seed-4 CV result.

## Packaging failure and research reflection

The q8r upload `1yv88h4s` failed before scoring because its reused host binary
linked `libpython3.13.so.1.0`. This is a deployment defect, not negative
forecasting evidence. Both original q8 variants are withdrawn. A packaging-only
rebuild of the exact frozen source passes offline Python 3.11 checks with
bitwise archived-output equivalence, but its separate runtime requirement is
still pending. See [the incident](E462_PACKAGING_INCIDENT_20261004.md).

The campaign now has a useful submission and a measured target result, while
many fundamental questions remain open. On explicit resumption, anchor new
comparisons to the fixed seed-4 package; keep matched complete-predictor
controls for cadence/EMA/loss work, and preserve the planned volume, clock,
cross-fit teacher and compression variations. These uploads do not reject
different implementations of those ideas. Treat runtime gains as measured
deployment outcomes and mechanism claims as questions requiring their own
controls. Keep public selection and historical-fold exposure visible.

## Reproducible visualization and records

The [interactive offline HTML](eda/leaderboard_climb/leaderboard_climb.html)
contains 24 successful own historical submissions, six teammate context
points, two historical forecasts and the failed q8 event. It supports a
single shared WP axis, public/detail toggles, chronological-order or calendar
spacing, package search, grouped tables and JSON/CSV downloads. Raw validation
and historical expected-public values are separate labelled series. Missing
validation values are not invented, and historical forecast timing/calibration
limitations are shown.

The machine-readable [timeline](eda/leaderboard_climb/leaderboard_history.json)
binds source hashes. Rebuild with `python3 scripts/leaderboard_climb.py`. The
existing marimo notebook also receives the saved timeline and reporting figure.

Raw portal associations: [PORTAL_RESULTS_20261004_LATE.json](PORTAL_RESULTS_20261004_LATE.json).
Public registry: [registry.json](../lb/registry.json).
Local scores: [VALIDATION_SCORECARD.md](VALIDATION_SCORECARD.md),
[seed replication](E450_SEED4_PAIRED_REPORT.md), and E450's saved scorecard.
Historical calibration moments come from `research/lb/holdout_stats.json`;
these are descriptive reporting evidence, not untouched holdout certification.

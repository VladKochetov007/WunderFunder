> Extracted from the final October 12 handoff. Its original SHA-256 was `e033f8d52f4d4444e5f0c5fa006c2b4f56b6e82f7187e2ed9fb1b74b20e6f704`. See [archive index](../README.md) and [split manifest](../SPLIT_MANIFEST.json). Local links were rebased for this page.

## Data, metric, and validation

The task is causal sequential prediction over supplied time-series sequences.
Predictions must be deterministic and reset state between sequences. The
official implementation is
[src/wnn/gmetric.py](../../../../../src/wnn/gmetric.py); the metric identifier is
global_wp_clip2_v1. It applies the fixed score/prediction mask, casts
predictions and labels through float32, clips each target to ±2, weights each
selected row by the absolute clipped true label, computes weighted Pearson
correlation pooled over selected rows separately for both targets, and averages
the two target correlations. Warm-up rows are excluded by the official mask
(99 rows for the scored sequence setup). Training-target clipping and teacher
mixing are separate recipe choices; they never change official scoring.

Groups are identified by source as well as row-group identity and connected
components preserve related sequences. Splits and bootstraps operate on whole
groups/components, not rows. This prevents within-sequence history or related
group leakage, but repeatedly inspected F0/F1/F2/F3 panels remain exposed.
Bootstrap intervals on fixed predictions measure evaluation-component
sampling only; they exclude fitting-seed variation, recipe variation, endpoint
selection exposure, and public-transfer uncertainty.

The clean-F0 student fit used 10,535 train-source groups and 1,405 valid-source
fit groups, with 468 valid-source evaluation groups excluded from fitting. The
separate prediction export covers all 1,873 valid-source groups, while the
official F0 score uses its fixed 468-group subset. F0 score receipts refer to
141 connected components. F1 comparison receipts use 457 groups and 149
components. Full sample membership, fingerprints, prediction hashes, and
official-scoring commands are in the endpoint-specific manifests linked below.
The final all-data H390/Z endpoints used validation rows for training or
head-fitting and therefore cannot be assigned held-out quality from those
all-data weights.
